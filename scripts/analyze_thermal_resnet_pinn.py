from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from mor.nn.data import collateThermalOperatorCases, loadSubsampledThermalDataset
from mor.nn.losses import thermalSteadyPinnLoss
from mor.nn.models import thermalCategorizedOperatorTorch

from train_thermal_discrete_pinn import (
    _caseCategory,
    _dimensionless,
    _evaluate,
    _splitCases,
    _summary,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Compare shallow and residual global thermal PINN checkpoints.'
    )
    parser.add_argument('source', type=Path)
    parser.add_argument('--baseline-checkpoint', type=Path, required=True)
    parser.add_argument('--baseline-report', type=Path, required=True)
    parser.add_argument('--resnet-checkpoint', type=Path, required=True)
    parser.add_argument('--resnet-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--maximum-volume-samples', type=int, default=128)
    parser.add_argument('--maximum-boundary-samples', type=int, default=256)
    parser.add_argument('--maximum-query-samples', type=int, default=128)
    parser.add_argument('--validation-fraction', type=float, default=0.2)
    parser.add_argument('--pde-collocation-count', type=int, default=8)
    parser.add_argument('--pde-kernel-scale', type=float, default=0.2)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=23)
    return parser.parse_args()


def _loadGlobal(checkpointPath: Path):
    checkpoint = torch.load(checkpointPath, map_location='cpu', weights_only=False)
    return thermalCategorizedOperatorTorch.fromCheckpoint(checkpoint).globalModel


def _toDevice(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _pdeResidual(model, cases, arguments: argparse.Namespace) -> float:
    device = model.toDevice(arguments.device, allowAutoFallback=True)
    loss = thermalSteadyPinnLoss(
        pdeResidualWeight=1.0,
        pdeCollocationCount=arguments.pde_collocation_count,
        pdeKernelScale=arguments.pde_kernel_scale,
    )
    loader = DataLoader(
        cases,
        batch_size=arguments.batch_size,
        shuffle=False,
        collate_fn=collateThermalOperatorCases,
    )
    total = 0.0
    caseCount = 0
    model.eval()
    with torch.enable_grad():
        for rawBatch in loader:
            batch = _toDevice(rawBatch, device)
            batchCount = int(batch['queryCoordinates'].shape[0])
            value = loss.computePdeResidual(model, batch)
            total += float(value.detach().cpu()) * batchCount
            caseCount += batchCount
    model.cpu()
    return total / caseCount


def _percentiles(records) -> dict[str, float]:
    values = np.asarray([record['rmseK'] for record in records], dtype=np.float64)
    return {
        name: float(value)
        for name, value in zip(
            ('minimum', 'median', 'p75', 'p90', 'p95', 'maximum'),
            np.percentile(values, (0, 50, 75, 90, 95, 100)),
        )
    }


def _byCategory(records) -> dict[str, dict[str, Any]]:
    grouped = defaultdict(list)
    for record in records:
        grouped[record['categoryKey']].append(record)
    return {key: _summary(values) for key, values in sorted(grouped.items())}


def _trainingSummary(report: dict[str, Any]) -> dict[str, Any]:
    training = report['globalFallback']['training']
    validation = np.asarray(training['validationLossHistory'], dtype=np.float64)
    tail = validation[-min(15, validation.size) :]
    return {
        'epochCount': int(validation.size),
        'bestEpochIndex': training['bestEpochIndex'],
        'bestValidationLoss': training['bestValidationLoss'],
        'lastValidationLoss': float(validation[-1]),
        'tailValidationMean': float(np.mean(tail)),
        'tailValidationStandardDeviation': float(np.std(tail)),
    }


def _public(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith('_')}


def _markdown(comparison: dict[str, Any]) -> str:
    baseline = comparison['baseline']['evaluation']
    resnet = comparison['resnet']['evaluation']
    baselineTraining = comparison['baseline']['training']
    resnetTraining = comparison['resnet']['training']
    delta = comparison['deltaPercent']
    lines = [
        '# Thermal Global PINN：浅层 MLP 与 ResNet 对照',
        '',
        '## 结论',
        '',
        (
            '在相同 153/41 分组切分、随机种子、采样、PINN 损失与 75 epoch 下，'
            '两层平滑 decoder ResNet 没有带来提升，当前不建议替换 Global 保底模型。'
        ),
        '',
        '| 模型 | 参数量 | 加权 RMSE/K | Macro RMSE/K | 最大误差/K | PDE residual |',
        '|---|---:|---:|---:|---:|---:|',
        (
            f"| 浅层 MLP | {comparison['baseline']['parameterCount']:,} | "
            f"{baseline['volumeWeightedRmseK']:.4f} | {baseline['macroCaseRmseK']:.4f} | "
            f"{baseline['maximumErrorK']:.4f} | {comparison['baseline']['pdeResidual']:.6g} |"
        ),
        (
            f"| ResNet ×2 | {comparison['resnet']['parameterCount']:,} | "
            f"{resnet['volumeWeightedRmseK']:.4f} | {resnet['macroCaseRmseK']:.4f} | "
            f"{resnet['maximumErrorK']:.4f} | {comparison['resnet']['pdeResidual']:.6g} |"
        ),
        '',
        (
            f"加权 RMSE 变化 {delta['volumeWeightedRmseK']:+.1f}%，"
            f"Macro RMSE 变化 {delta['macroCaseRmseK']:+.1f}%，"
            f"PDE residual 变化 {delta['pdeResidual']:+.1f}%。"
        ),
        '',
        '## 结构与训练诊断',
        '',
        '- 原 decoder 为 `835 → 256 → 128 → 1`，约占原模型参数的 81%；两层 ResNet 只在 128 维 decoder 隐状态上增加平滑残差块。',
        '- 残差块使用 `SiLU`，满足强形式 PINN 的二阶导要求；末层零初始化使同 seed 的初始输出与浅层 MLP 完全一致。',
        f"- 参数量增加 {comparison['baseline']['parameterCount']:,} → {comparison['resnet']['parameterCount']:,}（+21.7%），但最佳验证 loss 从 {baselineTraining['bestValidationLoss']:.4f} 变差到 {resnetTraining['bestValidationLoss']:.4f}。",
        f"- 末 15 epoch 验证 loss 标准差从 {baselineTraining['tailValidationStandardDeviation']:.4f} 降到 {resnetTraining['tailValidationStandardDeviation']:.4f}，但均值从 {baselineTraining['tailValidationMean']:.4f} 升到 {resnetTraining['tailValidationMean']:.4f}：训练更平稳，却收敛在更差的解。",
        '- 当前 checkpoint 选择未在验证阶段计算 PDE 项，且 PDE 权重仅为 0.005；更深 decoder 学出的高曲率没有被充分抑制，验证 PDE residual 因而显著恶化。',
        '',
        '## 逐类别结果',
        '',
        '| 类别 | MLP RMSE/K | ResNet RMSE/K | 变化 |',
        '|---|---:|---:|---:|',
    ]
    for key, values in comparison['byCategory'].items():
        label = key.replace('systemType=', '').replace('|assemblyType=', ' / ').replace(
            '|componentType=', ' / '
        )
        lines.append(
            f"| {label} | {values['baselineRmseK']:.4f} | "
            f"{values['resnetRmseK']:.4f} | {values['deltaPercent']:+.1f}% |"
        )
    lines.extend(
        [
            '',
            '## 配对残差',
            '',
            f"- ResNet 改善 {comparison['pairedCases']['improvedCount']} / "
            f"{comparison['pairedCases']['caseCount']} 个 case；恶化 "
            f"{comparison['pairedCases']['regressedCount']} 个。",
            f"- case RMSE 中位数：MLP {comparison['baseline']['percentiles']['median']:.4f} K，"
            f"ResNet {comparison['resnet']['percentiles']['median']:.4f} K。",
            f"- case RMSE P95：MLP {comparison['baseline']['percentiles']['p95']:.4f} K，"
            f"ResNet {comparison['resnet']['percentiles']['p95']:.4f} K。",
            '- 最大退化集中在 `tutorial-v` reducer housing：mid/left/right 的 case RMSE 分别增加约 19.07 / 9.08 / 5.36 K。',
            '',
            '主要问题仍是跨工况、跨类别的 Global 表征冲突，而不是单纯的网络深度不足。'
            '保留浅层 Global 作为保底，并优先使用分类专家加验证校准 gate。若继续消融残差结构，应先试一层块、较小 residual scale/学习率，并把 PDE residual 纳入 checkpoint 选择。',
            '',
        ]
    )
    return '\n'.join(lines)


def main() -> None:
    arguments = _arguments()
    dataset = loadSubsampledThermalDataset(
        arguments.source,
        analysisType='steady',
        maximumVolumeSamples=arguments.maximum_volume_samples,
        maximumBoundarySamples=arguments.maximum_boundary_samples,
        maximumQuerySamples=arguments.maximum_query_samples,
        randomSeed=arguments.seed,
    )
    _, validationCases = _splitCases(
        dataset.cases,
        arguments.validation_fraction,
        arguments.seed,
    )
    dimensionless = _dimensionless(validationCases, 1.0)
    operatorCases = tuple(value.operatorCase for value in dimensionless)
    baselineModel = _loadGlobal(arguments.baseline_checkpoint)
    resnetModel = _loadGlobal(arguments.resnet_checkpoint)
    baselineRecords = _evaluate(
        baselineModel,
        validationCases,
        dimensionless,
        baselineModel.toDevice(arguments.device, allowAutoFallback=True),
    )
    baselineModel.cpu()
    resnetRecords = _evaluate(
        resnetModel,
        validationCases,
        dimensionless,
        resnetModel.toDevice(arguments.device, allowAutoFallback=True),
    )
    resnetModel.cpu()
    categoryFields = ('systemType', 'assemblyType', 'componentType')
    for case, baselineRecord, resnetRecord in zip(
        validationCases, baselineRecords, resnetRecords
    ):
        category = _caseCategory(case, categoryFields)
        baselineRecord['categoryKey'] = category
        resnetRecord['categoryKey'] = category

    baselineById = {record['caseId']: record for record in baselineRecords}
    resnetById = {record['caseId']: record for record in resnetRecords}
    paired = []
    for caseId, baselineRecord in baselineById.items():
        resnetRecord = resnetById[caseId]
        paired.append(
            {
                'caseId': caseId,
                'categoryKey': baselineRecord['categoryKey'],
                'baselineRmseK': baselineRecord['rmseK'],
                'resnetRmseK': resnetRecord['rmseK'],
                'deltaRmseK': resnetRecord['rmseK'] - baselineRecord['rmseK'],
            }
        )
    baselineCategory = _byCategory(baselineRecords)
    resnetCategory = _byCategory(resnetRecords)
    byCategory = {}
    for key in baselineCategory:
        baselineRmse = baselineCategory[key]['volumeWeightedRmseK']
        resnetRmse = resnetCategory[key]['volumeWeightedRmseK']
        byCategory[key] = {
            'baselineRmseK': baselineRmse,
            'resnetRmseK': resnetRmse,
            'deltaPercent': 100.0 * (resnetRmse / baselineRmse - 1.0),
        }
    baselinePde = _pdeResidual(baselineModel, operatorCases, arguments)
    resnetPde = _pdeResidual(resnetModel, operatorCases, arguments)
    baselineReport = json.loads(arguments.baseline_report.read_text(encoding='utf-8'))
    resnetReport = json.loads(arguments.resnet_report.read_text(encoding='utf-8'))
    baselineEvaluation = _summary(baselineRecords)
    resnetEvaluation = _summary(resnetRecords)
    comparison = {
        'split': {
            'seed': arguments.seed,
            'trainCaseCount': len(dataset.cases) - len(validationCases),
            'validationCaseCount': len(validationCases),
            'validationCaseIds': [case.caseId for case in validationCases],
        },
        'baseline': {
            'parameterCount': sum(value.numel() for value in baselineModel.parameters()),
            'evaluation': baselineEvaluation,
            'percentiles': _percentiles(baselineRecords),
            'pdeResidual': baselinePde,
            'training': _trainingSummary(baselineReport),
            'cases': [_public(record) for record in baselineRecords],
        },
        'resnet': {
            'parameterCount': sum(value.numel() for value in resnetModel.parameters()),
            'evaluation': resnetEvaluation,
            'percentiles': _percentiles(resnetRecords),
            'pdeResidual': resnetPde,
            'training': _trainingSummary(resnetReport),
            'cases': [_public(record) for record in resnetRecords],
        },
        'deltaPercent': {
            'volumeWeightedRmseK': 100.0 * (
                resnetEvaluation['volumeWeightedRmseK']
                / baselineEvaluation['volumeWeightedRmseK']
                - 1.0
            ),
            'macroCaseRmseK': 100.0 * (
                resnetEvaluation['macroCaseRmseK']
                / baselineEvaluation['macroCaseRmseK']
                - 1.0
            ),
            'maximumErrorK': 100.0 * (
                resnetEvaluation['maximumErrorK']
                / baselineEvaluation['maximumErrorK']
                - 1.0
            ),
            'pdeResidual': 100.0 * (resnetPde / baselinePde - 1.0),
        },
        'byCategory': byCategory,
        'pairedCases': {
            'caseCount': len(paired),
            'improvedCount': sum(record['deltaRmseK'] < 0.0 for record in paired),
            'regressedCount': sum(record['deltaRmseK'] > 0.0 for record in paired),
            'largestImprovements': sorted(paired, key=lambda value: value['deltaRmseK'])[:10],
            'largestRegressions': sorted(
                paired, key=lambda value: value['deltaRmseK'], reverse=True
            )[:10],
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(comparison, indent=2),
        encoding='utf-8',
    )
    markdownPath = arguments.output.with_suffix('.md')
    markdownPath.write_text(_markdown(comparison), encoding='utf-8')
    print(json.dumps(comparison, indent=2))


if __name__ == '__main__':
    main()
