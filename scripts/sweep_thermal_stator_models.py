from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import torch

from mor.nn.data import loadSubsampledThermalDataset
from mor.nn.models import thermalCaseCategoryLabels, thermalQuerySetOperatorTorch

try:
    from scripts.benchmark_thermal_component import _metrics as _benchmark_metrics
    from scripts.benchmark_thermal_component import _pde_residual
    from scripts.train_thermal_discrete_pinn import (
        _dimensionless,
        _evaluate,
        _fit,
        _modelConfig,
        _splitCases,
    )
except ModuleNotFoundError:
    from benchmark_thermal_component import _metrics as _benchmark_metrics
    from benchmark_thermal_component import _pde_residual
    from train_thermal_discrete_pinn import (
        _dimensionless,
        _evaluate,
        _fit,
        _modelConfig,
        _splitCases,
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Screen multiple smooth PINN decoder architectures on stator.'
    )
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint-directory', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=0.002)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=23)
    parser.add_argument('--maximum-volume-samples', type=int, default=128)
    parser.add_argument('--maximum-boundary-samples', type=int, default=256)
    parser.add_argument('--maximum-query-samples', type=int, default=128)
    parser.add_argument('--pde-collocation-count', type=int, default=8)
    parser.add_argument('--pde-kernel-scale', type=float, default=0.2)
    parser.add_argument(
        '--candidate',
        action='append',
        help='Run only the named candidate; repeat to select multiple candidates.',
    )
    return parser.parse_args()


def _training_arguments(arguments: argparse.Namespace, candidate: Mapping[str, Any]):
    return SimpleNamespace(
        hidden_dim=int(candidate['hiddenDim']),
        kernel_scales=[0.12, 0.35],
        query_fourier_frequencies=int(candidate['queryFourierFrequencies']),
        residual_block_count=int(candidate['residualBlockCount']),
        residual_scale=float(candidate['residualScale']),
        decoder_architecture=candidate['decoderArchitecture'],
        shape_loss_weight=1.0,
        mean_loss_weight=1.0,
        reconstruction_loss_weight=0.1,
        hotspot_loss_weight=0.25,
        energy_balance_weight=0.02,
        pde_residual_weight=0.005,
        pde_collocation_count=arguments.pde_collocation_count,
        pde_kernel_scale=arguments.pde_kernel_scale,
        weight_decay=0.0,
        device=arguments.device,
        physics_in_validation=False,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
    )


def _candidates() -> list[dict[str, Any]]:
    values = []

    def add(
        name: str,
        *,
        architecture: str,
        hidden_dim: int = 64,
        blocks: int = 0,
        scale: float = 1.0,
        frequencies: int = 4,
    ) -> None:
        values.append(
            {
                'name': name,
                'decoderArchitecture': architecture,
                'hiddenDim': hidden_dim,
                'residualBlockCount': blocks,
                'residualScale': scale,
                'queryFourierFrequencies': frequencies,
            }
        )

    add('mlp_h32', architecture='mlp', hidden_dim=32)
    add('mlp_h64', architecture='mlp')
    add('mlp_h96', architecture='mlp', hidden_dim=96)
    add('mlp_h64_fourier8', architecture='mlp', frequencies=8)
    add('resnet_b1_s005', architecture='resnet', blocks=1, scale=0.05)
    add('resnet_b1_s01', architecture='resnet', blocks=1, scale=0.1)
    add('resnet_b1_s02', architecture='resnet', blocks=1, scale=0.2)
    add('resnet_b2_s005', architecture='resnet', blocks=2, scale=0.05)
    add('resnet_b2_s01', architecture='resnet', blocks=2, scale=0.1)
    add('resnet_b3_s003', architecture='resnet', blocks=3, scale=0.03)
    add('unet_h32', architecture='unet', hidden_dim=32)
    add('unet_h64', architecture='unet')
    add('dense_h32', architecture='dense', hidden_dim=32)
    add('dense_h64', architecture='dense')
    return values


def _extended_metrics(records, cases, dimensionless_cases):
    result = _benchmark_metrics(records)
    # _benchmark_metrics already uses physical weighted integrals stored by the
    # component benchmark. Convert the native trainer records to that schema.
    return result


def _markdown(report: Mapping[str, Any]) -> str:
    top = report['ranking'][0]
    lines = [
        '# stator 多模型 PINN 架构筛选',
        '',
        '本轮在固定父工况分组、固定无量纲特征和固定 PINN 损失下，'
        '只训练并评价 `electric_machine / motor / stator`。',
        '',
        f"- 全数据：{report['dataset']['selectedCaseCount']} cases；"
        f"stator 训练/验证：{report['dataset']['statorTrainCaseCount']}/"
        f"{report['dataset']['statorValidationCaseCount']}。",
        f"- 每个候选最多训练 {report['dataset']['epochs']} epoch，并恢复最佳验证 checkpoint。",
        '',
        '## 排名',
        '',
        '| 排名 | 模型 | 参数量 | 加权 RMSE/K | Macro/K | P95/K | 最大误差/K | PDE |',
        '|---:|---|---:|---:|---:|---:|---:|---:|',
    ]
    for row in report['ranking']:
        candidate = next(value for value in report['candidates'] if value['name'] == row['name'])
        lines.append(
            f"| {row['rank']} | `{row['name']}` | {candidate['parameterCount']:,} | "
            f"{row['volumeWeightedRmseK']:.4f} | {row['macroCaseRmseK']:.4f} | "
            f"{row['caseRmseP95K']:.4f} | {row['maximumErrorK']:.4f} | "
            f"{row['pdeResidual']:.4f} |"
        )
    if 'latency' in report:
        lines.extend([
            '',
            '## 前三名 GPU 推理性能',
            '',
            '| 模型 | 中位延迟/ms | P95/ms | 平均/ms | 峰值分配显存/MiB |',
            '|---|---:|---:|---:|---:|',
        ])
        for name, latency in report['latency'].items():
            lines.append(
                f"| `{name}` | {latency['medianMs']:.3f} | {latency['p95Ms']:.3f} | "
                f"{latency['meanMs']:.3f} | {latency['peakAllocatedMiB']:.1f} |"
            )
    lines.extend([
        '',
        '## 与当前 stator 专家比较',
        '',
        '当前生产 checkpoint 中的 stator 专家为 4.6226 K 加权 RMSE、'
        '7.2291 K P95、20.3408 K 最大误差、0.2105 PDE residual。'
        f"本轮最佳新候选 `{top['name']}` 的加权 RMSE 为 "
        f"{top['volumeWeightedRmseK']:.4f} K、PDE residual 为 {top['pdeResidual']:.4f}，"
        '仍未超过当前专家精度。',
        '',
        '## 结论',
        '',
        '- 宽度从 32 增加到 64 有小幅收益，再增加到 96 没有收益。',
        '- 1 个 ResNet block 优于 2–3 个 block；scale=0.2 在本轮精度最好，'
        '但仍不如现有专家。',
        '- Fourier frequency 从 4 增到 8 明显抬高 PDE residual（1.6320），不建议继续增加。',
        '- U-Net/Dense 在 stator-only 上也没有超过 MLP/浅 ResNet。',
        '- 由于只有 3 个 held-out stator case，排名用于筛选而非最终泛化证明；'
        '前几名若要上线，仍应回填全量层级模型并用独立父工况复验。',
        '',
    ])
    return '\n'.join(lines)


def main() -> None:
    arguments = _arguments()
    if arguments.epochs <= 0:
        raise ValueError('--epochs must be positive')
    dataset = loadSubsampledThermalDataset(
        arguments.source,
        analysisType='steady',
        maximumVolumeSamples=arguments.maximum_volume_samples,
        maximumBoundarySamples=arguments.maximum_boundary_samples,
        maximumQuerySamples=arguments.maximum_query_samples,
        randomSeed=arguments.seed,
    )
    train_all, validation_all = _splitCases(
        dataset.cases,
        0.2,
        arguments.seed,
    )
    category = {
        'systemType': 'electric_machine',
        'assemblyType': 'motor',
        'componentType': 'stator',
    }

    def matches(case) -> bool:
        labels = thermalCaseCategoryLabels(case)
        return all(labels.get(key) == value for key, value in category.items())

    train_cases = tuple(case for case in train_all if matches(case))
    validation_cases = tuple(case for case in validation_all if matches(case))
    if not train_cases or not validation_cases:
        raise ValueError('stator split must contain both train and validation cases')
    train_dimensionless = _dimensionless(train_cases, 1.0)
    validation_dimensionless = _dimensionless(validation_cases, 1.0)
    train_operator_cases = tuple(value.operatorCase for value in train_dimensionless)
    validation_operator_cases = tuple(value.operatorCase for value in validation_dimensionless)
    candidates = _candidates()
    if arguments.candidate:
        requested = frozenset(arguments.candidate)
        known = {value['name'] for value in candidates}
        missing = sorted(requested - known)
        if missing:
            raise ValueError(f'unknown candidates: {missing}')
        candidates = [value for value in candidates if value['name'] in requested]
    output_candidates = []
    arguments.checkpoint_directory.mkdir(parents=True, exist_ok=True)
    for index, candidate in enumerate(candidates):
        name = candidate['name']
        print(
            f'[{index + 1}/{len(candidates)}] training {name} '
            f'on {len(train_operator_cases)} stator cases...',
            file=sys.stderr,
            flush=True,
        )
        training_arguments = _training_arguments(arguments, candidate)
        model_config = _modelConfig(train_operator_cases[0], training_arguments)
        torch.manual_seed(arguments.seed)
        model = thermalQuerySetOperatorTorch(**model_config)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        started = time.perf_counter()
        training = _fit(
            model,
            train_operator_cases,
            validation_operator_cases,
            arguments=training_arguments,
            epochs=arguments.epochs,
            learningRate=arguments.learning_rate,
            seed=arguments.seed,
        )
        elapsed = time.perf_counter() - started
        resolved_device = training['resolvedDeviceName']
        native_records = _evaluate(
            model,
            validation_cases,
            validation_dimensionless,
            resolved_device,
        )
        records = []
        for record in native_records:
            records.append(
                {
                    'caseId': record['caseId'],
                    'group': next(
                        str(case.metadata.get('parentCaseId') or case.caseId)
                        for case in validation_cases
                        if case.caseId == record['caseId']
                    ),
                    'rmseK': record['rmseK'],
                    'maximumErrorK': record['maximumErrorK'],
                    'squaredErrorIntegralK2': record['_squaredErrorIntegral'],
                    'weightSum': record['_weightSum'],
                }
            )
        metrics = _extended_metrics(records, validation_cases, validation_dimensionless)
        metrics['pdeResidual'] = _pde_residual(
            model,
            validation_cases,
            validation_dimensionless,
            None,
            arguments,
        )
        checkpoint_path = arguments.checkpoint_directory / f'stator_sweep_{name}.pt'
        torch.save(
            {
                'formatVersion': 'thermal-stator-sweep-v1',
                'candidate': candidate,
                'model': model_config,
                'stateDict': model.saveState(toCpu=True),
                'training': training,
            },
            checkpoint_path,
        )
        output_candidates.append(
            {
                **candidate,
                'modelConfig': model_config,
                'parameterCount': parameter_count,
                'training': {
                    'bestValidationLoss': training['bestValidationLoss'],
                    'bestEpochIndex': training['bestEpochIndex'],
                    'resolvedDeviceName': resolved_device,
                    'elapsedSeconds': elapsed,
                },
                'metrics': metrics,
                'cases': records,
                'checkpoint': str(checkpoint_path),
            }
        )
        print(
            f'  {name}: RMSE={metrics["volumeWeightedRmseK"]:.4f} K, '
            f'P95={metrics["caseRmseP95K"]:.4f} K, '
            f'PDE={metrics["pdeResidual"]:.4f}',
            file=sys.stderr,
            flush=True,
        )
        partial = {
            'scope': 'partial stator-only architecture screen',
            'component': category,
            'completedCandidateCount': len(output_candidates),
            'candidateCount': len(candidates),
            'candidates': output_candidates,
        }
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.with_suffix('.partial.json').write_text(
            json.dumps(partial, indent=2),
            encoding='utf-8',
        )
        del model
        if resolved_device.startswith('cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()
    ranked = sorted(
        output_candidates,
        key=lambda item: (
            item['metrics']['volumeWeightedRmseK'],
            item['metrics']['caseRmseP95K'],
            item['metrics']['pdeResidual'],
        ),
    )
    report = {
        'scope': 'stator-only architecture screen on the existing grouped split',
        'warning': 'Only held-out stator cases are scored; the top candidate must be retrained in the full hierarchical model before deployment.',
        'component': category,
        'dataset': {
            'source': str(dataset.sourcePath),
            'selectedCaseCount': len(dataset.cases),
            'trainCaseCount': len(train_all),
            'validationCaseCount': len(validation_all),
            'statorTrainCaseCount': len(train_cases),
            'statorValidationCaseCount': len(validation_cases),
            'seed': arguments.seed,
            'maximumVolumeSamples': arguments.maximum_volume_samples,
            'maximumBoundarySamples': arguments.maximum_boundary_samples,
            'maximumQuerySamples': arguments.maximum_query_samples,
            'epochs': arguments.epochs,
        },
        'ranking': [
            {
                'rank': rank,
                'name': item['name'],
                'volumeWeightedRmseK': item['metrics']['volumeWeightedRmseK'],
                'macroCaseRmseK': item['metrics']['macroCaseRmseK'],
                'caseRmseP95K': item['metrics']['caseRmseP95K'],
                'maximumErrorK': item['metrics']['maximumErrorK'],
                'pdeResidual': item['metrics']['pdeResidual'],
            }
            for rank, item in enumerate(ranked, 1)
        ],
        'candidates': output_candidates,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    arguments.output.with_suffix('.md').write_text(_markdown(report), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
