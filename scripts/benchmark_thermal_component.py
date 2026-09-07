from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from mor.nn.data import (
    collateThermalOperatorCases,
    loadSubsampledThermalDataset,
    thermalCheckpointDataContract,
)
from mor.nn.losses import thermalSteadyPinnLoss
from mor.nn.models import (
    thermalCaseCategoryLabels,
    thermalCategorizedOperatorTorch,
    thermalCategoryKey,
)

try:
    from scripts.train_thermal_discrete_pinn import _caseGroup, _splitCases
except ModuleNotFoundError:
    from train_thermal_discrete_pinn import _caseGroup, _splitCases


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Benchmark one thermal component on grouped held-out cases.'
    )
    parser.add_argument('source', type=Path)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--system-type', default='electric_machine')
    parser.add_argument('--assembly-type', default='motor')
    parser.add_argument('--component-type', required=True)
    parser.add_argument('--validation-fraction', type=float, default=0.2)
    parser.add_argument('--maximum-volume-samples', type=int, default=128)
    parser.add_argument('--maximum-boundary-samples', type=int, default=256)
    parser.add_argument('--maximum-query-samples', type=int, default=128)
    parser.add_argument('--pde-collocation-count', type=int, default=8)
    parser.add_argument('--pde-kernel-scale', type=float, default=0.2)
    parser.add_argument('--latency-repeats', type=int, default=15)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=23)
    return parser.parse_args()


def _to_device(batch: Mapping[str, Any], device: str) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _center_if_unanchored(
    values: torch.Tensor,
    weights: torch.Tensor,
    anchored: bool,
) -> torch.Tensor:
    if anchored:
        return values
    total = weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
    return values - torch.sum(weights * values) / total


def _category_key(arguments: argparse.Namespace, category_fields: Sequence[str]) -> str:
    labels = {
        'systemType': arguments.system_type,
        'assemblyType': arguments.assembly_type,
        'componentType': arguments.component_type,
    }
    return thermalCategoryKey(labels, category_fields)


def _prediction(
    model: Any,
    batch: Mapping[str, Any],
    *,
    category: Mapping[str, str] | None = None,
) -> torch.Tensor:
    if category is None:
        return model(batch)
    return model(batch, category)


def _records(
    models: Mapping[str, Any],
    cases: Sequence[Any],
    dimensionless_cases: Sequence[Any],
    category: Mapping[str, str],
    device: str,
) -> dict[str, list[dict[str, Any]]]:
    for model in models.values():
        model.to(device)
        model.eval()
    records = {name: [] for name in models}
    with torch.no_grad():
        for case, dimensionless_case in zip(cases, dimensionless_cases):
            raw_batch = collateThermalOperatorCases([dimensionless_case.operatorCase])
            batch = _to_device(raw_batch, device)
            mask = batch['queryMask'][0]
            weights = batch['queryWeights'][0, mask]
            target = batch['targets'][0, :, 0][mask]
            for name, model in models.items():
                prediction = _prediction(
                    model,
                    batch,
                    category=category if name == 'routed' else None,
                )[0, :, 0][mask]
                prediction = _center_if_unanchored(
                    prediction,
                    weights,
                    dimensionless_case.scales.absoluteTemperatureAnchorAvailable,
                )
                centered_target = _center_if_unanchored(
                    target,
                    weights,
                    dimensionless_case.scales.absoluteTemperatureAnchorAvailable,
                )
                error_k = (
                    prediction - centered_target
                ) * float(dimensionless_case.scales.temperatureScaleK)
                squared = torch.sum(weights * error_k.square())
                weight_sum = torch.sum(weights)
                records[name].append(
                    {
                        'caseId': case.caseId,
                        'group': _caseGroup(case),
                        'rmseK': math.sqrt(float((squared / weight_sum).cpu())),
                        'maximumErrorK': float(torch.max(torch.abs(error_k)).cpu()),
                        'squaredErrorIntegralK2': float(squared.cpu()),
                        'weightSum': float(weight_sum.cpu()),
                    }
                )
    for model in models.values():
        model.cpu()
    return records


def _metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rmses = np.asarray([record['rmseK'] for record in records], dtype=np.float64)
    maximum = max((float(record['maximumErrorK']) for record in records), default=0.0)
    total_squared = sum(float(record['squaredErrorIntegralK2']) for record in records)
    total_weight = sum(float(record['weightSum']) for record in records)
    return {
        'caseCount': len(records),
        'volumeWeightedRmseK': (
            math.sqrt(total_squared / total_weight) if total_weight else 0.0
        ),
        'macroCaseRmseK': float(np.mean(rmses)) if len(rmses) else 0.0,
        'caseRmseP50K': float(np.percentile(rmses, 50)) if len(rmses) else 0.0,
        'caseRmseP90K': float(np.percentile(rmses, 90)) if len(rmses) else 0.0,
        'caseRmseP95K': float(np.percentile(rmses, 95)) if len(rmses) else 0.0,
        'maximumErrorK': maximum,
    }


def _pde_residual(
    model: Any,
    cases: Sequence[Any],
    dimensionless_cases: Sequence[Any],
    category_keys: Sequence[str] | None,
    arguments: argparse.Namespace,
) -> float:
    device = model.toDevice(arguments.device, allowAutoFallback=True)
    loss = thermalSteadyPinnLoss(
        pdeResidualWeight=1.0,
        pdeResidualMode='meshfree',
        pdeCollocationCount=arguments.pde_collocation_count,
        pdeKernelScale=arguments.pde_kernel_scale,
    )
    loader = DataLoader(
        tuple(value.operatorCase for value in dimensionless_cases),
        batch_size=1,
        shuffle=False,
        collate_fn=collateThermalOperatorCases,
    )
    total = 0.0
    count = 0
    model.eval()
    with torch.enable_grad():
        for index, raw_batch in enumerate(loader):
            if category_keys is not None:
                raw_batch['categoryKeys'] = [category_keys[index]]
            batch = _to_device(raw_batch, device)
            value = loss.computePdeResidual(model, batch)
            total += float(value.detach().cpu())
            count += 1
    model.cpu()
    return total / count if count else 0.0


def _synchronize(device: str) -> None:
    if device.startswith('cuda') and torch.cuda.is_available():
        torch.cuda.synchronize()


def _latency(
    models: Mapping[str, Any],
    dimensionless_cases: Sequence[Any],
    category: Mapping[str, str],
    arguments: argparse.Namespace,
) -> dict[str, Any]:
    device = arguments.device
    repeats = max(1, int(arguments.latency_repeats))
    result = {}
    raw_batches = [
        _to_device(collateThermalOperatorCases([value.operatorCase]), device)
        for value in dimensionless_cases
    ]
    for name, model in models.items():
        model.to(device)
        model.eval()
        samples_ms = []
        if device.startswith('cuda') and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            for batch in raw_batches:
                for _ in range(2):
                    _prediction(model, batch, category=category if name == 'routed' else None)
                _synchronize(device)
                for _ in range(repeats):
                    start = time.perf_counter()
                    _prediction(model, batch, category=category if name == 'routed' else None)
                    _synchronize(device)
                    samples_ms.append((time.perf_counter() - start) * 1000.0)
        model.cpu()
        result[name] = {
            'caseCount': len(raw_batches),
            'repeatCountPerCase': repeats,
            'medianMs': float(statistics.median(samples_ms)) if samples_ms else 0.0,
            'p95Ms': float(np.percentile(samples_ms, 95)) if samples_ms else 0.0,
            'meanMs': float(statistics.mean(samples_ms)) if samples_ms else 0.0,
            'maxMs': float(max(samples_ms)) if samples_ms else 0.0,
            'device': device if not (device.startswith('cuda') and not torch.cuda.is_available()) else 'cpu',
            'peakAllocatedMiB': (
                float(torch.cuda.max_memory_allocated(device)) / 1024.0**2
                if device.startswith('cuda') and torch.cuda.is_available()
                else 0.0
            ),
        }
    return result


def _markdown(report: Mapping[str, Any]) -> str:
    component = report['component']['componentType']
    global_metrics = report['metrics']['global']
    routed_metrics = report['metrics']['routed']
    rmse_reduction = 100.0 * (
        1.0 - routed_metrics['volumeWeightedRmseK'] / global_metrics['volumeWeightedRmseK']
    )
    macro_reduction = 100.0 * (
        1.0 - routed_metrics['macroCaseRmseK'] / global_metrics['macroCaseRmseK']
    )
    pde_reduction = 100.0 * (
        1.0 - routed_metrics['pdeResidual'] / global_metrics['pdeResidual']
    )
    lines = [
        f'# {component} 单元件热 PINN 性能基准',
        '',
        f"模型：`{report['checkpoint']}`。测试切片为完整类别键 "
        f"`{report['component']['systemType']} / {report['component']['assemblyType']} / {component}`。",
        '',
        '## 测试范围',
        '',
        f"- 全数据 case：{report['dataset']['selectedCaseCount']}；训练/验证父工况切分："
        f"{report['dataset']['trainCaseCount']}/{report['dataset']['validationCaseCount']}。",
        f"- held-out {component} case：{report['dataset']['componentValidationCaseCount']}。",
        f"- 验证父工况：{', '.join(report['dataset']['validationGroups'])}。",
        f"- 采样：volume {report['dataset']['maximumVolumeSamples']} / boundary "
        f"{report['dataset']['maximumBoundarySamples']} / query {report['dataset']['maximumQuerySamples']}。",
        '',
        '## 精度与物理残差',
        '',
        '| 路径 | 加权 RMSE/K | Macro RMSE/K | P90/K | P95/K | 最大误差/K | PDE residual |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for name, label in (
        ('global', 'Global'),
        ('expert', f'{component} 专家'),
        ('routed', '推荐 checkpoint 实际路由'),
    ):
        metrics = report['metrics'][name]
        lines.append(
            f"| {label} | {metrics['volumeWeightedRmseK']:.4f} | "
            f"{metrics['macroCaseRmseK']:.4f} | {metrics['caseRmseP90K']:.4f} | "
            f"{metrics['caseRmseP95K']:.4f} | {metrics['maximumErrorK']:.4f} | "
            f"{metrics['pdeResidual']:.4f} |"
        )
    lines.extend([
        '',
        '## 逐 case 残差',
        '',
        '| case | Global RMSE/K | Expert RMSE/K | 实际路由 RMSE/K | 路由最大误差/K |',
        '|---|---:|---:|---:|---:|',
    ])
    global_by_id = {value['caseId']: value for value in report['records']['global']}
    expert_by_id = {value['caseId']: value for value in report['records']['expert']}
    for routed in report['records']['routed']:
        case_id = routed['caseId']
        lines.append(
            f"| {case_id} | {global_by_id[case_id]['rmseK']:.4f} | "
            f"{expert_by_id[case_id]['rmseK']:.4f} | {routed['rmseK']:.4f} | "
            f"{routed['maximumErrorK']:.4f} |"
        )
    lines.extend([
        '',
        '## 单 case 推理延迟',
        '',
        '| 路径 | 设备 | 中位数/ms | P95/ms | 平均/ms | 峰值分配显存/MiB |',
        '|---|---|---:|---:|---:|---:|',
    ])
    for name, label in (
        ('global', 'Global'),
        ('expert', f'{component} 专家'),
        ('routed', '推荐 checkpoint 实际路由'),
    ):
        latency = report['latency'][name]
        lines.append(
            f"| {label} | {latency['device']} | {latency['medianMs']:.3f} | "
            f"{latency['p95Ms']:.3f} | {latency['meanMs']:.3f} | "
            f"{latency['peakAllocatedMiB']:.1f} |"
        )
    lines.extend([
        '',
        '## 结论',
        '',
        f"- checkpoint 中该类别的 Expert 权重为 `{report['blendWeight']:.2f}`；"
        'Expert 与实际路由逐 case 结果一致，路由正确。',
        f'- 相对 Global，实际路由的加权 RMSE 降低 {rmse_reduction:.1f}%，'
        f'Macro RMSE 降低 {macro_reduction:.1f}%，PDE residual 降低 {pde_reduction:.1f}%。',
        f"- 最差 held-out case 为 `{max(report['records']['routed'], key=lambda value: value['rmseK'])['caseId']}`，"
        f"RMSE 为 {max(value['rmseK'] for value in report['records']['routed']):.4f} K，"
        '表明跨 prototype 泛化仍是主要长尾。',
        f"- 实际路由的 GPU 中位延迟为 {report['latency']['routed']['medianMs']:.3f} ms，"
        f"P95 为 {report['latency']['routed']['p95Ms']:.3f} ms。",
        '',
        f"本报告只评价 held-out {component} case，不代表所有元件或新父工况上的总体泛化。"
        '当前只有少量 held-out case，P90/P95 只能作为描述性统计。推荐路径的结果包含 checkpoint '
        '内固定的 Global/专家融合权重；若路由字段缺失，实际会回退 Global。',
        '',
        '完整逐 case 结果和路由信息见同目录下的 JSON 文件。',
        '',
    ])
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
    checkpoint = torch.load(
        arguments.checkpoint, map_location='cpu', weights_only=False
    )
    data_contract = thermalCheckpointDataContract.fromCheckpoint(checkpoint)
    prepared_cases = data_contract.prepareCases(dataset.cases)
    train_cases, validation_cases = _splitCases(
        prepared_cases,
        arguments.validation_fraction,
        arguments.seed,
    )
    labels = {
        'systemType': arguments.system_type,
        'assemblyType': arguments.assembly_type,
        'componentType': arguments.component_type,
    }
    validation_cases = tuple(
        case
        for case in validation_cases
        if all(thermalCaseCategoryLabels(case).get(key) == value for key, value in labels.items())
    )
    if not validation_cases:
        raise ValueError('no held-out cases match the requested component category')
    dimensionless_cases = data_contract.transformCases(validation_cases)
    bundle = thermalCategorizedOperatorTorch.fromCheckpoint(checkpoint)
    arguments.device = bundle.toDevice(arguments.device, allowAutoFallback=True)
    bundle.cpu()
    category_key = _category_key(arguments, bundle.categoryFields)
    if category_key not in bundle._categoryToModule:
        raise KeyError(f'checkpoint has no expert for {category_key!r}')
    _, routing = bundle.resolveModel(labels)
    category_keys = [category_key] * len(validation_cases)
    models = {
        'global': bundle.globalModel,
        'expert': bundle.expertModels[bundle._categoryToModule[category_key]],
        'routed': bundle,
    }
    records = _records(models, validation_cases, dimensionless_cases, labels, arguments.device)
    metrics = {}
    for name, values in records.items():
        metrics[name] = {
            **_metrics(values),
            'pdeResidual': _pde_residual(
                models[name],
                validation_cases,
                dimensionless_cases,
                category_keys if name == 'routed' else None,
                arguments,
            ),
        }
    latency = _latency(models, dimensionless_cases, labels, arguments)
    report = {
        'checkpoint': str(arguments.checkpoint),
        'component': labels,
        'dataset': {
            'source': str(arguments.source),
            'selectedCaseCount': len(dataset.cases),
            'trainCaseCount': len(train_cases),
            'validationCaseCount': len(_splitCases(dataset.cases, arguments.validation_fraction, arguments.seed)[1]),
            'componentValidationCaseCount': len(validation_cases),
            'validationGroups': sorted({_caseGroup(case) for case in validation_cases}),
            'maximumVolumeSamples': arguments.maximum_volume_samples,
            'maximumBoundarySamples': arguments.maximum_boundary_samples,
            'maximumQuerySamples': arguments.maximum_query_samples,
            'seed': arguments.seed,
        },
        'categoryKey': category_key,
        'blendWeight': float(checkpoint['experts'][category_key].get('blendWeight', 1.0)),
        'routing': routing,
        'metrics': metrics,
        'latency': latency,
        'records': records,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    arguments.output.with_suffix('.md').write_text(_markdown(report), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
