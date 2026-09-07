from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from mor.nn.data import (
    STEADY_THERMAL_DIMENSIONLESS_V2,
    collateThermalOperatorCases,
    enrichThermalAssemblyCases,
    loadSubsampledThermalDataset,
    thermalAssemblyContextFeatureNames,
)
from mor.nn.losses import thermalSteadyPinnLoss
from mor.nn.models import (
    thermalCategorizedOperatorTorch,
    thermalQuerySetOperatorTorch,
    thermalSetAttentionOperatorTorch,
    thermalSetDeepONetTorch,
)

try:
    from scripts.train_thermal_discrete_pinn import (
        _caseGroup,
        _dimensionless,
        _evaluate,
        _featureSchema,
        _fit,
        _modelConfig,
        _splitCases,
    )
except ModuleNotFoundError:
    from train_thermal_discrete_pinn import (
        _caseGroup,
        _dimensionless,
        _evaluate,
        _featureSchema,
        _fit,
        _modelConfig,
        _splitCases,
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Screen all compatible discrete thermal operator families.'
    )
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint-directory', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=0.002)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=23)
    parser.add_argument('--maximum-volume-samples', type=int, default=128)
    parser.add_argument('--maximum-boundary-samples', type=int, default=256)
    parser.add_argument('--maximum-query-samples', type=int, default=128)
    parser.add_argument('--pde-residual-weight', type=float, default=0.002)
    parser.add_argument('--pde-collocation-count', type=int, default=8)
    parser.add_argument('--candidate', action='append')
    return parser.parse_args()


def _candidates() -> list[dict[str, Any]]:
    return [
        {
            'name': 'rbf_mlp_h64',
            'modelFamily': 'querySet',
            'contextArchitecture': 'rbf',
            'decoderArchitecture': 'mlp',
            'hiddenDim': 64,
            'residualBlockCount': 0,
            'residualScale': 1.0,
        },
        {
            'name': 'rbf_resnet_b1_h64',
            'modelFamily': 'querySet',
            'contextArchitecture': 'rbf',
            'decoderArchitecture': 'resnet',
            'hiddenDim': 64,
            'residualBlockCount': 1,
            'residualScale': 0.2,
        },
        {
            'name': 'rbf_dense_h64',
            'modelFamily': 'querySet',
            'contextArchitecture': 'rbf',
            'decoderArchitecture': 'dense',
            'hiddenDim': 64,
            'residualBlockCount': 0,
            'residualScale': 1.0,
        },
        {
            'name': 'rbf_unet_h64',
            'modelFamily': 'querySet',
            'contextArchitecture': 'rbf',
            'decoderArchitecture': 'unet',
            'hiddenDim': 64,
            'residualBlockCount': 0,
            'residualScale': 1.0,
        },
        {
            'name': 'pointnet_global_h64',
            'modelFamily': 'querySet',
            'contextArchitecture': 'global',
            'decoderArchitecture': 'mlp',
            'hiddenDim': 64,
            'residualBlockCount': 0,
            'residualScale': 1.0,
        },
        {
            'name': 'deeponet_h64',
            'modelFamily': 'querySet',
            'contextArchitecture': 'deeponet',
            'decoderArchitecture': 'mlp',
            'hiddenDim': 64,
            'residualBlockCount': 0,
            'residualScale': 1.0,
        },
        {
            'name': 'attention_h64',
            'modelFamily': 'querySet',
            'contextArchitecture': 'attention',
            'decoderArchitecture': 'mlp',
            'hiddenDim': 64,
            'residualBlockCount': 0,
            'residualScale': 1.0,
        },
        {
            'name': 'hybrid_rbf_attention_h64',
            'modelFamily': 'querySet',
            'contextArchitecture': 'hybrid',
            'decoderArchitecture': 'mlp',
            'hiddenDim': 64,
            'residualBlockCount': 0,
            'residualScale': 1.0,
        },
        {
            'name': 'set_deeponet_h64',
            'modelFamily': 'setDeepONet',
            'contextArchitecture': 'global',
            'decoderArchitecture': 'mlp',
            'hiddenDim': 64,
            'residualBlockCount': 0,
            'residualScale': 1.0,
        },
        {
            'name': 'set_attention_h64',
            'modelFamily': 'setAttention',
            'contextArchitecture': 'attention',
            'decoderArchitecture': 'mlp',
            'hiddenDim': 64,
            'residualBlockCount': 0,
            'residualScale': 1.0,
        },
        {
            'name': 'set_attention_h64_lr1e3_p0',
            'modelFamily': 'setAttention',
            'contextArchitecture': 'attention',
            'decoderArchitecture': 'mlp',
            'hiddenDim': 64,
            'latentCount': 8,
            'learningRate': 0.001,
            'pdeResidualWeight': 0.0,
            'residualBlockCount': 0,
            'residualScale': 1.0,
        },
        {
            'name': 'set_attention_h64_lr1e3_p2e4',
            'modelFamily': 'setAttention',
            'contextArchitecture': 'attention',
            'decoderArchitecture': 'mlp',
            'hiddenDim': 64,
            'latentCount': 8,
            'learningRate': 0.001,
            'pdeResidualWeight': 0.0002,
            'residualBlockCount': 0,
            'residualScale': 1.0,
        },
        {
            'name': 'set_attention_h64_lr1e3_p2e3',
            'modelFamily': 'setAttention',
            'contextArchitecture': 'attention',
            'decoderArchitecture': 'mlp',
            'hiddenDim': 64,
            'latentCount': 8,
            'learningRate': 0.001,
            'pdeResidualWeight': 0.002,
            'residualBlockCount': 0,
            'residualScale': 1.0,
        },
        {
            'name': 'set_attention_h64_lr5e4_p2e3',
            'modelFamily': 'setAttention',
            'contextArchitecture': 'attention',
            'decoderArchitecture': 'mlp',
            'hiddenDim': 64,
            'latentCount': 8,
            'learningRate': 0.0005,
            'pdeResidualWeight': 0.002,
            'residualBlockCount': 0,
            'residualScale': 1.0,
        },
        {
            'name': 'set_attention_h96_lr1e3_p2e3',
            'modelFamily': 'setAttention',
            'contextArchitecture': 'attention',
            'decoderArchitecture': 'mlp',
            'hiddenDim': 96,
            'latentCount': 8,
            'learningRate': 0.001,
            'pdeResidualWeight': 0.002,
            'residualBlockCount': 0,
            'residualScale': 1.0,
        },
        {
            'name': 'set_attention_h64_l16_lr1e3_p2e3',
            'modelFamily': 'setAttention',
            'contextArchitecture': 'attention',
            'decoderArchitecture': 'mlp',
            'hiddenDim': 64,
            'latentCount': 16,
            'learningRate': 0.001,
            'pdeResidualWeight': 0.002,
            'residualBlockCount': 0,
            'residualScale': 1.0,
        },
    ]


def _trainingArguments(arguments, candidate):
    return SimpleNamespace(
        hidden_dim=candidate['hiddenDim'],
        kernel_scales=[0.12, 0.35],
        query_fourier_frequencies=4,
        residual_block_count=candidate['residualBlockCount'],
        residual_scale=candidate['residualScale'],
        decoder_architecture=candidate['decoderArchitecture'],
        context_architecture=candidate['contextArchitecture'],
        attention_head_count=4,
        shape_loss_weight=1.0,
        mean_loss_weight=1.0,
        reconstruction_loss_weight=0.1,
        hotspot_loss_weight=0.25,
        energy_balance_weight=0.02,
        pde_residual_weight=candidate.get(
            'pdeResidualWeight', arguments.pde_residual_weight
        ),
        pde_residual_mode='meshfree',
        pde_collocation_count=arguments.pde_collocation_count,
        pde_kernel_scale=0.2,
        weight_decay=0.0,
        device=arguments.device,
        physics_in_validation=False,
        batch_size=arguments.batch_size,
        learning_rate=candidate.get('learningRate', arguments.learning_rate),
    )


def _buildModel(example, arguments, candidate, schema):
    family = candidate['modelFamily']
    if family == 'querySet':
        config = _modelConfig(example, arguments, schema)
        return thermalQuerySetOperatorTorch(**config), config
    common = {
        'volumeFeatureDim': example.volumeFeatureDim,
        'boundaryFeatureDim': example.boundaryFeatureDim,
        'globalFeatureDim': example.globalFeatureDim,
        'queryDim': example.queryDim,
    }
    if family == 'setDeepONet':
        config = {
            **common,
            'pointHiddenDim': candidate['hiddenDim'],
            'globalHiddenDim': candidate['hiddenDim'],
            'latentDim': 96,
            'poolingMode': 'mean_max',
            'queryFourierFrequencies': 4,
        }
        return thermalSetDeepONetTorch(**config), config
    if family == 'setAttention':
        config = {
            **common,
            'hiddenDim': candidate['hiddenDim'],
            'latentCount': candidate.get('latentCount', 8),
            'attentionHeads': 4,
            'queryFourierFrequencies': 4,
        }
        return thermalSetAttentionOperatorTorch(**config), config
    raise ValueError(f'unsupported model family: {family!r}')


def _summary(records):
    if not records:
        return None
    squared = sum(value['_squaredErrorIntegral'] for value in records)
    weight = sum(value['_weightSum'] for value in records)
    rmses = np.asarray([value['rmseK'] for value in records])
    return {
        'caseCount': len(records),
        'volumeWeightedRmseK': math.sqrt(squared / weight),
        'macroCaseRmseK': float(np.mean(rmses)),
        'caseRmseP90K': float(np.percentile(rmses, 90)),
        'caseRmseP95K': float(np.percentile(rmses, 95)),
        'maximumErrorK': max(value['maximumErrorK'] for value in records),
    }


def _byComponent(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[record['componentType']].append(record)
    return {key: _summary(values) for key, values in sorted(grouped.items())}


def _meshfreeResidual(model, operatorCases, trainingArguments, device):
    loss = thermalSteadyPinnLoss(
        pdeResidualWeight=1.0,
        pdeResidualMode='meshfree',
        pdeCollocationCount=trainingArguments.pde_collocation_count,
        pdeKernelScale=trainingArguments.pde_kernel_scale,
    )
    loader = DataLoader(
        operatorCases,
        batch_size=trainingArguments.batch_size,
        shuffle=False,
        collate_fn=collateThermalOperatorCases,
    )
    model.to(device).eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for rawBatch in loader:
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in rawBatch.items()
            }
            value = loss.computePdeResidual(model, batch)
            batchCount = len(rawBatch['caseIds'])
            total += float(value.cpu()) * batchCount
            count += batchCount
    model.cpu()
    return total / max(1, count)


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        '# 全元件离散热模型架构筛选',
        '',
        '所有候选使用相同父工况切分、装配上下文、无量纲输入和 mesh-free 物理损失。',
        '',
        '| 排名 | 模型 | 参数量 | 加权 RMSE/K | Macro/K | P95/K | 最大误差/K | 离散 PDE |',
        '|---:|---|---:|---:|---:|---:|---:|---:|',
    ]
    byName = {value['name']: value for value in report['candidates']}
    for row in report['ranking']:
        candidate = byName[row['name']]
        lines.append(
            f"| {row['rank']} | `{row['name']}` | {candidate['parameterCount']:,} | "
            f"{row['volumeWeightedRmseK']:.4f} | {row['macroCaseRmseK']:.4f} | "
            f"{row['caseRmseP95K']:.4f} | {row['maximumErrorK']:.4f} | "
            f"{row['meshfreePdeResidual']:.4f} |"
        )
    lines.extend(
        (
            '',
            '说明：这里的 U-Net 是适用于无结构点集的 U 形 MLP decoder；卷积 U-Net/FNO '
            '要求规则网格，当前 FEM 点数据不能直接使用。图网络要求 graph view 的网格连通性，'
            '本轮修正后的 stator overlay 只有 point view，因此不纳入虚假对比。',
            '',
        )
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
    componentTypes = tuple(sorted({case.componentType for case in dataset.cases}))
    contextNames = thermalAssemblyContextFeatureNames(componentTypes)
    schema = _featureSchema(contextNames)
    cases = enrichThermalAssemblyCases(dataset.cases, componentTypes=componentTypes)
    trainCases, validationCases = _splitCases(cases, 0.2, arguments.seed)
    trainDimensionless = _dimensionless(trainCases, 1.0, schema)
    validationDimensionless = _dimensionless(validationCases, 1.0, schema)
    trainOperatorCases = tuple(value.operatorCase for value in trainDimensionless)
    validationOperatorCases = tuple(
        value.operatorCase for value in validationDimensionless
    )
    candidates = _candidates()
    if arguments.candidate:
        requested = set(arguments.candidate)
        known = {value['name'] for value in candidates}
        if requested - known:
            raise ValueError(f'unknown candidates: {sorted(requested - known)}')
        candidates = [value for value in candidates if value['name'] in requested]
    arguments.checkpoint_directory.mkdir(parents=True, exist_ok=True)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for index, candidate in enumerate(candidates, start=1):
        print(
            f'[{index}/{len(candidates)}] training {candidate["name"]}...',
            file=sys.stderr,
            flush=True,
        )
        trainingArguments = _trainingArguments(arguments, candidate)
        torch.manual_seed(arguments.seed)
        model, config = _buildModel(
            trainOperatorCases[0],
            trainingArguments,
            candidate,
            schema,
        )
        parameterCount = sum(value.numel() for value in model.parameters())
        started = time.perf_counter()
        training = _fit(
            model,
            trainOperatorCases,
            validationOperatorCases,
            arguments=trainingArguments,
            epochs=arguments.epochs,
            learningRate=trainingArguments.learning_rate,
            seed=arguments.seed,
            schema=schema,
        )
        records = _evaluate(
            model,
            validationCases,
            validationDimensionless,
            training['resolvedDeviceName'],
        )
        metrics = _summary(records)
        metrics['meshfreePdeResidual'] = _meshfreeResidual(
            model,
            validationOperatorCases,
            trainingArguments,
            training['resolvedDeviceName'],
        )
        checkpoint = thermalCategorizedOperatorTorch(
            model.cpu(),
            {},
            categoryFields=('componentType',),
        ).toCheckpoint(
            globalModelConfig=config,
            metadata={
                'source': str(dataset.sourcePath),
                'featureMode': STEADY_THERMAL_DIMENSIONLESS_V2,
                'architectureSweepCandidate': candidate['name'],
                'schema': {
                    'operatingFeatureNames': list(schema.operatingFeatureNames),
                    'boundaryTypeCount': schema.boundaryTypeCount,
                    'temperatureBoundaryTypeIds': list(
                        schema.temperatureBoundaryTypeIds
                    ),
                    'fixedTemperatureBoundaryTypeIds': list(
                        schema.fixedTemperatureBoundaryTypeIds
                    ),
                },
                'assemblyContext': {
                    'enabled': True,
                    'componentTypes': list(componentTypes),
                    'featureNames': list(contextNames),
                },
            },
        )
        checkpointPath = arguments.checkpoint_directory / f'{candidate["name"]}.pt'
        torch.save(checkpoint, checkpointPath)
        outputs.append(
            {
                **candidate,
                'parameterCount': parameterCount,
                'elapsedSeconds': time.perf_counter() - started,
                'checkpoint': str(checkpointPath),
                'training': training,
                'evaluation': metrics,
                'byComponent': _byComponent(records),
                'cases': [
                    {key: value for key, value in record.items() if not key.startswith('_')}
                    for record in records
                ],
            }
        )
        partial = {
            'dataset': str(dataset.sourcePath),
            'completedCandidates': outputs,
        }
        arguments.output.with_suffix('.partial.json').write_text(
            json.dumps(partial, indent=2), encoding='utf-8'
        )

    ranked = sorted(
        outputs,
        key=lambda value: (
            value['evaluation']['volumeWeightedRmseK'],
            value['evaluation']['caseRmseP95K'],
            value['evaluation']['maximumErrorK'],
        ),
    )
    ranking = [
        {'rank': index, 'name': value['name'], **value['evaluation']}
        for index, value in enumerate(ranked, start=1)
    ]
    report = {
        'scope': 'all steady component cases with assembly context',
        'dataset': {
            'source': str(dataset.sourcePath),
            'selectedCaseCount': len(cases),
            'trainCaseCount': len(trainCases),
            'validationCaseCount': len(validationCases),
            'trainGroupCount': len({_caseGroup(case) for case in trainCases}),
            'validationGroupCount': len({_caseGroup(case) for case in validationCases}),
            'componentCounts': dict(Counter(case.componentType for case in cases)),
            'epochs': arguments.epochs,
            'seed': arguments.seed,
        },
        'ranking': ranking,
        'candidates': outputs,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    arguments.output.with_suffix('.md').write_text(_markdown(report), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
