from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

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

from train_thermal_discrete_pinn import _caseGroup, _splitCases


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Calibrate fixed per-category Global/expert blend weights and '
            'compare them using group-held-out cross calibration.'
        )
    )
    parser.add_argument('source', type=Path)
    parser.add_argument(
        '--candidate',
        nargs=2,
        action='append',
        metavar=('NAME', 'CHECKPOINT'),
        required=True,
    )
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint-directory', type=Path, required=True)
    parser.add_argument(
        '--expert-base-checkpoint',
        type=Path,
        help=(
            'Optionally reuse the category fields and experts from this '
            'checkpoint while taking the Global model from each candidate.'
        ),
    )
    parser.add_argument('--validation-fraction', type=float, default=0.2)
    parser.add_argument('--maximum-volume-samples', type=int, default=128)
    parser.add_argument('--maximum-boundary-samples', type=int, default=256)
    parser.add_argument('--maximum-query-samples', type=int, default=128)
    parser.add_argument('--pde-collocation-count', type=int, default=8)
    parser.add_argument('--pde-kernel-scale', type=float, default=0.2)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=23)
    return parser.parse_args()


def _toDevice(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _centerIfUnanchored(
    values: torch.Tensor,
    weights: torch.Tensor,
    anchored: bool,
) -> torch.Tensor:
    if anchored:
        return values
    total = weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
    return values - torch.sum(weights * values) / total


def _predictionRecords(bundle, cases, dimensionlessCases, device: str):
    records = []
    bundle.to(device).eval()
    with torch.no_grad():
        for case, dimensionlessCase in zip(cases, dimensionlessCases):
            batch = _toDevice(
                collateThermalOperatorCases([dimensionlessCase.operatorCase]),
                device,
            )
            labels = thermalCaseCategoryLabels(case)
            try:
                categoryKey = thermalCategoryKey(labels, bundle.categoryFields)
            except KeyError:
                categoryKey = None
            expertModuleKey = bundle._categoryToModule.get(categoryKey or '')
            globalPrediction = bundle.globalModel(batch)[0, :, 0]
            expertPrediction = (
                bundle.expertModels[expertModuleKey](batch)[0, :, 0]
                if expertModuleKey is not None
                else globalPrediction
            )
            target = batch['targets'][0, :, 0]
            mask = batch['queryMask'][0]
            weights = batch['queryWeights'][0, mask]
            anchored = dimensionlessCase.scales.absoluteTemperatureAnchorAvailable
            globalPrediction = _centerIfUnanchored(
                globalPrediction[mask], weights, anchored
            )
            expertPrediction = _centerIfUnanchored(
                expertPrediction[mask], weights, anchored
            )
            target = _centerIfUnanchored(target[mask], weights, anchored)
            scale = float(dimensionlessCase.scales.temperatureScaleK)
            globalError = (globalPrediction - target).cpu().numpy() * scale
            expertDelta = (expertPrediction - globalPrediction).cpu().numpy() * scale
            records.append(
                {
                    'caseId': case.caseId,
                    'group': _caseGroup(case),
                    'categoryKey': categoryKey,
                    'hasExpert': expertModuleKey is not None,
                    'weights': weights.cpu().numpy().astype(np.float64),
                    'globalErrorK': globalError.astype(np.float64),
                    'expertDeltaK': expertDelta.astype(np.float64),
                }
            )
    bundle.cpu()
    return records


def _caseSquaredError(record: Mapping[str, Any], alpha: float) -> float:
    error = record['globalErrorK'] + alpha * record['expertDeltaK']
    return float(np.sum(record['weights'] * np.square(error)))


def _metrics(
    records: Sequence[Mapping[str, Any]],
    alphas: Mapping[str, float] | Callable[[Mapping[str, Any]], float],
) -> dict[str, Any]:
    caseRmse = []
    totalSquaredError = 0.0
    totalWeight = 0.0
    maximumError = 0.0
    for record in records:
        alpha = (
            float(alphas(record))
            if callable(alphas)
            else float(alphas.get(record['categoryKey'], 0.0))
        )
        error = record['globalErrorK'] + alpha * record['expertDeltaK']
        weight = float(np.sum(record['weights']))
        squaredError = float(np.sum(record['weights'] * np.square(error)))
        totalSquaredError += squaredError
        totalWeight += weight
        caseRmse.append(math.sqrt(squaredError / weight))
        maximumError = max(maximumError, float(np.max(np.abs(error))))
    return {
        'caseCount': len(records),
        'volumeWeightedRmseK': math.sqrt(totalSquaredError / totalWeight),
        'macroCaseRmseK': float(np.mean(caseRmse)),
        'caseRmseP90K': float(np.percentile(caseRmse, 90)),
        'caseRmseP95K': float(np.percentile(caseRmse, 95)),
        'maximumErrorK': maximumError,
    }


def _pooledAlpha(records: Sequence[Mapping[str, Any]]) -> float:
    numerator = 0.0
    denominator = 0.0
    for record in records:
        weights = record['weights']
        globalError = record['globalErrorK']
        delta = record['expertDeltaK']
        numerator -= float(np.sum(weights * globalError * delta))
        denominator += float(np.sum(weights * np.square(delta)))
    if denominator <= np.finfo(np.float64).eps:
        return 0.0
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def _caseBalancedAlpha(records: Sequence[Mapping[str, Any]]) -> float:
    if not records:
        return 0.0
    grid = np.linspace(0.0, 1.0, 101)
    losses = []
    for alpha in grid:
        losses.append(
            np.mean(
                [
                    _caseSquaredError(record, float(alpha))
                    / float(np.sum(record['weights']))
                    for record in records
                ]
            )
        )
    return float(grid[int(np.argmin(losses))])


def _hardAlpha(records: Sequence[Mapping[str, Any]]) -> float:
    globalError = sum(_caseSquaredError(record, 0.0) for record in records)
    expertError = sum(_caseSquaredError(record, 1.0) for record in records)
    return float(expertError < globalError)


def _fitAlphas(
    records: Sequence[Mapping[str, Any]],
    fitter: Callable[[Sequence[Mapping[str, Any]]], float],
) -> dict[str, float]:
    grouped = defaultdict(list)
    for record in records:
        if record['hasExpert'] and record['categoryKey'] is not None:
            grouped[record['categoryKey']].append(record)
    return {key: fitter(values) for key, values in sorted(grouped.items())}


def _crossCalibratedMetrics(
    records: Sequence[Mapping[str, Any]],
    fitter: Callable[[Sequence[Mapping[str, Any]]], float],
) -> dict[str, Any]:
    groups = sorted({record['group'] for record in records})
    alphasByGroup = {}
    for heldOutGroup in groups:
        calibration = [
            record for record in records if record['group'] != heldOutGroup
        ]
        alphasByGroup[heldOutGroup] = _fitAlphas(calibration, fitter)
    return _metrics(
        records,
        lambda record: alphasByGroup[record['group']].get(
            record['categoryKey'], 0.0
        ),
    )


def _checkpointWithAlphas(
    sourceCheckpoint: Mapping[str, Any],
    alphas: Mapping[str, float],
    strategy: str,
) -> dict[str, Any]:
    checkpoint = dict(sourceCheckpoint)
    checkpoint['global'] = dict(sourceCheckpoint['global'])
    checkpoint['experts'] = {
        key: {**payload, 'blendWeight': float(alphas.get(key, 0.0))}
        for key, payload in sourceCheckpoint.get('experts', {}).items()
    }
    checkpoint['metadata'] = {
        **dict(sourceCheckpoint.get('metadata', {})),
        'expertCalibration': {
            'strategy': strategy,
            'blendWeights': dict(alphas),
        },
    }
    checkpoint['routingPolicy'] = 'exact-category-global-expert-blend'
    return checkpoint


def _checkpointWithExpertBase(
    candidateCheckpoint: Mapping[str, Any],
    expertBaseCheckpoint: Mapping[str, Any],
    expertBasePath: Path,
) -> dict[str, Any]:
    checkpoint = dict(candidateCheckpoint)
    checkpoint['categoryFields'] = list(expertBaseCheckpoint['categoryFields'])
    checkpoint['experts'] = dict(expertBaseCheckpoint.get('experts', {}))
    checkpoint['metadata'] = {
        **dict(candidateCheckpoint.get('metadata', {})),
        'expertBaseCheckpoint': str(expertBasePath),
    }
    checkpoint['routingPolicy'] = 'exact-category-then-global'
    return checkpoint


def _pdeResidual(
    bundle,
    cases,
    operatorCases,
    arguments: argparse.Namespace,
) -> float:
    device = bundle.toDevice(arguments.device, allowAutoFallback=True)
    loss = thermalSteadyPinnLoss(
        pdeResidualWeight=1.0,
        pdeCollocationCount=arguments.pde_collocation_count,
        pdeKernelScale=arguments.pde_kernel_scale,
    )
    loader = DataLoader(
        operatorCases,
        batch_size=4,
        shuffle=False,
        collate_fn=collateThermalOperatorCases,
    )
    labelsByCaseId = {
        case.caseId: thermalCaseCategoryLabels(case) for case in cases
    }
    total = 0.0
    count = 0
    with torch.enable_grad():
        for rawBatch in loader:
            rawBatch['categoryKeys'] = [
                bundle.categoryKey(labelsByCaseId[caseId])
                for caseId in rawBatch['caseIds']
            ]
            batch = _toDevice(rawBatch, device)
            batchCount = len(rawBatch['caseIds'])
            value = loss.computePdeResidual(bundle, batch)
            total += float(value.detach().cpu()) * batchCount
            count += batchCount
    bundle.cpu()
    return total / count


def main() -> None:
    arguments = _arguments()
    candidates = [(name, Path(path)) for name, path in arguments.candidate]
    if len({name for name, _ in candidates}) != len(candidates):
        raise ValueError('candidate names must be unique')
    dataset = loadSubsampledThermalDataset(
        arguments.source,
        analysisType='steady',
        maximumVolumeSamples=arguments.maximum_volume_samples,
        maximumBoundarySamples=arguments.maximum_boundary_samples,
        maximumQuerySamples=arguments.maximum_query_samples,
        randomSeed=arguments.seed,
    )
    _, rawValidationCases = _splitCases(
        dataset.cases,
        arguments.validation_fraction,
        arguments.seed,
    )
    fitters = {
        'hard': _hardAlpha,
        'pooledBlend': _pooledAlpha,
        'caseBalancedBlend': _caseBalancedAlpha,
    }
    comparison = {
        'scope': 'calibration on the existing grouped validation split',
        'warning': (
            'Cross-calibrated gate metrics leave out parent groups only for '
            'gate fitting; base checkpoints still used this validation split '
            'for early stopping.'
        ),
        'seed': arguments.seed,
        'validationCaseCount': len(rawValidationCases),
        'validationGroups': sorted({_caseGroup(case) for case in rawValidationCases}),
        'candidates': {},
    }
    arguments.checkpoint_directory.mkdir(parents=True, exist_ok=True)
    expertBaseCheckpoint = None
    if arguments.expert_base_checkpoint is not None:
        expertBaseCheckpoint = torch.load(
            arguments.expert_base_checkpoint,
            map_location='cpu',
            weights_only=False,
        )
    for name, checkpointPath in candidates:
        print(f'calibrating {name}...', file=sys.stderr)
        sourceCheckpoint = torch.load(
            checkpointPath,
            map_location='cpu',
            weights_only=False,
        )
        if expertBaseCheckpoint is not None:
            sourceCheckpoint = _checkpointWithExpertBase(
                sourceCheckpoint,
                expertBaseCheckpoint,
                arguments.expert_base_checkpoint,
            )
        dataContract = thermalCheckpointDataContract.fromCheckpoint(sourceCheckpoint)
        preparedCases = dataContract.prepareCases(dataset.cases)
        _, validationCases = _splitCases(
            preparedCases,
            arguments.validation_fraction,
            arguments.seed,
        )
        dimensionless = dataContract.transformCases(validationCases)
        operatorCases = tuple(value.operatorCase for value in dimensionless)
        bundle = thermalCategorizedOperatorTorch.fromCheckpoint(sourceCheckpoint)
        records = _predictionRecords(
            bundle,
            validationCases,
            dimensionless,
            bundle.toDevice(arguments.device, allowAutoFallback=True),
        )
        strategies = {
            'global': {
                'alphas': {},
                'fitMetrics': _metrics(records, {}),
                'crossCalibratedMetrics': _metrics(records, {}),
            },
            'expert': {
                'alphas': {
                    key: 1.0 for key in bundle.categoryKeys
                },
                'fitMetrics': _metrics(
                    records, {key: 1.0 for key in bundle.categoryKeys}
                ),
                'crossCalibratedMetrics': _metrics(
                    records, {key: 1.0 for key in bundle.categoryKeys}
                ),
            },
        }
        for strategy, fitter in fitters.items():
            alphas = _fitAlphas(records, fitter)
            strategies[strategy] = {
                'alphas': alphas,
                'fitMetrics': _metrics(records, alphas),
                'crossCalibratedMetrics': _crossCalibratedMetrics(records, fitter),
            }
        for strategy, values in strategies.items():
            strategyCheckpoint = _checkpointWithAlphas(
                sourceCheckpoint,
                values['alphas'],
                strategy,
            )
            values['pdeResidual'] = _pdeResidual(
                thermalCategorizedOperatorTorch.fromCheckpoint(
                    strategyCheckpoint
                ),
                validationCases,
                operatorCases,
                arguments,
            )
        eligibleStrategies = ('hard', 'pooledBlend', 'caseBalancedBlend')
        selectedStrategy = min(
            eligibleStrategies,
            key=lambda strategy: (
                strategies[strategy]['crossCalibratedMetrics'][
                    'volumeWeightedRmseK'
                ],
                strategies[strategy]['crossCalibratedMetrics']['caseRmseP95K'],
            ),
        )
        selectedCheckpoint = _checkpointWithAlphas(
            sourceCheckpoint,
            strategies[selectedStrategy]['alphas'],
            selectedStrategy,
        )
        selectedPath = arguments.checkpoint_directory / f'{name}_{selectedStrategy}.pt'
        torch.save(selectedCheckpoint, selectedPath)
        comparison['candidates'][name] = {
            'sourceCheckpoint': str(checkpointPath),
            'categoryFields': list(bundle.categoryFields),
            'selectedStrategy': selectedStrategy,
            'selectedCheckpoint': str(selectedPath),
            'strategies': strategies,
        }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(comparison, indent=2),
        encoding='utf-8',
    )
    print(json.dumps(comparison, indent=2))


if __name__ == '__main__':
    main()
