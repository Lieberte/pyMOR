from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from mor.nn.data import (
    STEADY_THERMAL_DIMENSIONLESS_V2,
    enrichThermalAssemblyCases,
    loadSubsampledThermalDataset,
    thermalAssemblyContextFeatureNames,
)
from mor.nn.models import (
    thermalCategorizedOperatorTorch,
    thermalSetDeepONetTorch,
)

try:
    from scripts.benchmark_thermal_component import _pde_residual
    from scripts.train_thermal_discrete_pinn import (
        OPERATING_SCALES,
        _caseGroup,
        _dimensionless,
        _evaluate,
        _featureSchema,
        _fit,
        _splitCases,
    )
except ModuleNotFoundError:
    from benchmark_thermal_component import _pde_residual
    from train_thermal_discrete_pinn import (
        OPERATING_SCALES,
        _caseGroup,
        _dimensionless,
        _evaluate,
        _featureSchema,
        _fit,
        _splitCases,
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Train and evaluate one standalone stator thermal operator. Only '
            'stator temperature targets enter optimization; complete parent '
            'assemblies are retained to construct inference-time context.'
        )
    )
    parser.add_argument('source', type=Path)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--latent-dim', type=int, default=96)
    parser.add_argument('--pde-residual-weight', type=float, default=0.002)
    parser.add_argument(
        '--pde-residual-mode',
        choices=('meshfree', 'strong'),
        default='meshfree',
    )
    parser.add_argument('--pde-collocation-count', type=int, default=8)
    parser.add_argument('--neumann-residual-weight', type=float, default=0.0)
    parser.add_argument('--robin-residual-weight', type=float, default=0.0)
    parser.add_argument('--dirichlet-residual-weight', type=float, default=0.0)
    parser.add_argument('--boundary-collocation-count', type=int, default=64)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=23)
    parser.add_argument('--maximum-volume-samples', type=int, default=128)
    parser.add_argument('--maximum-boundary-samples', type=int, default=256)
    parser.add_argument('--maximum-query-samples', type=int, default=128)
    parser.add_argument(
        '--validation-case-id',
        action='append',
        help=(
            'Explicit stator case id to hold out. Repeat for multiple parent '
            'groups. The default uses the seeded 20%% grouped split.'
        ),
    )
    return parser.parse_args()


def _trainingArguments(arguments: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        shape_loss_weight=1.0,
        mean_loss_weight=1.0,
        reconstruction_loss_weight=0.1,
        hotspot_loss_weight=0.25,
        energy_balance_weight=0.02,
        pde_residual_weight=arguments.pde_residual_weight,
        pde_residual_mode=arguments.pde_residual_mode,
        pde_collocation_count=arguments.pde_collocation_count,
        pde_kernel_scale=0.2,
        neumann_residual_weight=arguments.neumann_residual_weight,
        robin_residual_weight=arguments.robin_residual_weight,
        dirichlet_residual_weight=arguments.dirichlet_residual_weight,
        boundary_collocation_count=arguments.boundary_collocation_count,
        weight_decay=0.0,
        device=arguments.device,
        physics_in_validation=False,
        batch_size=arguments.batch_size,
    )


def _summary(records) -> dict[str, float | int]:
    squaredError = sum(value['_squaredErrorIntegral'] for value in records)
    totalWeight = sum(value['_weightSum'] for value in records)
    rmses = np.asarray([value['rmseK'] for value in records], dtype=np.float64)
    return {
        'caseCount': len(records),
        'volumeWeightedRmseK': math.sqrt(squaredError / totalWeight),
        'macroCaseRmseK': float(np.mean(rmses)),
        'caseRmseP90K': float(np.percentile(rmses, 90)),
        'caseRmseP95K': float(np.percentile(rmses, 95)),
        'maximumErrorK': max(value['maximumErrorK'] for value in records),
    }


def _publicRecords(records):
    return [
        {key: value for key, value in record.items() if not key.startswith('_')}
        for record in records
    ]


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
    componentTypes = tuple(sorted({case.componentType for case in dataset.cases}))
    contextNames = thermalAssemblyContextFeatureNames(componentTypes)
    schema = _featureSchema(contextNames)
    assemblyCases = enrichThermalAssemblyCases(
        dataset.cases,
        componentTypes=componentTypes,
    )
    if arguments.validation_case_id:
        requestedIds = frozenset(arguments.validation_case_id)
        statorById = {
            case.caseId: case
            for case in assemblyCases
            if case.componentType == 'stator'
        }
        missing = sorted(requestedIds.difference(statorById))
        if missing:
            raise ValueError(f'unknown stator validation case ids: {missing!r}')
        validationGroups = frozenset(
            _caseGroup(statorById[caseId]) for caseId in requestedIds
        )
        trainAssembly = tuple(
            case for case in assemblyCases if _caseGroup(case) not in validationGroups
        )
        validationAssembly = tuple(
            case for case in assemblyCases if _caseGroup(case) in validationGroups
        )
        splitStrategy = 'explicit-parent-group-holdout'
    else:
        trainAssembly, validationAssembly = _splitCases(
            assemblyCases,
            0.2,
            arguments.seed,
        )
        splitStrategy = 'seeded-20-percent-parent-group-holdout'
    trainCases = tuple(
        case for case in trainAssembly if case.componentType == 'stator'
    )
    validationCases = tuple(
        case for case in validationAssembly if case.componentType == 'stator'
    )
    if not trainCases or not validationCases:
        raise ValueError('stator split must contain both training and validation cases')
    anchoredTrainCases = tuple(
        case
        for case in trainCases
        if case.operatingFeatures.get('absoluteTemperatureAnchor', 0.0) > 0.5
    )
    anchoredValidationCases = tuple(
        case
        for case in validationCases
        if case.operatingFeatures.get('absoluteTemperatureAnchor', 0.0) > 0.5
    )
    if len(anchoredValidationCases) != len(validationCases):
        raise ValueError(
            'all held-out stator cases must have an assembly anchor for absolute '
            'temperature evaluation'
        )

    trainDimensionless = _dimensionless(trainCases, 1.0, schema)
    validationDimensionless = _dimensionless(validationCases, 1.0, schema)
    trainOperatorCases = tuple(value.operatorCase for value in trainDimensionless)
    validationOperatorCases = tuple(
        value.operatorCase for value in validationDimensionless
    )
    example = trainOperatorCases[0]
    modelConfig = {
        'volumeFeatureDim': example.volumeFeatureDim,
        'boundaryFeatureDim': example.boundaryFeatureDim,
        'globalFeatureDim': example.globalFeatureDim,
        'queryDim': example.queryDim,
        'pointHiddenDim': arguments.hidden_dim,
        'globalHiddenDim': arguments.hidden_dim,
        'latentDim': arguments.latent_dim,
        'poolingMode': 'mean_max',
        'queryFourierFrequencies': 4,
    }
    torch.manual_seed(arguments.seed)
    model = thermalSetDeepONetTorch(**modelConfig)
    trainingArguments = _trainingArguments(arguments)
    started = time.perf_counter()
    training = _fit(
        model,
        trainOperatorCases,
        validationOperatorCases,
        arguments=trainingArguments,
        epochs=arguments.epochs,
        learningRate=arguments.learning_rate,
        seed=arguments.seed,
        schema=schema,
    )
    elapsedSeconds = time.perf_counter() - started
    trainingRecords = _evaluate(
        model,
        trainCases,
        trainDimensionless,
        training['resolvedDeviceName'],
    )
    records = _evaluate(
        model,
        validationCases,
        validationDimensionless,
        training['resolvedDeviceName'],
    )
    metrics = _summary(records)
    metrics['meshfreePdeResidual'] = _pde_residual(
        model,
        validationCases,
        validationDimensionless,
        None,
        SimpleNamespace(
            device=arguments.device,
            pde_collocation_count=arguments.pde_collocation_count,
            pde_kernel_scale=0.2,
        ),
    )
    metadata = {
        'source': str(dataset.sourcePath),
        'analysisType': 'steady',
        'trainingScope': 'stator-only',
        'supportedComponentTypes': ['stator'],
        'usesOtherComponentTemperatureTargets': False,
        'usesParentAssemblyInputContext': True,
        'featureMode': STEADY_THERMAL_DIMENSIONLESS_V2,
        'operatingScales': OPERATING_SCALES,
        'minimumTemperatureScaleK': 1.0,
        'schema': {
            'operatingFeatureNames': list(schema.operatingFeatureNames),
            'boundaryTypeCount': schema.boundaryTypeCount,
            'temperatureBoundaryTypeIds': list(schema.temperatureBoundaryTypeIds),
            'fixedTemperatureBoundaryTypeIds': list(
                schema.fixedTemperatureBoundaryTypeIds
            ),
        },
        'assemblyContext': {
            'enabled': True,
            'componentTypes': list(componentTypes),
            'featureNames': list(contextNames),
        },
        'physics': {
            'energyBalanceWeight': trainingArguments.energy_balance_weight,
            'pdeResidualMode': trainingArguments.pde_residual_mode,
            'pdeResidualWeight': trainingArguments.pde_residual_weight,
            'pdeCollocationCount': trainingArguments.pde_collocation_count,
            'neumannResidualWeight': trainingArguments.neumann_residual_weight,
            'robinResidualWeight': trainingArguments.robin_residual_weight,
            'dirichletResidualWeight': trainingArguments.dirichlet_residual_weight,
            'boundaryCollocationCount': trainingArguments.boundary_collocation_count,
        },
    }
    checkpoint = thermalCategorizedOperatorTorch(
        model.cpu(),
        {},
        categoryFields=('componentType',),
    ).toCheckpoint(globalModelConfig=modelConfig, metadata=metadata)
    arguments.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, arguments.checkpoint)
    report = {
        'scope': 'standalone stator model with parent-assembly input context',
        'checkpoint': str(arguments.checkpoint),
        'dataset': {
            'source': str(dataset.sourcePath),
            'splitStrategy': splitStrategy,
            'statorTrainCaseIds': [case.caseId for case in trainCases],
            'statorValidationCaseIds': [case.caseId for case in validationCases],
            'totalComponentCaseCount': len(assemblyCases),
            'statorTrainCaseCount': len(trainCases),
            'statorValidationCaseCount': len(validationCases),
            'trainParentGroups': sorted({_caseGroup(case) for case in trainCases}),
            'validationParentGroups': sorted(
                {_caseGroup(case) for case in validationCases}
            ),
            'absoluteTemperatureTrainCaseCount': len(anchoredTrainCases),
            'shapeOnlyTrainCaseCount': len(trainCases) - len(anchoredTrainCases),
            'absoluteTemperatureValidationCaseCount': len(
                anchoredValidationCases
            ),
            'seed': arguments.seed,
        },
        'model': {
            'family': 'Set-DeepONet',
            'config': modelConfig,
            'parameterCount': sum(value.numel() for value in model.parameters()),
        },
        'training': {
            'epochs': arguments.epochs,
            'learningRate': arguments.learning_rate,
            'bestEpochIndex': training['bestEpochIndex'],
            'bestValidationLoss': training['bestValidationLoss'],
            'elapsedSeconds': elapsedSeconds,
            'resolvedDeviceName': training['resolvedDeviceName'],
        },
        'trainingSetEvaluation': _summary(trainingRecords),
        'trainingCases': _publicRecords(trainingRecords),
        'evaluation': metrics,
        'cases': _publicRecords(records),
        'caveat': (
            'Only three parent-group-held-out stator cases are available. This '
            'is a component experiment, not an independent production estimate.'
        ),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
