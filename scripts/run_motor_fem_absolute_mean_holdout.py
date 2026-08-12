from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from mor.nn.data import (
    STEADY_THERMAL_DIMENSIONLESS_V2,
    collateThermalOperatorCases,
    loadThermalDatasetAuto,
    steadyThermalDimensionlessOperatorCase,
    subsampleThermalComponentCase,
    thermalComponentFeatureSchema,
)
from mor.nn.models import thermalQuerySetOperatorTorch
try:
    from run_motor_fem_speed_holdout import (
        DEFAULT_COMPONENTS,
        DEFAULT_TEST_SPEEDS,
        _batchToDevice,
    )
except ModuleNotFoundError:
    from scripts.run_motor_fem_speed_holdout import (
        DEFAULT_COMPONENTS,
        DEFAULT_TEST_SPEEDS,
        _batchToDevice,
    )


TEMPERATURE_REFERENCE_K = 300.0
TEMPERATURE_SCALE_K = 100.0
OPERATING_SCALES = {'speedRpm': 25000.0, 'absoluteTemperatureAnchor': 1.0}
SCHEMA = thermalComponentFeatureSchema(
    operatingFeatureNames=('speedRpm', 'absoluteTemperatureAnchor'),
    boundaryTypeCount=3,
    temperatureBoundaryTypeIds=(1, 2),
    fixedTemperatureBoundaryTypeIds=(2,),
)
TARGET_MODE = 'absoluteVolumeWeightedMeanTemperature'


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Train and evaluate one absolute volume-weighted mean temperature '
            'for every motor FEM component without relative-temperature gauges.'
        )
    )
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--components', nargs='*', default=DEFAULT_COMPONENTS)
    parser.add_argument('--test-speeds', nargs='*', default=DEFAULT_TEST_SPEEDS)
    parser.add_argument('--maximum-volume-samples', type=int, default=256)
    parser.add_argument('--maximum-boundary-samples', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=2e-3)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=7)
    return parser.parse_args()


def _speedToken(value: float | str) -> str:
    speed = float(value)
    return str(int(speed)) if speed.is_integer() else f'{speed:g}'


def _temperatureMeanK(case) -> float:
    return float(
        np.average(
            case.temperature.temperatureK,
            weights=case.temperature.weightsM3,
        )
    )


def _operatorCase(case):
    return steadyThermalDimensionlessOperatorCase(
        case,
        schema=SCHEMA,
        operatingScales=OPERATING_SCALES,
        minimumTemperatureScaleK=1.0,
        featureMode=STEADY_THERMAL_DIMENSIONLESS_V2,
    ).operatorCase


def _targets(cases, device: str) -> torch.Tensor:
    return torch.as_tensor(
        [
            (_temperatureMeanK(case) - TEMPERATURE_REFERENCE_K)
            / TEMPERATURE_SCALE_K
            for case in cases
        ],
        dtype=torch.float32,
        device=device,
    )[:, None]


def _train(
    cases,
    *,
    epochs: int,
    batchSize: int,
    learningRate: float,
    hiddenDim: int,
    device: str,
    seed: int,
):
    torch.manual_seed(seed)
    generator = np.random.default_rng(seed)
    operatorCases = tuple(_operatorCase(case) for case in cases)
    example = operatorCases[0]
    model = thermalQuerySetOperatorTorch(
        volumeFeatureDim=example.volumeFeatureDim,
        boundaryFeatureDim=example.boundaryFeatureDim,
        globalFeatureDim=example.globalFeatureDim,
        queryDim=example.queryDim,
        hiddenDim=hiddenDim,
        kernelScales=(0.12, 0.35),
        queryFourierFrequencies=0,
        includeRelativeKernelFeatures=True,
        separateBoundaryTypeContexts=True,
        boundaryTypeFeatureStart=9,
        boundaryTypeCount=SCHEMA.boundaryTypeCount,
        decomposeMeanAndShape=True,
        anchorFeatureIndex=-1,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
    firstLoss = None
    finalLoss = None
    start = time.perf_counter()
    model.train()
    for _ in range(int(epochs)):
        order = generator.permutation(len(cases))
        for startIndex in range(0, len(cases), int(batchSize)):
            indices = order[startIndex : startIndex + int(batchSize)]
            selectedOperators = [operatorCases[index] for index in indices]
            selectedCases = [cases[index] for index in indices]
            batch = _batchToDevice(
                collateThermalOperatorCases(selectedOperators), device
            )
            prediction = model.forwardMean(batch)
            target = _targets(selectedCases, device)
            loss = torch.mean(torch.square(prediction - target))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            finalLoss = float(loss.detach().cpu())
            if firstLoss is None:
                firstLoss = finalLoss
    return model, operatorCases, {
        'epochs': int(epochs),
        'elapsedSeconds': time.perf_counter() - start,
        'initialLoss': float(firstLoss),
        'finalLoss': float(finalLoss),
    }


def _evaluate(model, cases, device: str):
    records = []
    model.eval()
    with torch.no_grad():
        for case in cases:
            batch = _batchToDevice(
                collateThermalOperatorCases([_operatorCase(case)]), device
            )
            predicted = float(model.forwardMean(batch)[0, 0].cpu())
            predictedK = TEMPERATURE_REFERENCE_K + TEMPERATURE_SCALE_K * predicted
            targetK = _temperatureMeanK(case)
            errorK = predictedK - targetK
            records.append(
                {
                    'caseId': case.caseId,
                    'component': case.componentType,
                    'speedRpm': float(case.operatingFeatures['speedRpm']),
                    'predictedMeanTemperatureK': predictedK,
                    'targetMeanTemperatureK': targetK,
                    'errorK': errorK,
                    'absoluteErrorK': abs(errorK),
                    'squaredErrorK2': errorK**2,
                }
            )
    return records


def _summary(records):
    errors = np.asarray([record['errorK'] for record in records])
    return {
        'caseCount': len(records),
        'rmseK': float(np.sqrt(np.mean(np.square(errors)))),
        'meanAbsoluteErrorK': float(np.mean(np.abs(errors))),
        'maximumAbsoluteErrorK': float(np.max(np.abs(errors))),
        'meanBiasK': float(np.mean(errors)),
    }


def _speedInterpolationBaseline(trainCases, testCases):
    training = defaultdict(list)
    for case in trainCases:
        training[case.componentType].append(
            (float(case.operatingFeatures['speedRpm']), _temperatureMeanK(case))
        )
    records = []
    for case in testCases:
        samples = sorted(training[case.componentType])
        predictedK = float(
            np.interp(
                float(case.operatingFeatures['speedRpm']),
                [sample[0] for sample in samples],
                [sample[1] for sample in samples],
            )
        )
        targetK = _temperatureMeanK(case)
        errorK = predictedK - targetK
        records.append(
            {
                'caseId': case.caseId,
                'component': case.componentType,
                'speedRpm': float(case.operatingFeatures['speedRpm']),
                'predictedMeanTemperatureK': predictedK,
                'targetMeanTemperatureK': targetK,
                'errorK': errorK,
                'absoluteErrorK': abs(errorK),
                'squaredErrorK2': errorK**2,
            }
        )
    return records


def _mapFieldToReference(
    sourceCoordinates: np.ndarray,
    sourceValues: np.ndarray,
    referenceCoordinates: np.ndarray,
    *,
    neighborCount: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    count = min(int(neighborCount), len(sourceCoordinates))
    if count <= 0:
        raise ValueError('neighborCount must be positive')
    distances, indices = cKDTree(sourceCoordinates).query(
        referenceCoordinates, k=count
    )
    if count == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    nearestDistances = distances[:, 0].copy()
    tolerance = np.finfo(np.float64).eps * max(
        1.0, float(np.max(np.abs(referenceCoordinates)))
    )
    exact = nearestDistances <= tolerance
    safeDistances = np.maximum(distances, tolerance)
    weights = 1.0 / np.square(safeDistances)
    mapped = np.sum(weights * sourceValues[indices], axis=1) / np.sum(
        weights, axis=1
    )
    mapped[exact] = sourceValues[indices[exact, 0]]
    return mapped, nearestDistances


def _podCoefficientInterpolationBaselines(
    trainCases,
    testCases,
    *,
    requestedRanks=(1, 2, 4, None),
    neighborCount: int = 8,
):
    training = defaultdict(list)
    testing = defaultdict(list)
    for case in trainCases:
        training[case.componentType].append(case)
    for case in testCases:
        testing[case.componentType].append(case)

    recordsByRank = defaultdict(list)
    componentDiagnostics = {}
    for component, componentTrainCases in sorted(training.items()):
        componentTrainCases = sorted(
            componentTrainCases,
            key=lambda case: float(case.operatingFeatures['speedRpm']),
        )
        componentTestCases = sorted(
            testing[component],
            key=lambda case: float(case.operatingFeatures['speedRpm']),
        )
        geometryIds = {case.geometryId for case in componentTrainCases + componentTestCases}
        if len(geometryIds) != 1:
            raise ValueError(
                f'POD interpolation requires one geometry per component; '
                f'{component!r} has {sorted(geometryIds)}'
            )
        reference = componentTrainCases[len(componentTrainCases) // 2]
        referenceCoordinates = reference.temperature.coordinatesMeters
        referenceWeights = reference.temperature.weightsM3.astype(np.float64)
        referenceWeights = referenceWeights / referenceWeights.sum()
        snapshots = []
        nearestDistances = []
        for case in componentTrainCases:
            mapped, distances = _mapFieldToReference(
                case.temperature.coordinatesMeters,
                case.temperature.temperatureK,
                referenceCoordinates,
                neighborCount=neighborCount,
            )
            mapped += _temperatureMeanK(case) - float(
                np.dot(referenceWeights, mapped)
            )
            snapshots.append(mapped)
            nearestDistances.append(distances)
        snapshotMatrix = np.column_stack(snapshots)
        meanField = np.mean(snapshotMatrix, axis=1)
        centered = snapshotMatrix - meanField[:, None]
        sqrtWeights = np.sqrt(referenceWeights)
        weightedSnapshots = sqrtWeights[:, None] * centered
        left, singularValues, _ = np.linalg.svd(
            weightedSnapshots, full_matrices=False
        )
        tolerance = (
            np.finfo(np.float64).eps
            * max(weightedSnapshots.shape)
            * (singularValues[0] if singularValues.size else 0.0)
        )
        availableRank = max(1, int(np.count_nonzero(singularValues > tolerance)))
        modes = left[:, :availableRank] / sqrtWeights[:, None]
        coefficients = left[:, :availableRank].T @ weightedSnapshots
        energy = np.square(singularValues[:availableRank])
        cumulativeEnergy = (
            np.cumsum(energy) / energy.sum()
            if energy.sum() > 0.0
            else np.ones_like(energy)
        )
        trainSpeeds = np.asarray(
            [case.operatingFeatures['speedRpm'] for case in componentTrainCases],
            dtype=np.float64,
        )
        resolvedRanks = sorted(
            {
                min(availableRank, availableRank if rank is None else int(rank))
                for rank in requestedRanks
            }
        )
        for case in componentTestCases:
            speed = float(case.operatingFeatures['speedRpm'])
            for rank in resolvedRanks:
                interpolatedCoefficients = np.asarray(
                    [
                        np.interp(speed, trainSpeeds, coefficients[index])
                        for index in range(rank)
                    ]
                )
                reconstructed = meanField + modes[:, :rank] @ interpolatedCoefficients
                predictedK = float(np.dot(referenceWeights, reconstructed))
                targetK = _temperatureMeanK(case)
                errorK = predictedK - targetK
                recordsByRank[rank].append(
                    {
                        'caseId': case.caseId,
                        'component': component,
                        'speedRpm': speed,
                        'podRank': rank,
                        'predictedMeanTemperatureK': predictedK,
                        'targetMeanTemperatureK': targetK,
                        'errorK': errorK,
                        'absoluteErrorK': abs(errorK),
                        'squaredErrorK2': errorK**2,
                    }
                )
        stackedDistances = np.concatenate(nearestDistances)
        componentDiagnostics[component] = {
            'referenceCaseId': reference.caseId,
            'referenceQueryCount': len(referenceCoordinates),
            'neighborCount': min(neighborCount, len(referenceCoordinates)),
            'availableRank': availableRank,
            'singularValues': singularValues[:availableRank].tolist(),
            'cumulativeEnergyByRank': {
                str(index + 1): float(value)
                for index, value in enumerate(cumulativeEnergy)
            },
            'trainingMappingNearestDistanceMeanM': float(
                np.mean(stackedDistances)
            ),
            'trainingMappingNearestDistanceMaximumM': float(
                np.max(stackedDistances)
            ),
        }
    return dict(recordsByRank), componentDiagnostics


def main() -> None:
    arguments = _arguments()
    source = loadThermalDatasetAuto(arguments.source, view='components')
    components = tuple(str(value) for value in arguments.components)
    selected = tuple(
        case for case in source.cases if case.componentType in components
    )
    missingComponents = sorted(
        set(components) - {case.componentType for case in selected}
    )
    if missingComponents:
        raise ValueError(f'components are missing from source: {missingComponents}')
    cases = tuple(
        subsampleThermalComponentCase(
            case,
            maximumVolumeSamples=arguments.maximum_volume_samples,
            maximumBoundarySamples=arguments.maximum_boundary_samples,
            maximumQuerySamples=None,
            randomSeed=arguments.seed + caseIndex,
        )
        for caseIndex, case in enumerate(selected)
    )
    availableSpeeds = {
        _speedToken(case.operatingFeatures['speedRpm']) for case in cases
    }
    testSpeeds = {_speedToken(value) for value in arguments.test_speeds}
    missingSpeeds = sorted(testSpeeds - availableSpeeds, key=float)
    if missingSpeeds:
        raise ValueError(f'test speeds are missing from source: {missingSpeeds}')
    trainCases = tuple(
        case
        for case in cases
        if _speedToken(case.operatingFeatures['speedRpm']) not in testSpeeds
    )
    testCases = tuple(
        case
        for case in cases
        if _speedToken(case.operatingFeatures['speedRpm']) in testSpeeds
    )
    device = (
        arguments.device
        if not arguments.device.startswith('cuda') or torch.cuda.is_available()
        else 'cpu'
    )
    print(
        f'training absolute mean temperature on {len(trainCases)} cases...',
        file=sys.stderr,
    )
    model, operatorCases, training = _train(
        trainCases,
        epochs=arguments.epochs,
        batchSize=arguments.batch_size,
        learningRate=arguments.learning_rate,
        hiddenDim=arguments.hidden_dim,
        device=device,
        seed=arguments.seed,
    )
    records = _evaluate(model, testCases, device)
    baselineRecords = _speedInterpolationBaseline(trainCases, testCases)
    podRecordsByRank, podDiagnostics = _podCoefficientInterpolationBaselines(
        trainCases, testCases
    )
    byComponent = defaultdict(list)
    bySpeed = defaultdict(list)
    for record in records:
        byComponent[record['component']].append(record)
        bySpeed[record['speedRpm']].append(record)
    report = {
        'device': device,
        'source': str(source.sourcePath),
        'targetRepresentation': {
            'mode': TARGET_MODE,
            'temperatureReferenceK': TEMPERATURE_REFERENCE_K,
            'temperatureScaleK': TEMPERATURE_SCALE_K,
            'relativeGaugeRemoved': False,
            'fieldShapeEvaluated': False,
            'allComponentsUseAbsoluteTargets': True,
        },
        'components': components,
        'trainSpeedsRpm': sorted(
            {float(case.operatingFeatures['speedRpm']) for case in trainCases}
        ),
        'testSpeedsRpm': sorted(
            {float(case.operatingFeatures['speedRpm']) for case in testCases}
        ),
        'training': training,
        'model': {
            'overall': _summary(records),
            'byComponent': {
                component: _summary(byComponent[component])
                for component in components
            },
            'bySpeed': {
                _speedToken(speed): _summary(speedRecords)
                for speed, speedRecords in sorted(bySpeed.items())
            },
            'cases': records,
        },
        'speedInterpolatedMeanTemperatureBaseline': {
            'overall': _summary(baselineRecords),
            'cases': baselineRecords,
        },
        'podCoefficientInterpolationBaselines': {
            'description': (
                'Per-component weighted POD on a common training reference '
                'point set, followed by linear interpolation of POD '
                'coefficients over speed. Source fields are mapped by '
                'inverse-distance nearest-neighbor interpolation because '
                'saved query samples differ between speeds.'
            ),
            'byRank': {
                str(rank): {
                    'overall': _summary(rankRecords),
                    'cases': rankRecords,
                }
                for rank, rankRecords in sorted(podRecordsByRank.items())
            },
            'componentDiagnostics': podDiagnostics,
        },
    }
    if arguments.checkpoint is not None:
        arguments.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        example = operatorCases[0]
        torch.save(
            {
                'stateDict': model.state_dict(),
                'model': {
                    'volumeFeatureDim': example.volumeFeatureDim,
                    'boundaryFeatureDim': example.boundaryFeatureDim,
                    'globalFeatureDim': example.globalFeatureDim,
                    'queryDim': example.queryDim,
                    'hiddenDim': arguments.hidden_dim,
                    'kernelScales': [0.12, 0.35],
                    'queryFourierFrequencies': 0,
                    'includeRelativeKernelFeatures': True,
                    'separateBoundaryTypeContexts': True,
                    'boundaryTypeFeatureStart': 9,
                    'boundaryTypeCount': SCHEMA.boundaryTypeCount,
                    'decomposeMeanAndShape': True,
                    'anchorFeatureIndex': -1,
                },
                'schema': asdict(SCHEMA),
                'targetMode': TARGET_MODE,
                'temperatureReferenceK': TEMPERATURE_REFERENCE_K,
                'temperatureScaleK': TEMPERATURE_SCALE_K,
            },
            arguments.checkpoint,
        )
        report['checkpoint'] = str(arguments.checkpoint)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
