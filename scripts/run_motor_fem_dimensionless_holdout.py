from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from mor.nn.data import (
    STEADY_THERMAL_DIMENSIONLESS_V2,
    buildMotorFemComponentCase,
    collateThermalOperatorCases,
    hasAbsoluteTemperatureAnchor,
    loadThermalDatasetAuto,
    randomSimilarityAugmentations,
    steadyThermalDimensionlessOperatorCase,
    subsampleThermalComponentCase,
    thermalComponentFeatureSchema,
    thermalOperatorCase,
)
from mor.nn.losses import thermalMeanShapeEnergyLoss
from mor.nn.models import thermalQuerySetOperatorTorch
from run_motor_fem_speed_holdout import (
    DEFAULT_COMPONENTS,
    DEFAULT_TEST_SPEEDS,
    HYBRID_ANCHORED_TARGET,
    _batchToDevice,
    _hybridUniformTemperatureBaseline,
    _numericSpeedDirectories,
    _publicRecords,
    _summary,
    _temperatureCenter,
)


OPERATING_SCALES = {
    'speedRpm': 25000.0,
    'absoluteTemperatureAnchor': 1.0,
}
SCHEMA = thermalComponentFeatureSchema(
    operatingFeatureNames=('speedRpm', 'absoluteTemperatureAnchor'),
    boundaryTypeCount=3,
    temperatureBoundaryTypeIds=(1, 2),
    fixedTemperatureBoundaryTypeIds=(2,),
)
MEAN_SHAPE_ANCHORED_TARGET = 'meanShapeAnchoredOrRelative'


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Train the shared motor thermal operator with per-FEM physical '
            'dimensionless features and evaluate held-out speeds.'
        )
    )
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--components', nargs='*', default=DEFAULT_COMPONENTS)
    parser.add_argument(
        '--additional-datasets',
        '--additional-component-datasets',
        dest='additional_datasets',
        nargs='*',
        type=Path,
        default=(),
        help=(
            'Additional component datasets, motor conversion output roots, or '
            'whole_motor views used for training.'
        ),
    )
    parser.add_argument('--test-speeds', nargs='*', default=DEFAULT_TEST_SPEEDS)
    parser.add_argument('--maximum-volume-samples', type=int, default=256)
    parser.add_argument('--maximum-boundary-samples', type=int, default=512)
    parser.add_argument('--maximum-query-samples', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=2e-3)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--kernel-scales', nargs='+', type=float, default=(0.12, 0.35))
    parser.add_argument('--query-fourier-frequencies', type=int, default=4)
    parser.add_argument('--minimum-temperature-scale-k', type=float, default=1.0)
    parser.add_argument('--similarity-augmentations-per-case', type=int, default=1)
    parser.add_argument('--similarity-length-range', nargs=2, type=float, default=(0.6, 1.8))
    parser.add_argument(
        '--similarity-conductivity-range', nargs=2, type=float, default=(0.6, 1.8)
    )
    parser.add_argument(
        '--similarity-temperature-range', nargs=2, type=float, default=(0.6, 1.8)
    )
    parser.add_argument('--energy-balance-weight', type=float, default=0.02)
    parser.add_argument('--shape-loss-weight', type=float, default=1.0)
    parser.add_argument('--mean-loss-weight', type=float, default=1.0)
    parser.add_argument('--reconstruction-loss-weight', type=float, default=0.1)
    parser.add_argument('--hotspot-loss-weight', type=float, default=0.25)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=7)
    return parser.parse_args()


def _dimensionlessCases(cases, minimumTemperatureScaleK: float):
    return tuple(
        steadyThermalDimensionlessOperatorCase(
            case,
            schema=SCHEMA,
            operatingScales=OPERATING_SCALES,
            minimumTemperatureScaleK=minimumTemperatureScaleK,
            featureMode=STEADY_THERMAL_DIMENSIONLESS_V2,
        )
        for case in cases
    )


def _prepareAdditionalTrainingCase(case):
    operating = dict(case.operatingFeatures)
    operating.setdefault('speedRpm', 0.0)
    fixed = np.isin(
        case.boundary.boundaryTypeIds,
        SCHEMA.fixedTemperatureBoundaryTypeIds,
    )
    anchored = bool(np.any(fixed) or np.any(case.boundary.htcWPerM2K > 0.0))
    operating.setdefault('absoluteTemperatureAnchor', float(anchored))
    return replace(case, operatingFeatures=operating)


def _additionalDatasetCases(datasetDirectory: Path):
    dataset = loadThermalDatasetAuto(datasetDirectory, view='components')
    motorIds = {
        group.motorId for group in dataset.groups if group.motorId is not None
    }
    datasetNamespace = (
        next(iter(motorIds))
        if len(motorIds) == 1
        else dataset.metadata.get('motorId') or datasetDirectory.stem
    )
    namespace = re.sub(
        r'[^A-Za-z0-9_-]+',
        '-',
        str(datasetNamespace),
    ).strip('-_')
    if not namespace:
        raise ValueError(f'cannot derive a case namespace from {datasetDirectory}')
    return dataset, tuple(
        replace(
            _prepareAdditionalTrainingCase(case),
            caseId=f'{namespace}--{case.caseId}',
            metadata={**case.metadata, 'originalCaseId': case.caseId},
        )
        for case in dataset.cases
    )


def _standardizedDatasetSource(source: Path) -> bool:
    return bool(
        (source / 'dataset.json').is_file()
        or (source / 'views.json').is_file()
        or (
            source.name.lower() == 'cases'
            and (source.parent / 'dataset.json').is_file()
        )
    )


def _speedToken(value: float | str) -> str:
    speed = float(value)
    return str(int(speed)) if speed.is_integer() else f'{speed:g}'


def _boundaryQueryBatch(batch: dict) -> dict:
    boundaryBatch = dict(batch)
    queryDim = batch['queryCoordinates'].shape[-1]
    boundaryBatch['queryCoordinates'] = batch['boundaryFeatures'][..., :queryDim]
    boundaryBatch['queryWeights'] = batch['boundaryWeights']
    boundaryBatch['queryMask'] = batch['boundaryMask']
    return boundaryBatch


def _trainV2(
    cases: Sequence[thermalOperatorCase],
    *,
    epochs: int,
    batchSize: int,
    learningRate: float,
    hiddenDim: int,
    kernelScales: Sequence[float],
    queryFourierFrequencies: int,
    energyBalanceWeight: float,
    shapeLossWeight: float,
    meanLossWeight: float,
    reconstructionLossWeight: float,
    hotspotLossWeight: float,
    device: str,
    seed: int,
) -> tuple[thermalQuerySetOperatorTorch, dict[str, float]]:
    if not cases:
        raise ValueError('training requires at least one case')
    torch.manual_seed(seed)
    generator = np.random.default_rng(seed)
    example = cases[0]
    model = thermalQuerySetOperatorTorch(
        volumeFeatureDim=example.volumeFeatureDim,
        boundaryFeatureDim=example.boundaryFeatureDim,
        globalFeatureDim=example.globalFeatureDim,
        queryDim=example.queryDim,
        hiddenDim=hiddenDim,
        kernelScales=kernelScales,
        queryFourierFrequencies=queryFourierFrequencies,
        includeRelativeKernelFeatures=True,
        separateBoundaryTypeContexts=True,
        boundaryTypeFeatureStart=9,
        boundaryTypeCount=SCHEMA.boundaryTypeCount,
        decomposeMeanAndShape=True,
        anchorFeatureIndex=-1,
    ).to(device)
    lossFunction = thermalMeanShapeEnergyLoss(
        shapeWeight=shapeLossWeight,
        meanWeight=meanLossWeight,
        gaugeWeight=0.1,
        reconstructionWeight=reconstructionLossWeight,
        hotspotWeight=hotspotLossWeight,
        energyBalanceWeight=energyBalanceWeight,
        fixedTemperatureBoundaryTypeIds=SCHEMA.fixedTemperatureBoundaryTypeIds,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
    firstLoss = None
    finalLoss = None
    start = time.perf_counter()
    model.train()
    for _ in range(int(epochs)):
        order = generator.permutation(len(cases))
        for startIndex in range(0, len(cases), int(batchSize)):
            selected = [
                cases[index]
                for index in order[startIndex : startIndex + batchSize]
            ]
            batch = _batchToDevice(collateThermalOperatorCases(selected), device)
            optimizer.zero_grad(set_to_none=True)
            modelOutput = model.forwardWithAux(batch)
            boundaryPrediction = (
                model(_boundaryQueryBatch(batch))
                if energyBalanceWeight > 0.0
                else None
            )
            loss = lossFunction.computeBatch(
                modelOutput,
                batch,
                boundaryPredictions=boundaryPrediction,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            finalLoss = float(loss.detach().cpu())
            if firstLoss is None:
                firstLoss = finalLoss
    return model, {
        'epochs': int(epochs),
        'elapsedSeconds': time.perf_counter() - start,
        'initialLoss': float(firstLoss),
        'finalLoss': float(finalLoss),
        'energyBalanceWeight': float(energyBalanceWeight),
        'shapeLossWeight': float(shapeLossWeight),
        'meanLossWeight': float(meanLossWeight),
        'reconstructionLossWeight': float(reconstructionLossWeight),
        'hotspotLossWeight': float(hotspotLossWeight),
    }


def _evaluate(model, componentCases, dimensionlessCases, device: str):
    records = []
    model.eval()
    with torch.no_grad():
        for componentCase, dimensionlessCase in zip(
            componentCases, dimensionlessCases
        ):
            batch = _batchToDevice(
                collateThermalOperatorCases([dimensionlessCase.operatorCase]),
                device,
            )
            prediction = model(batch)[0, :, 0]
            target = batch['targets'][0, :, 0]
            mask = batch['queryMask'][0]
            weights = torch.as_tensor(
                componentCase.temperature.weightsM3,
                dtype=prediction.dtype,
                device=prediction.device,
            )
            anchored = hasAbsoluteTemperatureAnchor(componentCase)
            if not anchored:
                weightTotal = weights.sum().clamp_min(
                    torch.finfo(weights.dtype).eps
                )
                prediction = prediction.clone()
                target = target.clone()
                prediction[mask] -= torch.sum(weights * prediction[mask]) / weightTotal
                target[mask] -= torch.sum(weights * target[mask]) / weightTotal
            errorK = (
                prediction[mask] - target[mask]
            ) * dimensionlessCase.scales.temperatureScaleK
            squaredErrorIntegral = float(torch.sum(weights * errorK.square()))
            weightSum = float(weights.sum())
            records.append(
                {
                    'caseId': componentCase.caseId,
                    'component': componentCase.componentType,
                    'speedRpm': componentCase.operatingFeatures['speedRpm'],
                    'rmseK': math.sqrt(squaredErrorIntegral / weightSum),
                    'meanAbsoluteErrorK': float(
                        torch.sum(weights * torch.abs(errorK)) / weightSum
                    ),
                    'maximumErrorK': float(torch.max(torch.abs(errorK))),
                    'temperatureRangeK': float(
                        np.ptp(componentCase.temperature.temperatureK)
                    ),
                    'temperatureWeightedMeanK': _temperatureCenter(componentCase),
                    'absoluteTemperatureAnchorAvailable': anchored,
                    'temperatureScaleK': (
                        dimensionlessCase.scales.temperatureScaleK
                    ),
                    'lengthScaleMeters': dimensionlessCase.scales.lengthMeters,
                    'conductivityScaleWPerMK': (
                        dimensionlessCase.scales.conductivityWPerMK
                    ),
                    'boundaryFluxTemperatureScaleK': (
                        dimensionlessCase.scales.boundaryFluxTemperatureScaleK
                    ),
                    'netConductionTemperatureScaleK': (
                        dimensionlessCase.scales.netConductionTemperatureScaleK
                    ),
                    'convectionTemperatureScaleK': (
                        dimensionlessCase.scales.convectionTemperatureScaleK
                    ),
                    '_squaredErrorIntegral': squaredErrorIntegral,
                    '_weightSum': weightSum,
                }
            )
    return records


def _speedInterpolatedUniformBaseline(trainCases, testCases):
    trainingByComponent = defaultdict(list)
    for case in trainCases:
        trainingByComponent[case.componentType].append(
            (
                float(case.operatingFeatures['speedRpm']),
                _temperatureCenter(case),
            )
        )
    records = []
    for case in testCases:
        anchored = hasAbsoluteTemperatureAnchor(case)
        weights = case.temperature.weightsM3
        target = case.temperature.temperatureK
        if anchored:
            samples = sorted(trainingByComponent[case.componentType])
            predictedMean = float(
                np.interp(
                    float(case.operatingFeatures['speedRpm']),
                    [sample[0] for sample in samples],
                    [sample[1] for sample in samples],
                )
            )
            errorK = predictedMean - target
        else:
            predictedMean = None
            errorK = _temperatureCenter(case) - target
        squaredErrorIntegral = float(np.dot(np.square(errorK), weights))
        weightSum = float(weights.sum())
        records.append(
            {
                'caseId': case.caseId,
                'component': case.componentType,
                'speedRpm': case.operatingFeatures['speedRpm'],
                'rmseK': math.sqrt(squaredErrorIntegral / weightSum),
                'meanAbsoluteErrorK': float(
                    np.dot(np.abs(errorK), weights) / weightSum
                ),
                'maximumErrorK': float(np.max(np.abs(errorK))),
                'temperatureRangeK': float(np.ptp(target)),
                'temperatureWeightedMeanK': _temperatureCenter(case),
                'absoluteTemperatureAnchorAvailable': anchored,
                'predictedUniformTemperatureK': predictedMean,
                '_squaredErrorIntegral': squaredErrorIntegral,
                '_weightSum': weightSum,
            }
        )
    return records


def main() -> None:
    arguments = _arguments()
    components = tuple(str(value) for value in arguments.components)
    device = (
        arguments.device
        if not arguments.device.startswith('cuda') or torch.cuda.is_available()
        else 'cpu'
    )
    if _standardizedDatasetSource(arguments.source):
        sourceDataset = loadThermalDatasetAuto(arguments.source, view='components')
        selectedCases = [
            case for case in sourceDataset.cases if case.componentType in components
        ]
        cases = [
            subsampleThermalComponentCase(
                case,
                maximumVolumeSamples=arguments.maximum_volume_samples,
                maximumBoundarySamples=arguments.maximum_boundary_samples,
                maximumQuerySamples=arguments.maximum_query_samples,
                randomSeed=arguments.seed + caseIndex,
            )
            for caseIndex, case in enumerate(selectedCases)
        ]
        evaluationCaseById = {
            case.caseId: subsampleThermalComponentCase(
                case,
                maximumVolumeSamples=arguments.maximum_volume_samples,
                maximumBoundarySamples=arguments.maximum_boundary_samples,
                maximumQuerySamples=None,
                randomSeed=arguments.seed + caseIndex,
            )
            for caseIndex, case in enumerate(selectedCases)
        }
        missingComponents = sorted(
            set(components) - {case.componentType for case in cases}
        )
        if missingComponents:
            raise ValueError(
                f'components are missing from the source dataset: {missingComponents}'
            )
        sourceMode = sourceDataset.viewType
        print(
            f'loaded {len(cases)} component cases from {sourceDataset.sourcePath}...',
            file=sys.stderr,
        )
    else:
        availableDirectories = _numericSpeedDirectories(arguments.source)
        print(
            f'building {len(components) * len(availableDirectories)} component cases...',
            file=sys.stderr,
        )
        cases = []
        for componentIndex, component in enumerate(components):
            for speed in availableDirectories:
                cases.append(
                    buildMotorFemComponentCase(
                        arguments.source / speed,
                        component,
                        maximumVolumeSamples=arguments.maximum_volume_samples,
                        maximumBoundarySamples=arguments.maximum_boundary_samples,
                        maximumQuerySamples=arguments.maximum_query_samples,
                        randomSeed=arguments.seed + componentIndex,
                    )
                )
        sourceMode = 'rawMotorDirectories'
        evaluationCaseById = {}

    availableSpeeds = tuple(
        sorted(
            {_speedToken(case.operatingFeatures['speedRpm']) for case in cases},
            key=float,
        )
    )
    testSpeeds = tuple(_speedToken(value) for value in arguments.test_speeds)
    missing = sorted(set(testSpeeds) - set(availableSpeeds), key=float)
    if missing:
        raise ValueError(f'test speeds are missing from the source dataset: {missing}')
    trainSpeeds = tuple(speed for speed in availableSpeeds if speed not in testSpeeds)
    trainCases = tuple(
        case
        for case in cases
        if _speedToken(case.operatingFeatures['speedRpm']) in trainSpeeds
    )
    testCases = tuple(
        evaluationCaseById.get(case.caseId, case)
        for case in cases
        if _speedToken(case.operatingFeatures['speedRpm']) in testSpeeds
    )
    additionalDatasetsAndCases = tuple(
        _additionalDatasetCases(datasetDirectory)
        for datasetDirectory in arguments.additional_datasets
    )
    additionalTrainCases = tuple(
        case
        for _, datasetCases in additionalDatasetsAndCases
        for case in datasetCases
    )
    baseTrainCases = trainCases + additionalTrainCases
    trainingCaseIds = [case.caseId for case in baseTrainCases]
    if len(trainingCaseIds) != len(set(trainingCaseIds)):
        raise ValueError(
            'motor and additional component datasets contain duplicate caseIds'
        )
    augmentedTrainCases = randomSimilarityAugmentations(
        baseTrainCases,
        augmentationsPerCase=arguments.similarity_augmentations_per_case,
        lengthRatioRange=tuple(arguments.similarity_length_range),
        conductivityRatioRange=tuple(arguments.similarity_conductivity_range),
        temperatureDifferenceRatioRange=tuple(
            arguments.similarity_temperature_range
        ),
        schema=SCHEMA,
        randomSeed=arguments.seed,
    )
    trainDimensionless = _dimensionlessCases(
        augmentedTrainCases, arguments.minimum_temperature_scale_k
    )
    testDimensionless = _dimensionlessCases(
        testCases, arguments.minimum_temperature_scale_k
    )
    model, training = _trainV2(
        tuple(case.operatorCase for case in trainDimensionless),
        epochs=arguments.epochs,
        batchSize=arguments.batch_size,
        learningRate=arguments.learning_rate,
        hiddenDim=arguments.hidden_dim,
        kernelScales=arguments.kernel_scales,
        queryFourierFrequencies=arguments.query_fourier_frequencies,
        energyBalanceWeight=arguments.energy_balance_weight,
        shapeLossWeight=arguments.shape_loss_weight,
        meanLossWeight=arguments.mean_loss_weight,
        reconstructionLossWeight=arguments.reconstruction_loss_weight,
        hotspotLossWeight=arguments.hotspot_loss_weight,
        device=device,
        seed=arguments.seed,
    )
    records = _evaluate(model, testCases, testDimensionless, device)
    baselineRecords = _hybridUniformTemperatureBaseline(testCases)
    fullQueryTrainCases = tuple(
        evaluationCaseById.get(case.caseId, case) for case in trainCases
    )
    speedBaselineRecords = _speedInterpolatedUniformBaseline(
        fullQueryTrainCases, testCases
    )
    byComponent = defaultdict(list)
    bySpeed = defaultdict(list)
    byAnchorMode = defaultdict(list)
    for record in records:
        byComponent[record['component']].append(record)
        bySpeed[record['speedRpm']].append(record)
        byAnchorMode[
            'anchoredAbsolute'
            if record['absoluteTemperatureAnchorAvailable']
            else 'floatingRelative'
        ].append(record)
    baselineByAnchorMode = defaultdict(list)
    for record in baselineRecords:
        baselineByAnchorMode[
            'anchoredAbsolute'
            if record['absoluteTemperatureAnchorAvailable']
            else 'floatingRelative'
        ].append(record)

    report = {
        'device': device,
        'source': {
            'path': str(arguments.source),
            'mode': sourceMode,
            'caseCount': len(cases),
            'fullQueryEvaluationForStandardizedDataset': bool(
                evaluationCaseById
            ),
        },
        'featureRepresentation': {
            'mode': STEADY_THERMAL_DIMENSIONLESS_V2,
            'description': (
                'Per-FEM physical dimensionless features with boundary-type '
                'integrals, source/flux spatial moments and mixed-boundary-safe '
                'temperature references.'
            ),
            'minimumTemperatureScaleK': arguments.minimum_temperature_scale_k,
            'operatingScales': OPERATING_SCALES,
        },
        'targetRepresentation': {
            'mode': MEAN_SHAPE_ANCHORED_TARGET,
            'meanTemperatureHeadForAnchoredCases': True,
            'zeroMeanShapeHeadForAllCases': True,
            'floatingCasesRemainRelative': True,
        },
        'similarityAugmentation': {
            'augmentationsPerTrainingCase': (
                arguments.similarity_augmentations_per_case
            ),
            'originalTrainingCaseCount': len(baseTrainCases),
            'effectiveTrainingCaseCount': len(augmentedTrainCases),
            'lengthRatioRange': list(arguments.similarity_length_range),
            'conductivityRatioRange': list(
                arguments.similarity_conductivity_range
            ),
            'temperatureDifferenceRatioRange': list(
                arguments.similarity_temperature_range
            ),
        },
        'additionalDatasets': [
            {
                'path': str(path),
                'resolvedView': dataset.viewType,
                'caseCount': len(dataset.cases),
                'groupCount': len(dataset.groups),
                'motorIds': sorted(
                    {
                        group.motorId
                        for group in dataset.groups
                        if group.motorId is not None
                    }
                ),
            }
            for path, (dataset, _) in zip(
                arguments.additional_datasets, additionalDatasetsAndCases
            )
        ],
        'additionalTrainingCaseCount': len(additionalTrainCases),
        'components': components,
        'trainSpeedsRpm': [float(speed) for speed in trainSpeeds],
        'testSpeedsRpm': [float(speed) for speed in testSpeeds],
        'training': training,
        'model': {
            'overall': _summary(records),
            'byComponent': {
                component: _summary(byComponent[component])
                for component in components
            },
            'bySpeed': {
                str(int(float(speed))): _summary(bySpeed[float(speed)])
                for speed in testSpeeds
            },
            'byAnchorMode': {
                mode: _summary(modeRecords)
                for mode, modeRecords in byAnchorMode.items()
            },
            'cases': _publicRecords(records),
        },
        'hybridUniformTemperatureBaseline': {
            'overall': _summary(baselineRecords),
            'byAnchorMode': {
                mode: _summary(modeRecords)
                for mode, modeRecords in baselineByAnchorMode.items()
            },
        },
        'speedInterpolatedUniformTemperatureBaseline': {
            'description': (
                'Per-component uniform temperature using linear interpolation '
                'of training-speed weighted means; floating cases use their '
                'zero-mean gauge.'
            ),
            'overall': _summary(speedBaselineRecords),
            'cases': _publicRecords(speedBaselineRecords),
        },
    }

    if arguments.checkpoint is not None:
        arguments.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        example = trainDimensionless[0].operatorCase
        torch.save(
            {
                'stateDict': model.state_dict(),
                'model': {
                    'volumeFeatureDim': example.volumeFeatureDim,
                    'boundaryFeatureDim': example.boundaryFeatureDim,
                    'globalFeatureDim': example.globalFeatureDim,
                    'queryDim': example.queryDim,
                    'hiddenDim': arguments.hidden_dim,
                    'kernelScales': list(arguments.kernel_scales),
                    'queryFourierFrequencies': arguments.query_fourier_frequencies,
                    'includeRelativeKernelFeatures': True,
                    'separateBoundaryTypeContexts': True,
                    'boundaryTypeFeatureStart': 9,
                    'boundaryTypeCount': SCHEMA.boundaryTypeCount,
                    'decomposeMeanAndShape': True,
                    'anchorFeatureIndex': -1,
                },
                'schema': asdict(SCHEMA),
                'targetMode': MEAN_SHAPE_ANCHORED_TARGET,
                'featureMode': STEADY_THERMAL_DIMENSIONLESS_V2,
                'dimensionless': {
                    'operatingScales': OPERATING_SCALES,
                    'minimumTemperatureScaleK': (
                        arguments.minimum_temperature_scale_k
                    ),
                    'energyBalanceWeight': arguments.energy_balance_weight,
                },
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
