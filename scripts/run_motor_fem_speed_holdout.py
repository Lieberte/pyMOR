from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from mor.nn.data import (
    buildMotorFemComponentCase,
    collateThermalOperatorCases,
    thermalComponentCase,
    thermalComponentFeatureSchema,
    thermalOperatorCase,
    thermalOperatorFeatureScaling,
)
from mor.nn.losses import thermalOperatorLoss
from mor.nn.models import thermalQuerySetOperatorTorch


DEFAULT_COMPONENTS = (
    'stator',
    'magnets',
    'shaft',
    'rotor_core',
    'disk1',
    'housing',
    'winding',
    'disk2',
)
DEFAULT_TEST_SPEEDS = ('7500', '15000', '22500')
HYBRID_ANCHORED_TARGET = 'hybridAnchoredOrRelative'


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Train thermal component operators on selected motor speeds and evaluate '
            'held-out speed cases with shared and per-component models.'
        )
    )
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--components', nargs='*', default=DEFAULT_COMPONENTS)
    parser.add_argument('--test-speeds', nargs='*', default=DEFAULT_TEST_SPEEDS)
    parser.add_argument('--maximum-volume-samples', type=int, default=256)
    parser.add_argument('--maximum-boundary-samples', type=int, default=512)
    parser.add_argument('--maximum-query-samples', type=int, default=256)
    parser.add_argument('--shared-epochs', type=int, default=300)
    parser.add_argument('--component-epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=2e-3)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--kernel-scales', nargs='+', type=float, default=(0.12, 0.35))
    parser.add_argument('--query-fourier-frequencies', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--skip-component-models', action='store_true')
    return parser.parse_args()


def _numericSpeedDirectories(root: Path) -> tuple[str, ...]:
    speeds = tuple(
        path.name
        for path in root.iterdir()
        if path.is_dir() and path.name.replace('.', '', 1).isdigit()
    )
    return tuple(sorted(speeds, key=float))


def _temperatureCenter(case: thermalComponentCase) -> float:
    return float(
        np.average(
            case.temperature.temperatureK,
            weights=case.temperature.weightsM3,
        )
    )


def _hasAbsoluteTemperatureAnchor(case: thermalComponentCase) -> bool:
    return bool(case.operatingFeatures.get('absoluteTemperatureAnchor', 0.0))


def _anchorReferenceTemperature(case: thermalComponentCase) -> float:
    if not _hasAbsoluteTemperatureAnchor(case):
        raise ValueError(f'case has no absolute temperature anchor: {case.caseId}')
    convection = case.boundary.htcWPerM2K > 0.0
    fixedTemperature = case.boundary.boundaryTypeIds == 2
    active = convection | fixedTemperature
    weights = case.boundary.weightsM2[active].copy()
    weights[convection[active]] *= case.boundary.htcWPerM2K[active][
        convection[active]
    ]
    return float(
        np.average(case.boundary.ambientTemperatureK[active], weights=weights)
    )


def _targetTemperatureK(case: thermalComponentCase) -> np.ndarray:
    reference = (
        _anchorReferenceTemperature(case)
        if _hasAbsoluteTemperatureAnchor(case)
        else _temperatureCenter(case)
    )
    return case.temperature.temperatureK - reference


def _caseInTargetTemperatureGauge(
    case: thermalComponentCase,
    scaling: thermalOperatorFeatureScaling,
) -> thermalComponentCase:
    if not _hasAbsoluteTemperatureAnchor(case):
        return case
    reference = _anchorReferenceTemperature(case)
    return replace(
        case,
        boundary=replace(
            case.boundary,
            ambientTemperatureK=(
                case.boundary.ambientTemperatureK
                - reference
                + scaling.temperatureReferenceK
            ),
        ),
    )


def _scaling(
    cases: Sequence[thermalComponentCase],
    targetMode: str = HYBRID_ANCHORED_TARGET,
) -> thermalOperatorFeatureScaling:
    if not cases:
        raise ValueError('scaling requires at least one training case')
    if targetMode != HYBRID_ANCHORED_TARGET:
        raise ValueError(f'unsupported motor temperature target mode: {targetMode!r}')
    targetTemperatures = tuple(_targetTemperatureK(case) for case in cases)
    temperatureMinimum = min(float(values.min()) for values in targetTemperatures)
    temperatureMaximum = max(float(values.max()) for values in targetTemperatures)
    anchorReferences = tuple(
        _anchorReferenceTemperature(case)
        for case in cases
        if _hasAbsoluteTemperatureAnchor(case)
    )
    return thermalOperatorFeatureScaling(
        lengthMeters=max(float(np.max(case.coordinateFrame.halfExtentMeters)) for case in cases),
        conductivityWPerMK=max(
            float(np.max(np.abs(case.volume.conductivityWPerMK))) for case in cases
        ),
        densityKgPerM3=max(float(case.volume.densityKgPerM3.max()) for case in cases),
        specificHeatJPerKgK=max(
            float(case.volume.specificHeatJPerKgK.max()) for case in cases
        ),
        heatSourceWPerM3=max(
            1.0,
            max(float(np.max(np.abs(case.volume.heatSourceWPerM3))) for case in cases),
        ),
        htcWPerM2K=max(
            1.0,
            max(float(np.max(np.abs(case.boundary.htcWPerM2K))) for case in cases),
        ),
        heatFluxWPerM2=max(
            1.0,
            max(float(np.max(np.abs(case.boundary.heatFluxWPerM2))) for case in cases),
        ),
        temperatureReferenceK=(
            float(np.mean(anchorReferences)) if anchorReferences else 0.0
        ),
        temperatureScaleK=max(temperatureMaximum - temperatureMinimum, 0.01),
        operatingScales={
            'speedRpm': 25000.0,
            'absoluteTemperatureAnchor': 1.0,
        },
    )


def _operatorCases(
    cases: Sequence[thermalComponentCase],
    scaling: thermalOperatorFeatureScaling,
    targetMode: str = HYBRID_ANCHORED_TARGET,
) -> tuple[thermalOperatorCase, ...]:
    schema = thermalComponentFeatureSchema(
        operatingFeatureNames=('speedRpm', 'absoluteTemperatureAnchor'),
        boundaryTypeCount=3,
        temperatureBoundaryTypeIds=(1, 2),
    )
    if targetMode != HYBRID_ANCHORED_TARGET:
        raise ValueError(f'unsupported motor temperature target mode: {targetMode!r}')
    operatorCases = []
    for case in cases:
        operatorCase = _caseInTargetTemperatureGauge(case, scaling).toOperatorCase(
            schema=schema,
            scaling=scaling,
        )
        targetTemperature = _targetTemperatureK(case) / scaling.temperatureScaleK
        operatorCases.append(
            replace(
                operatorCase,
                queryTemperatures=targetTemperature[:, None],
            )
        )
    return tuple(operatorCases)


def _batchToDevice(batch, device: str):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _balancedQueryWeights(batch: dict) -> torch.Tensor:
    weights = batch['queryWeights'] * batch['queryMask'].to(
        dtype=batch['queryWeights'].dtype
    )
    totals = weights.sum(dim=1, keepdim=True).clamp_min(
        torch.finfo(weights.dtype).eps
    )
    return weights / totals


def _applyHybridGauge(prediction: torch.Tensor, batch: dict) -> torch.Tensor:
    weights = _balancedQueryWeights(batch)
    predictionMean = torch.sum(
        weights.unsqueeze(-1) * prediction, dim=1, keepdim=True
    )
    anchorAvailable = batch['globalFeatures'][:, -1] > 0.5
    return torch.where(
        anchorAvailable[:, None, None],
        prediction,
        prediction - predictionMean,
    )


def _hybridBalancedLoss(prediction: torch.Tensor, batch: dict) -> torch.Tensor:
    prediction = _applyHybridGauge(prediction, batch)
    weights = _balancedQueryWeights(batch)
    squaredError = (prediction - batch['targets']).square().squeeze(-1)
    caseMse = torch.sum(weights * squaredError, dim=1)
    targetRmsSquared = torch.sum(
        weights * batch['targets'].squeeze(-1).square(), dim=1
    )
    minimumScaleSquared = 2.0e-3**2
    relativeCaseLoss = caseMse / targetRmsSquared.clamp_min(minimumScaleSquared)
    return relativeCaseLoss.mean() + 0.1 * caseMse.mean()


def _newModel(
    example: thermalOperatorCase,
    *,
    hiddenDim: int,
    kernelScales: Sequence[float],
    queryFourierFrequencies: int,
    device: str,
) -> thermalQuerySetOperatorTorch:
    return thermalQuerySetOperatorTorch(
        volumeFeatureDim=example.volumeFeatureDim,
        boundaryFeatureDim=example.boundaryFeatureDim,
        globalFeatureDim=example.globalFeatureDim,
        queryDim=example.queryDim,
        hiddenDim=hiddenDim,
        kernelScales=kernelScales,
        queryFourierFrequencies=queryFourierFrequencies,
    ).to(device)


def _train(
    cases: Sequence[thermalOperatorCase],
    *,
    epochs: int,
    batchSize: int,
    learningRate: float,
    hiddenDim: int,
    kernelScales: Sequence[float],
    queryFourierFrequencies: int,
    device: str,
    seed: int,
) -> tuple[thermalQuerySetOperatorTorch, dict[str, float]]:
    if not cases:
        raise ValueError('training requires at least one case')
    torch.manual_seed(seed)
    generator = np.random.default_rng(seed)
    model = _newModel(
        cases[0],
        hiddenDim=hiddenDim,
        kernelScales=kernelScales,
        queryFourierFrequencies=queryFourierFrequencies,
        device=device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
    firstLoss = None
    finalLoss = None
    start = time.perf_counter()
    model.train()
    for _ in range(int(epochs)):
        order = generator.permutation(len(cases))
        for startIndex in range(0, len(cases), int(batchSize)):
            selected = [cases[index] for index in order[startIndex : startIndex + batchSize]]
            batch = _batchToDevice(collateThermalOperatorCases(selected), device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch)
            loss = _hybridBalancedLoss(prediction, batch)
            loss.backward()
            optimizer.step()
            finalLoss = float(loss.detach().cpu())
            if firstLoss is None:
                firstLoss = finalLoss
    return model, {
        'epochs': int(epochs),
        'elapsedSeconds': time.perf_counter() - start,
        'initialLoss': float(firstLoss),
        'finalLoss': float(finalLoss),
    }


def _evaluate(
    model: thermalQuerySetOperatorTorch,
    componentCases: Sequence[thermalComponentCase],
    operatorCases: Sequence[thermalOperatorCase],
    scaling: thermalOperatorFeatureScaling,
    device: str,
    targetMode: str = HYBRID_ANCHORED_TARGET,
) -> list[dict]:
    if targetMode != HYBRID_ANCHORED_TARGET:
        raise ValueError(f'unsupported motor temperature target mode: {targetMode!r}')
    records = []
    model.eval()
    with torch.no_grad():
        for componentCase, operatorCase in zip(componentCases, operatorCases):
            batch = _batchToDevice(collateThermalOperatorCases([operatorCase]), device)
            prediction = model(batch)[0, :, 0]
            target = batch['targets'][0, :, 0]
            mask = batch['queryMask'][0]
            weights = batch['queryWeights'][0, mask]
            anchored = _hasAbsoluteTemperatureAnchor(componentCase)
            if not anchored:
                weightTotal = weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
                prediction = prediction.clone()
                prediction[mask] -= torch.sum(weights * prediction[mask]) / weightTotal
                target = target.clone()
                target[mask] -= torch.sum(weights * target[mask]) / weightTotal
            errorK = (prediction[mask] - target[mask]) * scaling.temperatureScaleK
            squaredErrorIntegral = float(
                torch.sum(weights * errorK.square()).detach().cpu()
            )
            weightSum = float(weights.sum().detach().cpu())
            records.append(
                {
                    'caseId': componentCase.caseId,
                    'component': componentCase.componentType,
                    'speedRpm': componentCase.operatingFeatures['speedRpm'],
                    'rmseK': math.sqrt(squaredErrorIntegral / weightSum),
                    'meanAbsoluteErrorK': float(
                        torch.sum(weights * torch.abs(errorK)).detach().cpu()
                    )
                    / weightSum,
                    'maximumErrorK': float(torch.max(torch.abs(errorK)).detach().cpu()),
                    'temperatureRangeK': float(np.ptp(componentCase.temperature.temperatureK)),
                    'temperatureWeightedMeanK': _temperatureCenter(componentCase),
                    'absoluteTemperatureAnchorAvailable': anchored,
                    'anchorReferenceTemperatureK': (
                        _anchorReferenceTemperature(componentCase) if anchored else None
                    ),
                    'targetRepresentation': targetMode,
                    '_squaredErrorIntegral': squaredErrorIntegral,
                    '_weightSum': weightSum,
                }
            )
    return records


def _summary(records: Sequence[dict]) -> dict[str, float | int]:
    squaredErrorIntegral = sum(record['_squaredErrorIntegral'] for record in records)
    weightSum = sum(record['_weightSum'] for record in records)
    return {
        'caseCount': len(records),
        'volumeWeightedRmseK': math.sqrt(squaredErrorIntegral / weightSum),
        'macroCaseRmseK': float(np.mean([record['rmseK'] for record in records])),
        'maximumErrorK': max(record['maximumErrorK'] for record in records),
    }


def _publicRecords(records: Sequence[dict]) -> list[dict]:
    return [
        {key: value for key, value in record.items() if not key.startswith('_')}
        for record in records
    ]


def _hybridUniformTemperatureBaseline(
    testCases: Sequence[thermalComponentCase],
) -> list[dict]:
    records = []
    for case in testCases:
        anchored = _hasAbsoluteTemperatureAnchor(case)
        reference = (
            _anchorReferenceTemperature(case) if anchored else _temperatureCenter(case)
        )
        errorK = reference - case.temperature.temperatureK
        weights = case.temperature.weightsM3
        squaredErrorIntegral = float(np.dot(np.square(errorK), weights))
        weightSum = float(weights.sum())
        records.append(
            {
                'caseId': case.caseId,
                'component': case.componentType,
                'speedRpm': case.operatingFeatures['speedRpm'],
                'rmseK': math.sqrt(squaredErrorIntegral / weightSum),
                'meanAbsoluteErrorK': float(np.dot(np.abs(errorK), weights) / weightSum),
                'maximumErrorK': float(np.max(np.abs(errorK))),
                'temperatureRangeK': float(np.ptp(case.temperature.temperatureK)),
                'temperatureWeightedMeanK': _temperatureCenter(case),
                'absoluteTemperatureAnchorAvailable': anchored,
                'anchorReferenceTemperatureK': reference if anchored else None,
                'targetRepresentation': HYBRID_ANCHORED_TARGET,
                '_squaredErrorIntegral': squaredErrorIntegral,
                '_weightSum': weightSum,
            }
        )
    return records


def main() -> None:
    arguments = _arguments()
    availableSpeeds = _numericSpeedDirectories(arguments.source)
    testSpeeds = tuple(str(value) for value in arguments.test_speeds)
    missing = sorted(set(testSpeeds) - set(availableSpeeds), key=float)
    if missing:
        raise ValueError(f'test speeds are missing from the source dataset: {missing}')
    trainSpeeds = tuple(speed for speed in availableSpeeds if speed not in testSpeeds)
    if not trainSpeeds or not testSpeeds:
        raise ValueError('speed split must leave at least one train and one test speed')
    components = tuple(str(value) for value in arguments.components)
    resolvedDevice = (
        arguments.device
        if not arguments.device.startswith('cuda') or torch.cuda.is_available()
        else 'cpu'
    )

    print(
        f'building {len(components) * len(availableSpeeds)} component cases...',
        file=sys.stderr,
    )
    cases = []
    for componentIndex, component in enumerate(components):
        for speed in availableSpeeds:
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

    trainCases = tuple(
        case for case in cases if str(int(case.operatingFeatures['speedRpm'])) in trainSpeeds
    )
    testCases = tuple(
        case for case in cases if str(int(case.operatingFeatures['speedRpm'])) in testSpeeds
    )
    sharedScaling = _scaling(trainCases)
    sharedTrainOperatorCases = _operatorCases(trainCases, sharedScaling)
    sharedTestOperatorCases = _operatorCases(testCases, sharedScaling)
    print('training shared all-component model...', file=sys.stderr)
    sharedModel, sharedTraining = _train(
        sharedTrainOperatorCases,
        epochs=arguments.shared_epochs,
        batchSize=arguments.batch_size,
        learningRate=arguments.learning_rate,
        hiddenDim=arguments.hidden_dim,
        kernelScales=arguments.kernel_scales,
        queryFourierFrequencies=arguments.query_fourier_frequencies,
        device=resolvedDevice,
        seed=arguments.seed,
    )
    sharedRecords = _evaluate(
        sharedModel,
        testCases,
        sharedTestOperatorCases,
        sharedScaling,
        resolvedDevice,
    )
    sharedByComponent = defaultdict(list)
    sharedBySpeed = defaultdict(list)
    for record in sharedRecords:
        sharedByComponent[record['component']].append(record)
        sharedBySpeed[record['speedRpm']].append(record)
    baselineRecords = _hybridUniformTemperatureBaseline(testCases)
    baselineByComponent = defaultdict(list)
    for record in baselineRecords:
        baselineByComponent[record['component']].append(record)
    sharedByAnchorMode = {
        'anchoredAbsolute': [
            record
            for record in sharedRecords
            if record['absoluteTemperatureAnchorAvailable']
        ],
        'floatingRelative': [
            record
            for record in sharedRecords
            if not record['absoluteTemperatureAnchorAvailable']
        ],
    }
    baselineByAnchorMode = {
        mode: [
            record
            for record in baselineRecords
            if record['absoluteTemperatureAnchorAvailable']
            == (mode == 'anchoredAbsolute')
        ]
        for mode in ('anchoredAbsolute', 'floatingRelative')
    }

    report = {
        'device': resolvedDevice,
        'components': components,
        'trainSpeedsRpm': [float(speed) for speed in trainSpeeds],
        'testSpeedsRpm': [float(speed) for speed in testSpeeds],
        'sampleLimits': {
            'volume': arguments.maximum_volume_samples,
            'boundary': arguments.maximum_boundary_samples,
            'query': arguments.maximum_query_samples,
        },
        'targetRepresentation': {
            'mode': HYBRID_ANCHORED_TARGET,
            'absoluteTemperaturePredictedForAnchoredCases': True,
            'description': (
                'Anchored FEM targets are temperature differences from their known '
                'convection/fixed-temperature reference and reconstruct absolute '
                'temperature. Floating pure-Neumann targets are centered by their '
                'volume-weighted mean and remain relative fields.'
            ),
        },
        'sharedModel': {
            'training': sharedTraining,
            'overall': _summary(sharedRecords),
            'byComponent': {
                component: _summary(sharedByComponent[component])
                for component in components
            },
            'bySpeed': {
                str(int(float(speed))): _summary(sharedBySpeed[float(speed)])
                for speed in testSpeeds
            },
            'byAnchorMode': {
                mode: _summary(records)
                for mode, records in sharedByAnchorMode.items()
                if records
            },
            'cases': _publicRecords(sharedRecords),
        },
        'hybridUniformTemperatureBaseline': {
            'overall': _summary(baselineRecords),
            'byComponent': {
                component: _summary(baselineByComponent[component])
                for component in components
            },
            'byAnchorMode': {
                mode: _summary(records)
                for mode, records in baselineByAnchorMode.items()
                if records
            },
            'cases': _publicRecords(baselineRecords),
        },
    }

    if arguments.checkpoint is not None:
        arguments.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        example = sharedTrainOperatorCases[0]
        torch.save(
            {
                'stateDict': sharedModel.state_dict(),
                'model': {
                    'volumeFeatureDim': example.volumeFeatureDim,
                    'boundaryFeatureDim': example.boundaryFeatureDim,
                    'globalFeatureDim': example.globalFeatureDim,
                    'queryDim': example.queryDim,
                    'hiddenDim': arguments.hidden_dim,
                    'kernelScales': list(arguments.kernel_scales),
                    'queryFourierFrequencies': arguments.query_fourier_frequencies,
                },
                'scaling': asdict(sharedScaling),
                'schema': asdict(
                    thermalComponentFeatureSchema(
                        operatingFeatureNames=(
                            'speedRpm',
                            'absoluteTemperatureAnchor',
                        ),
                        boundaryTypeCount=3,
                        temperatureBoundaryTypeIds=(1, 2),
                    )
                ),
                'targetMode': HYBRID_ANCHORED_TARGET,
            },
            arguments.checkpoint,
        )
        report['checkpoint'] = str(arguments.checkpoint)

    if not arguments.skip_component_models:
        componentReports = {}
        allComponentEvaluationRecords = []
        for componentIndex, component in enumerate(components):
            print(f'training component model: {component}', file=sys.stderr)
            componentTrainCases = tuple(
                case for case in trainCases if case.componentType == component
            )
            componentTestCases = tuple(
                case for case in testCases if case.componentType == component
            )
            scaling = _scaling(componentTrainCases)
            trainOperatorCases = _operatorCases(componentTrainCases, scaling)
            testOperatorCases = _operatorCases(componentTestCases, scaling)
            model, training = _train(
                trainOperatorCases,
                epochs=arguments.component_epochs,
                batchSize=arguments.batch_size,
                learningRate=arguments.learning_rate,
                hiddenDim=arguments.hidden_dim,
                kernelScales=arguments.kernel_scales,
                queryFourierFrequencies=arguments.query_fourier_frequencies,
                device=resolvedDevice,
                seed=arguments.seed + componentIndex + 1,
            )
            records = _evaluate(
                model,
                componentTestCases,
                testOperatorCases,
                scaling,
                resolvedDevice,
            )
            allComponentEvaluationRecords.extend(records)
            componentReports[component] = {
                'training': training,
                'test': _summary(records),
                'cases': _publicRecords(records),
            }
            del model
            if resolvedDevice.startswith('cuda'):
                torch.cuda.empty_cache()
        report['componentModels'] = componentReports
        report['componentModelsOverall'] = _summary(allComponentEvaluationRecords)

    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
