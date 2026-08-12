from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from mor.nn.data import (
    buildBoxFemCase,
    buildMotorFemComponentCase,
    collateThermalOperatorCases,
    thermalBoundarySamples,
    thermalComponentCase,
    thermalComponentFeatureSchema,
)
from run_motor_fem_speed_holdout import (
    DEFAULT_COMPONENTS,
    DEFAULT_TEST_SPEEDS,
    _batchToDevice,
    _evaluate,
    _newModel,
    _numericSpeedDirectories,
    _operatorCases,
    _scaling,
    _summary,
    _train,
    RELATIVE_WEIGHTED_MEAN_TARGET,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train the shared motor operator and apply it to a box without box training.'
    )
    parser.add_argument('motor_source', type=Path)
    parser.add_argument('box_case', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--result-time', type=float, default=1.0)
    parser.add_argument('--test-speeds', nargs='*', default=DEFAULT_TEST_SPEEDS)
    parser.add_argument('--box-speeds', nargs='*', type=float, default=(0.0, 12500.0))
    parser.add_argument('--maximum-volume-samples', type=int, default=256)
    parser.add_argument('--maximum-boundary-samples', type=int, default=512)
    parser.add_argument('--maximum-query-samples', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=2e-3)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--kernel-scales', nargs='+', type=float, default=(0.12, 0.35))
    parser.add_argument('--query-fourier-frequencies', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=7)
    return parser.parse_args()


def _boxForMotorSchema(case: thermalComponentCase, speedRpm: float) -> thermalComponentCase:
    boundary = case.boundary
    return thermalComponentCase(
        caseId=f'{case.caseId}-motor-zero-shot-{float(speedRpm):g}rpm',
        geometryId=case.geometryId,
        componentType=case.componentType,
        volume=case.volume,
        boundary=thermalBoundarySamples(
            coordinatesMeters=boundary.coordinatesMeters,
            normals=boundary.normals,
            htcWPerM2K=boundary.htcWPerM2K,
            ambientTemperatureK=boundary.ambientTemperatureK,
            weightsM2=boundary.weightsM2,
            boundaryTypeIds=np.zeros(
                boundary.coordinatesMeters.shape[0], dtype=np.int64
            ),
            heatFluxWPerM2=boundary.heatFluxWPerM2,
            patchIds=boundary.patchIds,
            extraFeatures=boundary.extraFeatures,
        ),
        temperature=case.temperature,
        operatingFeatures={'speedRpm': float(speedRpm)},
        metadata={
            **case.metadata,
            'zeroShotBoundaryAdaptation': (
                'fixed-temperature samples mapped to the only boundary channel of the '
                'motor model; temperature retained as ambient feature and heat flux set to zero'
            ),
        },
    )


def _metrics(
    predictionK: np.ndarray,
    targetK: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    selectedWeights = weights[mask]
    error = predictionK[mask] - targetK[mask]
    weightSum = float(selectedWeights.sum())
    return {
        'rmseK': math.sqrt(float(np.dot(selectedWeights, np.square(error))) / weightSum),
        'meanAbsoluteErrorK': float(np.dot(selectedWeights, np.abs(error)) / weightSum),
        'maximumErrorK': float(np.max(np.abs(error))),
    }


def main() -> None:
    arguments = _arguments()
    device = (
        arguments.device
        if not arguments.device.startswith('cuda') or torch.cuda.is_available()
        else 'cpu'
    )
    availableSpeeds = _numericSpeedDirectories(arguments.motor_source)
    testSpeeds = tuple(str(value) for value in arguments.test_speeds)
    trainSpeeds = tuple(speed for speed in availableSpeeds if speed not in testSpeeds)
    if not trainSpeeds or not testSpeeds:
        raise ValueError('motor speed split must contain training and test cases')

    cases = []
    for componentIndex, component in enumerate(DEFAULT_COMPONENTS):
        for speed in availableSpeeds:
            cases.append(
                buildMotorFemComponentCase(
                    arguments.motor_source / speed,
                    component,
                    maximumVolumeSamples=arguments.maximum_volume_samples,
                    maximumBoundarySamples=arguments.maximum_boundary_samples,
                    maximumQuerySamples=arguments.maximum_query_samples,
                    randomSeed=arguments.seed + componentIndex,
                )
            )
    trainCases = tuple(
        case for case in cases
        if str(int(case.operatingFeatures['speedRpm'])) in trainSpeeds
    )
    testCases = tuple(
        case for case in cases
        if str(int(case.operatingFeatures['speedRpm'])) in testSpeeds
    )
    scaling = _scaling(trainCases)
    trainOperatorCases = _operatorCases(trainCases, scaling)
    testOperatorCases = _operatorCases(testCases, scaling)
    model, training = _train(
        trainOperatorCases,
        epochs=arguments.epochs,
        batchSize=arguments.batch_size,
        learningRate=arguments.learning_rate,
        hiddenDim=arguments.hidden_dim,
        kernelScales=arguments.kernel_scales,
        queryFourierFrequencies=arguments.query_fourier_frequencies,
        device=device,
        seed=arguments.seed,
    )
    motorRecords = _evaluate(model, testCases, testOperatorCases, scaling, device)

    graphCase = buildBoxFemCase(
        arguments.box_case,
        resultTime=arguments.result_time,
        maximumVolumeSamples=arguments.maximum_volume_samples,
        randomSeed=arguments.seed,
    )
    boxCase = graphCase.componentCase
    schema = thermalComponentFeatureSchema(
        operatingFeatureNames=('speedRpm',), boundaryTypeCount=1
    )
    boundaryMask = np.isfinite(graphCase.fixedTemperatureK)
    interiorMask = ~boundaryMask & (graphCase.nodeWeightsM3 > 0.0)
    allVolumeMask = graphCase.nodeWeightsM3 > 0.0
    predictions = {}
    model.eval()
    with torch.no_grad():
        for speedRpm in arguments.box_speeds:
            adaptedCase = _boxForMotorSchema(boxCase, speedRpm)
            operatorCase = adaptedCase.toOperatorCase(schema=schema, scaling=scaling)
            temperatureCenter = float(
                np.average(
                    adaptedCase.temperature.temperatureK,
                    weights=adaptedCase.temperature.weightsM3,
                )
            )
            operatorCase = replace(
                operatorCase,
                queryTemperatures=(
                    (adaptedCase.temperature.temperatureK - temperatureCenter)
                    / scaling.temperatureScaleK
                )[:, None],
            )
            batch = _batchToDevice(collateThermalOperatorCases([operatorCase]), device)
            normalizedPrediction = model(batch)[0, :, 0].detach().cpu().numpy()
            weights = graphCase.nodeWeightsM3
            predictionCenter = float(
                np.average(normalizedPrediction[allVolumeMask], weights=weights[allVolumeMask])
            )
            predictionRelativeK = (
                normalizedPrediction - predictionCenter
            ) * scaling.temperatureScaleK
            targetRelativeK = graphCase.temperatureK - temperatureCenter
            predictions[str(float(speedRpm))] = {
                'predictedRelativeMinimumK': float(predictionRelativeK.min()),
                'predictedRelativeMaximumK': float(predictionRelativeK.max()),
                'allVolumeNodes': _metrics(
                    predictionRelativeK,
                    targetRelativeK,
                    graphCase.nodeWeightsM3,
                    allVolumeMask,
                ),
                'interiorNodes': _metrics(
                    predictionRelativeK,
                    targetRelativeK,
                    graphCase.nodeWeightsM3,
                    interiorMask,
                ),
                'temperatureWeightedMeanKForEvaluationOnly': temperatureCenter,
            }

    zeroRelativeTemperature = np.zeros_like(graphCase.temperatureK)
    report = {
        'device': device,
        'protocol': (
            'shared pure-Neumann motor model retrained only on motor training speeds; '
            'box temperatures are centered and used only to evaluate relative-field shape'
        ),
        'targetRepresentation': {
            'mode': RELATIVE_WEIGHTED_MEAN_TARGET,
            'absoluteTemperaturePredicted': False,
        },
        'motor': {
            'source': str(arguments.motor_source),
            'trainSpeedsRpm': [float(value) for value in trainSpeeds],
            'testSpeedsRpm': [float(value) for value in testSpeeds],
            'training': training,
            'heldOutSpeedVerification': _summary(motorRecords),
        },
        'box': {
            'source': str(arguments.box_case),
            'resultTimeS': float(arguments.result_time),
            'trueAbsoluteMinimumKForReference': float(graphCase.temperatureK.min()),
            'trueAbsoluteMaximumKForReference': float(graphCase.temperatureK.max()),
            'nodeCount': int(graphCase.nodesMeters.shape[0]),
            'boundaryAdaptation': (
                'the motor model has only a result-derived heat-flux boundary channel. '
                'Box fixed-temperature boundaries are mapped into that channel with their '
                '300 K value in the ambient-temperature feature and zero heat flux.'
            ),
            'predictionsByAssumedSpeedRpm': predictions,
            'zeroRelativeTemperatureBaseline': {
                'allVolumeNodes': _metrics(
                    zeroRelativeTemperature,
                    graphCase.temperatureK
                    - float(
                        np.average(
                            graphCase.temperatureK,
                            weights=graphCase.nodeWeightsM3,
                        )
                    ),
                    graphCase.nodeWeightsM3,
                    allVolumeMask,
                ),
                'interiorNodes': _metrics(
                    zeroRelativeTemperature,
                    graphCase.temperatureK
                    - float(
                        np.average(
                            graphCase.temperatureK,
                            weights=graphCase.nodeWeightsM3,
                        )
                    ),
                    graphCase.nodeWeightsM3,
                    interiorMask,
                ),
            },
        },
        'scaling': asdict(scaling),
        'outOfDistributionIndicators': {
            'boxHalfExtentOverMotorLengthScale': (
                boxCase.coordinateFrame.halfExtentMeters / scaling.lengthMeters
            ).tolist(),
            'boxMaximumHeatSourceOverMotorScale': float(
                np.max(np.abs(boxCase.volume.heatSourceWPerM3))
                / scaling.heatSourceWPerM3
            ),
            'boxVolumeOverMotorLengthScaleCubed': float(
                boxCase.volume.weightsM3.sum() / scaling.lengthMeters**3
            ),
        },
    }
    if arguments.checkpoint is not None:
        arguments.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        example = trainOperatorCases[0]
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
                },
                'scaling': asdict(scaling),
                'schema': {
                    'operatingFeatureNames': ['speedRpm'],
                    'boundaryTypeCount': 1,
                    'temperatureBoundaryTypeIds': [],
                },
                'targetMode': RELATIVE_WEIGHTED_MEAN_TARGET,
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
