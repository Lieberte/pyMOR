from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from mor.nn.data import (
    STEADY_THERMAL_DIMENSIONLESS_V1,
    STEADY_THERMAL_DIMENSIONLESS_V2,
    buildBoxFemCase,
    collateThermalOperatorCases,
    enforceMatchingDirichletQueries,
    steadyThermalDimensionlessOperatorCase,
    thermalBoundarySamples,
    thermalComponentCase,
    thermalComponentFeatureSchema,
)
from mor.nn.models import thermalQuerySetOperatorTorch
from run_motor2_zero_shot import _batchToDevice


MOTOR_FIXED_TEMPERATURE_BOUNDARY_TYPE = 2


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Apply a dimensionless operator trained only on motor-1 speed cases '
            'to a box FEM case without box training.'
        )
    )
    parser.add_argument('case', type=Path)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--result-time', type=float, default=1.0)
    parser.add_argument('--speeds', nargs='+', type=float, default=(0.0, 12500.0))
    parser.add_argument('--maximum-volume-samples', type=int, default=256)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=7)
    return parser.parse_args()


def _adaptBoxForMotorSchema(
    case: thermalComponentCase,
    *,
    schema: thermalComponentFeatureSchema,
    speedRpm: float,
) -> thermalComponentCase:
    if MOTOR_FIXED_TEMPERATURE_BOUNDARY_TYPE >= schema.boundaryTypeCount:
        raise ValueError(
            'checkpoint schema does not contain the motor fixed-temperature '
            f'boundary type {MOTOR_FIXED_TEMPERATURE_BOUNDARY_TYPE}'
        )
    if MOTOR_FIXED_TEMPERATURE_BOUNDARY_TYPE not in schema.temperatureBoundaryTypeIds:
        raise ValueError(
            'checkpoint schema does not mark boundary type 2 as a temperature boundary'
        )
    boundary = case.boundary
    return replace(
        case,
        caseId=f'{case.caseId}-dimensionless-zero-shot-{float(speedRpm):g}rpm',
        boundary=thermalBoundarySamples(
            coordinatesMeters=boundary.coordinatesMeters,
            normals=boundary.normals,
            htcWPerM2K=boundary.htcWPerM2K,
            ambientTemperatureK=boundary.ambientTemperatureK,
            weightsM2=boundary.weightsM2,
            boundaryTypeIds=np.full(
                boundary.coordinatesMeters.shape[0],
                MOTOR_FIXED_TEMPERATURE_BOUNDARY_TYPE,
                dtype=np.int64,
            ),
            heatFluxWPerM2=boundary.heatFluxWPerM2,
            patchIds=boundary.patchIds,
            extraFeatures=boundary.extraFeatures,
        ),
        operatingFeatures={
            **case.operatingFeatures,
            'speedRpm': float(speedRpm),
            'absoluteTemperatureAnchor': 1.0,
        },
        metadata={
            **case.metadata,
            'zeroShotBoundaryAdaptation': (
                'box fixed-temperature boundary type 1 is mapped to the motor '
                'schema fixed-temperature boundary type 2; prescribed temperatures '
                'are retained as known boundary inputs'
            ),
        },
    )


def _weightedMetrics(
    predictionK: np.ndarray,
    targetK: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    selectedWeights = np.asarray(weights[mask], dtype=np.float64)
    error = np.asarray(predictionK[mask] - targetK[mask], dtype=np.float64)
    weightSum = float(selectedWeights.sum())
    if weightSum <= np.finfo(np.float64).eps:
        raise ValueError('metric mask has no positive volume weight')
    return {
        'rmseK': math.sqrt(float(np.dot(selectedWeights, np.square(error))) / weightSum),
        'meanAbsoluteErrorK': float(
            np.dot(selectedWeights, np.abs(error)) / weightSum
        ),
        'maximumErrorK': float(np.max(np.abs(error))),
    }


def _unweightedMetrics(
    predictionK: np.ndarray,
    targetK: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    error = np.asarray(predictionK[mask] - targetK[mask], dtype=np.float64)
    if not error.size:
        raise ValueError('metric mask is empty')
    return {
        'rmseK': float(np.sqrt(np.mean(np.square(error)))),
        'meanAbsoluteErrorK': float(np.mean(np.abs(error))),
        'maximumErrorK': float(np.max(np.abs(error))),
    }


def main() -> None:
    arguments = _arguments()
    device = (
        arguments.device
        if not arguments.device.startswith('cuda') or torch.cuda.is_available()
        else 'cpu'
    )
    payload = torch.load(arguments.checkpoint, map_location=device, weights_only=False)
    featureMode = payload.get('featureMode')
    if featureMode not in (
        STEADY_THERMAL_DIMENSIONLESS_V1,
        STEADY_THERMAL_DIMENSIONLESS_V2,
    ):
        raise ValueError('checkpoint is not a supported steady thermal model')
    model = thermalQuerySetOperatorTorch(**payload['model']).to(device)
    model.load_state_dict(payload['stateDict'])
    model.eval()
    schema = thermalComponentFeatureSchema(**payload['schema'])
    dimensionlessConfiguration = payload['dimensionless']
    operatingScales = dimensionlessConfiguration['operatingScales']
    minimumTemperatureScaleK = dimensionlessConfiguration[
        'minimumTemperatureScaleK'
    ]

    graphCase = buildBoxFemCase(
        arguments.case,
        resultTime=arguments.result_time,
        maximumVolumeSamples=arguments.maximum_volume_samples,
        randomSeed=arguments.seed,
    )
    baseCase = graphCase.componentCase
    boundaryMask = np.isfinite(graphCase.fixedTemperatureK)
    allVolumeMask = graphCase.nodeWeightsM3 > 0.0
    interiorMask = ~boundaryMask & allVolumeMask
    predictions = {}
    dimensionlessIndicators = None

    with torch.no_grad():
        for speedRpm in arguments.speeds:
            case = _adaptBoxForMotorSchema(
                baseCase,
                schema=schema,
                speedRpm=float(speedRpm),
            )
            dimensionlessCase = steadyThermalDimensionlessOperatorCase(
                case,
                schema=schema,
                operatingScales=operatingScales,
                minimumTemperatureScaleK=minimumTemperatureScaleK,
                featureMode=featureMode,
            )
            operatorCase = dimensionlessCase.operatorCase
            batch = _batchToDevice(
                collateThermalOperatorCases([operatorCase]), device
            )
            normalizedPrediction = model(batch)[0, :, 0].detach().cpu().numpy()
            hardNormalizedPrediction, matchedDirichletQueryCount = (
                enforceMatchingDirichletQueries(
                    normalizedPrediction,
                    operatorCase,
                    schema=schema,
                )
            )
            predictionK = (
                normalizedPrediction * dimensionlessCase.scales.temperatureScaleK
                + dimensionlessCase.scales.temperatureReferenceK
            )
            targetK = graphCase.temperatureK
            dirichletEnforcedPredictionK = (
                hardNormalizedPrediction
                * dimensionlessCase.scales.temperatureScaleK
                + dimensionlessCase.scales.temperatureReferenceK
            )
            dirichletEnforcedPredictionK[boundaryMask] = graphCase.fixedTemperatureK[
                boundaryMask
            ]
            predictions[str(float(speedRpm))] = {
                'predictedDimensionlessMinimum': float(normalizedPrediction.min()),
                'predictedDimensionlessMaximum': float(normalizedPrediction.max()),
                'predictedMinimumK': float(predictionK.min()),
                'predictedMaximumK': float(predictionK.max()),
                'rawPrediction': {
                    'allVolumeNodes': _weightedMetrics(
                        predictionK,
                        targetK,
                        graphCase.nodeWeightsM3,
                        allVolumeMask,
                    ),
                    'interiorNodes': _weightedMetrics(
                        predictionK,
                        targetK,
                        graphCase.nodeWeightsM3,
                        interiorMask,
                    ),
                    'fixedTemperatureBoundaryNodesUnweighted': _unweightedMetrics(
                        predictionK,
                        targetK,
                        boundaryMask,
                    ),
                },
                'dirichletBoundaryEnforced': {
                    'matchedQueryCountFromGenericCoordinateEnforcement': (
                        matchedDirichletQueryCount
                    ),
                    'predictedMinimumK': float(dirichletEnforcedPredictionK.min()),
                    'predictedMaximumK': float(dirichletEnforcedPredictionK.max()),
                    'allVolumeNodes': _weightedMetrics(
                        dirichletEnforcedPredictionK,
                        targetK,
                        graphCase.nodeWeightsM3,
                        allVolumeMask,
                    ),
                    'interiorNodes': _weightedMetrics(
                        dirichletEnforcedPredictionK,
                        targetK,
                        graphCase.nodeWeightsM3,
                        interiorMask,
                    ),
                    'fixedTemperatureBoundaryNodesUnweighted': _unweightedMetrics(
                        dirichletEnforcedPredictionK,
                        targetK,
                        boundaryMask,
                    ),
                },
            }
            if dimensionlessIndicators is None:
                dimensionlessIndicators = {
                    'maximumAbsoluteLogDimensionlessHeatSource': float(
                        np.max(np.abs(operatorCase.volumeFeatures[:, 14]))
                    ),
                    'maximumLogBiot': float(
                        np.max(operatorCase.boundaryFeatures[:, 6])
                    ),
                    'maximumAbsoluteLogDimensionlessBoundaryHeatFlux': float(
                        np.max(np.abs(operatorCase.boundaryFeatures[:, 8]))
                    ),
                    'dimensionlessTargetRange': [
                        float(operatorCase.queryTemperatures.min()),
                        float(operatorCase.queryTemperatures.max()),
                    ],
                    'globalFeatures': operatorCase.globalFeatures.tolist(),
                    'scales': asdict(dimensionlessCase.scales),
                }

    anchorTemperatureK = float(
        np.average(
            baseCase.boundary.ambientTemperatureK,
            weights=baseCase.boundary.weightsM2,
        )
    )
    anchorBaseline = np.full_like(graphCase.temperatureK, anchorTemperatureK)
    report = {
        'protocol': (
            'dimensionless shared operator trained only on motor-1 speed cases and '
            'evaluated zero-shot on the box; box temperatures are used only after '
            'inference to calculate metrics'
        ),
        'featureRepresentation': {
            'mode': featureMode,
            'operatingScales': operatingScales,
            'minimumTemperatureScaleK': minimumTemperatureScaleK,
        },
        'device': device,
        'source': str(arguments.case),
        'checkpoint': str(arguments.checkpoint),
        'resultTimeS': float(arguments.result_time),
        'mesh': {
            'nodeCount': int(graphCase.nodesMeters.shape[0]),
            'volumeSampleCount': int(baseCase.volume.coordinatesMeters.shape[0]),
            'boundarySampleCount': int(baseCase.boundary.coordinatesMeters.shape[0]),
        },
        'physics': {
            'mappedHeatSourcePowerW': float(
                baseCase.metadata['mappedHeatSourcePowerW']
            ),
            'sampledHeatSourcePowerW': float(
                baseCase.metadata['sampledHeatSourcePowerW']
            ),
            'trueMinimumK': float(graphCase.temperatureK.min()),
            'trueMaximumK': float(graphCase.temperatureK.max()),
            'fixedTemperatureAnchorK': anchorTemperatureK,
        },
        'boundaryAdaptation': (
            'box fixed-temperature boundary type 1 is mapped to the motor schema '
            'fixed-temperature boundary type 2; prescribed 300 K values remain '
            'explicit boundary inputs and no box temperature result is used as input'
        ),
        'predictionsByAssumedSpeedRpm': predictions,
        'fixedTemperatureUniformFieldBaseline': {
            'allVolumeNodes': _weightedMetrics(
                anchorBaseline,
                graphCase.temperatureK,
                graphCase.nodeWeightsM3,
                allVolumeMask,
            ),
            'interiorNodes': _weightedMetrics(
                anchorBaseline,
                graphCase.temperatureK,
                graphCase.nodeWeightsM3,
                interiorMask,
            ),
        },
        'dimensionlessIndicators': dimensionlessIndicators,
    }
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
