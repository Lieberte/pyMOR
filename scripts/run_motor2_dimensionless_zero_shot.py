from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from mor.nn.data import (
    STEADY_THERMAL_DIMENSIONLESS_V1,
    STEADY_THERMAL_DIMENSIONLESS_V2,
    buildMotorFemComponentCase,
    collateThermalOperatorCases,
    hasAbsoluteTemperatureAnchor,
    steadyThermalDimensionlessOperatorCase,
    thermalComponentFeatureSchema,
)
from mor.nn.models import thermalQuerySetOperatorTorch
from run_motor2_zero_shot import DEFAULT_COMPONENTS, _batchToDevice, _metrics, _summary


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Apply the dimensionless motor-1 operator to motor2.'
    )
    parser.add_argument('case', type=Path)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--result-time', type=float, default=222.0)
    parser.add_argument('--speeds', nargs='+', type=float, default=(3000.0, 5000.0))
    parser.add_argument('--components', nargs='+', default=DEFAULT_COMPONENTS)
    parser.add_argument('--maximum-volume-samples', type=int, default=256)
    parser.add_argument('--maximum-boundary-samples', type=int, default=512)
    parser.add_argument('--maximum-query-samples', type=int, default=2048)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=7)
    return parser.parse_args()


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

    primarySpeed = float(arguments.speeds[0])
    baseCases = tuple(
        buildMotorFemComponentCase(
            arguments.case,
            component,
            maximumVolumeSamples=arguments.maximum_volume_samples,
            maximumBoundarySamples=arguments.maximum_boundary_samples,
            maximumQuerySamples=arguments.maximum_query_samples,
            randomSeed=arguments.seed + componentIndex,
            resultTime=arguments.result_time,
            speedRpm=primarySpeed,
        )
        for componentIndex, component in enumerate(arguments.components)
    )

    predictionsBySpeed = {}
    dimensionlessIndicators = {}
    with torch.no_grad():
        for speed in arguments.speeds:
            records = []
            baselineRecords = []
            for baseCase in baseCases:
                case = replace(
                    baseCase,
                    caseId=f'{float(speed):g}rpm-{baseCase.componentType}',
                    operatingFeatures={
                        **baseCase.operatingFeatures,
                        'speedRpm': float(speed),
                    },
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
                prediction = model(batch)[0, :, 0]
                target = batch['targets'][0, :, 0]
                mask = batch['queryMask'][0]
                weights = torch.as_tensor(
                    case.temperature.weightsM3,
                    dtype=prediction.dtype,
                    device=prediction.device,
                )
                weightTotal = weights.sum().clamp_min(
                    torch.finfo(weights.dtype).eps
                )
                prediction = prediction[mask]
                target = target[mask]
                anchored = hasAbsoluteTemperatureAnchor(case)
                if not anchored:
                    prediction -= torch.sum(weights * prediction) / weightTotal
                    target -= torch.sum(weights * target) / weightTotal
                predictionTargetK = (
                    prediction * dimensionlessCase.scales.temperatureScaleK
                )
                trueTargetK = target * dimensionlessCase.scales.temperatureScaleK
                metrics = _metrics(predictionTargetK - trueTargetK, weights)
                baselineMetrics = _metrics(-trueTargetK, weights)
                records.append(
                    {
                        'component': case.componentType,
                        'evaluationMode': (
                            'anchoredAbsolute' if anchored else 'floatingRelative'
                        ),
                        'rmseK': metrics['rmseK'],
                        'meanAbsoluteErrorK': metrics['meanAbsoluteErrorK'],
                        'maximumErrorK': metrics['maximumErrorK'],
                        'uniformFieldBaselineRmseK': baselineMetrics['rmseK'],
                        'temperatureScaleK': (
                            dimensionlessCase.scales.temperatureScaleK
                        ),
                        'lengthScaleMeters': dimensionlessCase.scales.lengthMeters,
                        'conductivityScaleWPerMK': (
                            dimensionlessCase.scales.conductivityWPerMK
                        ),
                        'predictedTargetMinimumK': float(predictionTargetK.min()),
                        'predictedTargetMaximumK': float(predictionTargetK.max()),
                        'trueTargetMinimumK': float(trueTargetK.min()),
                        'trueTargetMaximumK': float(trueTargetK.max()),
                        '_squaredErrorIntegral': metrics['_squaredErrorIntegral'],
                        '_weightSum': metrics['_weightSum'],
                    }
                )
                baselineRecords.append(
                    {
                        'rmseK': baselineMetrics['rmseK'],
                        'maximumErrorK': baselineMetrics['maximumErrorK'],
                        '_squaredErrorIntegral': baselineMetrics[
                            '_squaredErrorIntegral'
                        ],
                        '_weightSum': baselineMetrics['_weightSum'],
                    }
                )
                if case.componentType not in dimensionlessIndicators:
                    dimensionlessIndicators[case.componentType] = {
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
                        'temperatureScaleK': (
                            dimensionlessCase.scales.temperatureScaleK
                        ),
                    }
            byAnchorMode = {
                mode: [
                    record for record in records if record['evaluationMode'] == mode
                ]
                for mode in ('anchoredAbsolute', 'floatingRelative')
            }
            baselineByAnchorMode = {
                mode: [
                    baseline
                    for record, baseline in zip(records, baselineRecords)
                    if record['evaluationMode'] == mode
                ]
                for mode in ('anchoredAbsolute', 'floatingRelative')
            }
            predictionsBySpeed[str(float(speed))] = {
                'overall': _summary(records),
                'byAnchorMode': {
                    mode: _summary(modeRecords)
                    for mode, modeRecords in byAnchorMode.items()
                    if modeRecords
                },
                'uniformFieldBaseline': {
                    'overall': _summary(baselineRecords),
                    'byAnchorMode': {
                        mode: _summary(modeRecords)
                        for mode, modeRecords in baselineByAnchorMode.items()
                        if modeRecords
                    },
                },
                'components': [
                    {
                        key: value
                        for key, value in record.items()
                        if not key.startswith('_')
                    }
                    for record in records
                ],
            }

    report = {
        'protocol': (
            'dimensionless shared operator trained on motor-1 speed cases and '
            'evaluated zero-shot on motor2'
        ),
        'featureRepresentation': {
            'mode': featureMode,
            'operatingScales': operatingScales,
            'minimumTemperatureScaleK': minimumTemperatureScaleK,
        },
        'device': device,
        'source': str(arguments.case),
        'checkpoint': str(arguments.checkpoint),
        'resultTime': float(arguments.result_time),
        'predictionsByAssumedSpeedRpm': predictionsBySpeed,
        'dimensionlessIndicators': dimensionlessIndicators,
    }
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
