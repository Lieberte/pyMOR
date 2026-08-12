from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from mor.nn.data import (
    buildMotorFemComponentCase,
    collateThermalOperatorCases,
    thermalComponentFeatureSchema,
    thermalOperatorFeatureScaling,
)
from mor.nn.models import thermalQuerySetOperatorTorch


DEFAULT_COMPONENTS = ('winding', 'stator', 'rotor', 'shaft', 'magnet')
HYBRID_ANCHORED_TARGET = 'hybridAnchoredOrRelative'


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Apply a saved shared motor thermal operator to a different motor.'
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
    parser.add_argument(
        '--clip-normalized-physical-inputs',
        action='store_true',
        help=(
            'Diagnostic only: clip normalized material/source/HTC/heat-flux '
            'features to the nominal training scale range.'
        ),
    )
    return parser.parse_args()


def _batchToDevice(batch: dict, device: str) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _clipNormalizedPhysicalInputs(batch: dict) -> dict:
    clipped = dict(batch)
    volume = batch['volumeFeatures'].clone()
    volume[:, :, 3:15] = volume[:, :, 3:15].clamp(-1.0, 1.0)
    boundary = batch['boundaryFeatures'].clone()
    boundary[:, :, 6:9] = boundary[:, :, 6:9].clamp(-1.0, 1.0)
    globalFeatures = batch['globalFeatures'].clone()
    globalFeatures[:, 0:3] = globalFeatures[:, 0:3].clamp(0.0, 1.0)
    globalFeatures[:, 5:] = globalFeatures[:, 5:].clamp(0.0, 1.0)
    clipped['volumeFeatures'] = volume
    clipped['boundaryFeatures'] = boundary
    clipped['globalFeatures'] = globalFeatures
    return clipped


def _metrics(errorK: torch.Tensor, weights: torch.Tensor) -> dict[str, float]:
    weightSum = weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
    return {
        'rmseK': math.sqrt(float(torch.sum(weights * errorK.square()) / weightSum)),
        'meanAbsoluteErrorK': float(torch.sum(weights * torch.abs(errorK)) / weightSum),
        'maximumErrorK': float(torch.max(torch.abs(errorK))),
        '_squaredErrorIntegral': float(torch.sum(weights * errorK.square())),
        '_weightSum': float(weightSum),
    }


def _public(metrics: dict[str, float]) -> dict[str, float]:
    return {key: value for key, value in metrics.items() if not key.startswith('_')}


def _temperatureCenter(case) -> float:
    return float(
        np.average(
            case.temperature.temperatureK,
            weights=case.temperature.weightsM3,
        )
    )


def _hasAbsoluteTemperatureAnchor(case) -> bool:
    return bool(case.operatingFeatures.get('absoluteTemperatureAnchor', 0.0))


def _anchorReferenceTemperature(case) -> float:
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


def _caseInTargetTemperatureGauge(case, scaling):
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


def _summary(records: list[dict]) -> dict[str, float | int]:
    squaredErrorIntegral = sum(record['_squaredErrorIntegral'] for record in records)
    weightSum = sum(record['_weightSum'] for record in records)
    return {
        'caseCount': len(records),
        'volumeWeightedRmseK': math.sqrt(squaredErrorIntegral / weightSum),
        'macroComponentRmseK': float(np.mean([record['rmseK'] for record in records])),
        'maximumErrorK': max(record['maximumErrorK'] for record in records),
    }


def main() -> None:
    arguments = _arguments()
    device = (
        arguments.device
        if not arguments.device.startswith('cuda') or torch.cuda.is_available()
        else 'cpu'
    )
    payload = torch.load(arguments.checkpoint, map_location=device, weights_only=False)
    targetMode = payload.get('targetMode')
    if targetMode != HYBRID_ANCHORED_TARGET:
        raise ValueError(
            'motor2 zero-shot evaluation requires a hybridAnchoredOrRelative checkpoint'
        )
    model = thermalQuerySetOperatorTorch(**payload['model']).to(device)
    model.load_state_dict(payload['stateDict'])
    model.eval()
    scaling = thermalOperatorFeatureScaling(**payload['scaling'])
    schema = thermalComponentFeatureSchema(**payload['schema'])

    baseCases = []
    primarySpeed = float(arguments.speeds[0])
    for componentIndex, component in enumerate(arguments.components):
        baseCases.append(
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
        )

    predictionsBySpeed = {}
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
                operatorCase = _caseInTargetTemperatureGauge(
                    case, scaling
                ).toOperatorCase(schema=schema, scaling=scaling)
                anchored = _hasAbsoluteTemperatureAnchor(case)
                temperatureCenter = _temperatureCenter(case)
                referenceTemperature = (
                    _anchorReferenceTemperature(case)
                    if anchored
                    else temperatureCenter
                )
                operatorCase = replace(
                    operatorCase,
                    queryTemperatures=(
                        (case.temperature.temperatureK - referenceTemperature)
                        / scaling.temperatureScaleK
                    )[:, None],
                )
                batch = _batchToDevice(
                    collateThermalOperatorCases([operatorCase]), device
                )
                if arguments.clip_normalized_physical_inputs:
                    batch = _clipNormalizedPhysicalInputs(batch)
                prediction = model(batch)[0, :, 0]
                target = batch['targets'][0, :, 0]
                mask = batch['queryMask'][0]
                weights = batch['queryWeights'][0, mask]
                weightTotal = weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
                prediction = prediction[mask]
                target = target[mask]
                if not anchored:
                    prediction -= torch.sum(weights * prediction) / weightTotal
                    target -= torch.sum(weights * target) / weightTotal
                predictionTargetK = prediction * scaling.temperatureScaleK
                trueTargetK = target * scaling.temperatureScaleK
                metrics = _metrics(predictionTargetK - trueTargetK, weights)
                baselineMetrics = _metrics(-trueTargetK, weights)
                records.append(
                    {
                        'component': case.componentType,
                        'rmseK': metrics['rmseK'],
                        'meanAbsoluteErrorK': metrics['meanAbsoluteErrorK'],
                        'maximumErrorK': metrics['maximumErrorK'],
                        'evaluationMode': (
                            'anchoredAbsolute' if anchored else 'floatingRelative'
                        ),
                        'absoluteTemperatureAnchorAvailable': anchored,
                        'anchorReferenceTemperatureK': (
                            referenceTemperature if anchored else None
                        ),
                        'predictedTargetMinimumK': float(predictionTargetK.min()),
                        'predictedTargetMaximumK': float(predictionTargetK.max()),
                        'trueTargetMinimumK': float(trueTargetK.min()),
                        'trueTargetMaximumK': float(trueTargetK.max()),
                        'temperatureWeightedMeanKForEvaluationOnly': temperatureCenter,
                        'uniformFieldBaselineRmseK': baselineMetrics['rmseK'],
                        'conductivityWPerMK': float(
                            case.volume.conductivityWPerMK[0, 0, 0]
                        ),
                        'integratedHeatSourceW': case.volume.totalHeatSourcePowerW,
                        'integratedBoundaryHeatFlowW': case.boundary.totalHeatFlowRateW,
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
            byAnchorMode = {
                mode: [
                    record
                    for record in records
                    if record['evaluationMode'] == mode
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
                    {key: value for key, value in record.items() if not key.startswith('_')}
                    for record in records
                ],
            }

    report = {
        'protocol': (
            'saved shared model trained on motor-1 speed cases; motor2 temperatures are '
            'used only to construct and evaluate anchored absolute or floating relative '
            'temperature targets according to each FEM boundary configuration'
        ),
        'targetRepresentation': {
            'mode': targetMode,
            'absoluteTemperaturePredictedForAnchoredCases': True,
            'floatingCasesRemainRelative': True,
        },
        'device': device,
        'source': str(arguments.case),
        'checkpoint': str(arguments.checkpoint),
        'resultTime': float(arguments.result_time),
        'normalizedPhysicalInputClipping': bool(
            arguments.clip_normalized_physical_inputs
        ),
        'sampleLimits': {
            'volume': arguments.maximum_volume_samples,
            'boundary': arguments.maximum_boundary_samples,
            'query': arguments.maximum_query_samples,
        },
        'predictionsByAssumedSpeedRpm': predictionsBySpeed,
        'outOfDistributionIndicators': {
            case.componentType: {
                'halfExtentOverTrainingLengthScale': (
                    case.coordinateFrame.halfExtentMeters / scaling.lengthMeters
                ).tolist(),
                'conductivityOverTrainingScale': float(
                    np.max(np.abs(case.volume.conductivityWPerMK))
                    / scaling.conductivityWPerMK
                ),
                'maximumHeatSourceOverTrainingScale': float(
                    np.max(np.abs(case.volume.heatSourceWPerM3))
                    / scaling.heatSourceWPerM3
                ),
                'maximumHtcOverTrainingScale': float(
                    np.max(case.boundary.htcWPerM2K) / scaling.htcWPerM2K
                ),
                'maximumBoundaryHeatFluxOverTrainingScale': float(
                    np.max(np.abs(case.boundary.heatFluxWPerM2))
                    / scaling.heatFluxWPerM2
                ),
                'hybridTargetNormalizedRange': [
                    float(
                        (case.temperature.temperatureK.min()
                         - (
                             _anchorReferenceTemperature(case)
                             if _hasAbsoluteTemperatureAnchor(case)
                             else _temperatureCenter(case)
                         ))
                        / scaling.temperatureScaleK
                    ),
                    float(
                        (case.temperature.temperatureK.max()
                         - (
                             _anchorReferenceTemperature(case)
                             if _hasAbsoluteTemperatureAnchor(case)
                             else _temperatureCenter(case)
                         ))
                        / scaling.temperatureScaleK
                    ),
                ],
                'absoluteTemperatureAnchorAvailable': _hasAbsoluteTemperatureAnchor(case),
            }
            for case in baseCases
        },
    }
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
