from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mor.nn.data import (
    loadSubsampledThermalDataset,
    steadyThermalDimensionlessV2GlobalFeatureNames,
    thermalCheckpointDataContract,
)

try:
    from scripts.train_thermal_discrete_pinn import _caseGroup
except ModuleNotFoundError:
    from train_thermal_discrete_pinn import _caseGroup


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Measure held-out stator coverage in global-feature space.'
    )
    parser.add_argument('source', type=Path)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--validation-case-id', action='append', required=True)
    parser.add_argument('--seed', type=int, default=23)
    return parser.parse_args()


def _distance(left: np.ndarray, right: np.ndarray, scale: np.ndarray) -> float:
    active = scale > 1e-6
    if not np.any(active):
        return 0.0
    return float(np.sqrt(np.mean(np.square((left[active] - right[active]) / scale[active]))))


def main() -> None:
    arguments = _arguments()
    dataset = loadSubsampledThermalDataset(
        arguments.source,
        analysisType='steady',
        maximumVolumeSamples=128,
        maximumBoundarySamples=256,
        maximumQuerySamples=128,
        randomSeed=arguments.seed,
    )
    checkpoint = torch.load(
        arguments.checkpoint, map_location='cpu', weights_only=False
    )
    contract = thermalCheckpointDataContract.fromCheckpoint(checkpoint)
    prepared = contract.prepareCases(dataset.cases)
    stators = tuple(case for case in prepared if case.componentType == 'stator')
    byId = {case.caseId: case for case in stators}
    requested = frozenset(arguments.validation_case_id)
    missing = sorted(requested.difference(byId))
    if missing:
        raise ValueError(f'unknown stator validation case ids: {missing!r}')

    validationGroups = frozenset(_caseGroup(byId[caseId]) for caseId in requested)
    trainingCases = tuple(
        case for case in stators if _caseGroup(case) not in validationGroups
    )
    validationCases = tuple(
        case for case in stators if _caseGroup(case) in validationGroups
    )
    trainingDimensionless = contract.transformCases(trainingCases)
    validationDimensionless = contract.transformCases(validationCases)
    trainingFeatures = np.vstack(
        [value.operatorCase.globalFeatures for value in trainingDimensionless]
    )
    trainingMean = np.mean(trainingFeatures, axis=0)
    trainingStd = np.std(trainingFeatures, axis=0)
    trainingMinimum = np.min(trainingFeatures, axis=0)
    trainingMaximum = np.max(trainingFeatures, axis=0)
    active = trainingStd > 1e-6
    featureNames = steadyThermalDimensionlessV2GlobalFeatureNames(contract.schema)

    coverage = []
    for case, value in zip(validationCases, validationDimensionless):
        features = value.operatorCase.globalFeatures
        below = np.maximum((trainingMinimum - features) / np.maximum(trainingStd, 1e-6), 0.0)
        above = np.maximum((features - trainingMaximum) / np.maximum(trainingStd, 1e-6), 0.0)
        excursion = np.maximum(below, above)
        outside = active & (excursion > 1e-6)
        nearestDistances = np.asarray(
            [_distance(features, row, trainingStd) for row in trainingFeatures]
        )
        topOutside = np.argsort(excursion)[::-1]
        topOutside = [index for index in topOutside if outside[index]][:10]
        z = np.zeros_like(features)
        z[active] = (features[active] - trainingMean[active]) / trainingStd[active]
        coverage.append(
            {
                'caseId': case.caseId,
                'assemblyId': case.metadata.get('assemblyId'),
                'nearestTrainingCaseId': trainingCases[int(np.argmin(nearestDistances))].caseId,
                'nearestStandardizedRmsDistance': float(np.min(nearestDistances)),
                'activeGlobalFeatureCount': int(np.sum(active)),
                'outsideTrainingEnvelopeFeatureCount': int(np.sum(outside)),
                'outsideTrainingEnvelopeFeatureFraction': float(
                    np.sum(outside) / max(1, np.sum(active))
                ),
                'maximumEnvelopeExcursionInStd': float(np.max(excursion[active])),
                'maximumAbsoluteZScore': float(np.max(np.abs(z[active]))),
                'topOutsideFeatures': [
                    {
                        'feature': featureNames[index],
                        'value': float(features[index]),
                        'trainingMinimum': float(trainingMinimum[index]),
                        'trainingMaximum': float(trainingMaximum[index]),
                        'envelopeExcursionInStd': float(excursion[index]),
                    }
                    for index in topOutside
                ],
                'temperatureMeanK': float(
                    np.average(
                        case.temperature.temperatureK,
                        weights=case.temperature.weightsM3,
                    )
                ),
                'temperatureSpanK': float(np.ptp(case.temperature.temperatureK)),
                'volumeHeatSourcePowerW': float(case.volume.totalHeatSourcePowerW),
                'boundaryHeatFlowW': float(case.boundary.totalHeatFlowRateW),
            }
        )

    projectIds = [
        caseId
        for caseId in ('tutorial-ad-stator', 'tutorial-ag-stator', 'tutorial-aj-stator')
        if caseId in byId
    ]
    projectValues = contract.transformCases([byId[caseId] for caseId in projectIds])
    projectFeatures = {
        caseId: value.operatorCase.globalFeatures
        for caseId, value in zip(projectIds, projectValues)
    }
    projectDistances = []
    for leftIndex, leftId in enumerate(projectIds):
        for rightId in projectIds[leftIndex + 1 :]:
            projectDistances.append(
                {
                    'leftCaseId': leftId,
                    'rightCaseId': rightId,
                    'standardizedRmsDistance': _distance(
                        projectFeatures[leftId],
                        projectFeatures[rightId],
                        trainingStd,
                    ),
                }
            )

    result = {
        'source': str(dataset.sourcePath),
        'checkpoint': str(arguments.checkpoint),
        'distanceDefinition': (
            'RMS Euclidean distance over active dimensionless global features, '
            'scaled by the selected training-set standard deviation'
        ),
        'envelopeDefinition': (
            'axis-aligned minimum/maximum of each active dimensionless global '
            'feature in the selected training set'
        ),
        'trainingCaseIds': [case.caseId for case in trainingCases],
        'validationCaseIds': [case.caseId for case in validationCases],
        'coverage': coverage,
        'motorProjectPairDistances': projectDistances,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
