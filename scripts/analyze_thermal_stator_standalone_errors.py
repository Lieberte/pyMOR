from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mor.nn.data import (
    collateThermalOperatorCases,
    loadSubsampledThermalDataset,
    steadyThermalDimensionlessV2GlobalFeatureNames,
    thermalCheckpointDataContract,
)
from mor.nn.models import thermalCategorizedOperatorTorch

try:
    from scripts.train_thermal_discrete_pinn import _splitCases
except ModuleNotFoundError:
    from train_thermal_discrete_pinn import _splitCases


CASE_LABELS = {
    'tutorial-b-stator': 'B',
    'tutorial-h-stator': 'H',
    'tutorial-ag-stator': 'AG',
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate standalone stator absolute-temperature error clouds.'
    )
    parser.add_argument('source', type=Path)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=23)
    return parser.parse_args()


def _weightedQuantile(values, weights, quantiles):
    order = np.argsort(values)
    sortedValues = np.asarray(values)[order]
    sortedWeights = np.asarray(weights)[order]
    cumulative = np.cumsum(sortedWeights) - 0.5 * sortedWeights
    cumulative /= np.sum(sortedWeights)
    return np.interp(quantiles, cumulative, sortedValues)


def _principalCoordinates(coordinates):
    centered = coordinates - np.mean(coordinates, axis=0, keepdims=True)
    _, _, rotation = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ rotation.T
    scale = np.max(np.ptp(projected, axis=0))
    return projected / max(scale, np.finfo(np.float64).eps)


def _record(case, dimensionless, prediction):
    weights = np.asarray(case.temperature.weightsM3, dtype=np.float64)
    weights /= np.sum(weights)
    trueK = np.asarray(case.temperature.temperatureK, dtype=np.float64)
    scales = dimensionless.scales
    predictionK = (
        np.asarray(prediction, dtype=np.float64) * scales.temperatureScaleK
        + scales.temperatureReferenceK
    )
    errorK = predictionK - trueK
    meanTrue = float(np.sum(weights * trueK))
    meanPrediction = float(np.sum(weights * predictionK))
    meanBias = meanPrediction - meanTrue
    centeredError = errorK - meanBias
    rmse = math.sqrt(float(np.sum(weights * np.square(errorK))))
    shapeRmse = math.sqrt(float(np.sum(weights * np.square(centeredError))))
    biasMseFraction = meanBias**2 / max(rmse**2, np.finfo(np.float64).eps)
    trueQuantiles = _weightedQuantile(trueK, weights, (0.05, 0.5, 0.95))
    predictionQuantiles = _weightedQuantile(
        predictionK, weights, (0.05, 0.5, 0.95)
    )
    coordinates = np.asarray(case.temperature.coordinatesMeters, dtype=np.float64)
    principal = _principalCoordinates(coordinates)
    design = np.column_stack((np.ones(len(principal)), principal))
    weightedDesign = design * np.sqrt(weights)[:, None]
    weightedError = errorK * np.sqrt(weights)
    coefficients = np.linalg.lstsq(weightedDesign, weightedError, rcond=None)[0]
    fitted = design @ coefficients
    residual = errorK - fitted
    spatialTrendR2 = 1.0 - float(np.sum(weights * residual**2)) / max(
        float(np.sum(weights * centeredError**2)),
        np.finfo(np.float64).eps,
    )
    return {
        'caseId': case.caseId,
        'label': CASE_LABELS[case.caseId],
        'metrics': {
            'weightedTrueMeanK': meanTrue,
            'weightedPredictionMeanK': meanPrediction,
            'meanBiasK': meanBias,
            'rmseK': rmse,
            'shapeRmseK': shapeRmse,
            'biasMseFraction': biasMseFraction,
            'maximumAbsoluteErrorK': float(np.max(np.abs(errorK))),
            'trueMinimumK': float(np.min(trueK)),
            'trueMaximumK': float(np.max(trueK)),
            'trueSpanK': float(np.ptp(trueK)),
            'predictionMinimumK': float(np.min(predictionK)),
            'predictionMaximumK': float(np.max(predictionK)),
            'predictionSpanK': float(np.ptp(predictionK)),
            'trueP05K': float(trueQuantiles[0]),
            'trueMedianK': float(trueQuantiles[1]),
            'trueP95K': float(trueQuantiles[2]),
            'predictionP05K': float(predictionQuantiles[0]),
            'predictionMedianK': float(predictionQuantiles[1]),
            'predictionP95K': float(predictionQuantiles[2]),
            'spatialTrendR2OfCenteredError': spatialTrendR2,
        },
        'points': [
            {
                'p1': round(float(point[0]), 6),
                'p2': round(float(point[1]), 6),
                'p3': round(float(point[2]), 6),
                'trueK': round(float(truth), 5),
                'predictionK': round(float(predicted), 5),
                'errorK': round(float(error), 5),
                'weight': round(float(weight), 8),
            }
            for point, truth, predicted, error, weight in zip(
                principal, trueK, predictionK, errorK, weights
            )
        ],
    }


def _plot(records, outputPath):
    figure, axes = plt.subplots(
        len(records),
        3,
        figsize=(13.2, 10.8),
        constrained_layout=True,
    )
    allTemperatures = np.concatenate(
        [
            np.asarray([point[key] for point in record['points']])
            for record in records
            for key in ('trueK', 'predictionK')
        ]
    )
    temperatureMinimum = float(np.min(allTemperatures))
    temperatureMaximum = float(np.max(allTemperatures))
    errorLimit = max(
        abs(point['errorK']) for record in records for point in record['points']
    )
    for row, record in enumerate(records):
        points = record['points']
        x = np.asarray([point['p1'] for point in points])
        y = np.asarray([point['p2'] for point in points])
        values = (
            np.asarray([point['trueK'] for point in points]),
            np.asarray([point['predictionK'] for point in points]),
            np.asarray([point['errorK'] for point in points]),
        )
        labels = ('True temperature / K', 'Prediction / K', 'Signed error / K')
        for column, (value, label) in enumerate(zip(values, labels)):
            axis = axes[row, column]
            if column < 2:
                scatter = axis.scatter(
                    x,
                    y,
                    c=value,
                    s=18,
                    cmap='viridis',
                    vmin=temperatureMinimum,
                    vmax=temperatureMaximum,
                    linewidths=0,
                )
            else:
                scatter = axis.scatter(
                    x,
                    y,
                    c=value,
                    s=18,
                    cmap='coolwarm',
                    vmin=-errorLimit,
                    vmax=errorLimit,
                    linewidths=0,
                )
            axis.set_aspect('equal', adjustable='box')
            axis.set_xlabel('principal coordinate 1')
            axis.set_ylabel('principal coordinate 2')
            axis.set_title(f"{record['label']} - {label}")
            figure.colorbar(scatter, ax=axis, shrink=0.72)
    figure.suptitle('Standalone stator: temperature and error point clouds')
    figure.savefig(outputPath, dpi=180)
    plt.close(figure)


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
    training, validation = _splitCases(prepared, 0.2, arguments.seed)
    trainingCases = tuple(
        case for case in training if case.componentType == 'stator'
    )
    cases = tuple(
        case
        for case in validation
        if case.caseId in CASE_LABELS
    )
    cases = tuple(sorted(cases, key=lambda case: tuple(CASE_LABELS).index(case.caseId)))
    dimensionlessCases = contract.transformCases(cases)
    trainingDimensionless = contract.transformCases(trainingCases)
    model = thermalCategorizedOperatorTorch.fromCheckpoint(checkpoint).globalModel
    device = model.toDevice(arguments.device, allowAutoFallback=True)
    model.eval()
    records = []
    with torch.no_grad():
        for case, dimensionless in zip(cases, dimensionlessCases):
            batch = collateThermalOperatorCases([dimensionless.operatorCase])
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            prediction = model(batch)[0, :, 0].detach().cpu().numpy()
            records.append(_record(case, dimensionless, prediction))
    model.cpu()

    globalFeatureNames = steadyThermalDimensionlessV2GlobalFeatureNames(
        contract.schema
    )
    trainingGlobal = np.vstack(
        [value.operatorCase.globalFeatures for value in trainingDimensionless]
    )
    globalMean = np.mean(trainingGlobal, axis=0)
    globalStd = np.std(trainingGlobal, axis=0)
    active = globalStd > 1e-6
    trainingSpans = np.asarray(
        [np.ptp(case.temperature.temperatureK) for case in trainingCases]
    )
    diagnostics = {}
    dimensionlessById = {
        case.caseId: value for case, value in zip(cases, dimensionlessCases)
    }
    for case in cases:
        value = dimensionlessById[case.caseId]
        z = np.zeros_like(globalMean)
        z[active] = (
            value.operatorCase.globalFeatures[active] - globalMean[active]
        ) / globalStd[active]
        distances = np.sqrt(
            np.mean(
                np.square(
                    (
                        trainingGlobal[:, active]
                        - value.operatorCase.globalFeatures[active]
                    )
                    / globalStd[active]
                ),
                axis=1,
            )
        )
        nearestIndex = int(np.argmin(distances))
        topIndices = np.argsort(np.abs(z))[::-1][:8]
        trueSpan = float(np.ptp(case.temperature.temperatureK))
        diagnostics[CASE_LABELS[case.caseId]] = {
            'geometryId': case.geometryId,
            'assemblyId': case.metadata.get('assemblyId'),
            'nearestTrainingCaseId': trainingCases[nearestIndex].caseId,
            'nearestStandardizedGlobalDistance': float(distances[nearestIndex]),
            'trueSpanTrainingPercentile': float(
                np.mean(trainingSpans <= trueSpan) * 100.0
            ),
            'temperatureScaleK': value.scales.temperatureScaleK,
            'volumeHeatSourcePowerW': case.volume.totalHeatSourcePowerW,
            'boundaryHeatFlowW': case.boundary.totalHeatFlowRateW,
            'topGlobalFeatureZScores': [
                {
                    'feature': globalFeatureNames[index],
                    'zScore': float(z[index]),
                }
                for index in topIndices
            ],
        }

    byLabel = {record['label']: record for record in records}
    b = byLabel['B']['metrics']
    h = byLabel['H']['metrics']
    comparison = {
        'trueWeightedMeanDifferenceHMinusBK': (
            h['weightedTrueMeanK'] - b['weightedTrueMeanK']
        ),
        'predictionWeightedMeanDifferenceHMinusBK': (
            h['weightedPredictionMeanK'] - b['weightedPredictionMeanK']
        ),
        'trueMinimumDifferenceHMinusBK': h['trueMinimumK'] - b['trueMinimumK'],
        'trueMaximumDifferenceHMinusBK': h['trueMaximumK'] - b['trueMaximumK'],
        'trueMedianDifferenceHMinusBK': h['trueMedianK'] - b['trueMedianK'],
        'trueP05DifferenceHMinusBK': h['trueP05K'] - b['trueP05K'],
        'trueP95DifferenceHMinusBK': h['trueP95K'] - b['trueP95K'],
    }
    result = {
        'checkpoint': str(arguments.checkpoint),
        'errorDefinition': 'predictionK - trueK',
        'projection': 'case-local principal coordinates from physical query points',
        'cases': records,
        'bVsH': comparison,
        'trainingReference': {
            'statorCaseCount': len(trainingCases),
            'trueSpanMinimumK': float(np.min(trainingSpans)),
            'trueSpanMedianK': float(np.median(trainingSpans)),
            'trueSpanMaximumK': float(np.max(trainingSpans)),
        },
        'caseDiagnostics': diagnostics,
    }
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    jsonPath = arguments.output_directory / 'stator_error_analysis.json'
    imagePath = arguments.output_directory / 'stator_error_clouds.png'
    jsonPath.write_text(json.dumps(result, indent=2), encoding='utf-8')
    _plot(records, imagePath)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
