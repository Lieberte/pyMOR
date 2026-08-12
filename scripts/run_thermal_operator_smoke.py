from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from mor.nn.data import thermalOperatorCase, thermalOperatorDataModule
from mor.nn.losses import thermalOperatorLoss
from mor.nn.models import (
    thermalQuerySetOperatorTorch,
    thermalSetAttentionOperatorTorch,
    thermalSetDeepONetTorch,
)
from mor.nn.trainers import thermalOperatorTrainerTorch


TEMPERATURE_SCALE_K = 50.0


def _boxBoundary(rng: np.random.Generator, dimensions: np.ndarray, samplesPerFace: int):
    points = []
    normals = []
    weights = []
    faceDefinitions = (
        (0, -1.0, dimensions[1] * dimensions[2]),
        (0, 1.0, dimensions[1] * dimensions[2]),
        (1, -1.0, dimensions[0] * dimensions[2]),
        (1, 1.0, dimensions[0] * dimensions[2]),
        (2, -1.0, dimensions[0] * dimensions[1]),
        (2, 1.0, dimensions[0] * dimensions[1]),
    )
    for axis, side, area in faceDefinitions:
        facePoints = rng.uniform(-1.0, 1.0, size=(samplesPerFace, 3))
        facePoints[:, axis] = side
        faceNormals = np.zeros_like(facePoints)
        faceNormals[:, axis] = side
        points.append(facePoints)
        normals.append(faceNormals)
        weights.append(np.full(samplesPerFace, area / samplesPerFace))
    return np.vstack(points), np.vstack(normals), np.concatenate(weights)


def _makeCase(
    geometryId: str,
    dimensions: np.ndarray,
    operatingIndex: int,
    seed: int,
) -> thermalOperatorCase:
    rng = np.random.default_rng(seed)
    volumeCount = 48 + 8 * ((operatingIndex + seed) % 6)
    samplesPerFace = 6 + ((2 * operatingIndex + seed) % 5)
    queryCount = 56 + 8 * ((3 * operatingIndex + seed) % 6)
    volumeCoordinates = rng.uniform(-1.0, 1.0, size=(volumeCount, 3))
    queryCoordinates = rng.uniform(-1.0, 1.0, size=(queryCount, 3))
    boundaryCoordinates, boundaryNormals, boundaryWeights = _boxBoundary(
        rng, dimensions, samplesPerFace
    )
    volume = float(np.prod(dimensions))
    surfaceArea = float(
        2.0
        * (
            dimensions[0] * dimensions[1]
            + dimensions[1] * dimensions[2]
            + dimensions[0] * dimensions[2]
        )
    )
    volumeWeights = np.full(volumeCount, volume / volumeCount)
    conductivity = rng.uniform(0.65, 1.35)
    heatLevel = rng.uniform(0.55, 1.45)
    convectionLevel = rng.uniform(0.6, 1.4)
    ambient = rng.uniform(-0.25, 0.25)
    speed = rng.uniform(0.2, 1.0)
    sourceCenter = rng.uniform(-0.45, 0.45, size=3)
    sourceWidth = rng.uniform(0.22, 0.42)
    sourceDistance = np.sum((volumeCoordinates - sourceCenter) ** 2, axis=1)
    localHeat = heatLevel * (
        0.2 + 0.8 * np.exp(-sourceDistance / (2.0 * sourceWidth**2))
    )
    localConductivity = conductivity * (
        1.0 + 0.08 * volumeCoordinates[:, 0] * volumeCoordinates[:, 2]
    )
    flowAngle = rng.uniform(-math.pi, math.pi)
    flowDirection = np.array([math.cos(flowAngle), math.sin(flowAngle), 0.25 * speed])
    flowDirection /= np.linalg.norm(flowDirection)
    localConvection = convectionLevel * (
        1.0
        + 0.32 * (boundaryNormals @ flowDirection)
        + 0.10 * boundaryCoordinates[:, 2]
    )
    localAmbient = ambient + 0.04 * boundaryCoordinates[:, 0]
    volumeFeatures = np.column_stack(
        (volumeCoordinates, localConductivity, localHeat)
    )
    boundaryFeatures = np.column_stack(
        (
            boundaryCoordinates,
            boundaryNormals,
            localConvection,
            localAmbient,
        )
    )
    heatWeights = localHeat * volumeWeights
    inferredSourceCenter = np.sum(
        volumeCoordinates * heatWeights[:, None], axis=0
    ) / np.sum(heatWeights)
    boundaryConductanceWeights = localConvection * boundaryWeights
    effectiveAmbient = float(
        np.sum(localAmbient * boundaryConductanceWeights)
        / np.sum(boundaryConductanceWeights)
    )
    coolingDirection = np.sum(
        boundaryNormals * boundaryConductanceWeights[:, None], axis=0
    ) / np.sum(boundaryConductanceWeights)
    totalHeat = float(np.sum(heatWeights))
    totalConductance = float(np.sum(boundaryConductanceWeights))
    resistance = 0.55 / (conductivity * volume ** (1.0 / 3.0)) + 0.75 / totalConductance
    distanceToSource = np.sum((queryCoordinates - inferredSourceCenter) ** 2, axis=1)
    hotspot = np.exp(-distanceToSource / (2.0 * (sourceWidth + 0.08) ** 2))
    broadShape = 1.0 - np.sum(queryCoordinates**2, axis=1) / 3.0
    coolingTilt = queryCoordinates @ coolingDirection
    normalizedTemperature = (
        effectiveAmbient
        + totalHeat * resistance * (0.30 + 0.55 * hotspot + 0.15 * broadShape)
        - 0.18 * totalHeat * coolingTilt
    )
    globalFeatures = np.array(
        [
            dimensions[0],
            dimensions[1],
            dimensions[2],
            conductivity,
            heatLevel,
            convectionLevel,
            ambient,
            speed,
            volume,
            surfaceArea,
        ],
        dtype=np.float32,
    )
    return thermalOperatorCase(
        caseId=f'{geometryId}_op{operatingIndex:02d}',
        geometryId=geometryId,
        volumeFeatures=volumeFeatures,
        volumeWeights=volumeWeights,
        boundaryFeatures=boundaryFeatures,
        boundaryWeights=boundaryWeights,
        globalFeatures=globalFeatures,
        queryCoordinates=queryCoordinates,
        queryTemperatures=normalizedTemperature,
        queryWeights=np.full(queryCount, volume / queryCount),
    )


def buildSyntheticCases(casesPerGeometry: int, seed: int):
    dimensions = {
        'motorA': np.array([0.80, 0.95, 1.05]),
        'motorB': np.array([1.00, 0.82, 1.20]),
        'motorC': np.array([1.18, 1.00, 0.88]),
        'motorD': np.array([0.92, 1.22, 1.08]),
        'motorE': np.array([1.12, 0.90, 1.28]),
        'motorHoldout': np.array([1.04, 1.14, 0.96]),
    }
    cases = []
    for geometryIndex, (geometryId, geometryDimensions) in enumerate(dimensions.items()):
        for operatingIndex in range(casesPerGeometry):
            cases.append(
                _makeCase(
                    geometryId,
                    geometryDimensions,
                    operatingIndex,
                    seed + 1000 * geometryIndex + 17 * operatingIndex,
                )
            )
    return cases


def _batchToDevice(batch, device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def evaluate(model, dataModule, device, *, shuffleCaseFields=False):
    squaredError = 0.0
    squaredTarget = 0.0
    weightedCount = 0.0
    maximumError = 0.0
    rows = []
    model.evalMode()
    with torch.no_grad():
        for rawBatch in dataModule.getValidationLoader():
            batch = _batchToDevice(rawBatch, device)
            if shuffleCaseFields and batch['globalFeatures'].shape[0] > 1:
                for key in (
                    'volumeFeatures',
                    'volumeWeights',
                    'volumeMask',
                    'boundaryFeatures',
                    'boundaryWeights',
                    'boundaryMask',
                ):
                    batch[key] = torch.roll(batch[key], shifts=1, dims=0)
            predictions = model(batch)
            for batchIndex, caseId in enumerate(batch['caseIds']):
                mask = batch['queryMask'][batchIndex]
                weights = batch['queryWeights'][batchIndex, mask]
                error = predictions[batchIndex, mask, 0] - batch['targets'][batchIndex, mask, 0]
                target = batch['targets'][batchIndex, mask, 0]
                caseSquaredError = float(torch.sum(weights * error.square()).item())
                caseSquaredTarget = float(torch.sum(weights * target.square()).item())
                caseWeight = float(torch.sum(weights).item())
                caseMaximum = float(torch.max(torch.abs(error)).item())
                squaredError += caseSquaredError
                squaredTarget += caseSquaredTarget
                weightedCount += caseWeight
                maximumError = max(maximumError, caseMaximum)
                rows.append(
                    {
                        'caseId': caseId,
                        'geometryId': batch['geometryIds'][batchIndex],
                        'rmseNormalized': math.sqrt(caseSquaredError / caseWeight),
                        'rmseKelvin': TEMPERATURE_SCALE_K
                        * math.sqrt(caseSquaredError / caseWeight),
                        'relativeL2': math.sqrt(
                            caseSquaredError / max(caseSquaredTarget, 1e-12)
                        ),
                        'maximumErrorKelvin': TEMPERATURE_SCALE_K * caseMaximum,
                    }
                )
    return {
        'rmseNormalized': math.sqrt(squaredError / weightedCount),
        'rmseKelvin': TEMPERATURE_SCALE_K * math.sqrt(squaredError / weightedCount),
        'relativeL2': math.sqrt(squaredError / max(squaredTarget, 1e-12)),
        'maximumErrorKelvin': TEMPERATURE_SCALE_K * maximumError,
        'caseMetrics': rows,
    }


def runVariant(name, options, cases, epochs, deviceName, seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    dataModule = thermalOperatorDataModule(
        cases,
        validationGeometryIds=['motorHoldout'],
        batchSize=5,
        shuffle=True,
        randomSeed=seed,
    )
    dimensions = dataModule.featureDimensions
    modelType = options.get('modelType', 'deeponet')
    modelOptions = {key: value for key, value in options.items() if key != 'modelType'}
    common = {
        'volumeFeatureDim': dimensions['volume'],
        'boundaryFeatureDim': dimensions['boundary'],
        'globalFeatureDim': dimensions['global'],
        'queryDim': dimensions['query'],
    }
    if modelType == 'attention':
        model = thermalSetAttentionOperatorTorch(
            **common,
            hiddenDim=64,
            latentCount=8,
            attentionHeads=4,
            **modelOptions,
        )
    elif modelType == 'query_set':
        querySetOptions = {
            'hiddenDim': 32,
            'kernelScales': (0.3,),
            **modelOptions,
        }
        model = thermalQuerySetOperatorTorch(
            **common,
            **querySetOptions,
        )
    else:
        model = thermalSetDeepONetTorch(
            **common,
            pointHiddenDim=48,
            globalHiddenDim=32,
            latentDim=80,
            **modelOptions,
        )
    trainer = thermalOperatorTrainerTorch(
        thermalOperatorLoss(hotspotWeight=0.5),
        epochs=epochs,
        learningRate=2e-3,
        weightDecay=1e-6,
        deviceName=deviceName,
    )
    start = time.perf_counter()
    training = trainer.fit(model, dataModule)
    elapsed = time.perf_counter() - start
    metrics = evaluate(model, dataModule, training['resolvedDeviceName'])
    shuffledFieldMetrics = evaluate(
        model,
        dataModule,
        training['resolvedDeviceName'],
        shuffleCaseFields=True,
    )
    return {
        'variant': name,
        'options': options,
        'epochs': epochs,
        'elapsedSeconds': elapsed,
        'bestEpochIndex': training['bestEpochIndex'],
        'bestValidationLoss': training['bestValidationLoss'],
        'finalTrainLoss': training['trainLossHistory'][-1],
        'finalValidationLoss': training['validationLossHistory'][-1],
        'metrics': metrics,
        'shuffledFieldMetrics': shuffledFieldMetrics,
        'trainLossHistory': training['trainLossHistory'],
        'validationLossHistory': training['validationLossHistory'],
    }


def writeReports(results, outputDirectory: Path):
    outputDirectory.mkdir(parents=True, exist_ok=True)
    (outputDirectory / 'thermal_operator_smoke_results.json').write_text(
        json.dumps(results, indent=2), encoding='utf-8'
    )
    with (outputDirectory / 'thermal_operator_smoke_results.csv').open(
        'w', newline='', encoding='utf-8'
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                'variant',
                'rmseKelvin',
                'relativeL2',
                'maximumErrorKelvin',
                'shuffledFieldRmseKelvin',
                'bestEpochIndex',
                'elapsedSeconds',
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    'variant': result['variant'],
                    'rmseKelvin': result['metrics']['rmseKelvin'],
                    'relativeL2': result['metrics']['relativeL2'],
                    'maximumErrorKelvin': result['metrics']['maximumErrorKelvin'],
                    'shuffledFieldRmseKelvin': result['shuffledFieldMetrics']['rmseKelvin'],
                    'bestEpochIndex': result['bestEpochIndex'],
                    'elapsedSeconds': result['elapsedSeconds'],
                }
            )
    ranked = sorted(results, key=lambda item: item['metrics']['rmseKelvin'])
    lines = [
        '# 跨网格热神经算子冒泡结果',
        '',
        '验证集按完整几何 `motorHoldout` 留出，训练集与验证集节点数均可变化。',
        '',
        '| Variant | RMSE/K | Shuffled-field RMSE/K | Relative L2 | Max error/K | Best epoch |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for result in ranked:
        lines.append(
            f"| {result['variant']} | {result['metrics']['rmseKelvin']:.4f} | "
            f"{result['shuffledFieldMetrics']['rmseKelvin']:.4f} | "
            f"{result['metrics']['relativeL2']:.4f} | "
            f"{result['metrics']['maximumErrorKelvin']:.4f} | "
            f"{result['bestEpochIndex'] + 1} |"
        )
    lines.extend(
        [
            '',
            f"当前最优：`{ranked[0]['variant']}`。",
            '',
            '该实验只验证变长点集、整机留出和任意查询坐标闭环，不代表真实电机精度。',
        ]
    )
    (outputDirectory / 'THERMAL_OPERATOR_SMOKE_REPORT.md').write_text(
        '\n'.join(lines), encoding='utf-8-sig'
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--cases-per-geometry', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--variants', nargs='*', default=None)
    parser.add_argument('--output', type=Path, default=Path('reports/thermal_operator_smoke'))
    args = parser.parse_args()
    cases = buildSyntheticCases(args.cases_per_geometry, args.seed)
    variants = (
        ('mean', {'poolingMode': 'mean', 'queryFourierFrequencies': 0}),
        ('mean_max', {'poolingMode': 'mean_max', 'queryFourierFrequencies': 0}),
        ('mean_max_fourier2', {'poolingMode': 'mean_max', 'queryFourierFrequencies': 2}),
        (
            'latent_attention',
            {'modelType': 'attention', 'queryFourierFrequencies': 1},
        ),
        (
            'query_set_multiscale',
            {
                'modelType': 'query_set',
                'kernelScales': (0.18, 0.4, 0.85),
                'queryFourierFrequencies': 1,
            },
        ),
        (
            'query_set_single_scale',
            {
                'modelType': 'query_set',
                'kernelScales': (0.5,),
                'queryFourierFrequencies': 1,
            },
        ),
        (
            'query_set_single_scale_030',
            {
                'modelType': 'query_set',
                'kernelScales': (0.3,),
                'queryFourierFrequencies': 1,
            },
        ),
        (
            'query_set_single_scale_080',
            {
                'modelType': 'query_set',
                'kernelScales': (0.8,),
                'queryFourierFrequencies': 1,
            },
        ),
        (
            'query_set_multiscale_plain_query',
            {'modelType': 'query_set', 'queryFourierFrequencies': 0},
        ),
    )
    if args.variants:
        requested = frozenset(args.variants)
        variants = tuple(item for item in variants if item[0] in requested)
        missing = requested - {item[0] for item in variants}
        if missing:
            raise ValueError(f'unknown variants: {sorted(missing)}')
    results = [
        runVariant(name, options, cases, args.epochs, args.device, args.seed)
        for name, options in variants
    ]
    writeReports(results, args.output)
    print(json.dumps([{**item['metrics'], 'variant': item['variant']} for item in results], indent=2))


if __name__ == '__main__':
    main()
