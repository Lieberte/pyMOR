from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from mor.nn.data import (
    buildBoxFemCase,
    collateThermalOperatorCases,
    thermalComponentCase,
    thermalComponentFeatureSchema,
    thermalOperatorFeatureScaling,
    thermalTemperatureSamples,
)
from mor.nn.losses import thermalOperatorLoss
from mor.nn.models import thermalMeshGnnTorch, thermalQuerySetOperatorTorch


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Compare point-set thermal operator and mesh GNN on one solved box FEM case.'
    )
    parser.add_argument('case', type=Path)
    parser.add_argument('--result-time', type=float, default=1.0)
    parser.add_argument('--maximum-volume-samples', type=int, default=1024)
    parser.add_argument('--train-fraction', type=float, default=0.7)
    parser.add_argument('--point-epochs', type=int, default=1500)
    parser.add_argument('--gnn-epochs', type=int, default=400)
    parser.add_argument('--learning-rate', type=float, default=2e-3)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--output', type=Path)
    return parser.parse_args()


def _subsetCase(
    case: thermalComponentCase,
    indices: np.ndarray,
    suffix: str,
) -> thermalComponentCase:
    return thermalComponentCase(
        caseId=f'{case.caseId}-{suffix}',
        geometryId=case.geometryId,
        componentType=case.componentType,
        volume=case.volume,
        boundary=case.boundary,
        temperature=thermalTemperatureSamples(
            coordinatesMeters=case.temperature.coordinatesMeters[indices],
            temperatureK=case.temperature.temperatureK[indices],
            weightsM3=case.temperature.weightsM3[indices],
        ),
        operatingFeatures=case.operatingFeatures,
        metadata=case.metadata,
    )


def _scaling(case: thermalComponentCase, trainingIndices: np.ndarray):
    temperatures = case.temperature.temperatureK
    trainingWeights = case.temperature.weightsM3[trainingIndices]
    return thermalOperatorFeatureScaling(
        lengthMeters=float(np.max(case.coordinateFrame.halfExtentMeters)),
        conductivityWPerMK=float(np.max(np.abs(case.volume.conductivityWPerMK))),
        densityKgPerM3=float(np.max(case.volume.densityKgPerM3)),
        specificHeatJPerKgK=float(np.max(case.volume.specificHeatJPerKgK)),
        heatSourceWPerM3=max(
            float(np.max(np.abs(case.volume.heatSourceWPerM3))), 1.0
        ),
        htcWPerM2K=1.0,
        heatFluxWPerM2=1.0,
        temperatureReferenceK=float(
            np.average(temperatures[trainingIndices], weights=trainingWeights)
        ),
        temperatureScaleK=max(float(np.ptp(temperatures)), 0.01),
    )


def _batchToDevice(batch, device: str):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _weightedMetrics(errorK: torch.Tensor, weights: torch.Tensor) -> dict[str, float]:
    weightSum = weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
    return {
        'rmseK': math.sqrt(float(torch.sum(weights * errorK.square()) / weightSum)),
        'meanAbsoluteErrorK': float(torch.sum(weights * torch.abs(errorK)) / weightSum),
        'maximumErrorK': float(torch.max(torch.abs(errorK))),
    }


def _trainPointOperator(
    case: thermalComponentCase,
    trainingIndices: np.ndarray,
    validationIndices: np.ndarray,
    *,
    epochs: int,
    learningRate: float,
    device: str,
    seed: int,
):
    scaling = _scaling(case, trainingIndices)
    schema = thermalComponentFeatureSchema(
        boundaryTypeCount=2,
        temperatureBoundaryTypeIds=(1,),
    )
    trainOperatorCase = _subsetCase(case, trainingIndices, 'train').toOperatorCase(
        schema=schema, scaling=scaling
    )
    validationOperatorCase = _subsetCase(
        case, validationIndices, 'validation'
    ).toOperatorCase(schema=schema, scaling=scaling)
    trainBatch = _batchToDevice(
        collateThermalOperatorCases([trainOperatorCase]), device
    )
    validationBatch = _batchToDevice(
        collateThermalOperatorCases([validationOperatorCase]), device
    )
    torch.manual_seed(seed)
    model = thermalQuerySetOperatorTorch(
        volumeFeatureDim=trainOperatorCase.volumeFeatureDim,
        boundaryFeatureDim=trainOperatorCase.boundaryFeatureDim,
        globalFeatureDim=trainOperatorCase.globalFeatureDim,
        queryDim=trainOperatorCase.queryDim,
        hiddenDim=64,
        kernelScales=(0.10, 0.30),
        queryFourierFrequencies=4,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
    lossFunction = thermalOperatorLoss(hotspotWeight=0.5)
    history = []
    start = time.perf_counter()
    model.train()
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(trainBatch)
        loss = lossFunction.compute(
            prediction,
            trainBatch['targets'],
            trainBatch['queryWeights'],
            trainBatch['queryMask'],
        )
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu()))
    elapsed = time.perf_counter() - start

    model.eval()
    with torch.no_grad():
        trainError = (
            model(trainBatch)[0, :, 0] - trainBatch['targets'][0, :, 0]
        ) * scaling.temperatureScaleK
        validationError = (
            model(validationBatch)[0, :, 0]
            - validationBatch['targets'][0, :, 0]
        ) * scaling.temperatureScaleK
    return {
        'training': {
            'epochs': int(epochs),
            'elapsedSeconds': elapsed,
            'initialLoss': history[0],
            'finalLoss': history[-1],
        },
        'train': _weightedMetrics(trainError, trainBatch['queryWeights'][0]),
        'heldOutInterior': _weightedMetrics(
            validationError, validationBatch['queryWeights'][0]
        ),
    }


def _trainGnn(
    graphCase,
    trainingIndices: np.ndarray,
    validationIndices: np.ndarray,
    *,
    epochs: int,
    learningRate: float,
    device: str,
    seed: int,
    useGlobalContext: bool = False,
):
    case = graphCase.componentCase
    frame = case.coordinateFrame
    normalizedCoordinates = frame.normalize(graphCase.nodesMeters)
    temperatureReference = float(
        np.average(
            graphCase.temperatureK[trainingIndices],
            weights=graphCase.nodeWeightsM3[trainingIndices],
        )
    )
    temperatureScale = max(float(np.ptp(graphCase.temperatureK)), 0.01)
    sourceScale = max(float(np.max(np.abs(graphCase.nodeHeatSourceWPerM3))), 1.0)
    boundaryMask = np.isfinite(graphCase.fixedTemperatureK)
    fixedTemperatureFeature = np.zeros(graphCase.nodesMeters.shape[0], dtype=np.float64)
    fixedTemperatureFeature[boundaryMask] = (
        graphCase.fixedTemperatureK[boundaryMask] - temperatureReference
    ) / temperatureScale
    conductivity = float(case.volume.conductivityWPerMK[0, 0, 0])
    density = float(case.volume.densityKgPerM3[0])
    specificHeat = float(case.volume.specificHeatJPerKgK[0])
    nodeFeatures = np.column_stack(
        (
            normalizedCoordinates,
            graphCase.nodeHeatSourceWPerM3 / sourceScale,
            boundaryMask.astype(np.float64),
            fixedTemperatureFeature,
            np.full(graphCase.nodesMeters.shape[0], conductivity / max(conductivity, 1.0)),
            np.full(graphCase.nodesMeters.shape[0], density / max(density, 1.0)),
            np.full(
                graphCase.nodesMeters.shape[0],
                specificHeat / max(specificHeat, 1.0),
            ),
        )
    )
    edgeIndex = graphCase.edgeIndex
    relative = normalizedCoordinates[edgeIndex[0]] - normalizedCoordinates[edgeIndex[1]]
    edgeFeatures = np.column_stack((relative, np.linalg.norm(relative, axis=1)))
    target = (graphCase.temperatureK - temperatureReference) / temperatureScale

    nodeFeaturesTensor = torch.as_tensor(nodeFeatures, dtype=torch.float32, device=device)
    edgeIndexTensor = torch.as_tensor(edgeIndex, dtype=torch.long, device=device)
    edgeFeaturesTensor = torch.as_tensor(edgeFeatures, dtype=torch.float32, device=device)
    targetTensor = torch.as_tensor(target, dtype=torch.float32, device=device)
    weightsTensor = torch.as_tensor(
        graphCase.nodeWeightsM3, dtype=torch.float32, device=device
    )
    boundaryMaskTensor = torch.as_tensor(boundaryMask, dtype=torch.bool, device=device)
    fixedTemperatureTensor = torch.as_tensor(
        fixedTemperatureFeature, dtype=torch.float32, device=device
    )
    trainIndexTensor = torch.as_tensor(trainingIndices, dtype=torch.long, device=device)
    validationIndexTensor = torch.as_tensor(
        validationIndices, dtype=torch.long, device=device
    )

    torch.manual_seed(seed)
    model = thermalMeshGnnTorch(
        nodeFeatureDim=nodeFeatures.shape[1],
        edgeFeatureDim=edgeFeatures.shape[1],
        hiddenDim=32,
        messagePassingSteps=3,
        useGlobalContext=useGlobalContext,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
    history = []
    start = time.perf_counter()
    model.train()
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(
            nodeFeaturesTensor, edgeIndexTensor, edgeFeaturesTensor
        )[:, 0]
        prediction = torch.where(
            boundaryMaskTensor, fixedTemperatureTensor, prediction
        )
        trainWeights = weightsTensor[trainIndexTensor]
        trainError = prediction[trainIndexTensor] - targetTensor[trainIndexTensor]
        loss = torch.sum(trainWeights * trainError.square()) / trainWeights.sum()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu()))
    elapsed = time.perf_counter() - start

    model.eval()
    with torch.no_grad():
        prediction = model(
            nodeFeaturesTensor, edgeIndexTensor, edgeFeaturesTensor
        )[:, 0]
        prediction = torch.where(
            boundaryMaskTensor, fixedTemperatureTensor, prediction
        )
        trainErrorK = (
            prediction[trainIndexTensor] - targetTensor[trainIndexTensor]
        ) * temperatureScale
        validationErrorK = (
            prediction[validationIndexTensor] - targetTensor[validationIndexTensor]
        ) * temperatureScale
    return {
        'training': {
            'epochs': int(epochs),
            'elapsedSeconds': elapsed,
            'initialLoss': history[0],
            'finalLoss': history[-1],
            'edgeCount': int(edgeIndex.shape[1]),
            'useGlobalContext': bool(useGlobalContext),
        },
        'train': _weightedMetrics(
            trainErrorK, weightsTensor[trainIndexTensor]
        ),
        'heldOutInterior': _weightedMetrics(
            validationErrorK, weightsTensor[validationIndexTensor]
        ),
    }


def main() -> None:
    arguments = _arguments()
    if not 0.0 < arguments.train_fraction < 1.0:
        raise ValueError('trainFraction must lie strictly between zero and one')
    device = (
        arguments.device
        if not arguments.device.startswith('cuda') or torch.cuda.is_available()
        else 'cpu'
    )
    graphCase = buildBoxFemCase(
        arguments.case,
        resultTime=arguments.result_time,
        maximumVolumeSamples=arguments.maximum_volume_samples,
        randomSeed=arguments.seed,
    )
    boundaryMask = np.isfinite(graphCase.fixedTemperatureK)
    interiorNodes = np.flatnonzero(~boundaryMask & (graphCase.nodeWeightsM3 > 0.0))
    generator = np.random.default_rng(arguments.seed)
    shuffled = generator.permutation(interiorNodes)
    trainingCount = int(round(arguments.train_fraction * shuffled.size))
    trainingIndices = np.sort(shuffled[:trainingCount])
    validationIndices = np.sort(shuffled[trainingCount:])

    trainingMean = float(
        np.average(
            graphCase.temperatureK[trainingIndices],
            weights=graphCase.nodeWeightsM3[trainingIndices],
        )
    )
    baselineError = torch.as_tensor(
        trainingMean - graphCase.temperatureK[validationIndices], dtype=torch.float64
    )
    baselineWeights = torch.as_tensor(
        graphCase.nodeWeightsM3[validationIndices], dtype=torch.float64
    )
    report = {
        'case': str(arguments.case),
        'resultTimeS': arguments.result_time,
        'device': device,
        'dataAssessment': {
            'availableResultTimesS': [0.0, 1.0, 2.0],
            'independentGeometryCount': 1,
            'independentNontrivialSolutionCount': 1,
            'geometryHoldoutPossible': False,
            'note': (
                '1.0 s and 2.0 s results are numerically identical; this run is a '
                'single-geometry spatial holdout smoke test, not geometry generalization.'
            ),
        },
        'mesh': {
            'nodeCount': int(graphCase.nodesMeters.shape[0]),
            'quadraticTetrahedronCount': int(graphCase.tetrahedra10.shape[0]),
            'quadraticBoundaryTriangleCount': int(
                graphCase.boundaryTriangles6.shape[0]
            ),
            'interiorNodeCount': int(interiorNodes.size),
            'trainingInteriorNodeCount': int(trainingIndices.size),
            'heldOutInteriorNodeCount': int(validationIndices.size),
        },
        'physics': {
            'mappedHeatSourcePowerW': graphCase.componentCase.metadata[
                'mappedHeatSourcePowerW'
            ],
            'sampledHeatSourcePowerW': graphCase.componentCase.metadata[
                'sampledHeatSourcePowerW'
            ],
            'temperatureMinimumK': float(graphCase.temperatureK.min()),
            'temperatureMaximumK': float(graphCase.temperatureK.max()),
            'temperatureRangeK': float(np.ptp(graphCase.temperatureK)),
        },
        'constantMeanBaseline': {
            'heldOutInterior': _weightedMetrics(baselineError, baselineWeights)
        },
    }
    report['pointSetOperator'] = _trainPointOperator(
        graphCase.componentCase,
        trainingIndices,
        validationIndices,
        epochs=arguments.point_epochs,
        learningRate=arguments.learning_rate,
        device=device,
        seed=arguments.seed,
    )
    report['meshGnnLocal'] = _trainGnn(
        graphCase,
        trainingIndices,
        validationIndices,
        epochs=arguments.gnn_epochs,
        learningRate=arguments.learning_rate,
        device=device,
        seed=arguments.seed + 1,
    )
    report['meshGnnGlobalMean'] = _trainGnn(
        graphCase,
        trainingIndices,
        validationIndices,
        epochs=arguments.gnn_epochs,
        learningRate=arguments.learning_rate,
        device=device,
        seed=arguments.seed + 1,
        useGlobalContext=True,
    )
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
