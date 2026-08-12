from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from mor.nn.data import (
    collateThermalOperatorCases,
    loadThermalComponentDataset,
    thermalComponentFeatureSchema,
    thermalOperatorFeatureScaling,
)
from mor.nn.losses import thermalOperatorLoss
from mor.nn.models import thermalQuerySetOperatorTorch


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Overfit one converted motor FEM component case.')
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--learning-rate', type=float, default=2e-3)
    parser.add_argument('--hidden-dim', type=int, default=32)
    parser.add_argument('--kernel-scale', type=float, default=0.3)
    parser.add_argument('--query-fourier-frequencies', type=int, default=1)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--output', type=Path)
    return parser.parse_args()


def _scaling(case) -> thermalOperatorFeatureScaling:
    frame = case.coordinateFrame
    temperatures = case.temperature.temperatureK
    queryWeights = case.temperature.weightsM3
    temperatureScale = max(float(np.ptp(temperatures)), 0.01)
    conductivityScale = float(np.max(np.abs(case.volume.conductivityWPerMK)))
    heatSourceScale = max(float(np.max(np.abs(case.volume.heatSourceWPerM3))), 1.0)
    heatFluxScale = max(float(np.max(np.abs(case.boundary.heatFluxWPerM2))), 1.0)
    return thermalOperatorFeatureScaling(
        lengthMeters=float(np.max(frame.halfExtentMeters)),
        conductivityWPerMK=conductivityScale,
        densityKgPerM3=float(np.max(case.volume.densityKgPerM3)),
        specificHeatJPerKgK=float(np.max(case.volume.specificHeatJPerKgK)),
        heatSourceWPerM3=heatSourceScale,
        htcWPerM2K=1.0,
        heatFluxWPerM2=heatFluxScale,
        temperatureReferenceK=0.0,
        temperatureScaleK=temperatureScale,
    )


def _batchToDevice(batch, device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _metrics(prediction, batch, temperatureScale):
    mask = batch['queryMask']
    weights = batch['queryWeights'] * mask
    weightTotals = weights.sum(dim=1, keepdim=True).clamp_min(
        torch.finfo(weights.dtype).eps
    )
    prediction = prediction.squeeze(-1)
    targets = batch['targets'].squeeze(-1)
    prediction = prediction - (weights * prediction).sum(dim=1, keepdim=True) / weightTotals
    targets = targets - (weights * targets).sum(dim=1, keepdim=True) / weightTotals
    error = prediction - targets
    weightedMse = torch.sum(weights * error.square()) / torch.sum(weights)
    return {
        'rmseK': temperatureScale * math.sqrt(float(weightedMse.detach().cpu())),
        'maximumErrorK': temperatureScale
        * float(torch.max(torch.abs(error[mask])).detach().cpu()),
    }


def main() -> None:
    arguments = _arguments()
    torch.manual_seed(arguments.seed)
    cases = loadThermalComponentDataset(arguments.dataset)
    if len(cases) != 1:
        raise ValueError(f'single-case overfit requires exactly one case, found {len(cases)}')
    case = cases[0]
    scaling = _scaling(case)
    operatorCase = case.toOperatorCase(
        schema=thermalComponentFeatureSchema(
            boundaryTypeCount=1,
            temperatureBoundaryTypeIds=(),
        ),
        scaling=scaling,
    )
    temperatureCenter = float(
        np.average(
            case.temperature.temperatureK,
            weights=case.temperature.weightsM3,
        )
    )
    operatorCase = replace(
        operatorCase,
        queryTemperatures=(
            (case.temperature.temperatureK - temperatureCenter)
            / scaling.temperatureScaleK
        )[:, None],
    )
    batch = collateThermalOperatorCases([operatorCase])
    resolvedDevice = (
        arguments.device
        if not arguments.device.startswith('cuda') or torch.cuda.is_available()
        else 'cpu'
    )
    batch = _batchToDevice(batch, resolvedDevice)
    model = thermalQuerySetOperatorTorch(
        volumeFeatureDim=operatorCase.volumeFeatureDim,
        boundaryFeatureDim=operatorCase.boundaryFeatureDim,
        globalFeatureDim=operatorCase.globalFeatureDim,
        queryDim=operatorCase.queryDim,
        hiddenDim=arguments.hidden_dim,
        kernelScales=(arguments.kernel_scale,),
        queryFourierFrequencies=arguments.query_fourier_frequencies,
    ).to(resolvedDevice)
    lossFunction = thermalOperatorLoss(hotspotWeight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=arguments.learning_rate)

    model.eval()
    with torch.no_grad():
        initial = _metrics(model(batch), batch, scaling.temperatureScaleK)
    history = []
    start = time.perf_counter()
    model.train()
    for epoch in range(arguments.epochs):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch)
        loss = lossFunction.compute(
            prediction,
            batch['targets'],
            batch['queryWeights'],
            batch['queryMask'],
        )
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu()))
    elapsed = time.perf_counter() - start
    model.eval()
    with torch.no_grad():
        final = _metrics(model(batch), batch, scaling.temperatureScaleK)

    report = {
        'caseId': case.caseId,
        'device': resolvedDevice,
        'epochs': arguments.epochs,
        'elapsedSeconds': elapsed,
        'targetRepresentation': 'relativeWeightedMean',
        'absoluteTemperaturePredicted': False,
        'temperatureWeightedMeanKForEvaluationOnly': temperatureCenter,
        'temperatureScaleK': scaling.temperatureScaleK,
        'initial': initial,
        'final': final,
        'initialLoss': history[0],
        'finalLoss': history[-1],
    }
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
