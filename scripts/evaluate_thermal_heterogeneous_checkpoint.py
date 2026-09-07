from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mor.nn.data import (
    collateThermalOperatorCases,
    loadSubsampledThermalDataset,
    thermalCheckpointDataContract,
)
from mor.nn.losses import thermalSteadyPinnLoss
from mor.nn.models import thermalCategorizedOperatorTorch, thermalCategoryKey

try:
    from scripts.train_thermal_discrete_pinn import _splitCases
except ModuleNotFoundError:
    from train_thermal_discrete_pinn import _splitCases


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=23)
    return parser.parse_args()


def _toDevice(batch, device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _records(model, cases, dimensionlessCases, device, *, routed):
    records = []
    model.to(device).eval()
    with torch.no_grad():
        for case, dimensionlessCase in zip(cases, dimensionlessCases):
            batch = collateThermalOperatorCases([dimensionlessCase.operatorCase])
            categoryKey = thermalCategoryKey(
                {'componentType': case.componentType}, ('componentType',)
            )
            if routed:
                batch['categoryKeys'] = [categoryKey]
            batch = _toDevice(batch, device)
            prediction = model(batch)[0, :, 0]
            target = batch['targets'][0, :, 0]
            mask = batch['queryMask'][0]
            weights = batch['queryWeights'][0, mask]
            if not dimensionlessCase.scales.absoluteTemperatureAnchorAvailable:
                total = weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
                prediction = prediction.clone()
                target = target.clone()
                prediction[mask] -= torch.sum(weights * prediction[mask]) / total
                target[mask] -= torch.sum(weights * target[mask]) / total
            error = (prediction[mask] - target[mask]) * float(
                dimensionlessCase.scales.temperatureScaleK
            )
            squared = float(torch.sum(weights * error.square()).cpu())
            weight = float(weights.sum().cpu())
            records.append(
                {
                    'caseId': case.caseId,
                    'componentType': case.componentType,
                    'categoryKey': categoryKey,
                    'rmseK': math.sqrt(squared / weight),
                    'maximumErrorK': float(torch.max(torch.abs(error)).cpu()),
                    '_squaredErrorIntegral': squared,
                    '_weightSum': weight,
                }
            )
    model.cpu()
    return records


def _summary(records):
    rmses = np.asarray([value['rmseK'] for value in records], dtype=np.float64)
    squared = sum(value['_squaredErrorIntegral'] for value in records)
    weight = sum(value['_weightSum'] for value in records)
    return {
        'caseCount': len(records),
        'volumeWeightedRmseK': math.sqrt(squared / weight),
        'macroCaseRmseK': float(np.mean(rmses)),
        'caseRmseP90K': float(np.percentile(rmses, 90)),
        'caseRmseP95K': float(np.percentile(rmses, 95)),
        'maximumErrorK': max(value['maximumErrorK'] for value in records),
    }


def _byComponent(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[record['componentType']].append(record)
    return {key: _summary(values) for key, values in sorted(grouped.items())}


def _pdeResidual(model, cases, dimensionlessCases, device, *, routed):
    loss = thermalSteadyPinnLoss(
        pdeResidualWeight=1.0,
        pdeResidualMode='meshfree',
        pdeCollocationCount=8,
        pdeKernelScale=0.2,
    )
    loader = DataLoader(
        tuple(value.operatorCase for value in dimensionlessCases),
        batch_size=8,
        shuffle=False,
        collate_fn=collateThermalOperatorCases,
    )
    componentByCaseId = {case.caseId: case.componentType for case in cases}
    model.to(device).eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for rawBatch in loader:
            if routed:
                rawBatch['categoryKeys'] = [
                    thermalCategoryKey(
                        {'componentType': componentByCaseId[caseId]},
                        ('componentType',),
                    )
                    for caseId in rawBatch['caseIds']
                ]
            batch = _toDevice(rawBatch, device)
            value = loss.computePdeResidual(model, batch)
            batchCount = len(rawBatch['caseIds'])
            total += float(value.cpu()) * batchCount
            count += batchCount
    model.cpu()
    return total / max(1, count)


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
    bundle = thermalCategorizedOperatorTorch.fromCheckpoint(checkpoint)
    dataContract = thermalCheckpointDataContract.fromCheckpoint(checkpoint)
    cases = dataContract.prepareCases(dataset.cases)
    _, validationCases = _splitCases(cases, 0.2, arguments.seed)
    validationDimensionless = dataContract.transformCases(validationCases)
    device = bundle.toDevice(arguments.device, allowAutoFallback=True)
    bundle.cpu()
    globalRecords = _records(
        bundle.globalModel,
        validationCases,
        validationDimensionless,
        device,
        routed=False,
    )
    routedRecords = _records(
        bundle,
        validationCases,
        validationDimensionless,
        device,
        routed=True,
    )
    routingCounts = Counter()
    for case in validationCases:
        _, routing = bundle.resolveModel({'componentType': case.componentType})
        alpha = routing['expertBlendWeight']
        routingCounts[
            'globalFallback'
            if alpha == 0.0
            else ('categoryExpert' if alpha == 1.0 else 'globalExpertBlend')
        ] += 1
    result = {
        'source': str(dataset.sourcePath),
        'checkpoint': str(arguments.checkpoint),
        'deserializedCheckpoint': True,
        'validationCaseCount': len(validationCases),
        'globalFallback': {
            'evaluation': _summary(globalRecords),
            'meshfreePdeResidual': _pdeResidual(
                bundle.globalModel,
                validationCases,
                validationDimensionless,
                device,
                routed=False,
            ),
            'byComponent': _byComponent(globalRecords),
        },
        'routedModel': {
            'evaluation': _summary(routedRecords),
            'meshfreePdeResidual': _pdeResidual(
                bundle,
                validationCases,
                validationDimensionless,
                device,
                routed=True,
            ),
            'byComponent': _byComponent(routedRecords),
            'routingCounts': dict(routingCounts),
            'cases': [
                {
                    key: value
                    for key, value in record.items()
                    if not key.startswith('_')
                }
                for record in routedRecords
            ],
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
