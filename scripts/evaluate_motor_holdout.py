"""Fixed-budget training on one solved motor case and independent case evaluation."""
from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from mor.nn.data import loadThermalComponentDataset
from mor.nn.data.thermalComponentCase import thermalComponentFeatureSchema, thermalOperatorFeatureScaling
from mor.nn.data.thermalOperatorDataModule import collateThermalOperatorCases
from mor.nn.models.operator import (thermalQuerySetOperatorTorch, thermalSetAttentionOperatorTorch,
                                    thermalSetDeepONetTorch)


def prepareCases(path, schema, scaling):
    cases = loadThermalComponentDataset(path)
    selected = [case for case in cases if case.metadata.get('absoluteTemperatureAnchorAvailable')]
    if not selected:
        raise ValueError(f'no cases with an absolute temperature boundary anchor in {path}')
    return cases, [case.toOperatorCase(schema=schema, scaling=scaling) for case in selected]


def modelFactory(variant, case):
    dims = dict(volumeFeatureDim=case.volumeFeatureDim, boundaryFeatureDim=case.boundaryFeatureDim,
                globalFeatureDim=case.globalFeatureDim)
    if variant == 'query':
        return thermalQuerySetOperatorTorch(**dims, hiddenDim=16, kernelScales=(0.3,))
    if variant == 'attention':
        return thermalSetAttentionOperatorTorch(**dims, hiddenDim=16, latentCount=4, attentionHeads=2)
    return thermalSetDeepONetTorch(**dims, pointHiddenDim=16, globalHiddenDim=8, latentDim=16)


def batchOnDevice(cases, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in collateThermalOperatorCases(cases).items()}


def inputBatch(batch):
    # Query temperatures are scoring/training targets, never model inputs.
    return {key: value for key, value in batch.items() if key != 'targets'}


def score(model, batch, scale):
    with torch.no_grad():
        predictions = model(inputBatch(batch))
    rows = []
    for index, caseId in enumerate(batch['caseIds']):
        weights = batch['queryWeights'][index].double() * batch['queryMask'][index]
        difference = (predictions[index, :, 0].double() - batch['targets'][index, :, 0].double()) * scale
        rmse = float(torch.sqrt((difference.square() * weights).sum() / weights.sum()))
        assert np.isfinite(rmse)
        rows.append({'caseId': caseId, 'weightedRmseK': rmse,
                     'maxAbsoluteErrorK': float(difference[batch['queryMask'][index]].abs().max())})
    return rows, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--holdout', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('reports/motor_holdout'))
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('epochs must be positive')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable')
    torch.set_num_threads(1)
    schema = thermalComponentFeatureSchema(boundaryTypeCount=4, temperatureBoundaryTypeIds=(1, 2),
                                            fixedTemperatureBoundaryTypeIds=(2,))
    scaling = thermalOperatorFeatureScaling()
    rawTrain, train = prepareCases(args.train, schema, scaling)
    rawHoldout, holdout = prepareCases(args.holdout, schema, scaling)
    trainSources = {case.metadata.get('sourceSta') for case in rawTrain}
    holdoutSources = {case.metadata.get('sourceSta') for case in rawHoldout}
    trainParents = {case.metadata.get('parentCaseId') for case in rawTrain}
    holdoutParents = {case.metadata.get('parentCaseId') for case in rawHoldout}
    if (None in trainSources | holdoutSources | trainParents | holdoutParents
            or trainSources & holdoutSources or trainParents & holdoutParents):
        raise ValueError('train and holdout must have distinct source STA files and parent case IDs')
    if {case.caseId for case in train} & {case.caseId for case in holdout}:
        raise ValueError('train and holdout case IDs overlap')
    args.output.mkdir(parents=True, exist_ok=True)
    trainBatch, holdoutBatch = batchOnDevice(train, args.device), batchOnDevice(holdout, args.device)
    rows = []
    for variant in ('query', 'attention', 'deeponet'):
        torch.manual_seed(7)
        model = modelFactory(variant, train[0]).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        if args.device == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        losses = []
        model.train()
        for _ in range(args.epochs):
            optimizer.zero_grad(set_to_none=True)
            predictions = model(inputBatch(trainBatch))
            weights = trainBatch['queryWeights'] * trainBatch['queryMask']
            lossPerCase = ((predictions[..., 0] - trainBatch['targets'][..., 0]).square() * weights).sum(1) / weights.sum(1)
            loss = lossPerCase.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f'{variant} produced nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        trainScores, _ = score(model, trainBatch, scaling.temperatureScaleK)
        holdoutScores, expected = score(model, holdoutBatch, scaling.temperatureScaleK)
        checkpoint = args.output / f'{variant}.pt'
        torch.save(model.saveState(toCpu=True), checkpoint)
        restored = modelFactory(variant, train[0]).to(args.device)
        restored.loadState(torch.load(checkpoint, map_location=args.device, weights_only=True))
        restored.eval()
        _, actual = score(restored, holdoutBatch, scaling.temperatureScaleK)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        memory = None
        if args.device == 'cuda':
            torch.cuda.synchronize()
            memory = {'peakAllocatedBytes': torch.cuda.max_memory_allocated(),
                      'peakReservedBytes': torch.cuda.max_memory_reserved()}
        rows.append({'variant': variant, 'epochs': args.epochs, 'seed': 7, 'elapsedMs': (time.perf_counter() - start)*1000,
                     'lossHistory': losses, 'train': trainScores, 'holdout': holdoutScores,
                     'roundtripMaxDifference': float((actual - expected).abs().max()), 'cudaMemory': memory})
        print(variant, 'mean holdout RMSE K', np.mean([row['weightedRmseK'] for row in holdoutScores]), flush=True)
    payload = {'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'device': args.device},
               'trainSources': sorted(trainSources), 'holdoutSources': sorted(holdoutSources),
               'trainCaseIds': [case.caseId for case in train], 'holdoutCaseIds': [case.caseId for case in holdout],
               'selection': 'Only components with an absolute temperature anchor; fixed epochs; holdout never used for optimization or model selection.',
               'boundaryPolicy': 'Exported FEM result heat flux is used for result-derived Neumann features. This is conditional field evaluation, not an unsolved whole-motor prediction.',
               'excludedTrainCaseIds': [case.caseId for case in rawTrain if not case.metadata.get('absoluteTemperatureAnchorAvailable')],
               'excludedHoldoutCaseIds': [case.caseId for case in rawHoldout if not case.metadata.get('absoluteTemperatureAnchorAvailable')],
               'cases': rows}
    (args.output / 'motor_holdout.json').write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    with (args.output / 'motor_holdout.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['variant', 'caseId', 'weightedRmseK', 'maxAbsoluteErrorK'])
        writer.writeheader()
        writer.writerows({'variant': row['variant'], **scoreRow} for row in rows for scoreRow in row['holdout'])


if __name__ == '__main__':
    main()
