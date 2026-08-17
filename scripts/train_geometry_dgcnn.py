#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from mor.nn.data import (
    buildGeometryPointCloudSamples,
    geometryPointCloudDataset,
    loadGeometryRecognitionDatasetRecords,
    stratifiedGeometryPointCloudSplit,
)
from mor.nn.models import geometryDgcnnClassifierTorch


FORMAT_VERSION = 'geometry-dgcnn-classifier-v1'


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train a geometry-only DGCNN component classifier.'
    )
    parser.add_argument('dataset', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--point-count', type=int, default=512)
    parser.add_argument('--cache-point-count', type=int, default=2048)
    parser.add_argument('--repeat-factor', type=int, default=2)
    parser.add_argument('--neighbor-count', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--minimum-class-count', type=int, default=2)
    parser.add_argument('--validation-per-class', type=int, default=1)
    parser.add_argument('--evaluation-rotations', type=int, default=4)
    parser.add_argument('--open-set-views', type=int, default=4)
    parser.add_argument('--unknown-exposure-weight', type=float, default=0.0)
    parser.add_argument('--target-known-acceptance', type=float, default=0.875)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=17)
    return parser.parse_args()


def _setSeed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolveDevice(requested: str) -> str:
    if requested.startswith('cuda') and not torch.cuda.is_available():
        return 'cpu'
    return requested


def _metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    classCount = logits.shape[1]
    topK = min(3, classCount)
    ranked = logits.topk(topK, dim=1).indices
    return {
        'loss': float(F.cross_entropy(logits, targets).detach().cpu()),
        'top1': float((ranked[:, 0] == targets).float().mean().cpu()),
        'top3': float((ranked == targets[:, None]).any(dim=1).float().mean().cpu()),
    }


def _collectOutputs(model, loader, device: str):
    model.eval()
    logits = []
    embeddings = []
    targets = []
    meshUuids = []
    with torch.no_grad():
        for pointClouds, labels, meshes in loader:
            pointClouds = pointClouds.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            embedding = model.encode(pointClouds)
            logits.append(model.classifier(embedding).cpu())
            embeddings.append(embedding.cpu())
            targets.append(labels.cpu())
            meshUuids.extend(meshes)
    logitsTensor = torch.cat(logits)
    embeddingsTensor = torch.cat(embeddings)
    targetsTensor = torch.cat(targets)
    return logitsTensor, embeddingsTensor, targetsTensor, tuple(meshUuids)


def _evaluate(model, loader, device: str):
    logits, embeddings, targets, meshUuids = _collectOutputs(model, loader, device)
    return _metrics(logits, targets), logits, embeddings, targets, meshUuids


def _knownAcceptanceThreshold(
    knownScores: torch.Tensor, targetAcceptance: float
) -> float:
    """Calibrate rejection using known data only, without peeking at unknowns."""
    if knownScores.ndim != 1 or knownScores.numel() == 0:
        raise ValueError('known calibration scores must be a non-empty vector')
    if not 0.0 < targetAcceptance <= 1.0:
        raise ValueError('target known acceptance must be in (0, 1]')
    acceptedCount = int(np.ceil(targetAcceptance * knownScores.numel()))
    descending = torch.sort(knownScores.detach().cpu(), descending=True).values
    return float(descending[acceptedCount - 1])


def _openSetMetrics(
    knownLogits: torch.Tensor,
    knownTargets: torch.Tensor,
    knownScores: torch.Tensor,
    unknownScores: torch.Tensor,
    threshold: float,
) -> dict[str, float | int]:
    knownCorrect = knownLogits.argmax(dim=1) == knownTargets
    knownAccepted = knownScores >= threshold
    unknownRejected = unknownScores < threshold
    recognized = knownCorrect & knownAccepted
    totalCorrect = int(recognized.sum() + unknownRejected.sum())
    return {
        'threshold': float(threshold),
        'knownRecognitionRate': float(recognized.float().mean()),
        'knownAcceptanceRate': float(knownAccepted.float().mean()),
        'unknownRejectionRate': float(unknownRejected.float().mean()),
        'balancedOpenSetAccuracy': 0.5
        * (
            float(recognized.float().mean())
            + float(unknownRejected.float().mean())
        ),
        'overallOpenSetAccuracy': totalCorrect
        / float(knownScores.numel() + unknownScores.numel()),
        'knownCount': int(knownScores.numel()),
        'unknownCount': int(unknownScores.numel()),
    }


def _splitKnownHoldout(samples):
    grouped = {}
    for sample in samples:
        grouped.setdefault(sample.componentType, []).append(sample)
    calibration = []
    evaluation = []
    for label in sorted(grouped):
        items = grouped[label]
        calibration.append(items[0])
        evaluation.extend(items[1:])
    if not evaluation:
        evaluation = list(calibration)
    return tuple(calibration), tuple(evaluation)


def _splitUnknownExposure(samples):
    grouped = {}
    for sample in samples:
        grouped.setdefault(sample.componentType, []).append(sample)
    exposure = []
    evaluation = []
    for label in sorted(grouped):
        items = grouped[label]
        evaluation.append(items[0])
        exposure.extend(items[1:])
    return tuple(exposure), tuple(evaluation)


def _aggregateOutputs(logits, embeddings, targets, meshUuids):
    orderedMeshes = tuple(dict.fromkeys(meshUuids))
    groupedLogits = []
    groupedEmbeddings = []
    groupedTargets = []
    for meshUuid in orderedMeshes:
        indices = [index for index, value in enumerate(meshUuids) if value == meshUuid]
        selected = torch.tensor(indices, dtype=torch.long)
        groupedLogits.append(logits[selected].mean(dim=0))
        groupedEmbeddings.append(
            F.normalize(embeddings[selected].mean(dim=0), p=2, dim=0)
        )
        groupedTargets.append(targets[indices[0]])
    return (
        torch.stack(groupedLogits),
        torch.stack(groupedEmbeddings),
        torch.stack(groupedTargets),
        orderedMeshes,
    )


def _prototypeMetrics(
    trainEmbeddings: torch.Tensor,
    trainTargets: torch.Tensor,
    validationEmbeddings: torch.Tensor,
    validationTargets: torch.Tensor,
    classCount: int,
):
    prototypes = []
    for classIndex in range(classCount):
        prototype = trainEmbeddings[trainTargets == classIndex].mean(dim=0)
        prototypes.append(F.normalize(prototype, dim=0))
    prototypeTensor = torch.stack(prototypes)
    similarities = validationEmbeddings @ prototypeTensor.T
    metrics = _metrics(similarities, validationTargets)
    return metrics, prototypeTensor


def _classCounts(samples) -> dict[str, int]:
    return dict(sorted(Counter(sample.componentType for sample in samples).items()))


def main() -> None:
    arguments = _arguments()
    if arguments.cache_point_count < arguments.point_count:
        raise ValueError('cache-point-count must be at least point-count')
    if (
        min(arguments.epochs, arguments.evaluation_rotations, arguments.open_set_views)
        <= 0
    ):
        raise ValueError('epochs and evaluation view counts must be positive')
    if arguments.unknown_exposure_weight < 0.0:
        raise ValueError('unknown-exposure-weight cannot be negative')
    if not 0.0 < arguments.target_known_acceptance <= 1.0:
        raise ValueError('target-known-acceptance must be in (0, 1]')
    _setSeed(arguments.seed)
    device = _resolveDevice(arguments.device)
    records = loadGeometryRecognitionDatasetRecords(
        arguments.dataset, deduplicateByMeshUuid=True
    )
    preparationStart = time.perf_counter()
    samples = buildGeometryPointCloudSamples(
        records,
        pointCount=arguments.cache_point_count,
        seed=arguments.seed,
    )
    trainSamples, validationSamples, excludedSamples = (
        stratifiedGeometryPointCloudSplit(
            samples,
            minimumClassCount=arguments.minimum_class_count,
            validationPerClass=arguments.validation_per_class,
            seed=arguments.seed,
        )
    )
    knownCalibrationSamples, knownEvaluationSamples = _splitKnownHoldout(
        validationSamples
    )
    unknownExposureSamples, unknownEvaluationSamples = _splitUnknownExposure(
        excludedSamples
    )
    if arguments.unknown_exposure_weight > 0.0 and not unknownExposureSamples:
        raise ValueError(
            'unknown exposure requires at least one rare class with multiple meshes'
        )
    classNames = tuple(sorted({sample.componentType for sample in trainSamples}))
    preparationSeconds = time.perf_counter() - preparationStart
    trainDataset = geometryPointCloudDataset(
        trainSamples,
        classNames,
        pointCount=arguments.point_count,
        repeatFactor=arguments.repeat_factor,
        augment=True,
        seed=arguments.seed + 1,
    )
    validationDataset = geometryPointCloudDataset(
        knownCalibrationSamples,
        classNames,
        pointCount=arguments.point_count,
        augment=False,
        seed=arguments.seed + 2,
    )
    rotatedValidationDataset = geometryPointCloudDataset(
        knownEvaluationSamples,
        classNames,
        pointCount=arguments.point_count,
        repeatFactor=arguments.evaluation_rotations,
        augment=True,
        jitterStandardDeviation=0.0,
        seed=arguments.seed + 3,
    )
    sampler = WeightedRandomSampler(
        trainDataset.sampleWeights(),
        num_samples=len(trainDataset),
        replacement=True,
        generator=torch.Generator().manual_seed(arguments.seed),
    )
    loaderArguments = {
        'batch_size': arguments.batch_size,
        'num_workers': 0,
        'pin_memory': device.startswith('cuda'),
    }
    trainLoader = DataLoader(trainDataset, sampler=sampler, **loaderArguments)
    trainEvaluationLoader = DataLoader(
        geometryPointCloudDataset(
            trainSamples,
            classNames,
            pointCount=arguments.point_count,
            augment=False,
            seed=arguments.seed + 4,
        ),
        shuffle=False,
        **loaderArguments,
    )
    validationLoader = DataLoader(validationDataset, shuffle=False, **loaderArguments)
    rotatedValidationLoader = DataLoader(
        rotatedValidationDataset, shuffle=False, **loaderArguments
    )
    knownOpenSetLoader = DataLoader(
        geometryPointCloudDataset(
            knownEvaluationSamples,
            classNames,
            pointCount=arguments.point_count,
            repeatFactor=arguments.open_set_views,
            augment=True,
            rotationAugmentation=False,
            jitterStandardDeviation=0.0,
            seed=arguments.seed + 5,
        ),
        shuffle=False,
        **loaderArguments,
    )
    calibrationOpenSetLoader = DataLoader(
        geometryPointCloudDataset(
            knownCalibrationSamples,
            classNames,
            pointCount=arguments.point_count,
            repeatFactor=arguments.open_set_views,
            augment=True,
            rotationAugmentation=False,
            jitterStandardDeviation=0.0,
            seed=arguments.seed + 6,
        ),
        shuffle=False,
        **loaderArguments,
    )
    unknownEvaluationLoader = None
    if unknownEvaluationSamples:
        unknownEvaluationLoader = DataLoader(
            geometryPointCloudDataset(
                unknownEvaluationSamples,
                tuple(
                    sorted(
                        {sample.componentType for sample in unknownEvaluationSamples}
                    )
                ),
                pointCount=arguments.point_count,
                repeatFactor=arguments.open_set_views,
                augment=True,
                rotationAugmentation=False,
                jitterStandardDeviation=0.0,
                seed=arguments.seed + 7,
            ),
            shuffle=False,
            **loaderArguments,
        )
    unknownExposureLoader = None
    if arguments.unknown_exposure_weight > 0.0:
        exposureDataset = geometryPointCloudDataset(
            unknownExposureSamples,
            tuple(sorted({sample.componentType for sample in unknownExposureSamples})),
            pointCount=arguments.point_count,
            repeatFactor=max(arguments.repeat_factor, 2),
            augment=True,
            seed=arguments.seed + 8,
        )
        unknownExposureLoader = DataLoader(
            exposureDataset,
            batch_size=arguments.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=device.startswith('cuda'),
        )
    model = geometryDgcnnClassifierTorch(
        classCount=len(classNames),
        edgeChannels=(32, 32, 64, 128),
        globalFeatureDim=256,
        embeddingDim=128,
        neighborCount=arguments.neighbor_count,
        dropout=0.25,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=arguments.epochs, eta_min=arguments.learning_rate * 0.05
    )
    bestState = None
    bestValidationLoss = float('inf')
    history = []
    trainingStart = time.perf_counter()
    for epoch in range(1, arguments.epochs + 1):
        model.train()
        totalLoss = 0.0
        totalExposureLoss = 0.0
        exposureExampleCount = 0
        correct = 0
        exampleCount = 0
        exposureIterator = (
            iter(unknownExposureLoader) if unknownExposureLoader else None
        )
        for pointClouds, labels, _ in trainLoader:
            pointClouds = pointClouds.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(pointClouds)
            loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
            if exposureIterator is not None:
                try:
                    unknownPointClouds, _, _ = next(exposureIterator)
                except StopIteration:
                    exposureIterator = iter(unknownExposureLoader)
                    unknownPointClouds, _, _ = next(exposureIterator)
                unknownPointClouds = unknownPointClouds.to(device, non_blocking=True)
                unknownLogits = model(unknownPointClouds)
                unknownLogProbabilities = F.log_softmax(unknownLogits, dim=1)
                unknownProbabilities = unknownLogProbabilities.exp()
                exposureLoss = (
                    unknownProbabilities
                    * (
                        unknownLogProbabilities
                        + np.log(float(len(classNames)))
                    )
                ).sum(dim=1).mean()
                loss = loss + arguments.unknown_exposure_weight * exposureLoss
                totalExposureLoss += (
                    float(exposureLoss.detach().cpu()) * unknownPointClouds.shape[0]
                )
                exposureExampleCount += unknownPointClouds.shape[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            totalLoss += float(loss.detach().cpu()) * labels.shape[0]
            correct += int((logits.argmax(dim=1) == labels).sum().detach().cpu())
            exampleCount += labels.shape[0]
        scheduler.step()
        validationMetrics, *_ = _evaluate(model, validationLoader, device)
        entry = {
            'epoch': epoch,
            'trainLoss': totalLoss / exampleCount,
            'trainTop1': correct / exampleCount,
            'unknownExposureLoss': (
                totalExposureLoss / exposureExampleCount
                if exposureExampleCount
                else None
            ),
            'validationLoss': validationMetrics['loss'],
            'validationTop1': validationMetrics['top1'],
            'learningRate': float(optimizer.param_groups[0]['lr']),
        }
        history.append(entry)
        if validationMetrics['loss'] < bestValidationLoss:
            bestValidationLoss = validationMetrics['loss']
            bestState = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        if epoch == 1 or epoch % 5 == 0 or epoch == arguments.epochs:
            print(json.dumps(entry), flush=True)
    trainingSeconds = time.perf_counter() - trainingStart
    model.load_state_dict(bestState)
    model.to(device)
    trainMetrics, _, trainEmbeddings, trainTargets, _ = _evaluate(
        model, trainEvaluationLoader, device
    )
    validationMetrics, _, validationEmbeddings, validationTargets, _ = (
        _evaluate(model, validationLoader, device)
    )
    evaluationMetrics, *_ = _evaluate(
        model,
        DataLoader(
            geometryPointCloudDataset(
                knownEvaluationSamples,
                classNames,
                pointCount=arguments.point_count,
                augment=False,
                seed=arguments.seed + 9,
            ),
            shuffle=False,
            **loaderArguments,
        ),
        device,
    )
    rotatedMetrics, _, rotatedEmbeddings, rotatedTargets, _ = _evaluate(
        model, rotatedValidationLoader, device
    )
    prototypeMetrics, prototypeTensor = _prototypeMetrics(
        trainEmbeddings,
        trainTargets,
        validationEmbeddings,
        validationTargets,
        len(classNames),
    )
    rotatedPrototypeMetrics, _ = _prototypeMetrics(
        trainEmbeddings,
        trainTargets,
        rotatedEmbeddings,
        rotatedTargets,
        len(classNames),
    )
    openSet = None
    unknownRejection = None
    if unknownEvaluationLoader is not None:
        independentKnownEvaluation = {
            sample.meshUuid for sample in knownCalibrationSamples
        }.isdisjoint({sample.meshUuid for sample in knownEvaluationSamples})
        calibrationOutputs = _aggregateOutputs(
            *_collectOutputs(model, calibrationOpenSetLoader, device)
        )
        evaluationOutputs = _aggregateOutputs(
            *_collectOutputs(model, knownOpenSetLoader, device)
        )
        unknownOutputs = _aggregateOutputs(
            *_collectOutputs(model, unknownEvaluationLoader, device)
        )
        calibrationScores = F.softmax(calibrationOutputs[0], dim=1).max(dim=1).values
        knownScores = F.softmax(evaluationOutputs[0], dim=1).max(dim=1).values
        unknownScores = F.softmax(unknownOutputs[0], dim=1).max(dim=1).values
        threshold = _knownAcceptanceThreshold(
            calibrationScores, arguments.target_known_acceptance
        )
        openSet = _openSetMetrics(
            evaluationOutputs[0],
            evaluationOutputs[2],
            knownScores,
            unknownScores,
            threshold,
        )
        openSet.update(
            {
                'method': 'softmax_max_multiview',
                'calibration': (
                    'known-only-independent-threshold'
                    if independentKnownEvaluation
                    else 'known-only-threshold-shared-evaluation'
                ),
                'targetKnownAcceptance': arguments.target_known_acceptance,
                'views': arguments.open_set_views,
                'knownCalibrationCount': len(knownCalibrationSamples),
                'unknownExposureCount': len(unknownExposureSamples),
                'rareEvaluationClassCounts': _classCounts(unknownEvaluationSamples),
                'calibrationScoreRange': [
                    float(calibrationScores.min()),
                    float(calibrationScores.max()),
                ],
                'knownScoreRange': [
                    float(knownScores.min()),
                    float(knownScores.max()),
                ],
                'unknownScoreRange': [
                    float(unknownScores.min()),
                    float(unknownScores.max()),
                ],
            }
        )
        unknownRejection = {
            'method': openSet['method'],
            'threshold': openSet['threshold'],
            'label': 'unknown',
            'calibration': openSet['calibration'],
            'views': arguments.open_set_views,
        }
    checkpoint = {
        'formatVersion': FORMAT_VERSION,
        'classNames': list(classNames),
        'modelConfig': {
            **model.encoder.configDict(),
            'classCount': len(classNames),
            'pointCount': arguments.point_count,
            'cachePointCount': arguments.cache_point_count,
            'samplingSeed': arguments.seed,
            'normalFeatures': True,
        },
        'stateDict': {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        'prototypes': prototypeTensor,
        'unknownRejection': unknownRejection,
        'split': {
            'trainMeshUuids': [sample.meshUuid for sample in trainSamples],
            'validationMeshUuids': [sample.meshUuid for sample in validationSamples],
            'excludedMeshUuids': [sample.meshUuid for sample in excludedSamples],
            'knownCalibrationMeshUuids': [
                sample.meshUuid for sample in knownCalibrationSamples
            ],
            'knownEvaluationMeshUuids': [
                sample.meshUuid for sample in knownEvaluationSamples
            ],
            'unknownExposureMeshUuids': [
                sample.meshUuid for sample in unknownExposureSamples
            ],
            'unknownEvaluationMeshUuids': [
                sample.meshUuid for sample in unknownEvaluationSamples
            ],
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, arguments.output)
    report = {
        'formatVersion': FORMAT_VERSION,
        'model': str(arguments.output.resolve()),
        'device': device,
        'seed': arguments.seed,
        'independentGeometries': len(samples),
        'trainGeometries': len(trainSamples),
        'validationGeometries': len(validationSamples),
        'excludedGeometries': len(excludedSamples),
        'classNames': list(classNames),
        'trainClassCounts': _classCounts(trainSamples),
        'validationClassCounts': _classCounts(validationSamples),
        'excludedClassCounts': _classCounts(excludedSamples),
        'knownCalibrationClassCounts': _classCounts(knownCalibrationSamples),
        'knownEvaluationClassCounts': _classCounts(knownEvaluationSamples),
        'unknownExposureClassCounts': _classCounts(unknownExposureSamples),
        'unknownEvaluationClassCounts': _classCounts(unknownEvaluationSamples),
        'configuration': {
            'epochs': arguments.epochs,
            'batchSize': arguments.batch_size,
            'pointCount': arguments.point_count,
            'cachePointCount': arguments.cache_point_count,
            'repeatFactor': arguments.repeat_factor,
            'neighborCount': arguments.neighbor_count,
            'embeddingDim': 128,
            'evaluationRotations': arguments.evaluation_rotations,
            'openSetViews': arguments.open_set_views,
            'unknownExposureWeight': arguments.unknown_exposure_weight,
            'targetKnownAcceptance': arguments.target_known_acceptance,
        },
        'timing': {
            'preparationSeconds': preparationSeconds,
            'trainingSeconds': trainingSeconds,
        },
        'bestValidationLoss': bestValidationLoss,
        'train': trainMetrics,
        'validationClassifier': validationMetrics,
        'evaluationClassifier': evaluationMetrics,
        'validationPrototype': prototypeMetrics,
        'rotatedValidationClassifier': rotatedMetrics,
        'rotatedValidationPrototype': rotatedPrototypeMetrics,
        'openSet': openSet,
        'history': history,
    }
    reportPath = arguments.report or arguments.output.with_suffix('.json')
    reportPath.parent.mkdir(parents=True, exist_ok=True)
    reportPath.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != 'history'},
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
