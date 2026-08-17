#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from numbers import Integral
from pathlib import Path

import numpy as np
import torch

from mor.nn.data import (
    loadGeometryRecognitionCase,
    randomRotationMatrix,
    sampleGeometryPointCloud,
)
from mor.nn.models import geometryDgcnnClassifierTorch


SUPPORTED_FORMAT = 'geometry-dgcnn-classifier-v1'


def _loadModel(checkpointPath: Path, device: str):
    checkpoint = torch.load(checkpointPath, map_location='cpu', weights_only=True)
    if checkpoint.get('formatVersion') != SUPPORTED_FORMAT:
        raise ValueError(
            f"unsupported DGCNN checkpoint format: {checkpoint.get('formatVersion')!r}"
        )
    config = checkpoint['modelConfig']
    model = geometryDgcnnClassifierTorch(
        classCount=len(checkpoint['classNames']),
        inputFeatureDim=config['inputFeatureDim'],
        edgeChannels=config['edgeChannels'],
        globalFeatureDim=config['globalFeatureDim'],
        embeddingDim=config['embeddingDim'],
        neighborCount=config['neighborCount'],
        dropout=config.get('dropout', 0.0),
    )
    model.load_state_dict(checkpoint['stateDict'])
    return model.to(device).eval(), checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Predict componentType from geometry with a trained DGCNN.'
    )
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('case', type=Path)
    parser.add_argument(
        '--samples',
        type=int,
        help=(
            'Independent surface samples; defaults to the checkpoint calibration value.'
        ),
    )
    parser.add_argument('--rotations', type=int, default=1)
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument(
        '--unknown-threshold',
        type=float,
        help='Override the checkpoint unknown-rejection threshold.',
    )
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=23)
    arguments = parser.parse_args()
    for name, value in [('rotations', arguments.rotations), ('top-k', arguments.top_k)]:
        if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0:
            raise ValueError(f'{name} must be a positive integer')
    device = (
        arguments.device
        if not arguments.device.startswith('cuda') or torch.cuda.is_available()
        else 'cpu'
    )
    model, checkpoint = _loadModel(arguments.checkpoint, device)
    rejection = checkpoint.get('unknownRejection')
    sampleCount = arguments.samples
    if sampleCount is None:
        sampleCount = int(rejection.get('views', 1)) if rejection else 1
    if (
        not isinstance(sampleCount, Integral)
        or isinstance(sampleCount, bool)
        or sampleCount <= 0
    ):
        raise ValueError('samples must be a positive integer')
    case = loadGeometryRecognitionCase(arguments.case)
    pointCount = int(checkpoint['modelConfig']['pointCount'])
    rng = np.random.default_rng(arguments.seed + 1)
    views = []
    for sampleIndex in range(sampleCount):
        base = sampleGeometryPointCloud(
            case,
            pointCount,
            seed=arguments.seed + sampleIndex * 104729,
        )
        views.append(base)
        for _ in range(arguments.rotations - 1):
            rotation = randomRotationMatrix(rng)
            changed = base.copy()
            changed[:, :3] = changed[:, :3] @ rotation.T
            changed[:, 3:] = changed[:, 3:] @ rotation.T
            views.append(changed)
    inputs = torch.from_numpy(np.stack(views)).to(device)
    with torch.no_grad():
        embeddings = model.encode(inputs)
        logits = model.classifier(embeddings).mean(dim=0)
        probabilities = torch.softmax(logits, dim=0)
    classNames = tuple(checkpoint['classNames'])
    count = min(arguments.top_k, len(classNames))
    scores, indices = probabilities.topk(count)
    candidates = [
        {'componentType': classNames[int(index)], 'score': float(score)}
        for score, index in zip(scores.cpu(), indices.cpu())
    ]
    rejectionScore = None
    rejectionMethod = rejection.get('method') if rejection else None
    rejectionThreshold = arguments.unknown_threshold
    if rejectionThreshold is None and rejection is not None:
        rejectionThreshold = float(rejection['threshold'])
    if rejectionThreshold is not None:
        if rejectionMethod in ('softmax_max', 'softmax_max_multiview'):
            rejectionScore = float(probabilities.max().cpu())
        elif rejectionMethod in (None, 'prototype_cosine'):
            prototypes = checkpoint.get('prototypes')
            if prototypes is None:
                raise ValueError('checkpoint has no prototypes for unknown rejection')
            meanEmbedding = torch.nn.functional.normalize(
                embeddings.mean(dim=0), p=2, dim=0
            )
            prototypeTensor = prototypes.to(device=device, dtype=meanEmbedding.dtype)
            rejectionScore = float((prototypeTensor @ meanEmbedding).max().cpu())
            rejectionMethod = 'prototype_cosine'
        else:
            raise ValueError(
                f'unsupported unknown rejection method: {rejectionMethod!r}'
            )
    rejected = (
        rejectionThreshold is not None
        and rejectionScore is not None
        and rejectionScore < rejectionThreshold
    )
    print(
        json.dumps(
            {
                'predictedType': (
                    'unknown' if rejected else candidates[0]['componentType']
                ),
                'confidence': candidates[0]['score'],
                'confidenceCalibrated': False,
                'unknownSupported': rejectionThreshold is not None,
                'unknownRejected': rejected,
                'unknownScoreMethod': rejectionMethod,
                'unknownScore': rejectionScore,
                'unknownThreshold': rejectionThreshold,
                'surfaceSamples': int(sampleCount),
                'rotationsPerSample': arguments.rotations,
                'candidates': candidates,
                'geometryId': case.geometryId,
                'meshUuidUsedAsInput': False,
                'device': device,
                'modelVersion': checkpoint['formatVersion'],
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
