from __future__ import annotations

from typing import Any

import torch

from mor.nn.registry import nnRegistry

from .baseTrainer import baseTrainer


def _batchToDevice(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _boundaryQueryBatch(batch: dict[str, Any]) -> dict[str, Any]:
    boundaryBatch = dict(batch)
    queryDim = int(batch['queryCoordinates'].shape[-1])
    boundaryBatch['queryCoordinates'] = batch['boundaryFeatures'][..., :queryDim]
    boundaryBatch['queryWeights'] = batch['boundaryWeights']
    boundaryBatch['queryMask'] = batch['boundaryMask']
    return boundaryBatch


class thermalOperatorTrainerTorch(baseTrainer):
    def __init__(
        self,
        lossFunction: Any,
        *,
        epochs: int = 100,
        learningRate: float = 1e-3,
        weightDecay: float = 0.0,
        gradientClip: float | None = 1.0,
        deviceName: str = 'cpu',
        physicsInValidation: bool = True,
        **kwargs,
    ):
        super().__init__(trainerName='thermalOperatorTrainerTorch', **kwargs)
        self.lossFunction = lossFunction
        self.epochs = int(epochs)
        self.learningRate = float(learningRate)
        self.weightDecay = float(weightDecay)
        self.gradientClip = None if gradientClip is None else float(gradientClip)
        self.deviceName = str(deviceName)
        self.physicsInValidation = bool(physicsInValidation)
        if self.epochs <= 0 or self.learningRate <= 0.0 or self.weightDecay < 0.0:
            raise ValueError('invalid thermal operator trainer options')

    def _epoch(self, model, loader, device, optimizer=None) -> float:
        model.trainMode() if optimizer is not None else model.evalMode()
        totalLoss = 0.0
        batchCount = 0
        for rawBatch in loader:
            batch = _batchToDevice(rawBatch, device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            if hasattr(self.lossFunction, 'computeBatch'):
                modelOutput = (
                    model.forwardWithAux(batch)
                    if hasattr(model, 'forwardWithAux')
                    else model(batch)
                )
                boundaryPredictions = None
                if getattr(
                    self.lossFunction, 'requiresBoundaryPredictions', False
                ):
                    boundaryPredictions = model(_boundaryQueryBatch(batch))
                if hasattr(self.lossFunction, 'computePinnBatch') and (
                    optimizer is not None or self.physicsInValidation
                ):
                    loss = self.lossFunction.computePinnBatch(
                        model,
                        modelOutput,
                        batch,
                        boundaryPredictions=boundaryPredictions,
                    )
                else:
                    loss = self.lossFunction.computeBatch(
                        modelOutput,
                        batch,
                        boundaryPredictions=boundaryPredictions,
                    )
            else:
                predictions = model(batch)
                loss = self.lossFunction.compute(
                    predictions,
                    batch['targets'],
                    batch['queryWeights'],
                    batch['queryMask'],
                )
            if not torch.isfinite(loss):
                caseIds = batch.get('caseIds', ())
                raise FloatingPointError(
                    f'non-finite thermal operator loss for cases: {caseIds}'
                )
            if optimizer is not None:
                loss.backward()
                if self.gradientClip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradientClip)
                optimizer.step()
            totalLoss += float(loss.detach().item())
            batchCount += 1
        return totalLoss / max(batchCount, 1)

    def fit(self, model, dataModule) -> dict[str, Any]:
        device = model.toDevice(self.deviceName, allowAutoFallback=True)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learningRate,
            weight_decay=self.weightDecay,
        )
        history = {'train': [], 'validation': []}
        bestLoss = None
        bestEpoch = -1
        bestState = None
        trainLoader = dataModule.getTrainLoader()
        validationLoader = dataModule.getValidationLoader()
        for epoch in range(self.epochs):
            trainLoss = self._epoch(model, trainLoader, device, optimizer)
            validationContext = (
                torch.enable_grad()
                if (
                    self.physicsInValidation
                    and getattr(
                        self.lossFunction,
                        'requiresCoordinateGradients',
                        False,
                    )
                )
                else torch.no_grad()
            )
            with validationContext:
                validationLoss = self._epoch(model, validationLoader, device)
            history['train'].append(trainLoss)
            history['validation'].append(validationLoss)
            if bestLoss is None or validationLoss < bestLoss:
                bestLoss = validationLoss
                bestEpoch = epoch
                bestState = model.saveState(toCpu=True)
        if bestState is not None:
            model.loadState(bestState)
        model.evalMode()
        return {
            'trainLossHistory': history['train'],
            'validationLossHistory': history['validation'],
            'bestValidationLoss': bestLoss,
            'bestEpochIndex': bestEpoch,
            'bestModelState': bestState,
            'resolvedDeviceName': device,
        }


nnRegistry.register('trainers', 'thermalOperatorTrainerTorch', thermalOperatorTrainerTorch)
