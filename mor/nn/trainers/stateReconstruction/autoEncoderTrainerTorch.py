import torch
from typing import Any
from mor.nn.trainers.baseTrainer import baseTrainer
from mor.nn.registry import nnRegistry

class autoEncoderTrainerTorch(baseTrainer):
    def __init__(self, lossFunction: Any, epochs: int = 100, learningRate: float = 1e-03, deviceName: str = 'cpu', **kwargs):
        super().__init__(trainerName='autoEncoderTrainerTorch', **kwargs)
        self.lossFunction = lossFunction
        self.epochs = epochs
        self.learningRate = learningRate
        self.deviceName = deviceName

    def fit(self, model: Any, dataModule: Any) -> dict:
        trainInputs, trainTargets = dataModule.getTrainData()
        validationInputs, validationTargets = dataModule.getValidationData()
        model.to(self.deviceName)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learningRate)
        trainInputsTensor = torch.as_tensor(trainInputs, dtype=torch.float32, device=self.deviceName)
        trainTargetsTensor = torch.as_tensor(trainTargets, dtype=torch.float32, device=self.deviceName)
        validationInputsTensor = torch.as_tensor(validationInputs, dtype=torch.float32, device=self.deviceName)
        validationTargetsTensor = torch.as_tensor(validationTargets, dtype=torch.float32, device=self.deviceName)
        trainLossHistory = []
        validationLossHistory = []
        for _ in range(self.epochs):
            model.trainMode()
            optimizer.zero_grad()
            trainPredictions = model.forward(trainInputsTensor)
            trainLoss = self.lossFunction.compute(trainPredictions, trainTargetsTensor)
            trainLoss.backward()
            optimizer.step()
            model.evalMode()
            with torch.no_grad():
                validationPredictions = model.forward(validationInputsTensor)
                validationLoss = self.lossFunction.compute(validationPredictions, validationTargetsTensor)
            trainLossHistory.append(float(trainLoss.item()))
            validationLossHistory.append(float(validationLoss.item()))
        return {
            'trainLossHistory': trainLossHistory,
            'validationLossHistory': validationLossHistory,
            'finalTrainLoss': trainLossHistory[-1] if trainLossHistory else None,
            'finalValidationLoss': validationLossHistory[-1] if validationLossHistory else None
        }

nnRegistry.register('trainers.stateReconstruction', 'autoEncoderTrainerTorch', autoEncoderTrainerTorch)
