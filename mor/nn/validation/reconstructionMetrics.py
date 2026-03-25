import torch
from typing import Any
from mor.nn.validation.baseValidation import baseValidation
from mor.nn.registry import nnRegistry

class reconstructionMetrics(baseValidation):
    def __init__(self, deviceName: str | None = None, lossFunction: Any | None = None, config: Any | None = None, **kwargs):
        super().__init__(validationName='reconstructionMetrics', **kwargs)
        self.config = config
        requestedDeviceName = deviceName if deviceName is not None else self._getConfigValue('runtime.deviceName', 'cpu')
        self.deviceName = 'cpu' if requestedDeviceName.startswith('cuda') and not torch.cuda.is_available() else requestedDeviceName
        self.dtypeName = self._getConfigValue('runtime.dtypeName', 'float32')
        self.lossFunction = lossFunction

    def _getConfigValue(self, dottedPath: str, defaultValue: Any) -> Any:
        if self.config is None:
            return defaultValue
        currentValue = self.config
        for key in dottedPath.split('.'):
            if not hasattr(currentValue, key):
                return defaultValue
            currentValue = getattr(currentValue, key)
        return currentValue

    def evaluate(self, model: Any, dataModule: Any) -> dict:
        model.evalMode()
        if self.dtypeName == 'float64':
            resolvedDtype = torch.float64
        elif self.dtypeName == 'float16':
            resolvedDtype = torch.float16
        elif self.dtypeName == 'bfloat16':
            resolvedDtype = torch.bfloat16
        else:
            resolvedDtype = torch.float32
        totalSquaredError = 0.0
        totalTargetSquaredNorm = 0.0
        totalSampleCount = 0
        validationLossValues = []
        validationLoader = dataModule.getValidationLoader() if hasattr(dataModule, 'getValidationLoader') else None
        with torch.no_grad():
            if validationLoader is not None:
                for validationInputsBatch, validationTargetsBatch in validationLoader:
                    validationInputsBatch = validationInputsBatch.to(self.deviceName)
                    validationTargetsBatch = validationTargetsBatch.to(self.deviceName)
                    validationInputsBatch = validationInputsBatch.to(dtype=resolvedDtype)
                    validationTargetsBatch = validationTargetsBatch.to(dtype=resolvedDtype)
                    predictionsBatch = model.forward(validationInputsBatch)
                    diffBatch = predictionsBatch - validationTargetsBatch
                    totalSquaredError += float(torch.sum(diffBatch ** 2).item())
                    totalTargetSquaredNorm += float(torch.sum(validationTargetsBatch ** 2).item())
                    totalSampleCount += int(validationTargetsBatch.numel())
                    if self.lossFunction is not None:
                        validationLossValues.append(float(self.lossFunction.compute(predictionsBatch, validationTargetsBatch).item()))
            else:
                validationInputs, validationTargets = dataModule.getValidationData()
                inputsTensor = torch.as_tensor(validationInputs, dtype=torch.float32, device=self.deviceName)
                targetsTensor = torch.as_tensor(validationTargets, dtype=torch.float32, device=self.deviceName)
                inputsTensor = inputsTensor.to(dtype=resolvedDtype)
                targetsTensor = targetsTensor.to(dtype=resolvedDtype)
                predictions = model.forward(inputsTensor)
                diff = predictions - targetsTensor
                totalSquaredError = float(torch.sum(diff ** 2).item())
                totalTargetSquaredNorm = float(torch.sum(targetsTensor ** 2).item())
                totalSampleCount = int(targetsTensor.numel())
                if self.lossFunction is not None:
                    validationLossValues.append(float(self.lossFunction.compute(predictions, targetsTensor).item()))
        mse = totalSquaredError / max(totalSampleCount, 1)
        relativeError = (totalSquaredError ** 0.5) / max(totalTargetSquaredNorm ** 0.5, 1e-12)
        result = {'mse': mse, 'relativeError': relativeError}
        if validationLossValues:
            result['validationLoss'] = sum(validationLossValues) / len(validationLossValues)
        return result

nnRegistry.register('validation', 'reconstructionMetrics', reconstructionMetrics)
