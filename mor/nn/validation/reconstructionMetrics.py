import torch
from typing import Any
from mor.nn.validation.baseValidation import baseValidation
from mor.nn.registry import nnRegistry
from mor.runtime import resolveDeviceName

class reconstructionMetrics(baseValidation):
    def __init__(self, deviceName: str | None = None, lossFunction: Any | None = None, config: Any | None = None, **kwargs):
        super().__init__(validationName='reconstructionMetrics', **kwargs)
        self.config = config
        requestedDeviceName = deviceName if deviceName is not None else self._getConfigValue('runtime.deviceName', 'auto')
        self.deviceName = resolveDeviceName(requestedDeviceName)
        self.dtypeName = self._getConfigValue('runtime.dtypeName', 'float32')
        self.lossFunction = lossFunction
        self._explicitDevice = deviceName is not None

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
        parameter = next(model.parameters(), None)
        deviceName = self.deviceName if self._explicitDevice or parameter is None else str(parameter.device)
        if self.dtypeName == 'float64':
            resolvedDtype = torch.float64
        elif self.dtypeName == 'float16':
            resolvedDtype = torch.float16
        elif self.dtypeName == 'bfloat16':
            resolvedDtype = torch.bfloat16
        else:
            resolvedDtype = torch.float32
        accumulateOnDevice = torch.device(deviceName).type == 'cuda'
        totals = torch.zeros(3, dtype=torch.float64, device=deviceName) if accumulateOnDevice else [0.0, 0.0, 0.0]
        totalSampleCount = 0
        lossSampleCount = 0
        validationLoader = dataModule.getValidationLoader() if hasattr(dataModule, 'getValidationLoader') else None
        with torch.no_grad():
            if validationLoader is not None:
                for validationInputsBatch, validationTargetsBatch in validationLoader:
                    validationInputsBatch = validationInputsBatch.to(device=deviceName, dtype=resolvedDtype, non_blocking=True)
                    validationTargetsBatch = validationTargetsBatch.to(device=deviceName, dtype=resolvedDtype, non_blocking=True)
                    if validationTargetsBatch.numel() == 0:
                        continue
                    outputs = self._diagnosticOutputs(model.forward(validationInputsBatch))
                    predictionsBatch = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                    validationTargetsBatch = validationTargetsBatch.to(torch.float64)
                    diffBatch = predictionsBatch - validationTargetsBatch
                    squaredError = torch.sum(diffBatch ** 2)
                    squaredNorm = torch.sum(validationTargetsBatch ** 2)
                    totals[0] += squaredError.to(torch.float64) if accumulateOnDevice else float(squaredError.item())
                    totals[1] += squaredNorm.to(torch.float64) if accumulateOnDevice else float(squaredNorm.item())
                    totalSampleCount += int(validationTargetsBatch.numel())
                    if self.lossFunction is not None:
                        batchSize = len(validationTargetsBatch)
                        loss = self.lossFunction.compute(outputs, validationTargetsBatch)
                        totals[2] += (loss.to(torch.float64) if accumulateOnDevice else float(loss.item())) * batchSize
                        lossSampleCount += batchSize
            else:
                validationInputs, validationTargets = dataModule.getValidationData()
                inputsTensor = torch.as_tensor(validationInputs, dtype=resolvedDtype, device=deviceName)
                targetsTensor = torch.as_tensor(validationTargets, dtype=resolvedDtype, device=deviceName)
                if targetsTensor.numel() == 0:
                    raise ValueError('validation data is empty')
                outputs = self._diagnosticOutputs(model.forward(inputsTensor))
                predictions = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                targetsTensor = targetsTensor.to(torch.float64)
                diff = predictions - targetsTensor
                squaredError = torch.sum(diff ** 2)
                squaredNorm = torch.sum(targetsTensor ** 2)
                totals[0] = squaredError if accumulateOnDevice else float(squaredError.item())
                totals[1] = squaredNorm if accumulateOnDevice else float(squaredNorm.item())
                totalSampleCount = int(targetsTensor.numel())
                if self.lossFunction is not None:
                    loss = self.lossFunction.compute(outputs, targetsTensor)
                    totals[2] = loss if accumulateOnDevice else float(loss.item())
                    lossSampleCount = 1
        if totalSampleCount == 0:
            raise ValueError('validation data is empty')
        totalSquaredError, totalTargetSquaredNorm, totalLoss = totals.tolist() if accumulateOnDevice else totals
        mse = totalSquaredError / max(totalSampleCount, 1)
        relativeError = (totalSquaredError ** 0.5) / max(totalTargetSquaredNorm ** 0.5, 1e-12)
        result = {'mse': mse, 'relativeError': relativeError}
        if lossSampleCount:
            result['validationLoss'] = totalLoss / lossSampleCount
        return result

    @staticmethod
    def _diagnosticOutputs(outputs):
        if isinstance(outputs, (tuple, list)):
            return type(outputs)(reconstructionMetrics._diagnosticOutputs(value) for value in outputs)
        return outputs.to(torch.float64) if isinstance(outputs, torch.Tensor) else outputs

nnRegistry.register('validation', 'reconstructionMetrics', reconstructionMetrics)
