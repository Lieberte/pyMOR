import torch
from typing import Any
from mor.nn.validation.baseValidation import baseValidation
from mor.nn.registry import nnRegistry

class reconstructionMetrics(baseValidation):
    def __init__(self, deviceName: str = 'cpu', **kwargs):
        super().__init__(validationName='reconstructionMetrics', **kwargs)
        self.deviceName = deviceName

    def evaluate(self, model: Any, dataModule: Any) -> dict:
        validationInputs, validationTargets = dataModule.getValidationData()
        model.evalMode()
        with torch.no_grad():
            inputsTensor = torch.as_tensor(validationInputs, dtype=torch.float32, device=self.deviceName)
            targetsTensor = torch.as_tensor(validationTargets, dtype=torch.float32, device=self.deviceName)
            predictions = model.forward(inputsTensor)
            mse = torch.mean((predictions - targetsTensor) ** 2).item()
            denominator = torch.norm(targetsTensor).item()
            relativeError = torch.norm(predictions - targetsTensor).item() / max(denominator, 1e-12)
        return {'mse': mse, 'relativeError': relativeError}

nnRegistry.register('validation.stateReconstruction', 'reconstructionMetrics', reconstructionMetrics)
