import torch
from typing import Any
from mor.nn.validation.baseValidation import baseValidation
from mor.nn.registry import nnRegistry

class dynamicsPredictionMetrics(baseValidation):
    def __init__(self, config: Any | None = None, **kwargs):
        super().__init__(validationName='dynamicsPredictionMetrics', **kwargs)
        self.config = config

    def evaluate(self, model: Any, dataModule: Any) -> dict:
        model.evalMode()
        parameter = next(model.parameters(), None)
        device = parameter.device if parameter is not None else torch.device('cpu')
        dtype = parameter.dtype if parameter is not None else torch.float32
        loader = dataModule.getValidationLoader() if hasattr(dataModule, 'getValidationLoader') else None
        batches = loader if loader is not None else [dataModule.getValidationData()]
        total = torch.zeros((), device=device, dtype=torch.float64)
        count = 0
        with torch.no_grad():
            for inputs, targets in batches:
                inputs = torch.as_tensor(inputs).to(device=device, dtype=dtype, non_blocking=True)
                targets = torch.as_tensor(targets).to(device=device, dtype=dtype, non_blocking=True)
                if targets.numel() == 0:
                    continue
                preds = model.forward(inputs)
                if isinstance(preds, (tuple, list)): preds = preds[0]
                if preds.shape != targets.shape:
                    raise ValueError('prediction and target shapes must match')
                total += (preds.double() - targets.double()).square().sum()
                count += targets.numel()
        if count == 0:
            raise ValueError('validation data is empty')
        return {'mse': float((total / count).item())}

nnRegistry.register('validation', 'dynamicsPredictionMetrics', dynamicsPredictionMetrics)
