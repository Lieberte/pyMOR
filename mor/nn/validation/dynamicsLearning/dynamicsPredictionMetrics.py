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
        inputs, targets = dataModule.getValidationData()
        with torch.no_grad():
            preds = model.forward(inputs)
            if isinstance(preds, tuple): preds = preds[0]
            mse = torch.mean((preds - targets) ** 2).item()
        return {'mse': float(mse)}

nnRegistry.register('validation.dynamicsLearning', 'dynamicsPredictionMetrics', dynamicsPredictionMetrics)
