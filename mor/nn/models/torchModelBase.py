from typing import Any
import torch
from torch import nn
from .baseModel import baseModel

class torchModelBase(nn.Module, baseModel):
    def __init__(self, modelName: str = 'torchModelBase', **kwargs):
        nn.Module.__init__(self)
        baseModel.__init__(self, modelName=modelName, **kwargs)

    def _resolveDeviceName(self, deviceName: str, allowAutoFallback: bool = True) -> str:
        if deviceName.startswith('cuda') and not torch.cuda.is_available() and allowAutoFallback:
            return 'cpu'
        return deviceName

    def toDevice(self, deviceName: str, allowAutoFallback: bool = True) -> str:
        resolvedDeviceName = self._resolveDeviceName(deviceName, allowAutoFallback=allowAutoFallback)
        self.to(resolvedDeviceName)
        self.deviceName = resolvedDeviceName
        return self.deviceName

    def trainMode(self):
        self.train()
        self.trainingState['isTraining'] = True

    def evalMode(self):
        self.eval()
        self.trainingState['isTraining'] = False

    def _cloneStateObject(self, value: Any, toCpu: bool) -> Any:
        if isinstance(value, torch.Tensor):
            tensorValue = value.detach()
            if toCpu:
                tensorValue = tensorValue.cpu()
            return tensorValue.clone()
        if isinstance(value, dict):
            return {key: self._cloneStateObject(item, toCpu) for key, item in value.items()}
        if isinstance(value, list):
            return [self._cloneStateObject(item, toCpu) for item in value]
        if isinstance(value, tuple):
            return tuple(self._cloneStateObject(item, toCpu) for item in value)
        return value

    def saveState(self, toCpu: bool = True) -> dict:
        return self._cloneStateObject(self.state_dict(), toCpu=toCpu)

    def loadState(self, state: dict, strict: bool = True):
        self.load_state_dict(state, strict=strict)
