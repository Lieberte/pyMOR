import math
from numbers import Integral
import platform

import torch
from torch import nn
import torch.nn.functional as F

from mor.nn.models.torchModelBase import torchModelBase
from mor.nn.registry import nnRegistry


def _positiveDimension(name, value):
    if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


class _ReferenceMamba(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_inner, self.d_state = expand * d_model, d_state
        self.dt_rank = math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)
        nn.init.uniform_(self.dt_proj.weight, -self.dt_rank ** -0.5, self.dt_rank ** -0.5)
        dt = torch.exp(torch.rand(self.d_inner) * math.log(100) + math.log(0.001))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.arange(1, d_state + 1, dtype=torch.float32).log().repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, inputs):
        x, gate = self.in_proj(inputs).chunk(2, dim=-1)
        x = F.silu(self.conv1d(x.transpose(1, 2))[..., :inputs.shape[1]].transpose(1, 2))
        delta, b, c = self.x_proj(x).split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        workDtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        a = -self.A_log.to(workDtype).exp()
        state = x.new_zeros((x.shape[0], self.d_inner, self.d_state), dtype=workDtype)
        output = []
        for step in range(inputs.shape[1]):
            dt = delta[:, step].to(workDtype).unsqueeze(-1)
            values = x[:, step].to(workDtype)
            state = (torch.exp(dt * a) * state
                     + dt * b[:, step].to(workDtype).unsqueeze(1) * values.unsqueeze(-1))
            output.append((state * c[:, step].to(workDtype).unsqueeze(1)).sum(-1) + self.D * values)
        result = torch.stack(output, dim=1).to(x.dtype) * F.silu(gate)
        return self.out_proj(result)


class _MambaLayer(nn.Module):
    def __init__(self, blockClass, dModel, stateDim, convWidth, expand):
        super().__init__()
        self.norm = nn.LayerNorm(dModel)
        self.mixer = blockClass(d_model=dModel, d_state=stateDim, d_conv=convWidth, expand=expand)

    def forward(self, inputs):
        return inputs + self.mixer(self.norm(inputs))


class mambaDynamicsTorch(torchModelBase):
    def __init__(self, inputDim, outputDim=None, dModel=64, stateDim=16, convWidth=4,
                 expand=2, numLayers=2, engine='reference', **kwargs):
        super().__init__(modelName='mambaDynamicsTorch', **kwargs)
        dimensions = dict(inputDim=inputDim, outputDim=inputDim if outputDim is None else outputDim,
                          dModel=dModel, stateDim=stateDim, convWidth=convWidth, expand=expand, numLayers=numLayers)
        dimensions = {name: _positiveDimension(name, value) for name, value in dimensions.items()}
        if engine not in ('reference', 'mamba-ssm'):
            raise ValueError('engine must be reference or mamba-ssm')
        blockClass = _ReferenceMamba
        if engine == 'mamba-ssm':
            if platform.system() != 'Linux':
                raise RuntimeError('The official mamba-ssm engine requires Linux and CUDA; use engine=reference here')
            from mamba_ssm import Mamba
            blockClass = Mamba
        self.engine, self.inputDim = engine, dimensions['inputDim']
        self.inputProjection = nn.Linear(self.inputDim, dimensions['dModel'])
        self.layers = nn.ModuleList([_MambaLayer(blockClass, dimensions['dModel'], dimensions['stateDim'],
                                                dimensions['convWidth'], dimensions['expand'])
                                     for _ in range(dimensions['numLayers'])])
        self.outputNorm = nn.LayerNorm(dimensions['dModel'])
        self.outputProjection = nn.Linear(dimensions['dModel'], dimensions['outputDim'])

    def forward(self, inputs):
        if inputs.ndim != 3 or inputs.shape[-1] != self.inputDim or inputs.shape[1] == 0:
            raise ValueError('inputs must have shape (batch, nonempty time, inputDim)')
        if self.engine == 'mamba-ssm' and (not inputs.is_cuda or inputs.dtype not in (torch.float32, torch.float16, torch.bfloat16)):
            raise ValueError('mamba-ssm requires CUDA inputs with float32, float16 or bfloat16 dtype')
        output = self.inputProjection(inputs)
        for layer in self.layers:
            output = layer(output)
        return self.outputProjection(self.outputNorm(output))


nnRegistry.register('models', 'mambaDynamicsTorch', mambaDynamicsTorch)
