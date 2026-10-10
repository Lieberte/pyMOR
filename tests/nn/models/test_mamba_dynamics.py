import numpy as np
import pytest
import torch

from mor.nn.models.mambaDynamicsTorch import mambaDynamicsTorch, _ReferenceMamba


def test_mamba_selective_recurrence_matches_scalar_reference_and_gradients():
    block = _ReferenceMamba(1, d_state=1, d_conv=1, expand=1).double()
    with torch.no_grad():
        block.in_proj.weight.copy_(torch.tensor([[1.], [2.]]))
        block.conv1d.weight.fill_(1.)
        block.conv1d.bias.zero_()
        block.x_proj.weight.copy_(torch.tensor([[0.4], [0.5], [0.6]]))
        block.dt_proj.weight.fill_(0.7)
        block.dt_proj.bias.fill_(-0.3)
        block.A_log.fill_(np.log(1.2))
        block.D.fill_(0.8)
        block.out_proj.weight.fill_(1.1)
    values = np.array([0.2, -0.1, 0.3, 0.1])
    state, expected = 0., []
    for value in values:
        x = value / (1 + np.exp(-value))
        dt = np.logaddexp(0., 0.7 * 0.4 * x - 0.3)
        state = np.exp(-1.2 * dt) * state + dt * 0.5 * x * x
        gate = 2 * value / (1 + np.exp(-2 * value))
        expected.append(1.1 * (0.6 * x * state + 0.8 * x) * gate)
    inputs = torch.tensor(values).reshape(1, -1, 1).requires_grad_()
    torch.testing.assert_close(block(inputs).reshape(-1), torch.tensor(expected))
    assert torch.autograd.gradcheck(block, (inputs,), eps=1e-6, atol=1e-5)
    block(inputs).sum().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in block.parameters())


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_mamba_is_causal_trainable_and_reloads(device, dtype):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    torch.manual_seed(42)
    model = mambaDynamicsTorch(inputDim=2, outputDim=3, dModel=4, stateDim=2, convWidth=3, numLayers=2).to(device=device, dtype=dtype)
    inputs = torch.randn((2, 7, 2), device=device, dtype=dtype)
    altered = inputs.clone()
    altered[:, 4:] += 20
    original = model(inputs)
    torch.testing.assert_close(model(altered)[:, :4], original[:, :4])
    loss = original.square().mean()
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer.step()
    assert not torch.equal(model(inputs), original)
    restored = mambaDynamicsTorch(inputDim=2, outputDim=3, dModel=4, stateDim=2, convWidth=3, numLayers=2).to(dtype=dtype)
    restored.loadState(model.saveState())
    torch.testing.assert_close(restored(inputs.cpu()), model(inputs).cpu(), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('name', ['inputDim', 'outputDim', 'dModel', 'stateDim', 'convWidth', 'expand', 'numLayers'])
def test_mamba_rejects_invalid_dimensions(name):
    options = {'inputDim': 2, name: 0}
    with pytest.raises(ValueError, match=name): mambaDynamicsTorch(**options)


def test_official_engine_platform_contract_and_bad_input(monkeypatch):
    import importlib
    module = importlib.import_module('mor.nn.models.mambaDynamicsTorch')
    monkeypatch.setattr(module.platform, 'system', lambda: 'Windows')
    with pytest.raises(RuntimeError, match='Linux'): mambaDynamicsTorch(inputDim=2, engine='mamba-ssm')
    with pytest.raises(ValueError, match='engine'): mambaDynamicsTorch(inputDim=2, engine='bad')
    model = mambaDynamicsTorch(inputDim=2, dModel=4, numLayers=1)
    for values in [torch.zeros(1, 0, 2), torch.zeros(1, 2), torch.zeros(1, 3, 4)]:
        with pytest.raises(ValueError, match='shape'): model(values)
