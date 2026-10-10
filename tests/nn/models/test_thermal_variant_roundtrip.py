import pytest
import torch

from tests.nn.models.test_thermal_set_deeponet import makeCase
from mor.nn.data.thermalOperatorDataModule import collateThermalOperatorCases
from mor.nn.models.operator import thermalMeshGnnTorch, thermalQuerySetOperatorTorch, thermalSetAttentionOperatorTorch


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('variant', ['mesh', 'query', 'attention'])
def test_thermal_variant_training_and_cross_device_state_roundtrip(device, variant):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    if variant == 'mesh':
        factory = lambda: thermalMeshGnnTorch(nodeFeatureDim=3, edgeFeatureDim=4, hiddenDim=8, messagePassingSteps=2)
        inputs = (torch.randn(4, 3, device=device), torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], device=device),
                  torch.randn(4, 4, device=device))
    else:
        if variant == 'query':
            factory = lambda: thermalQuerySetOperatorTorch(volumeFeatureDim=5, boundaryFeatureDim=8,
                                                            globalFeatureDim=3, hiddenDim=8, kernelScales=(0.25,))
        else:
            factory = lambda: thermalSetAttentionOperatorTorch(volumeFeatureDim=5, boundaryFeatureDim=8,
                                                               globalFeatureDim=3, hiddenDim=8, latentCount=2, attentionHeads=2)
        batch = collateThermalOperatorCases([makeCase()])
        inputs = ({key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()},)
    model = factory().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.zero_grad(set_to_none=True)
    loss = model(*inputs).square().mean()
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    model.eval()
    restored = factory()
    restored.loadState(model.saveState(toCpu=True))
    restored.eval()
    cpuInputs = tuple({key: value.cpu() if isinstance(value, torch.Tensor) else value for key, value in item.items()}
                      if isinstance(item, dict) else item.cpu() for item in inputs)
    with torch.no_grad():
        torch.testing.assert_close(restored(*cpuInputs), model(*inputs).cpu(), atol=1e-5, rtol=1e-5)
