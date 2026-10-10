import pytest
import torch

from mor.nn.data.snapshotDataModule import snapshotDataModule
from mor.nn.losses.vaeReconstructionLoss import vaeReconstructionLoss
from mor.nn.models.convolutionalAutoEncoderTorch import convolutionalAutoEncoderTorch
from mor.nn.models.variationalAutoEncoderTorch import variationalAutoEncoderTorch
from mor.nn.validation.reconstructionMetrics import reconstructionMetrics


@pytest.mark.parametrize('shape', [(16, 16), (32, 32), (17, 25)])
def test_convolutional_autoencoder_trains_and_reloads_exact_spatial_shape(shape, tmp_path):
    model = convolutionalAutoEncoderTorch(1, 2, [2, 4])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    inputs = torch.randn(2, 1, *shape)
    predictions = model(inputs)
    assert predictions.shape == inputs.shape
    loss = (predictions - inputs).square().mean()
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    model.eval()
    path = tmp_path / 'weights.pt'
    torch.save(model.saveState(toCpu=True), path)
    restored = convolutionalAutoEncoderTorch(1, 2, [2, 4])
    restored.loadState(torch.load(path, weights_only=True))
    restored.eval()
    torch.testing.assert_close(restored(inputs), model(inputs))
    assert restored.decode(restored.encode(inputs)).shape == inputs.shape
    with pytest.raises(ValueError, match='remain'):
        restored(torch.zeros(2, 1, shape[0] + 1, shape[1]))


def test_convolutional_decoder_requires_shape_before_first_encode():
    with pytest.raises(ValueError, match='inputShape'):
        convolutionalAutoEncoderTorch(1, 2, [2]).decode(torch.zeros(1, 2))
    assert convolutionalAutoEncoderTorch(1, 2, [2], inputShape=(13, 19)).decode(torch.zeros(1, 2)).shape == (1, 1, 13, 19)


def test_legacy_fixed_four_by_four_checkpoint_migrates_without_changing_outputs():
    model = convolutionalAutoEncoderTorch(1, 2, [2, 4], inputShape=(16, 16))
    inputs = torch.randn(2, 1, 16, 16)
    expected = model(inputs)
    state = model.saveState()
    del state['_inputSpatialShape']
    restored = convolutionalAutoEncoderTorch(1, 2, [2, 4])
    restored.loadState(state)
    torch.testing.assert_close(restored(inputs), expected)
    assert tuple(restored._inputSpatialShape.tolist()) == (16, 16)
    assert '_inputSpatialShape' not in state


def test_legacy_checkpoint_requires_matching_explicit_shape_when_not_four_by_four():
    model = convolutionalAutoEncoderTorch(1, 2, [2, 4], inputShape=(17, 25))
    inputs = torch.randn(2, 1, 17, 25)
    expected = model(inputs)
    state = model.saveState()
    del state['_inputSpatialShape']
    with pytest.raises(ValueError, match='explicit inputShape'):
        convolutionalAutoEncoderTorch(1, 2, [2, 4]).loadState(state)
    with pytest.raises(ValueError, match='losslessly'):
        convolutionalAutoEncoderTorch(1, 2, [2, 4], inputShape=(16, 16)).loadState(state)
    restored = convolutionalAutoEncoderTorch(1, 2, [2, 4], inputShape=(17, 25))
    restored.loadState(state)
    torch.testing.assert_close(restored(inputs), expected)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_vae_metrics_keep_full_loss_tuple_and_use_deterministic_mean_reconstruction(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    model = variationalAutoEncoderTorch(2, 1, [3]).to(device)
    inputs = torch.randn(5, 2)
    data = snapshotDataModule(inputs, inputs, batchSize=2, shuffle=False)
    validator = reconstructionMetrics(deviceName=device, lossFunction=vaeReconstructionLoss())
    first = validator.evaluate(model, data)
    second = validator.evaluate(model, data)
    assert first == second
    with torch.no_grad():
        state = torch.get_rng_state().clone()
        outputs = model(inputs.to(device))
        assert torch.equal(torch.get_rng_state(), state)
        expected = vaeReconstructionLoss().compute(tuple(t.double() for t in outputs), inputs.to(device).double())
        mse = (outputs[0].double() - inputs.to(device).double()).square().mean()
    assert first['validationLoss'] == pytest.approx(expected.item(), rel=1e-6)
    assert first['mse'] == pytest.approx(mse.item(), rel=1e-6)
    model.train()
    assert not torch.equal(model(inputs.to(device))[0], model(inputs.to(device))[0])
