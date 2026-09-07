import numpy as np
import pytest
import torch

from mor.nn.data import collateThermalOperatorCases, thermalOperatorCase
from mor.nn.losses import thermalSteadyPinnLoss
from mor.nn.models import (
    THERMAL_CATEGORIZED_OPERATOR_FORMAT,
    thermalCategorizedOperatorTorch,
    thermalCategoryKey,
    thermalQuerySetOperatorTorch,
)


def _case(caseId: str, target: float = 0.0) -> thermalOperatorCase:
    rng = np.random.default_rng(abs(hash(caseId)) % (2**32))
    volume = rng.normal(size=(6, 15)).astype(np.float32)
    volume[:, :3] = rng.uniform(-0.7, 0.7, size=(6, 3))
    volume[:, 3:12] = np.eye(3, dtype=np.float32).reshape(1, 9)
    volume[:, 14] = np.sign(target) * np.log1p(abs(target))
    boundary = rng.normal(size=(5, 12)).astype(np.float32)
    boundary[:, :3] = rng.uniform(-1.0, 1.0, size=(5, 3))
    boundary[:, 6:9] = 0.0
    boundary[:, 9:12] = 0.0
    boundary[:, 10] = 1.0
    return thermalOperatorCase(
        caseId=caseId,
        geometryId=f'{caseId}-geometry',
        volumeFeatures=volume,
        volumeWeights=np.full(6, 1.0 / 6.0),
        boundaryFeatures=boundary,
        boundaryWeights=np.full(5, 1.0 / 5.0),
        globalFeatures=np.array([0.1, 0.2, 1.0], dtype=np.float32),
        queryCoordinates=rng.uniform(-0.6, 0.6, size=(7, 3)),
        queryTemperatures=np.full(7, target, dtype=np.float32),
    )


def _model() -> thermalQuerySetOperatorTorch:
    return thermalQuerySetOperatorTorch(
        volumeFeatureDim=15,
        boundaryFeatureDim=12,
        globalFeatureDim=3,
        hiddenDim=8,
        kernelScales=(0.3,),
        queryFourierFrequencies=1,
        decomposeMeanAndShape=True,
        anchorFeatureIndex=-1,
    )


def _constantModel(value: float) -> thermalQuerySetOperatorTorch:
    model = _model()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.decoder[-1].bias.fill_(value)
    return model


def test_steady_pinn_loss_has_finite_second_derivative_backward():
    batch = collateThermalOperatorCases([_case('pde', target=0.2)])
    model = _model()
    lossFunction = thermalSteadyPinnLoss(
        energyBalanceWeight=0.0,
        pdeResidualWeight=0.05,
        pdeCollocationCount=4,
    )
    output = model.forwardWithAux(batch)
    terms = lossFunction.lossTerms(model, output, batch)
    loss = terms['dataAndEnergy'] + 0.05 * terms['pdeResidual']
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(terms['pdeResidual'])
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_steady_pinn_loss_handles_queries_coincident_with_input_points():
    case = _case('coincident', target=0.1)
    case = thermalOperatorCase(
        caseId=case.caseId,
        geometryId=case.geometryId,
        volumeFeatures=case.volumeFeatures,
        volumeWeights=case.volumeWeights,
        boundaryFeatures=case.boundaryFeatures,
        boundaryWeights=case.boundaryWeights,
        globalFeatures=case.globalFeatures,
        queryCoordinates=np.vstack(
            (case.volumeFeatures[0, :3], case.boundaryFeatures[0, :3])
        ),
        queryTemperatures=np.array([0.1, 0.1]),
    )
    batch = collateThermalOperatorCases([case])
    model = thermalQuerySetOperatorTorch(
        volumeFeatureDim=15,
        boundaryFeatureDim=12,
        globalFeatureDim=3,
        hiddenDim=8,
        kernelScales=(0.3,),
        queryFourierFrequencies=1,
        includeRelativeKernelFeatures=True,
        decomposeMeanAndShape=True,
        anchorFeatureIndex=-1,
        residualBlockCount=2,
        residualScale=0.5,
    )
    lossFunction = thermalSteadyPinnLoss(
        pdeResidualWeight=0.05,
        pdeCollocationCount=2,
    )

    residual = lossFunction.computePdeResidual(model, batch)
    residual.backward()

    assert torch.isfinite(residual)


def test_boundary_autodiff_residuals_match_linear_temperature_field():
    case = _case('linear-boundary', target=0.0)
    boundary = np.zeros((3, 12), dtype=np.float32)
    boundary[:, 0] = [0.0, 0.5, 0.25]
    boundary[:, 3] = 1.0
    boundary[0, 8] = np.log1p(1.0)
    boundary[0, 9] = 1.0
    boundary[1, 6] = np.log1p(2.0)
    boundary[1, 7] = 1.0
    boundary[1, 10] = 1.0
    boundary[2, 7] = 0.25
    boundary[2, 11] = 1.0
    case = thermalOperatorCase(
        caseId=case.caseId,
        geometryId=case.geometryId,
        volumeFeatures=case.volumeFeatures,
        volumeWeights=case.volumeWeights,
        boundaryFeatures=boundary,
        boundaryWeights=np.ones(3, dtype=np.float32),
        globalFeatures=case.globalFeatures,
        queryCoordinates=case.queryCoordinates,
        queryTemperatures=case.queryTemperatures,
    )
    batch = collateThermalOperatorCases([case])

    class LinearTemperature(torch.nn.Module):
        def forward(self, values):
            return values['queryCoordinates'][..., :1]

    terms = thermalSteadyPinnLoss(
        neumannResidualWeight=1.0,
        robinResidualWeight=1.0,
        dirichletResidualWeight=1.0,
        fixedTemperatureBoundaryTypeIds=(2,),
        boundaryCollocationCount=3,
    )._boundaryConditionResiduals(LinearTemperature(), batch)

    for value in terms.values():
        torch.testing.assert_close(value, torch.zeros_like(value), atol=1e-6, rtol=0.0)


def test_boundary_autodiff_residuals_backpropagate_through_operator():
    case = _case('boundary-backward', target=0.15)
    boundary = case.boundaryFeatures.copy()
    boundary[:, 3:6] /= np.linalg.norm(boundary[:, 3:6], axis=1, keepdims=True)
    boundary[:2, 9:12] = 0.0
    boundary[:2, 9] = 1.0
    boundary[2:4, 6] = np.log1p(0.7)
    boundary[2:4, 7] = 0.1
    boundary[2:4, 9:12] = 0.0
    boundary[2:4, 10] = 1.0
    boundary[4, 7] = 0.2
    boundary[4, 9:12] = 0.0
    boundary[4, 11] = 1.0
    case = thermalOperatorCase(
        caseId=case.caseId,
        geometryId=case.geometryId,
        volumeFeatures=case.volumeFeatures,
        volumeWeights=case.volumeWeights,
        boundaryFeatures=boundary,
        boundaryWeights=case.boundaryWeights,
        globalFeatures=case.globalFeatures,
        queryCoordinates=case.queryCoordinates,
        queryTemperatures=case.queryTemperatures,
    )
    batch = collateThermalOperatorCases([case])
    model = _model()
    lossFunction = thermalSteadyPinnLoss(
        neumannResidualWeight=0.1,
        robinResidualWeight=0.1,
        dirichletResidualWeight=0.1,
        fixedTemperatureBoundaryTypeIds=(2,),
        boundaryCollocationCount=5,
    )
    output = model.forwardWithAux(batch)
    terms = lossFunction.lossTerms(model, output, batch)
    loss = lossFunction.computePinnBatch(model, output, batch)
    loss.backward()

    for name in ('neumannResidual', 'robinResidual', 'dirichletResidual'):
        assert torch.isfinite(terms[name])
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_residual_decoder_starts_as_legacy_model_and_updates_smoothly():
    torch.manual_seed(31)
    legacy = _model()
    torch.manual_seed(31)
    residual = thermalQuerySetOperatorTorch(
        volumeFeatureDim=15,
        boundaryFeatureDim=12,
        globalFeatureDim=3,
        hiddenDim=8,
        kernelScales=(0.3,),
        queryFourierFrequencies=1,
        decomposeMeanAndShape=True,
        anchorFeatureIndex=-1,
        residualBlockCount=2,
    )
    batch = collateThermalOperatorCases([_case('residual', target=0.2)])

    torch.testing.assert_close(residual(batch), legacy(batch))
    loss = residual(batch).square().mean()
    loss.backward()

    lastLayers = [block.residual[-1] for block in residual.decoderResidualBlocks]
    assert all(layer.weight.grad is not None for layer in lastLayers)
    assert all(torch.isfinite(layer.weight.grad).all() for layer in lastLayers)


@pytest.mark.parametrize('architecture', ('unet', 'dense'))
def test_smooth_skip_decoders_have_finite_second_derivative_backward(architecture):
    batch = collateThermalOperatorCases([_case(architecture, target=0.15)])
    model = thermalQuerySetOperatorTorch(
        volumeFeatureDim=15,
        boundaryFeatureDim=12,
        globalFeatureDim=3,
        hiddenDim=8,
        kernelScales=(0.3,),
        queryFourierFrequencies=1,
        includeRelativeKernelFeatures=True,
        decomposeMeanAndShape=True,
        anchorFeatureIndex=-1,
        decoderArchitecture=architecture,
    )
    lossFunction = thermalSteadyPinnLoss(
        pdeResidualWeight=0.05,
        pdeCollocationCount=4,
    )

    output = model.forwardWithAux(batch)
    terms = lossFunction.lossTerms(model, output, batch)
    loss = terms['dataAndEnergy'] + 0.05 * terms['pdeResidual']
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(terms['pdeResidual'])
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


@pytest.mark.parametrize(
    'context_architecture',
    ('global', 'deeponet', 'attention', 'hybrid'),
)
def test_discrete_context_architectures_support_meshfree_pde_backward(
    context_architecture,
):
    batch = collateThermalOperatorCases([_case(context_architecture, target=0.15)])
    model = thermalQuerySetOperatorTorch(
        volumeFeatureDim=15,
        boundaryFeatureDim=12,
        globalFeatureDim=3,
        hiddenDim=8,
        kernelScales=(0.3,),
        queryFourierFrequencies=1,
        includeRelativeKernelFeatures=True,
        decomposeMeanAndShape=True,
        anchorFeatureIndex=-1,
        contextArchitecture=context_architecture,
        attentionHeadCount=4,
    )
    lossFunction = thermalSteadyPinnLoss(
        pdeResidualWeight=0.05,
        pdeResidualMode='meshfree',
        pdeCollocationCount=6,
    )

    output = model.forwardWithAux(batch)
    terms = lossFunction.lossTerms(model, output, batch)
    loss = terms['dataAndEnergy'] + 0.05 * terms['pdeResidual']
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(terms['pdeResidual'])
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_categorized_operator_routes_mixed_batch_and_falls_back_global():
    globalModel = _constantModel(-1.0)
    shaftModel = _constantModel(2.0)
    key = thermalCategoryKey({'componentType': 'shaft'}, ('componentType',))
    bundle = thermalCategorizedOperatorTorch(
        globalModel,
        {key: shaftModel},
        categoryFields=('componentType',),
    )
    batch = collateThermalOperatorCases([_case('known'), _case('unknown')])
    prediction, routing = bundle.forwardWithRouting(
        batch,
        ({'componentType': 'shaft'}, {'componentType': 'new-part'}),
    )

    torch.testing.assert_close(prediction[0], torch.full_like(prediction[0], 2.0))
    torch.testing.assert_close(prediction[1], torch.full_like(prediction[1], -1.0))
    assert routing[0]['usedGlobalFallback'] is False
    assert routing[1]['usedGlobalFallback'] is True


def test_categorized_operator_blends_expert_with_global():
    key = thermalCategoryKey({'componentType': 'shaft'}, ('componentType',))
    bundle = thermalCategorizedOperatorTorch(
        _constantModel(-2.0),
        {key: _constantModel(2.0)},
        expertBlendWeights={key: 0.25},
    )
    batch = collateThermalOperatorCases([_case('blend')])

    prediction, routing = bundle.forwardWithRouting(
        batch,
        {'componentType': 'shaft'},
    )

    torch.testing.assert_close(prediction, torch.full_like(prediction, -1.0))
    assert routing[0]['expertBlendWeight'] == 0.25


def test_categorized_checkpoint_round_trip_preserves_routing():
    config = {
        'volumeFeatureDim': 15,
        'boundaryFeatureDim': 12,
        'globalFeatureDim': 3,
        'hiddenDim': 8,
        'kernelScales': (0.3,),
        'queryFourierFrequencies': 1,
        'decomposeMeanAndShape': True,
        'anchorFeatureIndex': -1,
    }
    key = thermalCategoryKey({'componentType': 'shaft'}, ('componentType',))
    original = thermalCategorizedOperatorTorch(
        _constantModel(-1.0),
        {key: _constantModel(2.0)},
        expertBlendWeights={key: 0.75},
    )
    checkpoint = original.toCheckpoint(globalModelConfig=config)
    restored = thermalCategorizedOperatorTorch.fromCheckpoint(checkpoint)
    batch = collateThermalOperatorCases([_case('round-trip')])

    assert checkpoint['formatVersion'] == THERMAL_CATEGORIZED_OPERATOR_FORMAT
    assert restored.expertBlendWeights == {key: 0.75}
    torch.testing.assert_close(
        restored(batch, {'componentType': 'shaft'}),
        original(batch, {'componentType': 'shaft'}),
    )
