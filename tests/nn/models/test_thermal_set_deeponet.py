import numpy as np
import torch

from mor.nn.data import collateThermalOperatorCases, thermalOperatorCase
from mor.nn.losses import thermalOperatorLoss
from mor.nn.models import (
    thermalQuerySetOperatorTorch,
    thermalSetAttentionOperatorTorch,
    thermalSetDeepONetTorch,
)


def makeCase():
    rng = np.random.default_rng(4)
    return thermalOperatorCase(
        caseId='case',
        geometryId='motor',
        volumeFeatures=rng.normal(size=(6, 5)),
        volumeWeights=rng.uniform(0.1, 1.0, size=6),
        boundaryFeatures=rng.normal(size=(4, 8)),
        boundaryWeights=rng.uniform(0.1, 1.0, size=4),
        globalFeatures=rng.normal(size=3),
        queryCoordinates=rng.uniform(-1.0, 1.0, size=(7, 3)),
        queryTemperatures=rng.normal(size=7),
    )


def test_thermal_set_deeponet_supports_variable_sets_and_backward():
    case = makeCase()
    batch = collateThermalOperatorCases([case, makeCase()])
    model = thermalSetDeepONetTorch(
        volumeFeatureDim=5,
        boundaryFeatureDim=8,
        globalFeatureDim=3,
        latentDim=24,
        pointHiddenDim=16,
        queryFourierFrequencies=2,
    )

    predictions = model(batch)
    loss = thermalOperatorLoss(hotspotWeight=0.5).compute(
        predictions,
        batch['targets'],
        batch['queryWeights'],
        batch['queryMask'],
    )
    loss.backward()

    assert predictions.shape == (2, 7, 1)
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_thermal_set_encoder_is_permutation_invariant():
    case = makeCase()
    volumePermutation = np.array([4, 1, 5, 0, 3, 2])
    boundaryPermutation = np.array([2, 0, 3, 1])
    permuted = thermalOperatorCase(
        caseId='permuted',
        geometryId=case.geometryId,
        volumeFeatures=case.volumeFeatures[volumePermutation],
        volumeWeights=case.volumeWeights[volumePermutation],
        boundaryFeatures=case.boundaryFeatures[boundaryPermutation],
        boundaryWeights=case.boundaryWeights[boundaryPermutation],
        globalFeatures=case.globalFeatures,
        queryCoordinates=case.queryCoordinates,
        queryTemperatures=case.queryTemperatures,
    )
    model = thermalSetDeepONetTorch(
        volumeFeatureDim=5,
        boundaryFeatureDim=8,
        globalFeatureDim=3,
        latentDim=24,
        pointHiddenDim=16,
    ).eval()

    originalPrediction = model(collateThermalOperatorCases([case]))
    permutedPrediction = model(collateThermalOperatorCases([permuted]))

    torch.testing.assert_close(originalPrediction, permutedPrediction, atol=1e-6, rtol=1e-6)


def test_latent_attention_operator_is_permutation_invariant_and_query_shaped():
    case = makeCase()
    permutation = np.array([5, 0, 2, 4, 1, 3])
    permuted = thermalOperatorCase(
        caseId='attention-permuted',
        geometryId=case.geometryId,
        volumeFeatures=case.volumeFeatures[permutation],
        volumeWeights=case.volumeWeights[permutation],
        boundaryFeatures=case.boundaryFeatures,
        boundaryWeights=case.boundaryWeights,
        globalFeatures=case.globalFeatures,
        queryCoordinates=case.queryCoordinates,
        queryTemperatures=case.queryTemperatures,
    )
    model = thermalSetAttentionOperatorTorch(
        volumeFeatureDim=5,
        boundaryFeatureDim=8,
        globalFeatureDim=3,
        hiddenDim=24,
        latentCount=4,
        attentionHeads=4,
    ).eval()

    original = model(collateThermalOperatorCases([case]))
    reordered = model(collateThermalOperatorCases([permuted]))

    assert original.shape == (1, 7, 1)
    torch.testing.assert_close(original, reordered, atol=1e-6, rtol=1e-6)


def test_multiscale_query_set_operator_uses_variable_sets_without_connectivity():
    case = makeCase()
    batch = collateThermalOperatorCases([case])
    model = thermalQuerySetOperatorTorch(
        volumeFeatureDim=5,
        boundaryFeatureDim=8,
        globalFeatureDim=3,
        hiddenDim=16,
        kernelScales=(0.25, 0.75),
    )

    prediction = model(batch)
    prediction.square().mean().backward()

    assert prediction.shape == (1, 7, 1)
    assert torch.isfinite(prediction).all()
