import numpy as np
import torch

from mor.nn.data import collateThermalOperatorCases, thermalOperatorCase
from mor.nn.losses import thermalMeanShapeEnergyLoss, thermalOperatorLoss
from mor.nn.models import (
    thermalMeshGnnTorch,
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


def test_query_set_operator_v2_decomposes_mean_shape_and_uses_typed_boundary_contexts():
    rng = np.random.default_rng(19)
    boundaryFeatures = rng.normal(size=(6, 12))
    boundaryFeatures[:, 6] = np.log1p(np.linspace(0.2, 1.0, 6))
    boundaryFeatures[:, 7] = 0.0
    boundaryFeatures[:, 8] = 0.0
    boundaryFeatures[:, 9:12] = 0.0
    boundaryFeatures[:, 10] = 1.0
    volumeFeatures = rng.normal(size=(8, 15))
    volumeFeatures[:, 14] = 0.1
    case = thermalOperatorCase(
        caseId='v2-case',
        geometryId='v2-geometry',
        volumeFeatures=volumeFeatures,
        volumeWeights=np.full(8, 1.0 / 8.0),
        boundaryFeatures=boundaryFeatures,
        boundaryWeights=np.full(6, 1.0 / 6.0),
        globalFeatures=np.array([0.2, -0.1, 1.0]),
        queryCoordinates=rng.uniform(-1.0, 1.0, size=(9, 3)),
        queryTemperatures=np.linspace(0.3, 1.1, 9),
    )
    batch = collateThermalOperatorCases([case])
    model = thermalQuerySetOperatorTorch(
        volumeFeatureDim=15,
        boundaryFeatureDim=12,
        globalFeatureDim=3,
        hiddenDim=16,
        kernelScales=(0.2, 0.6),
        includeRelativeKernelFeatures=True,
        separateBoundaryTypeContexts=True,
        boundaryTypeFeatureStart=9,
        boundaryTypeCount=3,
        decomposeMeanAndShape=True,
        anchorFeatureIndex=-1,
    )
    output = model.forwardWithAux(batch)
    meanPrediction = model.forwardMean(batch)
    boundaryBatch = dict(batch)
    boundaryBatch['queryCoordinates'] = batch['boundaryFeatures'][..., :3]
    boundaryBatch['queryWeights'] = batch['boundaryWeights']
    boundaryBatch['queryMask'] = batch['boundaryMask']
    boundaryPrediction = model(boundaryBatch)
    loss = thermalMeanShapeEnergyLoss(
        energyBalanceWeight=0.05,
        fixedTemperatureBoundaryTypeIds=(2,),
    ).computeBatch(
        output,
        batch,
        boundaryPredictions=boundaryPrediction,
    )
    loss.backward()

    assert output['prediction'].shape == (1, 9, 1)
    assert output['meanOffset'].shape == (1, 1, 1)
    assert meanPrediction.shape == (1, 1)
    torch.testing.assert_close(meanPrediction[:, None, :], output['meanOffset'])
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_thermal_mesh_gnn_supports_message_passing_and_backward():
    nodeFeatures = torch.randn(5, 7)
    edgeIndex = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]],
        dtype=torch.long,
    )
    edgeFeatures = torch.randn(edgeIndex.shape[1], 4)
    model = thermalMeshGnnTorch(
        nodeFeatureDim=7,
        edgeFeatureDim=4,
        hiddenDim=16,
        messagePassingSteps=2,
    )

    prediction = model(nodeFeatures, edgeIndex, edgeFeatures)
    prediction.square().mean().backward()

    assert prediction.shape == (5, 1)
    assert torch.isfinite(prediction).all()
    assert all(parameter.grad is not None for parameter in model.parameters())
