import torch

from mor.nn.models import geometryDgcnnClassifierTorch, geometryDgcnnEncoderTorch


def _encoder():
    return geometryDgcnnEncoderTorch(
        edgeChannels=(8, 8, 16),
        globalFeatureDim=24,
        embeddingDim=12,
        neighborCount=4,
        dropout=0.0,
    )


def test_geometry_dgcnn_produces_normalized_embeddings_and_backward():
    torch.manual_seed(3)
    model = geometryDgcnnClassifierTorch(classCount=5, encoder=_encoder())
    inputs = torch.randn(3, 32, 6)

    logits = model(inputs)
    embeddings = model.encode(inputs)
    logits.square().mean().backward()

    assert logits.shape == (3, 5)
    assert embeddings.shape == (3, 12)
    torch.testing.assert_close(
        torch.linalg.vector_norm(embeddings, dim=1), torch.ones(3), atol=1e-6, rtol=1e-6
    )
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_geometry_dgcnn_is_invariant_to_point_numbering():
    torch.manual_seed(7)
    model = _encoder().eval()
    inputs = torch.randn(2, 40, 6)
    permutation = torch.randperm(inputs.shape[1])

    original = model(inputs)
    reordered = model(inputs[:, permutation])

    torch.testing.assert_close(original, reordered, atol=2e-6, rtol=2e-6)


def test_geometry_dgcnn_rejects_too_few_points():
    model = _encoder()
    try:
        model(torch.randn(2, 4, 6))
    except ValueError as error:
        assert 'exceed neighborCount' in str(error)
    else:
        raise AssertionError('DGCNN accepted pointCount <= neighborCount')
