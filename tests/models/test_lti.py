import numpy as np
import pytest

from mor.models.lti import ltiModel
from mor.operators import matrixOperator, scaledOperator


def test_lti_model_input_dim_is_one_for_vector_b():
    A = matrixOperator(-np.eye(3), backendName='scipy')
    B = matrixOperator(np.ones(3), backendName='scipy')
    C = matrixOperator(np.ones((1, 3)), backendName='scipy')

    model = ltiModel(A, B, C, backendName='scipy')

    assert model.inputDim == 1


def test_lti_model_rejects_one_dimensional_c():
    A = matrixOperator(-np.eye(3), backendName='scipy')
    B = matrixOperator(np.ones((3, 1)), backendName='scipy')
    C = matrixOperator(np.ones(3), backendName='scipy')

    with pytest.raises(ValueError, match='C must be a 2D operator'):
        ltiModel(A, B, C, backendName='scipy')


def test_lti_model_rejects_invalid_feedthrough_shape():
    A = matrixOperator(-np.eye(3), backendName='scipy')
    B = matrixOperator(np.ones((3, 2)), backendName='scipy')
    C = matrixOperator(np.ones((4, 3)), backendName='scipy')
    D = matrixOperator(np.ones((2, 4)), backendName='scipy')

    with pytest.raises(ValueError, match=r'D must have shape \(4, 2\)'):
        ltiModel(A, B, C, D, backendName='scipy')


def test_lti_model_records_discrete_time_domain():
    A = matrixOperator(0.5 * np.eye(3), backendName='scipy')
    B = matrixOperator(np.ones((3, 1)), backendName='scipy')
    C = matrixOperator(np.ones((1, 3)), backendName='scipy')

    model = ltiModel(A, B, C, backendName='scipy', isContinuous=False)

    assert not model.isContinuous
    assert model.timeDomain == 'discrete'


def test_lti_model_rejects_mixed_backends():
    pytest.importorskip('torch')
    A = matrixOperator(-np.eye(3), backendName='scipy')
    B = matrixOperator(np.ones((3, 1)), backendName='torch')
    C = matrixOperator(np.ones((1, 3)), backendName='scipy')

    with pytest.raises(ValueError, match='B backend torch does not match model backend scipy'):
        ltiModel(A, B, C, backendName='scipy')


def test_lti_to_numpy_accepts_scaled_operator():
    A = scaledOperator(matrixOperator(-np.eye(3), backendName='scipy'), 2.0)
    B = matrixOperator(np.ones((3, 1)), backendName='scipy')
    C = matrixOperator(np.ones((1, 3)), backendName='scipy')

    arrays = ltiModel(A, B, C, backendName='scipy').toNumpy()

    np.testing.assert_allclose(arrays['A'], -2.0 * np.eye(3))


def test_lti_continuous_poles_stability_and_frequency_response():
    A = np.diag([-1.0, -2.0])
    B = np.ones((2, 1))
    C = np.array([[1.0, 2.0]])
    D = np.array([[0.25]])
    model = ltiModel(
        matrixOperator(A, backendName='scipy'),
        matrixOperator(B, backendName='scipy'),
        matrixOperator(C, backendName='scipy'),
        matrixOperator(D, backendName='scipy'),
        backendName='scipy',
    )
    frequencies = np.array([0.0, 1.5])

    response = model.frequencyResponse(frequencies)
    expected = np.asarray([C @ np.linalg.solve(1j * w * np.eye(2) - A, B) + D for w in frequencies])

    np.testing.assert_allclose(np.sort(model.poles()), [-2.0, -1.0])
    assert model.stabilityMetric() == pytest.approx(-1.0)
    assert model.isStable
    np.testing.assert_allclose(response, expected)


def test_lti_discrete_frequency_response_uses_unit_circle():
    A = np.diag([0.25, 0.5])
    B = np.ones((2, 1))
    C = np.ones((1, 2))
    model = ltiModel(
        matrixOperator(A, backendName='scipy'),
        matrixOperator(B, backendName='scipy'),
        matrixOperator(C, backendName='scipy'),
        backendName='scipy',
        isContinuous=False,
    )
    frequencies = np.array([0.0, np.pi / 2])

    expected = np.asarray([
        C @ np.linalg.solve(np.exp(1j * w) * np.eye(2) - A, B)
        for w in frequencies
    ])

    assert model.stabilityMetric() == pytest.approx(0.5)
    assert model.isStable
    np.testing.assert_allclose(model.frequencyResponse(frequencies), expected)


def test_lti_frequency_response_promotes_all_model_dtypes():
    A = np.array([[-1.0, 1.0], [-1.0, -1.0]], dtype=np.float32)
    B = np.array([[1.0 + 2.0j], [0.5 - 0.25j]], dtype=np.complex128)
    C = np.array([[0.75 + 0.5j, 1.0 - 0.2j]], dtype=np.complex128)
    model = ltiModel(
        matrixOperator(A, backendName='scipy'),
        matrixOperator(B, backendName='scipy'),
        matrixOperator(C, backendName='scipy'),
        backendName='scipy',
    )
    frequency = 0.123456789

    response = model.frequencyResponse([frequency])
    expected = C @ np.linalg.solve(1j * frequency * np.eye(2) - A, B)

    assert response.dtype == np.complex128
    np.testing.assert_allclose(response[0], expected, rtol=1e-14, atol=1e-14)


def test_lti_generalized_poles_and_negative_frequency_validation():
    model = ltiModel(
        matrixOperator(np.diag([-2.0, -6.0]), backendName='scipy'),
        matrixOperator(np.ones((2, 1)), backendName='scipy'),
        matrixOperator(np.ones((1, 2)), backendName='scipy'),
        E=matrixOperator(np.diag([2.0, 3.0]), backendName='scipy'),
        backendName='scipy',
    )

    np.testing.assert_allclose(np.sort(model.poles()), [-2.0, -1.0])
    assert model.descriptorKind == 'regular'
    with pytest.raises(ValueError, match='non-negative'):
        model.frequencyResponse([-1.0])


def test_lti_descriptor_kind_distinguishes_singular_and_unknown():
    from scipy.sparse import eye

    singular = ltiModel(
        matrixOperator(-np.eye(3), backendName='scipy'),
        matrixOperator(np.ones((3, 1)), backendName='scipy'),
        matrixOperator(np.ones((1, 3)), backendName='scipy'),
        E=matrixOperator(np.diag([1.0, 1.0, 0.0]), backendName='scipy'),
        backendName='scipy',
    )
    large = ltiModel(
        matrixOperator(-eye(513, format='csr'), backendName='scipy'),
        matrixOperator(np.ones((513, 1)), backendName='scipy'),
        matrixOperator(np.ones((1, 513)), backendName='scipy'),
        E=matrixOperator(eye(513, format='csr'), backendName='scipy'),
        backendName='scipy',
    )

    assert singular.descriptorKind == 'singular'
    assert large.descriptorKind == 'regular'
    assert large.classifyDescriptor(maxSize=600) == 'regular'


def test_lti_large_composed_diagonal_queries_do_not_materialize(monkeypatch):
    from scipy.sparse import diags

    order = 10_000
    aBase = matrixOperator(diags(-np.linspace(1.0, 2.0, order), format='csr'), backendName='scipy')
    eBase = matrixOperator(diags(np.linspace(2.0, 3.0, order), format='csr'), backendName='scipy')
    A = 2.0 * (aBase + 0.0 * aBase)
    E = eBase + 0.0 * eBase
    model = ltiModel(
        A,
        matrixOperator(np.ones((order, 1)), backendName='scipy'),
        matrixOperator(np.ones((1, order)), backendName='scipy'),
        E=E,
        backendName='scipy',
    )
    monkeypatch.setattr(A, 'toNumpy', lambda: (_ for _ in ()).throw(AssertionError('materialized A')))
    monkeypatch.setattr(E, 'toNumpy', lambda: (_ for _ in ()).throw(AssertionError('materialized E')))

    poles = model.poles()

    assert model.descriptorKind == 'regular'
    assert poles.shape == (order,)
    assert model.stabilityMetric() == pytest.approx(-1.0)


def test_lti_diagonal_descriptor_classification_uses_numerical_rank_tolerance():
    model = ltiModel(
        matrixOperator(-np.eye(3), backendName='scipy'),
        matrixOperator(np.ones((3, 1)), backendName='scipy'),
        matrixOperator(np.ones((1, 3)), backendName='scipy'),
        E=matrixOperator(np.diag([1.0, 1.0, 1e-20]), backendName='scipy'),
        backendName='scipy',
    )

    assert model.descriptorKind == 'singular'


def test_lti_singular_diagonal_descriptor_is_not_reported_stable():
    model = ltiModel(
        matrixOperator(np.diag([-1.0, -2.0]), backendName='scipy'),
        matrixOperator(np.ones((2, 1)), backendName='scipy'),
        matrixOperator(np.ones((1, 2)), backendName='scipy'),
        E=matrixOperator(np.diag([1.0, 0.0]), backendName='scipy'),
        backendName='scipy',
    )

    np.testing.assert_array_equal(model.poles(), [-1.0, np.inf])
    assert model.stabilityMetric() == np.inf
    assert not model.isStable


@pytest.mark.parametrize('frequencies', [[np.nan], [np.inf], [-np.inf]])
def test_lti_frequency_response_rejects_non_finite_frequencies(frequencies):
    model = ltiModel(
        matrixOperator(-np.eye(2), backendName='scipy'),
        matrixOperator(np.ones((2, 1)), backendName='scipy'),
        matrixOperator(np.ones((1, 2)), backendName='scipy'),
        backendName='scipy',
    )

    with pytest.raises(ValueError, match='finite'):
        model.frequencyResponse(frequencies)


def test_lti_sparse_frequency_response_does_not_materialize_operators(monkeypatch):
    from scipy.sparse import diags

    A = matrixOperator(diags([-1.0, -2.0, -3.0], format='csr'), backendName='scipy')
    E = matrixOperator(diags([1.0, 2.0, 3.0], format='csr'), backendName='scipy')
    model = ltiModel(
        A,
        matrixOperator(np.ones((3, 1)), backendName='scipy'),
        matrixOperator(np.ones((1, 3)), backendName='scipy'),
        E=E,
        backendName='scipy',
    )
    monkeypatch.setattr(A, 'toNumpy', lambda: (_ for _ in ()).throw(AssertionError('materialized A')))
    monkeypatch.setattr(E, 'toNumpy', lambda: (_ for _ in ()).throw(AssertionError('materialized E')))

    response = model.frequencyResponse([0.0, 1.0])
    expected = np.asarray([
        np.ones((1, 3)) @ np.linalg.solve(
            1j * frequency * np.diag([1.0, 2.0, 3.0]) - np.diag([-1.0, -2.0, -3.0]),
            np.ones((3, 1)),
        )
        for frequency in [0.0, 1.0]
    ])

    assert response.shape == (2, 1, 1)
    np.testing.assert_allclose(response, expected)
