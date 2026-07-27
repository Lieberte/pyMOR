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
