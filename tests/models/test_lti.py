import numpy as np
import pytest

from mor.models.lti import ltiModel
from mor.operators import matrixOperator


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
