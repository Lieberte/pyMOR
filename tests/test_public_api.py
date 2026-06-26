import mor


def test_top_level_public_api_imports_classical_core():
    assert mor.__version__ == "0.1.0"
    assert mor.backendRegistry is not None
    assert mor.algorithmRegistry is not None
    assert mor.solverRegistry is not None
    assert mor.reductorRegistry is not None
    assert mor.matrixOperator.__name__ == "matrixOperator"
