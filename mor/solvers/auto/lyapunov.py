from mor.operators import matrixOperator

def selectLyapunovSolver(**kwargs) -> str:
    # Now we always return 'unified', and let algorithm/auto handle the rest
    return 'unified'
