from mor.operators import matrixOperator

def selectLyapunovSolver(**kwargs) -> str:
    A = kwargs.get('A')
    E = kwargs.get('E')
    isContinuous = kwargs.get('isContinuous', True)
    
    # TODO: implement sophisticated heuristic based on matrix size, sparsity, and backend
    if isContinuous:
        return 'continuousLrGeneralized' if E is not None else 'continuousLr'
    return 'discreteLrGeneralized' if E is not None else 'discreteLr'
