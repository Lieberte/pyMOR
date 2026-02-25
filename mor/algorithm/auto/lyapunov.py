def selectLyapunovAlgorithm(**kwargs) -> str:
    A = kwargs.get('A')
    E = kwargs.get('E')
    isContinuous = kwargs.get('isContinuous', True)
    isSparse = A.isSparse if hasattr(A, 'isSparse') else False
    if isSparse:
        if isContinuous:
            return 'lradi' 
        else:
            return 'lrsmith'
    else:
        if isContinuous:
            return 'continuousHrGeneralized' if E is not None else 'continuousHr'
        else:
            return 'discreteHrGeneralized' if E is not None else 'discreteHr'
