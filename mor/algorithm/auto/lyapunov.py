def selectLyapunovAlgorithm(**kwargs) -> str:
    variant = kwargs.get('variant', 'auto')
    if variant != 'auto': return variant
    
    A = kwargs.get('A')
    E = kwargs.get('E')
    backendName = kwargs.get('backendName', 'scipy')
    isContinuous = kwargs.get('isContinuous', True)
    isSparse = A.isSparse if hasattr(A, 'isSparse') else False
    
    if isSparse:
        return 'lradi' if isContinuous else 'lrsmith'
    
    if isContinuous:
        if backendName == 'scipy':
            return 'bartelsStewart'
        elif backendName == 'torch':
            return 'sign'
        return 'bartelsStewart'
    else:
        return 'discreteHr'
