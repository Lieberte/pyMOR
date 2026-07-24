from mor.algorithm.lyapunov_discrete import estimateDiscreteRadius


def _is_sparse_operator(op) -> bool:
    return op is not None and getattr(op, 'isSparse', False)


def selectLyapunovAlgorithm(**kwargs) -> str:
    variant = kwargs.get('variant', 'auto')
    if variant != 'auto': return variant
    
    A = kwargs.get('A')
    E = kwargs.get('E')
    backendName = kwargs.get('backendName', 'scipy')
    isContinuous = kwargs.get('isContinuous', True)
    isSparse = any(_is_sparse_operator(op) for op in (A, E))
    
    if isSparse:
        if backendName != 'scipy':
            raise NotImplementedError(
                "Sparse Lyapunov auto-dispatch is currently supported only "
                "for the scipy backend."
            )
        return 'lradi' if isContinuous else 'lrsmith'
    
    if isContinuous:
        if backendName == 'scipy':
            return 'bartelsStewart'
        elif backendName == 'torch':
            return 'sign'
        return 'bartelsStewart'
    else:
        raise NotImplementedError(
            "Dense discrete Lyapunov equations are not implemented. "
            "Only the sparse path (lrsmith) supports the discrete case; "
            "use a sparse descriptor or implement a dense discrete solver "
            "(e.g. a discrete Bartels-Stewart variant)."
        )


def recommendLyapunovOptions(**kwargs) -> dict:
    variant = selectLyapunovAlgorithm(**kwargs)
    if variant == 'lrsmith':
        A = kwargs.get('A')
        E = kwargs.get('E')
        B = kwargs.get('B')
        n = getattr(A, 'shape', (0, 0))[0] if A is not None else 0
        inputs = 1
        if B is not None and len(getattr(B, 'shape', ())) == 2:
            inputs = B.shape[1]
        radius = estimateDiscreteRadius(A, E)
        options = {
            'maxIter': 220 if n >= 1000 else 160,
            'tol': 1e-12,
            'acceptanceResidual': 1e-7 if n >= 1000 else 1e-8,
            'stability': 'unknown' if radius is None else 'stable',
            'risk': 'standard',
        }
        if inputs > 1:
            options['maxIter'] = max(options['maxIter'], 220)
        if radius is not None:
            options['spectralRadiusEstimate'] = radius
            if radius >= 1.0:
                options.update({'stability': 'unstable', 'risk': 'reject_unstable_discrete_operator'})
            elif radius >= 0.99:
                options.update({'risk': 'near_unit_slow_convergence', 'maxIter': max(options['maxIter'], 600)})
            elif radius >= 0.95:
                options.update({'risk': 'near_unit_monitor', 'maxIter': max(options['maxIter'], 360)})
        return options

    if variant != 'lradi':
        return {}

    A = kwargs.get('A')
    B = kwargs.get('B')
    n = getattr(A, 'shape', (0, 0))[0] if A is not None else 0
    inputs = 1
    if B is not None and len(getattr(B, 'shape', ())) == 2:
        inputs = B.shape[1]

    if n >= 5000:
        return {
            'maxIter': 240,
            'compressionColumns': max(60, 20 * inputs),
            'shiftOptions': {'maxShifts': 12},
        }
    if inputs >= 4:
        return {
            'maxIter': 240,
            'compressionColumns': max(64, 20 * inputs),
            'shiftOptions': {'maxShifts': 12},
        }
    if n >= 1000:
        return {
            'maxIter': 180,
            'compressionColumns': max(32, 16 * inputs),
            'shiftOptions': {'maxShifts': 10},
        }
    return {
        'maxIter': 120,
        'compressionColumns': 16,
        'shiftOptions': {'maxShifts': 8},
    }
