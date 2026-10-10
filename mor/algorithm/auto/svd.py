def selectSVDVariant(backendName: str = 'scipy', **kwargs) -> str:
    xOperator = kwargs.get('xOperator')
    rank = kwargs.get('rank')

    if kwargs.get('incremental', False) or kwargs.get('streaming', False):
        return 'incrementalStandard'

    if kwargs.get('centerSnapshots', False):
        return 'static'

    if xOperator is None:
        return 'static'

    shape = getattr(xOperator, 'shape', ())
    if len(shape) != 2:
        return 'static'
    nRows, nCols = shape
    minDim = min(nRows, nCols)
    aspect = max(nRows, nCols) / max(minDim, 1)
    rankRatio = None if rank is None or minDim == 0 else rank / minDim

    if kwargs.get('product') is not None:
        return 'dual'

    if backendName == 'torch':
        data = xOperator.toBackendData() if hasattr(xOperator, 'toBackendData') else None
        device = getattr(data, 'device', None)
        if getattr(device, 'type', None) == 'cuda' and nRows >= 2 * nCols:
            return 'dual'
        if nRows >= 2 * nCols and (nRows >= 1000 or rankRatio is not None and rankRatio <= 0.5):
            return 'dual'
        return 'economic'

    if getattr(xOperator, 'isSparse', False):
        return 'economic'

    if backendName == 'scipy' and nRows == nCols and 32 <= minDim <= 128:
        metadata = getattr(xOperator, 'metadata', {})
        isDiagonal = bool(metadata.get('isDiagonal', getattr(xOperator, 'isDiagonal', False)))
        isComposed = bool(metadata.get('isComposed', getattr(xOperator, 'isComposed', False)))
        if isDiagonal or isComposed:
            return 'dual'

    if backendName == 'scipy' and nRows >= 2 * nCols and (nRows >= 1000 and nCols >= 20):
        return 'dual'

    if backendName == 'scipy' and nRows >= 2 * nCols:
        return 'qrSVD'

    return 'economic'
