def selectLinearAlgorithm(**kwargs) -> str:
    if kwargs.get('matrixFree', False):
        if kwargs.get('backendName', 'scipy') != 'scipy':
            raise NotImplementedError('Matrix-free linear auto requires the scipy backend')
        return 'gmres'
    return 'shifted'
