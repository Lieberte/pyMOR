from typing import Literal, Optional
import numpy as np

TimeType = Literal['cont', 'disc']
FormType = Literal['lowrank', 'dense']
SystemType = Literal['standard', 'generalized']
Backend = Literal['scipy']

_DEFAULT_BACKEND: dict[tuple[TimeType, FormType, SystemType], Backend] = {
    ('cont', 'lowrank', 'standard'): 'scipy',
    ('cont', 'dense', 'standard'): 'scipy',
    ('disc', 'lowrank', 'standard'): 'scipy',
    ('disc', 'dense', 'standard'): 'scipy',
    ('cont', 'lowrank', 'generalized'): 'scipy',
    ('cont', 'dense', 'generalized'): 'scipy',
    ('disc', 'lowrank', 'generalized'): 'scipy',
    ('disc', 'dense', 'generalized'): 'scipy',
}

def _checkLyapunovArgs(A, E, B, trans):
    n = A.shape[0]
    assert isinstance(A, np.ndarray)
    assert isinstance(B, np.ndarray)
    assert A.ndim == 2 and A.shape[1] == n
    assert B.ndim == 2 and B.shape[1-trans] == n
    if E != None:
        assert isinstance(E, np.ndarray)
        assert E.ndim == 2 and E.shape == (n,n)

