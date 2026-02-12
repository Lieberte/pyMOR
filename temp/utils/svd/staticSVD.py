from typing import Tuple,Optional
import numpy as np

from .svd import svdBase, svdMethod, truncationSVD, registerSVD

@registerSVD(svdMethod.static)
class staticSVD(svdBase):
    def __init__(self, method: svdMethod, fullMatrices: bool = False):
        super().__init__(name="Static SVD", method=method)
        self.fullMatrices = fullMatrices

    def _decompose(self, A: np.ndarray,r:Optional[int]= None,**kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        U, S, Vt = np.linalg.svd(A, full_matrices=self.fullMatrices)
        return truncationSVD(U, S, Vt, r)

    @property
    def supportsIncremental(self) -> bool:
        return False

    @property
    def supportsSparse(self) -> bool:
        return False

    @property
    def isApproximate(self) -> bool:
        return False

