from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any, Type
from enum import Enum
import numpy as np
from scipy import sparse

class svdMethod(Enum):
    static = "static"
    randomized = "randomized"


def checkArray(
        A: np.ndarray,
        acceptSparse: bool = False,
        dtype: str = 'float64'
) -> np.ndarray:
    if sparse.issparse(A):
        if not acceptSparse:
            raise ValueError("This SVD method does not support sparse matrices")
        A = np.asarray(A, dtype=dtype)
        if A.ndim != 2:
            raise ValueError("Sparse matrices must be 2D")
        if not np.isfinite(A).all ():
            raise ValueError("Input contains non-finite values")
    return np.asarray(A, dtype=dtype)

# def safeSVD(A: np.ndarray, fullMatrices:bool = False, computeUv: bool = True)-> Tuple[np.ndarray, np.ndarray, np.ndarray]:

def truncationSVD(U: np.ndarray, S: np.ndarray, Vt: np.ndarray, r: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return U[:, :r],S[:r],Vt[:r, :]

def orthogonalize(Q: np.ndarray) -> np.ndarray:
    return np.linalg.qr(Q)

def estimateRank(S: np.ndarray, tol: float = 0.0) -> int:
    return np.sum(S > tol)

def relativeError(A: np.ndarray, Aapprox: np.ndarray, norm: str = 'fro') -> float:
    return np.linalg.norm(A - Aapprox, ord=norm) / np.linalg.norm(A, ord=norm) if np.linalg.norm(A, ord=norm) > 0 else 0.0

def randomizedRangeFinder(A: np.ndarray, size:int, iter: int=2, randomState: Optional[int] = None) -> np.ndarray:
    if randomState is not None:
        np.random.seed(randomState)
    Omega = np.random.randn(A.shape[1], size)
    Y = A @ Omega
    for _ in range(iter):
        Y = A @ (A.T @ Y)
    Q, _ = np.linalg.qr(Y)
    return Q

class svdBase(ABC):
    def __init__(self, name:str, method: svdMethod):
        self.name = name
        self.method = method
        self._U = None
        self._S = None
        self._Vt = None
        self._isFitted = False
        self._nSample = None
        self._nFeature = None

    @abstractmethod
    def _decompose(self, A: np.ndarray,r:Optional[int]= None,**kwargs)-> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass

    def decompose(self, A: np.ndarray, r: Optional[int] = None, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        A = checkArray(A)
        return self._decompose(A, r, **kwargs)

    def fit(self, A: np.ndarray, r: Optional[int] = None, **kwargs):
        self._U, self._S, self._Vt = self.decompose(A, r, **kwargs)
        self._isFitted = True
        return self

    def fitTransform(self, A: np.ndarray, r: Optional[int] = None, **kwargs) -> np.ndarray:
        self.fit(A, r, **kwargs)
        return self.transform(A)

    def transform(self,A: np.ndarray) -> np.ndarray:
        self.checkFitted()
        A = checkArray(A)
        return A @ self._Vt.T

    def inverseTransform(self,A: np.ndarray) -> np.ndarray:
        self.checkFitted()
        A = checkArray(A)
        return A @ self._Vt

    def reconstruct(self,r: Optional[int] = None) -> np.ndarray:
        self.checkFitted()
        return self._U[:,:r] @ np.diag(self._S[:r]) @ self._Vt[:r,:]

    def reconstructError(self,A: np.ndarray, r: Optional[int] = None, norm: str = 'fro', relative: bool = False) -> float:
        self.checkFitted()
        A = checkArray(A)
        return relativeError(A, self.reconstruct(r),norm= norm) if relative else np.linalg.norm(A - self.reconstruct(r), ord=norm)

    def explainedVariance(self) -> np.ndarray:
        self.checkFitted()
        return self._S ** 2

    def explainedVarianceRatio(self) -> np.ndarray:
        self.checkFitted()
        return self.explainedVariance() / np.sum(self.explainedVariance()) if np.sum(self.explainedVariance()) > 0 else self.explainedVariance()

    def cumulativeVarianceRatio(self) -> np.ndarray:
        self.checkFitted()
        return np.cumsum(self.explainedVarianceRatio())

    @property
    def U(self) -> np.ndarray:
        self.checkFitted()
        return self._U

    @property
    def S(self) -> np.ndarray:
        self.checkFitted()
        return self._S

    @property
    def Vt(self) -> np.ndarray:
        self.checkFitted()
        return self._Vt

    @property
    def components_(self) -> np.ndarray:
        self.checkFitted()
        return self._Vt

    @property
    def singularValues_(self) -> np.ndarray:
        self.checkFitted()
        return self._S

    @property
    def rank(self) -> int:
        self.checkFitted()
        return estimateRank(self._S)

    @property
    def nComponents(self) -> int:
        self.checkFitted()
        return len(self._S)

    @property
    @abstractmethod
    def supportsIncremental(self) -> bool:
        pass

    @property
    @abstractmethod
    def supportsSparse(self) -> bool:
        pass

    @property
    @abstractmethod
    def isApproximate(self) -> bool:
        pass

    def checkFitted(self):
        if not self._isFitted:
            raise ValueError("This SVD model is not fitted yet")

    def getInfo(self) -> Dict[str, Any]:
        info = {
            "name": self.name,
            "method": self.method.value,
            "_isFitted": self._isFitted,
            "supportsIncremental": self.supportsIncremental,
            "supportsSparse": self.supportsSparse,
            "isApproximate": self.isApproximate,
        }
        if self._isFitted:
            info.update({
                "rank": self.rank,
                "nComponents": self.nComponents,
                "explainedVariance": self.explainedVariance(),
                "explainedVarianceRatio": self.explainedVarianceRatio(),
                "cumulativeVarianceRatio": self.cumulativeVarianceRatio(),
                "shape": (self._U.shape[0], self._Vt.shape[1])
            })
        return  info

    def __repr__(self) -> str:
        status = "Fitted" if self._isFitted else "Not Fitted"
        return f"{self.name} ({status}), rank={self.rank},r={self.nComponents}" if status=="Fitted" else f"{self.name} ({status})"

class svdFactory:
    _registry: Dict[str, Type[svdBase]] = {}

    @classmethod
    def register(cls,method: svdMethod, svdClass: Type[svdBase]):
        cls._registry[method] = svdClass
        print(f"Registered SVD method {method.value} with class {svdClass.__name__}")

    @classmethod
    def create(cls,method: svdMethod,**kwargs) -> svdBase:
        return cls._registry[method](method,**kwargs)

    @classmethod
    def getMethodInfo(cls,method: svdMethod) -> Dict:
        if method not in cls._registry:
            raise ValueError(f"Method {method.value} is not registered")
        svdClass = cls._registry[method]
        return {
            'method': method.value,
            'class': svdClass.__name__,
            'module': svdClass.__module__,
        }

def registerSVD(method: svdMethod):
    def decorator(cls):
        svdFactory.register(method, cls)
        return cls
    return decorator