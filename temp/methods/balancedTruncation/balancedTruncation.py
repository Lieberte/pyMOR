import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple
import warnings


class balancedTruncation(ABC):
    def __init__(self):
        self.A: np.ndarray | None = None
        self.B: np.ndarray | None = None
        self.C: np.ndarray | None = None
        self.D: np.ndarray | None = None

        self.isDiscrete: bool = False
        self.ts: float = -1.0

        self.n: int = 0
        self.m: int = 0
        self.p: int = 0
        self.r: int = 0

        self.Ar: np.ndarray | None = None
        self.Br: np.ndarray | None = None
        self.Cr: np.ndarray | None = None
        self.Dr: np.ndarray | None = None

        self.hankelSv: np.ndarray | None = None

        self.isReduced: bool = False

        self.tol = {
            'lyapunovConvergence': 1e-12,
            'eigenvalueThreshold': 1e-10,
            'singularValueThreshold': 1e-12,
            'maxLyapunovIterations': 100
        }

    @abstractmethod
    def reduce(self, r: int) -> bool:
        pass

    @abstractmethod
    def reduceByEnergy(self, energyThreshold: float = 0.99) -> bool:
        pass

    @abstractmethod
    def getReducedA(self) -> np.ndarray | None:
        pass

    @abstractmethod
    def getReducedB(self) -> np.ndarray | None:
        pass

    @abstractmethod
    def getReducedC(self) -> np.ndarray | None:
        pass

    @abstractmethod
    def getReducedD(self) -> np.ndarray | None:
        pass

    @abstractmethod
    def getHankelSingularValues(self) -> np.ndarray | None:
        pass

    def getReducedDimension(self) -> int:
        return self.r

    def isSystemReduced(self) -> bool:
        return self.isReduced

    def checkSystemStability(self) -> bool:
        if self.A is None:
            return False

        eigvals = np.linalg.eigvals(self.A)

        if self.isDiscrete:
            return np.all(np.abs(eigvals) < 1.0 + self.tol['eigenvalueThreshold'])
        else:
            return np.all(np.real(eigvals) < self.tol['eigenvalueThreshold'])

    def computeSystemConditionNumber(self) -> float:
        if self.A is None:
            return np.inf

        try:
            cond = np.linalg.cond(self.A)
            return cond
        except:
            return np.inf

    def _initializeSystem(self, A: np.ndarray, B: np.ndarray,
                          C: np.ndarray, D: np.ndarray | None = None,
                          ts: float = -1.0) -> None:
        if not self._validateSystemDimensions(A, B, C, D):
            raise ValueError("System dimensions are inconsistent")
        self.A = A.copy()
        self.B = B.copy()
        self.C = C.copy()
        self.D = D.copy() if D is not None else np.zeros((C.shape[0], B.shape[1]))

        self.n = A.shape[0]
        self.m = B.shape[1]
        self.p = C.shape[0]

        self.ts = ts
        self.isDiscrete = (ts > 0)

        self._logInfo(f"System initialized: n={self.n}, m={self.m}, p={self.p}, "
                      f"discrete={self.isDiscrete}")

    def _validateSystemDimensions(self, A: np.ndarray, B: np.ndarray,
                                   C: np.ndarray, D: np.ndarray | None) -> bool:
        if A.shape[0] != A.shape[1]:
            self._logError("A matrix must be square")
            return False

        n = A.shape[0]

        if B.shape[0] != n:
            self._logError(f"B matrix rows ({B.shape[0]}) must equal A dimension ({n})")
            return False

        if C.shape[1] != n:
            self._logError(f"C matrix columns ({C.shape[1]}) must equal A dimension ({n})")
            return False

        if D is not None:
            if D.shape[0] != C.shape[0] or D.shape[1] != B.shape[1]:
                self._logError(f"D matrix dimension ({D.shape}) incompatible with C and B")
                return False

        return True

    @abstractmethod
    def _solveLyapunovContinuous(self, A: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, bool]:
        pass

    @abstractmethod
    def _solveLyapunovDiscrete(self, A: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, bool]:
        pass

    def _solveLyapunov(self, A: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, bool]:
        if self.isDiscrete:
            return self._solveLyapunovDiscrete(A, Q)
        else:
            return self._solveLyapunovContinuous(A, Q)

    def _computeSymmetricProduct(self, M1: np.ndarray, M2: np.ndarray,
                                  transpose2: bool = True) -> np.ndarray:
        if transpose2:
            result = M1 @ M2.T
        else:
            result = M1 @ M2

        return 0.5 * (result + result.T)

    def _computeFrobeniusNorm(self, M: np.ndarray) -> float:
        return np.linalg.norm(M, 'fro')

    def _logInfo(self, message: str) -> None:
        print(f"[INFO] {message}")

    def _logWarning(self, message: str) -> None:
        warnings.warn(message)

    def _logError(self, message: str) -> None:
        print(f"[ERROR] {message}")