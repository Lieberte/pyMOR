from typing import Optional, Any
from .modelsBase import modelBase
from mor.operators.operatorsBase import operatorBase

class ltiModel(modelBase):
    def __init__(self, A: operatorBase, B: operatorBase, C: operatorBase, 
                 D: operatorBase | None = None, E: operatorBase | None = None, 
                 backendName: str | None = None):
        super().__init__(backendName or A.backendName)
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.E = E
        self._validate()

    def _validate(self):
        n = self.A.shape[0]
        if self.A.shape[1] != n: raise ValueError("A must be square")
        if self.B.shape[0] != n: raise ValueError("B must have same rows as A")
        if self.C.shape[1] != n: raise ValueError("C must have same columns as A")
        if self.E is not None and self.E.shape != (n, n): raise ValueError("E must have same shape as A")

    @property
    def order(self) -> int:
        return self.A.shape[0]

    @property
    def inputDim(self) -> int:
        return self.B.shape[1]

    @property
    def outputDim(self) -> int:
        return self.C.shape[0]

    def toNumpy(self) -> dict:
        return {
            'A': self.A.toNumpy(),
            'B': self.B.toNumpy(),
            'C': self.C.toNumpy(),
            'D': self.D.toNumpy() if self.D else None,
            'E': self.E.toNumpy() if self.E else None
        }
