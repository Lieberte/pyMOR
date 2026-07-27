from typing import Any
from .modelsBase import modelBase
from mor.operators.operatorsBase import operatorBase

class ltiModel(modelBase):
    def __init__(self, A: operatorBase, B: operatorBase, C: operatorBase, 
                 D: operatorBase | None = None, E: operatorBase | None = None, 
                 backendName: str | None = None, isContinuous: bool = True):
        if not isinstance(A, operatorBase):
            raise TypeError("A must be an operatorBase")
        super().__init__(backendName or A.backendName)
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.E = E
        self.isContinuous = isContinuous
        self._validate()

    def _validate(self):
        operators = {'A': self.A, 'B': self.B, 'C': self.C, 'D': self.D, 'E': self.E}
        for name, operator in operators.items():
            if operator is not None and not isinstance(operator, operatorBase):
                raise TypeError(f"{name} must be an operatorBase")
        if not isinstance(self.isContinuous, bool):
            raise TypeError("isContinuous must be a bool")
        if len(self.A.shape) != 2:
            raise ValueError("A must be a 2D square operator")
        n = self.A.shape[0]
        if self.A.shape[1] != n: raise ValueError("A must be square")
        if len(self.B.shape) not in (1, 2):
            raise ValueError("B must be a vector or a 2D operator")
        if self.B.shape[0] != n: raise ValueError("B must have same rows as A")
        if len(self.C.shape) != 2:
            raise ValueError("C must be a 2D operator")
        if self.C.shape[1] != n: raise ValueError("C must have same columns as A")
        if self.E is not None and self.E.shape != (n, n): raise ValueError("E must have same shape as A")
        expectedDShape = (self.outputDim, self.inputDim)
        if self.D is not None and self.D.shape != expectedDShape:
            raise ValueError(f"D must have shape {expectedDShape}")
        for name, operator in operators.items():
            if operator is not None and operator.backendName != self.backendName:
                raise ValueError(f"{name} backend {operator.backendName} does not match model backend {self.backendName}")

    @property
    def order(self) -> int:
        return self.A.shape[0]

    @property
    def inputDim(self) -> int:
        return self.B.shape[1] if len(self.B.shape) > 1 else 1

    @property
    def outputDim(self) -> int:
        return self.C.shape[0]

    @property
    def timeDomain(self) -> str:
        return 'continuous' if self.isContinuous else 'discrete'

    def toNumpy(self) -> dict:
        return {
            'A': self.A.toNumpy(),
            'B': self.B.toNumpy(),
            'C': self.C.toNumpy(),
            'D': self.D.toNumpy() if self.D is not None else None,
            'E': self.E.toNumpy() if self.E is not None else None
        }
