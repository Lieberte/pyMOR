from typing import Any
from .svd import svdBase
from mor.algorithm.registry import registerAlgorithm

@registerAlgorithm('svd', 'static')
class staticSVD(svdBase):

    def decompose(self, xOperator, rank: int | None = None, fullMatrices: bool = False) -> tuple[Any, Any, Any]:
        return xOperator.svd(fullMatrices=fullMatrices, rank=rank)