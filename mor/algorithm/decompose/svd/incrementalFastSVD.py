from mor.algorithm.registry import registerAlgorithm
from .incrementalSVD import incrementalSVD


@registerAlgorithm('svd', 'incrementalFast')
class incrementalFastSVD(incrementalSVD):
    pass
