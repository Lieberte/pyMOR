from mor.algorithm.registry import registerAlgorithm
from .incrementalSVD import incrementalSVD


@registerAlgorithm('svd', 'incrementalStandard')
class incrementalStandardSVD(incrementalSVD):
    pass
