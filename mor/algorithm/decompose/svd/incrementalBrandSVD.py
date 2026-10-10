from mor.algorithm.registry import registerAlgorithm
from .incrementalSVD import incrementalSVD


@registerAlgorithm('svd', 'incrementalBrand')
class incrementalBrandSVD(incrementalSVD):
    pass
