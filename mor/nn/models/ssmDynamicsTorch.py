from mor.nn.models.mambaDynamicsTorch import mambaDynamicsTorch
from mor.nn.registry import nnRegistry


class ssmDynamicsTorch(mambaDynamicsTorch):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.modelName = 'ssmDynamicsTorch'


nnRegistry.register('models', 'ssmDynamicsTorch', ssmDynamicsTorch)
