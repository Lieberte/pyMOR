class nnRegistry:
    _modules = {}

    @classmethod
    def register(cls, category: str, name: str, moduleClass):
        if category not in cls._modules:
            cls._modules[category] = {}
        cls._modules[category][name] = moduleClass

    @classmethod
    def get(cls, category: str, name: str):
        if category not in cls._modules or name not in cls._modules[category]:
            raise ValueError(f"NN module '{name}' not registered in '{category}'.")
        return cls._modules[category][name]

    @classmethod
    def list(cls, category: str) -> list:
        return list(cls._modules.get(category, {}).keys())
