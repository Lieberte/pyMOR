from pathlib import Path
from .meshioCommon import meshFileToIr
from ..meshIr import meshIr

def mshToIr(path: str | Path, **kwargs) -> meshIr:
    return meshFileToIr(path, **kwargs)
