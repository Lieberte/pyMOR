from pathlib import Path
import meshio
from .meshioCommon import meshIoToIr
from ..meshIr import meshIr

def inpToIr(path: str | Path, **kwargs) -> meshIr:
    return meshIoToIr(meshio.read(path), **kwargs)
