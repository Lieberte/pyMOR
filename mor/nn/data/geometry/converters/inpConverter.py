from __future__ import annotations
from pathlib import Path
import meshio
from .meshioCommon import meshIoToIr

def inpToIr(path: str | Path, **kwargs) -> meshIr:
    return meshIoToIr(meshio.read(path), **kwargs)
