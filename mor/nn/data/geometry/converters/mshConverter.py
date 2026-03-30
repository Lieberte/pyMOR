from __future__ import annotations
from pathlib import Path
import meshio
from .meshioCommon import meshIoToIr

def mshToIr(path: str | Path, **kwargs) -> meshIr:
    return meshIoToIr(meshio.read(path), **kwargs)
