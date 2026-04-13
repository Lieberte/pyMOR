from pathlib import Path
import meshio
from .meshioCommon import meshIoToIr
from ..meshIr import meshIr

def inpToIr(
    path: str | Path,
    *,
    meshioReadKwargs: dict | None = None,
    **kwargs,
) -> meshIr:
    resolved = Path(path).expanduser()
    opts: dict = dict(meshioReadKwargs or {})
    if opts.get('file_format') is None and resolved.suffix.lower() == '.inp':
        opts['file_format'] = 'abaqus'
    mesh = meshio.read(resolved, **opts)
    return meshIoToIr(mesh, **kwargs)
