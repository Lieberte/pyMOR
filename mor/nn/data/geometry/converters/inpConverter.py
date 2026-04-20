from pathlib import Path
from .meshioCommon import meshFileToIr
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
    return meshFileToIr(resolved, meshioReadKwargs=opts, **kwargs)
