from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import h5py


def latestResult(archive: ZipFile) -> str:
    candidates = []
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        if info.is_dir() or not info.filename.startswith('results/') or path.suffix.lower() != '.h5':
            continue
        try:
            candidates.append((float(path.stem), info.filename))
        except ValueError:
            continue
    if not candidates:
        raise FileNotFoundError('STA archive has no numeric H5 result')
    return max(candidates)[1]


def main() -> None:
    parser = argparse.ArgumentParser(description='Inspect the latest H5 result inside a motor STA.')
    parser.add_argument('source', type=Path)
    parser.add_argument('--limit', type=int, default=300)
    parser.add_argument('--contains')
    arguments = parser.parse_args()

    with ZipFile(arguments.source) as archive:
        member = latestResult(archive)
        info = archive.getinfo(member)
        with tempfile.TemporaryDirectory(prefix='motor-sta-h5-') as directory:
            extracted = Path(directory) / 'result.h5'
            with archive.open(info) as source, extracted.open('wb') as target:
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
            datasets = []
            with h5py.File(extracted, 'r') as file:
                def visitor(name, value):
                    if (
                        isinstance(value, h5py.Dataset)
                        and len(datasets) < arguments.limit
                        and (arguments.contains is None or arguments.contains.lower() in name.lower())
                    ):
                        datasets.append({'path': name, 'shape': value.shape, 'dtype': str(value.dtype)})
                file.visititems(visitor)
    print(json.dumps({
        'source': str(arguments.source),
        'resultMember': member,
        'resultSizeBytes': info.file_size,
        'datasets': datasets,
    }, indent=2))


if __name__ == '__main__':
    main()
