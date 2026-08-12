from __future__ import annotations

import argparse
import shutil
import tempfile
import urllib.request
from pathlib import Path

from mor_application_cases import SLICOT_CASES, defaultDataDirectory, fileSha256


def fetchCase(case, outputDirectory: Path, overwrite: bool = False) -> Path:
    outputDirectory.mkdir(parents=True, exist_ok=True)
    destination = outputDirectory / case.filename
    if destination.exists() and not overwrite:
        actualHash = fileSha256(destination)
        if actualHash == case.sha256:
            print(f"keep {case.filename}: checksum verified")
            return destination
        raise ValueError(
            f"{destination} already exists with checksum {actualHash}; use --overwrite to replace it"
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mat", dir=outputDirectory) as temporary:
        temporaryPath = Path(temporary.name)
    try:
        request = urllib.request.Request(case.url, headers={"User-Agent": "pyMOR-application-benchmarks"})
        with urllib.request.urlopen(request, timeout=60) as response, temporaryPath.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        actualHash = fileSha256(temporaryPath)
        if actualHash != case.sha256:
            raise ValueError(f"Checksum mismatch for {case.filename}: expected {case.sha256}, got {actualHash}")
        temporaryPath.replace(destination)
        print(f"fetched {case.filename}: {destination}")
        return destination
    finally:
        if temporaryPath.exists():
            temporaryPath.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch checksum-verified SLICOT MOR application benchmarks.")
    parser.add_argument("--output", type=Path, default=defaultDataDirectory())
    parser.add_argument("--case", action="append", choices=[case.key for case in SLICOT_CASES])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    selected = [case for case in SLICOT_CASES if not args.case or case.key in args.case]
    for case in selected:
        fetchCase(case, args.output, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
