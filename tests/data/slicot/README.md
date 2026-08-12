# SLICOT model-reduction benchmark data

The application tests in `tests/application/` support the public
`SLICOT/Benchmark-ModelReduction` collection:

<https://github.com/SLICOT/Benchmark-ModelReduction>

The upstream repository describes the files as real-world benchmark examples,
but it does not currently expose a repository-level license through GitHub.
For that reason the MAT files are not committed here. Fetch them explicitly:

```powershell
.\.venv\Scripts\python.exe scripts\fetch_mor_benchmark_data.py
```

The downloader verifies SHA-256 checksums. Tests skip external cases when the
files are absent; the generated two-dimensional heat-equation case always runs.

Included case definitions:

- continuous thin-rod heat equation
- discrete heat-equation benchmark
- classical CD-player model
- building-motion model
- clamped-beam model
- transmission-line descriptor model
- PEEC descriptor circuit model
