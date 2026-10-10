# mor

A pyMOR-style model order reduction toolkit combining classical reduction
methods (POD, balanced truncation, DMD, Gramian-based solvers, SVD variants) with
neural-network reductors and a mesh-based geometry engine.

The distribution name is `mor`; the import name is also `mor`.

## Install

```bash
pip install -e .
pip install -e ".[dev]"
pip install -e ".[all]"
```

Core runtime requires Python `>=3.10` plus `numpy` and `scipy`.
Importing `mor` and listing backends do not initialize PyTorch. Both backends are always registered; Torch loads when requested with
`backendRegistry.get("torch")`. Feature distributions install their declared
dependencies. Runtime code does not probe for missing packages or silently omit
features; missing dependencies surface when the feature is executed.

## Quick Start

```python
import mor
import numpy as np

print(mor.__version__)
print(mor.backendRegistry.list())
print(mor.matrixOperator)

snapshots = mor.matrixOperator(
    np.array([[1.0, 0.9, 0.81], [1.0, 0.8, 0.64]]),
    backendName="scipy",
)
dynamics = mor.dmdReductor("scipy").reduce(snapshots)
future = dynamics.predict(steps=5)

pod_dynamics = mor.podDmdReductor("scipy", centerSnapshots=True).reduce(
    snapshots,
    podRank=2,
    dmdRank=2,
)
future_from_pod = pod_dynamics.predict(steps=5)

import mor.nn
```

## Tests

```bash
pip install -e ".[dev,deployment]"
python -m pytest tests/ -q
```

For ONNX export and inference checks, install `.[dev,deployment]` and run
`python -m pytest tests/test_workflow_onnx.py`. CI runs classical and full suites with each suite's required dependencies.
Use `.[reports]` for performance-report dependencies and `.[sklearn]` for
randomized SVD and classical geometry classifiers, including XGBoost.

Sparse POD with a rank smaller than `min(snapshots.shape)` uses a truncated
sparse SVD. Full-spectrum or energy-tolerance decomposition may require a
dense conversion, limited to 64 MiB per decomposition input. For larger inputs,
choose a truncated rank or explicitly provide dense data with an appropriate
memory budget. Weighted POD whitens snapshots using diagonal square roots or
Cholesky factors; sparse inputs and general sparse products follow the same
bounded dense-conversion policy.

## Layout

```text
mor/
  algorithm/    SVD / DMD / linear / Lyapunov algorithms + auto-dispatch
  backends/     scipy / torch backend abstraction
  assembly/     element kernels and global sparse thermal assembly
  operators/    matrix / low-rank / sum / scaled operators
  models/       system models
  solvers/      POD / Lyapunov solvers
  reductors/    POD / DMD / balanced truncation reductors
  nn/           neural-network reductors and geometry engine
  workflow/     task execution, validation, model storage and export
  runtime.py    runtime configuration and sessions
  shonta.py     shonTA integration entry points
tests/          in-repo regression and unit tests
scripts/        runnable scripts and shared report utilities
docs/           maintained interfaces, data formats and usage documentation
applicationTests/  application example scripts and minimal input fixtures
```

Documentation starts at [docs/README.md](docs/README.md). Script categories and
entry points are listed in [scripts/README.md](scripts/README.md), and report
and data conventions are described in [docs/PROJECT_LAYOUT.md](docs/PROJECT_LAYOUT.md).

[OPTIMIZATION_PLAN.md](OPTIMIZATION_PLAN.md) remains the root entry point for
the classical MOR optimization workflow used by the repository skill.

The checkout keeps source code, tests, documentation and minimal example inputs.
Historical checkpoints, reports, archives and generated datasets have been
removed. Experiment scripts create output directories as needed; use
`reports/<topic>/<run>/` for results and `temp/experiments/<run>/` for scratch data,
or explicitly choose a directory outside the checkout. Keep generated artifacts
out of the repository root. Commands in documentation assume the repository
root as the working directory. External benchmark data is optional and can be
downloaded again with `scripts/fetch_mor_benchmark_data.py`.
