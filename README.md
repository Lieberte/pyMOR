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

import mor.nn
```

## Tests

```bash
pytest tests/ -q
```

## Layout

```text
mor/
  algorithm/    SVD / DMD / linear / Lyapunov algorithms + auto-dispatch
  backends/     scipy / torch backend abstraction
  operators/    matrix / low-rank / sum / scaled operators
  models/       system models
  solvers/      POD / Lyapunov solvers
  reductors/    POD / DMD / balanced truncation reductors
  nn/           neural-network reductors and geometry engine
tests/          in-repo regression and unit tests
```
