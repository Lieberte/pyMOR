# mor

A pyMOR-style **model order reduction (MOR)** toolkit combining classical
reduction methods (POD, balanced truncation, Gramian-based solvers, SVD
variants) with neural-network reductors (autoencoders / VAE / CAE, RNN/LSTM/GRU/
SSM/Transformer dynamics learning, PINN) and a mesh-based geometry engine
for `.inp` / `.msh` conversion and sampling.

> Local/research build only — not published to PyPI. The distribution name is
> `mor`; the import name is `mor`.

## Install (editable)

```bash
pip install -e .            # classical core (numpy + scipy)
pip install -e ".[all]"     # + torch, meshio, scikit-learn
pip install -e ".[dev]"     # + pytest, meshio, matplotlib (for running tests)
```

Requires Python **>= 3.10** (the codebase uses `X | None` type hints).

## Quick start

```python
import mor
print(mor.__version__)            # 0.1.0
print(mor.solverRegistry)         # classical-MOR registries & classes are public
print(mor.matrixOperator)

import mor.nn                     # neural-network reductors (pulled on demand)
```

## Tests

```bash
pytest tests/ -q
```

`tests/nn/data/geometry/` holds pure unit tests for the geometry utilities and
samplers.

## Layout

```
mor/
  algorithm/    SVD / linear / Lyapunov algorithms + auto-dispatch
  backends/     scipy / torch backend abstraction
  operators/    algebraic operator wrappers (matrix, low-rank, sum, scaled)
  models/       system models (LTI, ...)
  solvers/      Lyapunov / POD solvers + registry
  reductors/    POD, balanced-truncation reductors
  nn/           neural-network reductors, trainers, geometry engine, PINN
tests/          in-repo test suite
```
