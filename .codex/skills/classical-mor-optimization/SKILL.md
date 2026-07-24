---
name: classical-mor-optimization
description: Classical MOR optimization workflow for this pyMOR-style repository. Use when Codex is asked to continue, review, or implement non-NN MOR algorithm optimization phases, including SVD, Lyapunov, low-rank solvers, operator compatibility, auto dispatch logic, regression tests, benchmark/result JSON or CSV artifacts, and short reports. Also use when the user asks to formalize or follow the repository's classical MOR optimization plan.
---

# Classical MOR Optimization

## Overview

Use this skill to continue the repository's non-NN classical MOR work as an evidence-driven optimization process. Each phase must evaluate the algorithm together with operator form, backend, data regime, and `auto` dispatch policy.

Primary local references:

- Read `OPTIMIZATION_PLAN.md` before changing optimization scope or phase order.
- Use existing report shape from `reports/svd/` as the artifact pattern.
- Use existing generator shape from `scripts/generate_svd_report.py` as the reporting script pattern.

## Scope

Work only on the classical MOR stack unless the user explicitly expands scope:

- `mor/algorithm/**`
- `mor/operators/**`
- `mor/models/**`
- `mor/reductors/**`
- `mor/solvers/**` when solver wrappers are directly affected
- `tests/algorithm/**`, `tests/operators/**`, `tests/models/**`, and reducer or low-rank regression tests
- `scripts/generate_*_report.py`
- `reports/<algorithm>/`

Avoid `mor/nn/**` during this workflow unless the user asks for NN work.

## Evidence Layers

Use distinct evidence layers. Do not collapse all of them into ordinary unit tests.

1. Regression tests: fast `pytest` cases that protect correctness, public contracts, expected failures, and dispatch rules.
2. Micro benchmark tests: deterministic pytest cases under `tests/benchmarks/` that measure bottom-layer algorithm timing and quality metrics without enforcing fragile runtime thresholds.
3. Operator compatibility tests: targeted cases for dense, sparse, low-rank, scaled, sum/composed, descriptor/generalized, transpose, LR, HR, singular, and ill-conditioned inputs.
4. Dispatch smoke tests: a small representative set proving `auto` follows the lower-layer evidence.
5. Characterization reports: benchmark-style scripts that generate `json`, `csv`, `md`, `tex`, and visualizations across a wider matrix.

Regression tests answer "does it still work?". Micro benchmarks answer "what is the bottom-layer algorithm cost?". Operator tests answer "which input contracts are supported?". Dispatch tests answer "does `auto` implement the selected policy?". Characterization reports answer "what is the complete evidence trail?".

## Layered Report Strategy

Reports should be built bottom-up and preserve the evidence at each layer:

1. Algorithm Layer: compare candidate algorithms directly on representative normal, LR, HR, pathological, and scale regimes. This layer decides which algorithm is reliable and fast enough for each data regime.
2. Operator Layer: run the selected or candidate algorithm through supported operator forms. This layer decides whether dense, sparse, low-rank, composed, scaled, and descriptor inputs satisfy the public contract.
3. Dispatch Layer: validate a small set of representative cases where `auto` must choose a default. This layer should not repeat the full algorithm matrix.
4. Integration Layer: upper-level models and reductors should call `auto` by default and only validate end-to-end accuracy, shape contracts, and failure behavior.

Once a lower layer has established a default and `auto` is updated, upper layers should not repeat every algorithm/backend/operator combination. They should verify the `auto` path unless the user is explicitly asking for a new characterization sweep.

## Test Reduction Rules

Keep the evidence chain strong while avoiding combinatorial test growth:

- Do not test every algorithm at every upper layer once `auto` is established.
- Algorithm Layer may compare all candidate variants on representative cases.
- Operator Layer should test only supported algorithms plus a few expected-failure contracts.
- Dispatch Layer should test boundary cases that can change `auto`, not every benchmark case.
- Integration Layer should call `auto` and verify end-to-end behavior only.
- If a full matrix is needed for investigation, put it in a report generator, not the default pytest suite.

## Combination Matrix

For every phase, evaluate the algorithm family across the relevant subset of this matrix.

Operator form:

- dense `matrixOperator`
- scipy sparse `matrixOperator`
- `lowRankOperator`
- `scaledOperator`
- `sumOperator`
- descriptor/generalized pairs through `A` and `E`

Backend:

- `scipy` as the primary classical backend
- `torch` where the dense path already supports it
- sparse paths should normally stay on `scipy` unless explicit support exists elsewhere

Data regime:

- LR: low-rank input, low-rank RHS, low-rank solution factor, rank truncation behavior
- HR: high-rank or near-full-rank input, rank growth, compression behavior, memory-sensitive paths
- tall, wide, square, and composed snapshot matrices for SVD phases
- stable, stiff, ill-conditioned, singular, near-singular, and descriptor systems for Lyapunov phases
- transpose and generalized paths when the algorithm exposes them

Record unsupported combinations intentionally. A clear `NotImplementedError`, validation error, or `xfail` is better than a silent weak fallback.

## Standard Benchmark Sources

Prefer public standard cases over ad hoc random matrices when building benchmarks.

- MATLAB `gallery` test matrices for local dense standards such as Hilbert, Grcar, Kahan, Poisson, and other classic numerical test matrices.
- Julia `MatrixDepot.jl` style matrices for local generated standards such as Poisson, Hilbert, and other catalogued test matrices.
- SuiteSparse Matrix Collection for optional external sparse benchmark matrices when local files are available.
- MOR benchmark collections such as rail, building, CD player, or other LTI examples when matrices are available locally.

Use random matrices only for regression isolation, fuzz-like shape coverage, or when no suitable standard case exists. When using an external benchmark file, record the source, matrix name, dimensions, sparsity, license/redistribution constraints if known, and local path. Do not require network access in normal pytest runs.

## Phase Workflow

For each optimization phase:

1. Inspect `git status --short`, `OPTIMIZATION_PLAN.md`, targeted implementation files, and current tests.
2. Define the phase-specific subset of the combination matrix before changing `auto`.
3. Reproduce current behavior with focused regression tests or characterization scripts.
4. Fix or optimize the targeted algorithm family without changing unrelated APIs.
5. Verify operator compatibility and backend data conversion through public operator methods.
6. Compare sibling variants at the Algorithm Layer on representative LR, HR, normal, pathological, and edge regimes.
7. Verify supported operator contracts at the Operator Layer.
8. Update `mor/algorithm/auto/*.py` only after lower-layer evidence supports the rule.
9. Add targeted regression and edge-case tests for implementation behavior, operator behavior, and dispatch decisions.
10. Add or update micro benchmark tests under `tests/benchmarks/` for the phase's main bottom-layer metrics.
11. Add or update report generators under `scripts/`.
12. Generate local report artifacts under `reports/<algorithm>/` and the cross-algorithm LaTeX summary.
13. Run targeted tests, benchmark smoke tests, then the classical regression suite.

## Phase Order

Follow this order unless the user changes it:

1. SVD family and SVD auto-selection.
2. Large-scale continuous Lyapunov and `lradi` auto-selection.
3. Dense/Torch continuous Lyapunov paths: `sign` and `bartelsStewart`.
4. Discrete sparse Lyapunov: `lrsmith`.
5. Cross-family auto-selection review.

Report artifacts should use algorithm names such as `svd`, `lradi`, `dense_lyapunov`, and `lrsmith`.

## Dispatch Rules

Treat `auto` selection as scenario selection, not a simple "fastest observed" rule.

- Prefer defaults that are correct, stable, covered by tests, and appropriate for backend, density, shape, and operator regime.
- Use characterization timing as evidence, but do not promote a variant solely from one small local timing case.
- Prefer sparse-aware algorithms for sparse/operator-scale scenarios even when a dense reference solve is faster on tiny local cases.
- Prefer dense direct baselines for small dense cases when they are more robust and sufficiently covered.
- Add explicit dispatch tests for every new `auto` decision.
- Make unsupported combinations fail clearly instead of silently selecting a weak fallback.
- Include the dispatch rationale in the phase report.

## Auto Decision Contract

Every new or changed `auto` rule should have a short, structured evidence trail:

- `scenario`: backend, operator form, shape regime, rank/input count, continuous/discrete mode, and special risk tags.
- `default`: selected variant and recommended options when applicable.
- `evidence`: algorithm-layer benchmark summary, operator compatibility result, and dispatch smoke test name.
- `risk`: known limits such as `near_unit_slow_convergence`, `monitor_residual_estimator_floor`, `unsupported_backend`, or `reject_unstable_discrete_operator`.
- `fallback`: expected exception, unsupported status, or alternative variant when the default is unsafe.

Prefer returning or recording option hints such as `maxIter`, `tol`, `acceptanceResidual`, `risk`, and `stability` when the selected algorithm needs scenario-specific tuning. Do not make upper-level callers duplicate the low-level decision logic.

## Risk Labels

Use stable machine-readable risk labels in CSV/JSON and keep their meaning consistent across reports:

- `keep`: current default is acceptable.
- `monitor_dispatch_speed`: default is correct but slower than the fastest observed option.
- `review_dispatch_speed`: default is correct but materially slower and should be reviewed before freezing the rule.
- `fix_auto_correctness`: `auto` selected a failing or unsupported path.
- `near_unit_slow_convergence`: discrete Lyapunov spectral radius is close to one and iteration budget may dominate.
- `monitor_residual_estimator_floor`: large low-rank residual estimate is near numerical/estimator floor.
- `increase_iterations_or_shifts`: LRADI-style iteration or shift budget is insufficient.
- `reject_unstable_discrete_operator`: discrete operator is not stable enough for LRSmith.
- `expected_unsupported`: unsupported combination is intentionally rejected.

Add new labels only when the existing vocabulary cannot express the decision.

## Operator Contracts

Preserve the operator/backend contracts already stabilized in this repository.

- Use `toBackendData()` or public operator methods instead of direct `.data` access in algorithms.
- Keep low-rank solver outputs compatible with `lowRankOperator` and downstream reductors.
- Test composed operators, not only raw dense matrices, when changing an algorithm that claims operator support.
- Preserve backend support for SciPy and Torch where the current path supports both.
- Keep SISO and descriptor/generalized model shape behavior explicit in tests when touched.
- Preserve scipy sparse matrices as sparse backend data; do not accidentally convert them to object ndarrays.
- Preserve 2D RHS/result shape for single-input solves.

## Code Style

Avoid adding comments and docstrings by default. Prefer clear names, small helpers, and direct tests over explanatory comments. Add a comment or docstring only when the logic is non-obvious, documents an intentional numerical tradeoff, or records a boundary condition that future maintainers are likely to misread.

## Report Artifacts

Every optimization phase should produce:

- `scripts/generate_<algorithm>_report.py`
- optional cross-family scripts such as `scripts/generate_lyapunov_dispatch_report.py`
- `reports/<algorithm>/<algorithm>_results.json`
- `reports/<algorithm>/<algorithm>_results.csv`
- `reports/<algorithm>/<ALGORITHM>_REPORT.md`
- `reports/<algorithm>/*.svg` or `reports/<algorithm>/*.png` visualizations for the main comparison metrics

Each row or case should include enough information to compare variants:

- scenario name
- backend
- operator form and data regime
- selected auto algorithm
- candidate algorithm results
- absolute and relative reconstruction or residual error
- reference error when a reliable baseline exists
- rank, iteration count, compression count, or other algorithm-specific diagnostics
- rough runtime measurement when practical
- pass/fail or acceptance status
- failure reason for unsupported combinations

The markdown report should be short and should summarize why the `auto` selection is appropriate even when another variant is faster in a small case. Link or name the generated visualization files from the markdown report.

Generate a cross-algorithm LaTeX summary after all relevant reports are current:

- `scripts/generate_classical_mor_latex_report.py`
- `reports/classical_mor/classical_mor_layered_report.tex`

The LaTeX report should be organized by Algorithm Layer, Operator Layer, Dispatch Layer, and Integration Layer. It should summarize results from existing JSON/CSV outputs instead of replacing them.

Refresh reports in dependency order:

1. Per-algorithm reports such as SVD, LRADI, dense Lyapunov, and LRSmith.
2. Cross-family dispatch reports such as Lyapunov dispatch.
3. Cross-algorithm LaTeX summary.

Markdown reports should be written as UTF-8 with BOM (`utf-8-sig`) so Windows tools display Chinese text correctly. JSON and CSV should remain plain UTF-8 for machine readers.

If a PDF is requested and a TeX engine is available, compile the LaTeX report with XeLaTeX or another CJK-capable engine. Do not require PDF generation for normal report refreshes.

Do not commit `reports/` artifacts unless the user explicitly asks. Commit the generator scripts and tests so the results can be reproduced.

## Benchmark Tests

Keep pytest benchmark tests deterministic and lightweight. They should:

- live under `tests/benchmarks/`
- use `@pytest.mark.benchmark`
- prefer standard generated cases from `tests/benchmarks/standard_cases.py`
- collect elapsed time, accuracy/residual, rank, iteration count, selected `auto` variant, backend, operator form, and data regime
- assert numerical correctness and metric schema, not exact runtime superiority
- avoid writing report files during pytest
- use small matrices so the default classical test suite stays fast

Use benchmark tests as the first layer of algorithm comparison, not as a full dispatch matrix. Promote broader, slower, pathological, or externally downloaded standard cases into `scripts/generate_*_report.py` once the metric shape is stable.

## Test Commands

Prefer the repository virtual environment when available:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/algorithm tests/operators tests/models tests/test_reductors.py tests/test_lrsmith_lowrank.py
```

For focused phase work, run the smallest relevant slice first:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/algorithm/test_svd.py
.\.venv\Scripts\python.exe -m pytest tests/algorithm/test_lyapunov.py tests/test_reductors.py tests/test_lrsmith_lowrank.py
.\.venv\Scripts\python.exe -m pytest tests/benchmarks
```

Regenerate reports with the same interpreter:

```powershell
.\.venv\Scripts\python.exe scripts/generate_svd_report.py
.\.venv\Scripts\python.exe scripts/generate_lyapunov_dispatch_report.py
.\.venv\Scripts\python.exe scripts/generate_lradi_report.py
.\.venv\Scripts\python.exe scripts/generate_dense_lyapunov_report.py
.\.venv\Scripts\python.exe scripts/generate_lrsmith_report.py
.\.venv\Scripts\python.exe scripts/generate_classical_mor_latex_report.py
```

If `.venv` is unavailable, fall back to `python -m pytest` and state that fallback in the final response.

## Commit Discipline

Keep each phase self-contained. Before committing, include implementation changes, tests, scripts, and report artifacts together. Do not include unrelated worktree cleanup or unrelated user edits in the same commit.
