# Classical MOR Optimization Plan

This document defines the execution order for targeted optimization of the
non-NN classical MOR stack. The current goal is to improve algorithm quality
without regressing the recently stabilized contracts and tests.

The optimization logic is intentionally coupled across three layers:

1. algorithm implementation
2. operator compatibility
3. auto-selection policy

We do not optimize these independently. Each phase should leave the targeted
algorithm family correct on supported operator inputs and place it sensibly in
the relevant `auto` dispatch path.

## Principles

1. Preserve correctness first.
2. Optimize one algorithm family at a time, but include its operator and
   dispatch behavior in the same phase.
3. Add or extend regression tests before or alongside each optimization.
4. Where optimization changes asymptotic behavior or numerical tradeoffs,
   add focused benchmarks or measurable acceptance checks.
5. Do not broaden scope into `mor/nn/**` during this plan.
6. `auto` should choose the most appropriate default for a scenario, not
   simply the fastest implementation in isolation.

## Optimization Logic

For every targeted family, optimization is done in this order:

1. Make the algorithm implementation correct and numerically defensible.
2. Verify it works across the operator shapes/compositions it claims to support.
3. Compare it against sibling variants for representative scenarios.
4. Only then adjust the `auto` selection logic.

This means:

- We do not promote a variant into `auto` just because it benchmarks fastest on
  one case.
- We prefer defaults that are stable, well-covered, and appropriate for the
  matrix/operator/backend regime.
- Performance-driven dispatch changes require representative evidence, not just
  local micro-bench impressions.

## Execution Order

### Phase 1: SVD Family and SVD Auto-Selection

Targets:
- `mor/algorithm/decompose/svd/economicSVD.py`
- `mor/algorithm/decompose/svd/dualSVD.py`
- `mor/algorithm/decompose/svd/incrementalFastSVD.py`
- `mor/algorithm/decompose/svd/incrementalBrandSVD.py`
- `mor/algorithm/decompose/svd/incrementalStandardSVD.py`
- `mor/algorithm/auto/svd.py`
- SVD-related operator contract tests

Why first:
- These algorithms currently prioritize correctness over their intended
  incremental/efficient behavior.
- Their scope is self-contained and already has contract tests.
- They are a good place to reintroduce algorithm-specific optimization under
  strong regression coverage.

Deliverables:
- Clarify intended semantics for each variant.
- Optimize update logic beyond full dense recomposition where safe.
- Verify behavior on composed operators, not just raw dense matrices.
- Add multi-batch reconstruction tests and, if useful, lightweight timing checks.
- Refine SVD `auto` defaults based on scenario fit, stability, and measured behavior.

Exit criteria:
- Existing SVD tests pass.
- New multi-batch incremental tests pass.
- Each variant has a documented behavioral difference.
- `auto` selection is explicitly tested for representative SVD scenarios.

### Phase 2: Large-Scale Continuous Lyapunov and Lyapunov Auto-Selection (`lradi`)

Targets:
- `mor/algorithm/lyapunov/lradi.py`
- `mor/algorithm/auto/lyapunov.py`
- related linear solve / shift-selection paths
- operator compatibility along the sparse continuous Lyapunov route

Why second:
- This is the main sparse continuous Lyapunov workhorse.
- Performance and robustness here matter directly for larger MOR workflows.

Focus areas:
- Shift initialization and refresh strategy
- Compression cadence
- Avoidable dense intermediates
- Numerical robustness on stiff systems

Deliverables:
- Better shift-selection strategy or clearer heuristic control
- Regression tests for representative sparse/operator cases
- Optional benchmark harness for iteration count / runtime comparison
- Clear rule for when `auto` should prefer `lradi`

Exit criteria:
- Existing Lyapunov and reducer tests pass.
- No regression in low-rank output contract.
- Measurable improvement in at least one representative sparse case.
- Dispatch behavior for sparse continuous cases is explicit and test-covered.

### Phase 3: Dense/Torch Continuous Lyapunov Paths (`sign`, `bartelsStewart`)

Targets:
- `mor/algorithm/lyapunov/sign.py`
- `mor/algorithm/lyapunov/bartelsStewart.py`
- dense continuous dispatch behavior in `mor/algorithm/auto/lyapunov.py`

Why third:
- These dense continuous paths are what `auto` exposes after sparse routes are decided.
- `sign` is numerically sensitive and already contains fallback logic.
- `bartelsStewart` remains the baseline dense scipy route.

Focus areas:
- Scaling strategy
- Fallback thresholding
- Residual-based acceptance criteria
- Transposed and generalized cases
- Correct default dispatch per backend and matrix regime

Deliverables:
- Better-conditioned iteration or clearer stopping logic
- Extra tests for transposed/generalized usage and fallback behavior
- Explicit comparison of dense-solver default choices

Exit criteria:
- Existing sign-related dispatch paths remain green.
- Residual acceptance behavior is tested and explicit.
- Dense continuous `auto` logic is intentional and covered.

### Phase 4: Discrete Sparse Lyapunov (`lrsmith`)

Targets:
- `mor/algorithm/lyapunov/lrsmith.py`
- discrete dispatch behavior in `mor/algorithm/auto/lyapunov.py`

Why fourth:
- The contract and descriptor behavior are now in good shape.
- Next improvements should be algorithmic rather than structural.

Focus areas:
- Iteration efficiency
- Rank truncation policy
- Residual estimation quality
- Generalized/discrete descriptor path robustness

Deliverables:
- Targeted tests for descriptor and transpose paths
- Improved rank-control or iteration behavior where justified
- Clear discrete sparse dispatch coverage

Exit criteria:
- Existing `tests/test_lrsmith_lowrank.py` stays green.
- Any algorithmic change is covered by a concrete regression case.
- Discrete sparse selection behavior is explicit and test-covered.

### Phase 5: Cross-Family Auto-Selection Review

Targets:
- `mor/algorithm/auto/*.py`
- solver/algorithm dispatch selection logic

Why last:
- Final cross-family cleanup should happen after the underlying algorithms are stable.
- Earlier phases already update local dispatch choices; this phase is for consistency.

Focus areas:
- Backend-aware selection
- Dense vs sparse heuristics
- Continuous vs discrete route quality
- Better failure messages for unsupported combinations
- Ensure local per-family decisions compose coherently across the full stack

Deliverables:
- Refined selection heuristics
- Dispatch tests for representative combinations

Exit criteria:
- Dispatch behavior is explicit and test-covered.
- Unsupported cases fail clearly and intentionally.

## Shared Work for Every Phase

For each phase:

1. Reproduce current behavior with focused tests.
2. Improve the targeted algorithm(s).
3. Verify supported operator compatibility for that family.
4. Reassess and, if justified, update the relevant `auto` default logic.
5. Run targeted tests for that phase.
6. Run the classical MOR regression suite:
   - `tests/algorithm`
   - `tests/operators`
   - `tests/models`
   - `tests/test_reductors.py`
   - `tests/test_lrsmith_lowrank.py`
7. Commit phase work separately if the change is self-contained.

## Non-Goals

- No neural-network (`mor/nn/**`) optimization in this plan.
- No packaging/publishing expansion beyond current internal-use baseline.
- No broad refactor unless required by a specific phase.
