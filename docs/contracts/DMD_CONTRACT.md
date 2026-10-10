# Dynamic Mode Decomposition Contract

The DMD API treats columns as state snapshots. A single snapshot matrix
`snapshots` creates the paired matrices `X = snapshots[:, :-1]` and
`Y = snapshots[:, 1:]`. Callers may instead provide `X` and `Y` explicitly.

The primary spectrum is discrete-time. A positive finite `dt` enables the
continuous-time rates `log(eigenvalues) / dt` and prediction at physical
times. Complex eigenvalues, modes, and amplitudes are preserved. Results from
real snapshots are converted back to real arrays only when the remaining
imaginary component is at numerical-noise level.

Rank selection follows the existing SVD contract: `rank` is an upper bound and
`tol` is an energy-discard tolerance. Numerically zero singular values are
excluded before inversion. A zero numerical rank is rejected.

`dmdResult` is intentionally independent of `ltiModel`. It describes learned
autonomous snapshot dynamics and exposes modes, eigenvalues, amplitudes, the
reduced operator, singular values, rank, diagnostics, reconstruction, and
prediction.

The explicit POD-DMD pipeline uses separate `podRank` and `dmdRank` controls.
It projects snapshots with `V* x`, or `V* M x` for weighted POD, fits DMD in
the POD coordinates, and lifts modes and predictions back with `V`. Centered
POD-DMD adds the stored snapshot mean after lifting. Explicit `(X, Y)` pairs
fit their shared POD basis on `[X, Y]`. Controlled POD-DMD is not part of this
contract; it requires a dedicated POD-DMDc simulation result.

## 控制辨识与嵌入内存

DMDc diagnostics 提供 `inputNumericalRank`、`augmentedDimension`、`controlRank`、`identifiable`、`retainedIdentifiable` 和 `retainedCondition`。默认允许欠激励数据的最小范数拟合；训练残差很小仍可能无法独立辨识 A/B。`requireIdentifiable=True` 要求增广快照满行秩且保留该完整秩，否则明确拒绝。含噪数据的代数满秩不证明实际系统辨识可靠，需结合条件数、噪声与留出轨迹。simulate 拒绝非有限输入和溢出，支持空控制序列。

Hankel fit 可用 `maxAugmentedBytes` 在构造延迟嵌入前限制其字节数；默认 None，超过显式预算抛 MemoryError。diagnostics 给出 `augmentedBytes` 和 `augmentedShape`。预算仅覆盖嵌入矩阵，SVD 与结果仍需要额外内存；大 delay 的进程峰值用 `generate_validation_report.py` 独立测量。
