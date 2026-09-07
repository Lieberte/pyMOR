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
