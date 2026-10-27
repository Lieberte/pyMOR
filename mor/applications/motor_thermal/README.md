# Motor thermal MOR application layer

This package adapts electric-motor thermal data to the reusable MOR core.

The initial contract covers linear transient thermal systems

`M dT/dt + K T = B u`, `y = C T + D u`

and maps them to a continuous descriptor LTI model with `E=M` and `A=-K`.
Solver-specific file parsing belongs in `adapters/`; numerical SVD, Lyapunov,
operator, and projection implementations remain in the generic MOR packages.

Temperature-dependent material properties, changing contact resistance, and
strong fluid-thermal coupling are outside this linear contract and should gain
an explicit parametric or nonlinear model abstraction before integration.
