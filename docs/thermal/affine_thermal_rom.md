# Reusable affine solid thermal ROM

`affineThermalReductor` projects a constant solid capacity/conduction model and
fixed spatial Robin patterns once. `affineThermalModel` stores only reduced
matrices and interface output maps; online trajectories need neither a full mesh
nor a full solver or reconstruction basis.

The supported temperature-rise equation is

```text
M xdot + (Kbulk + sum_j scale_j Kboundary_j) x
    = sourceScale Q + sum_j scale_j Bboundary_j (Tref_j - Tbaseline).
```

Each scale is nonnegative. Each interface has a fixed spatial HTC pattern and
one uniform reference temperature. The bulk operator conserves energy, and the
orthonormal basis must contain the constant temperature mode. The projected
capacity must be positive definite; conductivity terms must be symmetric and
positive semidefinite. Fixed Dirichlet boundaries, changing spatial patterns,
temperature-dependent material laws, and nonsymmetric advection are unsupported.

Offline construction accepts dense/SciPy sparse matrices or public MOR operators:

```python
from mor.reductors import affineThermalReductor

rom = affineThermalReductor().reduce(
    M, Kbulk, Kboundary, Q, Bboundary, basis,
    referenceTemperature=338.15,
    interfaceNames=['stator_contact', 'coolant'],
)
rom.save('solid_rom.npz')
```

`Q` has shape `(fullOrder,)`, `Bboundary` has shape
`(fullOrder, interfaceCount)`, and `Kboundary` is an ordered list of matrices.
Every Robin pattern must satisfy `Kboundary[j] @ ones == Bboundary[:, j]`.
Interfaces with zero nominal conductance must be omitted; zero online scales
are supported. `sourceScale` multiplies the original total-power source pattern.

Deployment uses only the saved NPZ:

```python
from mor.models import affineThermalModel

rom = affineThermalModel.load('solid_rom.npz')
result = rom.solveTransient(
    dt=0.5,
    sourceScales=source_history,
    boundaryScales=htc_scale_history,
    referenceTemperatures=fluid_temperature_history,
    initialRise=0.,
)
```

Source histories have shape `(steps,)`; boundary histories have shape
`(steps, interfaceCount)`, in `rom.interfaceNames` order. Reference temperatures
are absolute kelvin; the initial state is a uniform rise above the baseline.
`coordinates` and temperature outputs include the initial time. Interface power
arrays have one row per completed step and are positive out of the solid.

`interfaceTemperatures` are nominal **HTC-weighted** wall means, which allow exact
net-power evaluation for these spatial patterns; they are not area means.
`interfacePowers` uses the end-of-step wall state. `stepInterfacePowers` uses the
average of old/new wall states with current boundary inputs and is used in
`energyResidualW`.

The timestep uses current `K_n` on both sides:
`(M/dt + K_n/2) x_n = (M/dt - K_n/2) x_previous + current_load`.
For changing coefficients this differs from trapezoidal time integration using
the previous `K`. Consecutive identical boundary scales reuse the factorization
within a trajectory; changed scales invalidate it. A new solve starts a fresh
trajectory and cache. Complete field/hotspot output requires separately retaining
the basis and reconstructing it; the deployed NPZ intentionally has no basis.

Run the real-motor characterization with the repository's Python environment:

```powershell
.\.venv\Scripts\python.exe scripts/generate_motor_affine_pod_report.py `
    --source F:/mor/motor/df10000shuntai.sta `
    --cases F:/mor/geometrymotor/graph/motors/motor-df-prototype/cases
```

The source paths are local external data and are not needed for online execution.
Results go to ignored `reports/compatibility_review/motor_affine_pod/`. The report
separates offline work, reload, online solve/interface output, and reconstruction.
It compares the same component equations in SciPy and does not validate original
C++ contact iterations, whole-motor flow/network coupling, an actual operating
cycle, or geometry transfer.

The whole-motor integration probe preserves the original C++ thermal network and
non-detail TCC iteration. All 10 first-iteration matrices and loads agree with
the offline projections, with maximum relative errors of 4.16e-15 and 1.65e-13.
However, the 245-mode basis trained on tutorial-m/n archive temperatures has
startup errors up to 44.98 K at individual shaft nodes and 27.23 K in rotor iron
when checked against independent solves of the original assembled equations.
These are first-outer-iteration errors, not errors of a converged timestep.
The current basis therefore does not pass whole-motor startup qualification.
No completed C++ full-order trajectory or valid whole-motor speedup is available.

Rebuild the models with `scripts/generate_motor_whole_pod_models.py` after exporting
the original C++ nodal-interpolation/face-average HTC fields. Reproduce the
initial equation solve comparison with
`scripts/generate_motor_whole_pod_initial_audit.py`; results are deliberately
separate from a completed trajectory report. The latter requires successful
full-order and reduced-order runs through `scripts/generate_motor_whole_pod_report.py`.
The experimental C++ patch and detailed evidence are in ignored
`reports/compatibility_review/motor_whole_machine/`.
