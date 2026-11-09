# Building a forward model

Geometry defines material regions and boundaries. Conductivity and electrode
contact impedance are supplied separately to the model. A measurement plan
combines current patterns with observation rules.

```python
from eitlab import (
    ElectrodeDisk, ElementwiseConductivity, ElectricalModel,
    MeasurementPlan, MeasurementRunner, ProtocolFactory, AdjacentDifference,
    PotentialReference, ElectricalLinearization, JacobianAssembler,
)

geometry = ElectrodeDisk(radius=1., n_electrodes=8)
geometry.add_disk_inclusion(name="target", p=(.3, .1), r=.2)
mesh = geometry.make_mesh(maxh=.25)
electrodes = tuple(geometry.electrode_names)
space = ElementwiseConductivity(mesh)
sigma = space.create({"body": 1., "target": .8})
model = ElectricalModel(mesh, electrodes, sigma=sigma, z_contact=.01,
                        potential_reference=PotentialReference.MEAN_ZERO)
protocol = ProtocolFactory.adjacent(electrodes, amplitude=1.)
plan = MeasurementPlan(protocol, AdjacentDifference)
results = MeasurementRunner(model, plan).run()
linearization = ElectricalLinearization(model, parameter_space=space)
J = JacobianAssembler(linearization, results).assemble_total_jacobian()
voltages = results.vector
```

Preserve electrode construction order for protocols, model arrays and Jacobian
rows. Use consistent geometry units. With lengths in metres, conductivity is
in S/m, current in A and the 3D CEM contact parameter in ohm m². In 2D, declare
the slice/per-unit-depth convention for current and contact impedance before
making physical comparisons.

## Parameters and mesh operators

`space.create` accepts a scalar, a vector in parameter DOF order or a mapping
covering the material names. It returns an owned `ConductivityParameter`.
`with_values` creates another parameter; it does not modify the original.
The model accepts these objects directly without high-level construction of
native grid functions.

`space.regularization_operators()` returns geometric L2 and P0 TV operators.
`space.pixel_space(x_edges, y_edges)` constructs overlap maps on a straight
triangular mesh. See [architecture](../reference/architecture.md).
Geometry constructors store shapes and labels, not conductivity.

## Grounding, references and model updates

The numerical ground removes the additive-potential null space. It defaults
to the first finite electrode; use `numerical_ground_el` or a
`numerical_ground_point` to choose another. `PotentialReference` separately
controls the reported electrode potentials: mean-zero, first-electrode,
current-electrode mean and raw/fixed-reference options are available.

`model.set_sigma` validates and copies supported inputs, assembles the
operator, clears cached factorization and increments the revision.
`set_contact_impedances` updates contacts. Conductivity updates reuse spaces
and boundary layout. Invalid conductivity updates leave the previous valid
model usable. Recreate the model for structural changes.

FEM states and linearizations belong to one model revision; using them after
an update raises a stale-state error. Compute derivatives before the update,
or evaluate fresh states afterward. `duplicate()` creates an independent solver
configuration sharing the mesh.

## Observation records

`SingleEnded` measures referenced electrode potentials. `SingleEndedNoCurrent`
excludes active current electrodes. `AdjacentDifference` measures adjacent
potential differences. `ElectrodeObservation` applies the same transforms
in the forward evaluation and adjoint right-hand sides.

`results.vector` is the canonical injection-major data. `voltage_blocks`,
`offsets` and `labels` describe its layout, including variable channel counts.
`results.voltages` stacks only rectangular protocols. Voltage/observation
records are snapshots.

`run(store_solutions=False)` omits full FEM states for voltage-only work.
The default retains them and is required by `JacobianAssembler`.

## Numerical adapters

```python
import numpy as np
from eit_forward.backend import load_backend
from eit_forward.adapters import NGSolveConductivityForward

forward = NGSolveConductivityForward(load_backend(), mesh, electrodes, plan)
snapshot = forward.linearize(np.ones(forward.n_parameters), method="adjoint")
prediction, J = snapshot.prediction, snapshot.matrix
```

The loader imports installed packages without changing `sys.path` or loading
files by filename. Numerical snapshots own their arrays and survive later
evaluations. `method="central"`, `"forward"` or `"backward"` selects finite
differences; see [differentiation](../reference/differentiation.md).

Complex forward solves use `complex_mode=True` with compatible conductivity
and contact inputs. The general inverse solvers and electrical adjoint use
real arrays. Declare component stacking, noise and derivative conventions
explicitly for complex inversion.

[Documentation index](../README.md)
