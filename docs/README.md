# Documentation

Begin with the [installation and two starting examples](../README.md).
The [example index](../examples/README.md) lists every demonstration.

## Guides

| Guide | Read it to |
| --- | --- |
| [Forward model](guides/forward_model.md) | Build geometry, conductivity, protocols, measurements and Jacobians |
| [2D reconstruction](guides/simple_eit_2d.md) | Understand the fine/coarse meshes, two protocols, L2 and TV starter |
| [Named points](guides/named_points.md) | Preserve named vertices, build point rings and choose a point ground |
| [Plotting](guides/plotting.md) | Render meshes/fields, inspect boundaries and sample images |
| [Joint contact estimation](guides/joint_contact.md) | Reconstruct conductivity and per-electrode contact changes |

## Mathematical and API references

| Reference | Contents |
| --- | --- |
| [Architecture](reference/architecture.md) | Module responsibilities, objective, priors and parameter/pixel maps |
| [Differentiation](reference/differentiation.md) | Adjoint/finite-difference Jacobians, step sizes and validation |
| [Weight selection](reference/weight_selection.md) | Paths, L-curves, correlation and cross-validation |
| [Noise tuning](reference/noise_tuning.md) | Repeated reconstructions and amplification searches |
| [Bibliography](reference/bibliography.md) | Shared BibTeX registry and method/software citations |

## Development

- [Testing](development/testing.md): dependencies, focused checks and wheel validation.
- [API changes in 0.3](development/api_changes.md): replacements for removed imports and helpers.
- [Validation](development/validation.md): checks performed on this release.
