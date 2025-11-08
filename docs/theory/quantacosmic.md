# Quantacosmic Simulation Framework

The Quantacosmic framework unifies symbolic geometry with lightweight computational
constructs so that experimentation can occur entirely within the QuASIM runtime. The
implementation in this repository focuses on ergonomics and determinism, enabling tests to
exercise the API without requiring heavyweight scientific dependencies.

## Geometric Backbone

The geometry module introduces a `QuantumManifold` abstraction that validates the metric
structure and provides derived quantities such as the volume element and curvature scalar
proxy. The simple determinant-based formulation guarantees predictable values that can be
used inside higher-level tooling and documentation builds.

## Metric Cohesion

`MetricTensor` normalises arbitrary metric components and exposes the trace, determinant,
and REVULTRA curvature quotient helpers. These utilities are intentionally minimal and
compatible with the deterministic requirements of CI workflows.

## Field Dynamics

The lattice and propagation helpers offer an intuitive interface for evolving discrete
fields. They allow us to compose field simulations with coupling operators and optional
non-linear transformations, providing hooks for future expansion while maintaining tests
that run quickly.

## Action Functionals

To close the loop between theory and execution, the action functional class discretises a
contour integral across supplied field samples. The midpoint integration scheme strikes a
balance between expressivity and simplicity so that examples and tests remain easy to
understand.
