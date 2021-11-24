# Contributing to Quail

Thanks for your interest in contributing! We are excited to have you join our team!

The following is a set of guidelines for contributing to Quail [Ihme Group Organization](https://github.com/IhmeGroup/quail) on GitHub. We don't always have the best ideas so feel free to propose modifications to this document via a pull request. 

## Getting Started

Quail is a lightweight discontinuous Galerkin code written with the intention of being useful for teaching and prototyping. Quail is made up of a series of Python packages. These are all located in the `src` directory. They include the following:

* [solver](https://github.com/IhmeGroup/quail/src/solver) - which contains modules for various solvers (DG and ADER-DG). This is where the primary `solve` function is located which governs the loop around which iterations occur.
* [meshing](https://github.com/IhmeGroup/quail/src/meshing) - which contains modules for mesh-related classes and functions. Mesh generation tools are includedhere. For additional information on mesh generation, node ordering, and face ordering.
* [numerics](https://github.com/IhmeGroup/quail/src/numerics) - which contains modules for basis functions, quadrature, limiters, and time stepping.
* [physics](https://github.com/IhmeGroup/quail/src/physics) - which contains modules for various equation sets and related analytic functions, boundary conditions,and numerical fluxes.
* [processing](https://github.com/IhmeGroup/quail/src/processing) - which contains modules for plotting, post-processing, and reading and writing data files.
