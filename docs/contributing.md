# Contributing to Quail

Thanks for your interest in contributing! We are excited to have you join our team!

The following is a set of guidelines for contributing to Quail [Ihme Group Organization](https://github.com/IhmeGroup/quail) on GitHub. We don't always have the best ideas so feel free to propose modifications to this document via a pull request. 

## Getting Started

Quail is a lightweight discontinuous Galerkin code written with the intention of being useful for teaching and prototyping. Quail is made up of a series of Python packages. These are all located in the `src` directory. They include the following:

* [solver](https://github.com/IhmeGroup/quail/tree/main/src/solver) - which contains modules for various solvers (DG and ADER-DG). This is where the primary `solve` function is located which governs the loop around which iterations occur.
* [meshing](https://github.com/IhmeGroup/quail/tree/main/src/meshing) - which contains modules for mesh-related classes and functions. Mesh generation tools are includedhere. For additional information on mesh generation, node ordering, and face ordering.
* [numerics](https://github.com/IhmeGroup/quail/tree/main/src/numerics) - which contains modules for basis functions, quadrature, limiters, and time stepping.
* [physics](https://github.com/IhmeGroup/quail/tree/main/src/physics) - which contains modules for various equation sets and related analytic functions, boundary conditions,and numerical fluxes.
* [processing](https://github.com/IhmeGroup/quail/tree/main/src/processing) - which contains modules for plotting, post-processing, and reading and writing data files.

In addition to these packages are the driver function (`src/quail`), user-defined exceptions (`src/errors.py`), default parameters for input decks (`src/defaultparams`), and a list of constants and general `Enums` (`src/general.py`).

## Neat, but how do I interface with these packages?

Interfacing with Quail is meant to be user friendly and painless. We want users to be able to add new physics, limiters, time-steppers, and more to Quail for rapid prototyping! Our primary tool to do this are [abstract base classes](https://docs.python.org/3/library/abc.html). These are parent classes that provide a basic outline of the necessary attributes that a new class in its category would need. 

Let's look at a simple example of a base class:

  ```python
  class ClassBase(ABC):
    '''
    Comments for the base class.
    '''
    @property
    @abstractmethod
    def NEEDED_PROPERTY(self):
      '''
      This property is needed by every instantiation of this class.
      '''
    @abstractmethod
    def need_this_function(self, args**):
      '''
      A function that everychild class of this type needs.
      '''
  ```
