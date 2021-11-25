# Contributing to Quail

Thanks for your interest in contributing! We are excited to have you join our team!

The following is a set of guidelines for contributing to Quail [Ihme Group Organization](https://github.com/IhmeGroup/quail) on GitHub. Feel free to propose modifications to this document via a pull request as we value others contributions. 

#### Table Of Contents

[Getting Started](#getting-started)
  * [Video Tutorials](#some-video-tutorials)
  * [Interfaces](#neat-but-how-do-i-interface-with-quails-packages)

[Style Guide](#style-guide)

[Branch naming conventions](#branch-naming-conventions)
## Getting Started

Quail is a lightweight discontinuous Galerkin code written with the intention of being useful for teaching and prototyping. Quail is made up of a series of Python packages. These are all located in the `src` directory. They include the following:

* [solver](https://github.com/IhmeGroup/quail/tree/main/src/solver) - which contains modules for various solvers (DG and ADER-DG). This is where the primary `solve` function is located which governs the loop around which iterations occur.
* [meshing](https://github.com/IhmeGroup/quail/tree/main/src/meshing) - which contains modules for mesh-related classes and functions. Mesh generation tools are includedhere. For additional information on mesh generation, node ordering, and face ordering.
* [numerics](https://github.com/IhmeGroup/quail/tree/main/src/numerics) - which contains modules for basis functions, quadrature, limiters, and time stepping.
* [physics](https://github.com/IhmeGroup/quail/tree/main/src/physics) - which contains modules for various equation sets and related analytic functions, boundary conditions,and numerical fluxes.
* [processing](https://github.com/IhmeGroup/quail/tree/main/src/processing) - which contains modules for plotting, post-processing, and reading and writing data files.

In addition to these packages are the driver function (`src/quail`), user-defined exceptions (`src/errors.py`), default parameters for input decks (`src/defaultparams`), and a list of constants and general `Enums` (`src/general.py`).

### Some Video Tutorials
We have a [YouTube channel](https://www.youtube.com/channel/UCElNsS_mm_0c6X41qVKBMew) for Quail! Why? I don't quite know, but we have one and hopefully some of these videos will be helpful to you. These include videos that will help you as a user/contributor:
* [Get started with Quail](https://www.youtube.com/watch?v=IkobZVVkWL4)
* [Create a cool animation](https://www.youtube.com/watch?v=-FjCX-wkX38)
* [Construct your first input deck](https://www.youtube.com/watch?v=wf01iopPuBo)
* [Add initial conditions and exact solutions](https://www.youtube.com/watch?v=vpGOYmVOmjk)
* [Add a new physics class](https://www.youtube.com/watch?v=Rt3I3xj3ECg)
* [Add boundary conditions](https://www.youtube.com/watch?v=63YqSo1TiAA)


## Neat, but how do I interface with Quail's packages?

Interfacing with Quail is meant to be user friendly and painless. We want users to be able to add new physics, limiters, time-steppers, and more to Quail for rapid prototyping! Our primary tool to do this are [abstract base classes](https://docs.python.org/3/library/abc.html). These are parent classes that provide a basic outline of the necessary attributes that a new class in its category would need. 

Let's look at a simple example of a base class:

  ```python
  class NeatFeatureBase(ABC):
    '''
    This is a base class for neat features.
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
We utilize `@abstractmethod` to define attributes that must be overwritten by any derived class. This ensures that the user defines the minimally required members of the class. Lets imagine that we wish add a new feature called `NeatFeature` that builds on the `NeatFeatureBase` class. 

  ```python
  class NeatFeature(NeatFeatureBase):
    '''
    This neat feature inherits from the NeatFeatureBase class.
    '''
    # We define our abstract properties
    NEEDED_PROPERTY = "Clearly this is needed"
    # We define our abstract methods
    def need_this_function(self, args**):
      '''
      How the new feature used this function.
      '''
      print('This is different from before')
  ```

Our `NeatFeature` class is now compliant with the abstract base class. Other functions can be added to this class as needed but only the abstract methods are required by all classes derived from `NeatFeatureBase`.

Now, assuming we have defined our class correctly, how does Quail know that it exists? This is done primarily through `Enums` located in `src/general.py`. Here, under the desired class type, the user would add their new class.

An example of a defined `Enum` in `src/general.py` could look like this:
  ```python
  class NeatFeatureType(Enum):
    '''
    Enum containing available features of this type.
    '''
    Feature1 = auto()
    Feature2 = auto()
    NeatFeature = auto()
   ```
Lastly, the user needs to use the setter function for the feature type to take the appropriate parameter from the input deck and instantiate the corresponding class. These functions are located in the `src/<package-name>/tools.py` file and always start with `set_<name_of_instangtiated_object>`. For example, we could have a function called `set_neatfeatures` which would look like this:

  ```python
  import numerics.neatfeatures as neat_feature_defs
  from general import NeatFeatureType
  
  def set_neatfeatures(params):
	'''
	Given the NeatFeature parameter, set the neat_feature object

	Inputs:
	-------
		params: list of parameters from solver

	Outputs:
	--------
	    neat_feature: instantiated neat_feature object
	'''
	feature_name = params["NeatFeature"]
	if NeatFeatureType[feature_name] == NeatFeatureType.NeatFeature:
		neat_feature = neat_feature_defs.NeatFeature()
	else:
		raise NotImplementedError("Feature not supported")
	return neat_feature
  ```
  
These interfaces allow the user to focus on the meat of their new feature while not having too much overhead in making the feature available to the users.

## Style Guide

#### Some general guidelines to follow are:
* Use tabs with a length of four spaces
* Indent continued linestwice
* Limit lines to a maximum of 78 characters
For more detailed guidelines, please refer to the [PEP 8 style guide for Python code](https://www.python.org/dev/peps/pep-0008/#a-foolish-consistency-is-the-hobgoblin-of-little-minds), which we largely aim to follow.
#### Commenting Style
* Class commenting style
  ```python
  ’’’
  Class: ClassName
  -------------------
  This class contains information about "blank"
  
  Attributes:
  -----------
  attribute name: brief description
  ’’’
  ```
* Function definitions
  ```python
  ’’'
  Brief description of function
  
  Inputs:
  -------
  input name: brief description [array shape if available]
  
  Outputs:
  --------
  output name: brief description [array shape if available]
  '''
  ```
## Branch naming conventions

