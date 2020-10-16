# ------------------------------------------------------------------------ #
#
#       File : src/physics/base/functions.py
#
#       Contains definitions of Functions, boundary conditions, and source
#       terms generally applicable to all physics types.
#
# ------------------------------------------------------------------------ #
from enum import Enum, auto
import numpy as np

from physics.base.data import FcnBase, BCWeakRiemann, BCWeakPrescribed, \
		ConvNumFluxBase


class FcnType(Enum):
	'''
	Enum class that stores the types of analytic functions for initial
	conditions, exact solutions, and/or boundary conditions. These
	functions are generalizable to different kinds of physics.
	'''
	Uniform = auto()


class BCType(Enum):
	'''
	Enum class that stores the types of boundary conditions. These
	boundary conditions are generalizable to different kinds of physics.
	'''
	StateAll = auto()
	Extrapolate = auto()


class SourceType(Enum):
	'''
	Enum class that stores the types of source terms. These
	source terms are generalizable to different kinds of physics.
	'''
	pass


class ConvNumFluxType(Enum):
	'''
	Enum class that stores the types of convective numerical fluxes. These
	numerical fluxes are generalizable to different kinds of physics.
	'''
	LaxFriedrichs = auto()


'''
---------------
State functions
---------------
These classes inherit from the FcnBase class. See FcnBase for detailed
comments of attributes and methods. Information specific to the
corresponding child classes can be found below. These classes should
correspond to the FcnType enum members above.
'''

class Uniform(FcnBase):
	'''
	This class sets a uniform state.

	Attributes:
	-----------
	state: numpy array
		values of state variables [ns]
	'''
	def __init__(self, state=None):
		'''
		This method initializes the attributes.

		Inputs:
		-------
		    state: values of the state variables [ns]

		Outputs:
		--------
		    self: attributes initialized
		'''
		if state is None:
			raise ValueError
		self.state = np.array(state)

	def get_state(self, physics, x, t):
		state = self.state
		Uq = np.tile(state, [x.shape[0], 1])

		return Uq


'''
-------------------
Boundary conditions
-------------------
These classes inherit from either the BCWeakRiemann or BCWeakPrescribed
classes. See those parent classes for detailed comments of attributes
and methods. Information specific to the corresponding child classes can be
found below. These classes should correspond to the BCType enum members
above.
'''

class StateAll(BCWeakRiemann):
	'''
	This class prescribes all state variables. Requires a Function object,
	such as Uniform (see above), to evaluate the state.

	Attributes:
	-----------
	function: Function object
		analytical function to evaluate the state
	'''
	def __init__(self, **kwargs):
		'''
		This method initializes the attributes.

		Inputs:
		-------
		    kwargs: keyword-arguments; must include "function" key

		Outputs:
		--------
		    self: attributes initialized
		'''
		if "function" not in kwargs.keys():
			raise Exception("function must be specified for StateAll BC")
		fcn_class = kwargs["function"]
		kwargs.pop("function")
		self.function = fcn_class(**kwargs)

	def get_boundary_state(self, physics, UqI, normals, x, t):
		UqB = self.function.get_state(physics, x, t)

		return UqB


class Extrapolate(BCWeakPrescribed):
	'''
	This class sets the exterior state to be equal to the interior state.

	Attributes:
	-----------
		function: Function object
			analytical function to evaluate the state
	'''
	def __init__(self, **kwargs):
		pass
	def get_boundary_state(self, physics, UqI, normals, x, t):
		return UqI.copy()


'''
------------------------
Numerical flux functions
------------------------
These classes inherit from the ConvNumFluxBase class. See
ConvNumFluxBase for detailed comments of attributes and methods.
Information specific to the corresponding child classes can be found below.
These classes should correspond to the ConvNumFluxType enum members above.
'''

class LaxFriedrichs(ConvNumFluxBase):
	'''
	This class corresponds to the local Lax-Friedrichs flux function.
	'''
	def compute_flux(self, physics, UqL, UqR, normals):
		# TODO: This whole thing assumes Euler2D - make a LaxFriedrichs for each
		# Physics model
		# Normalize the normal vectors
		n_mag = np.linalg.norm(normals, axis=2, keepdims=True)
		n_hat = normals/n_mag

		# Left flux
		FqL, (u2L, v2L, rhoL, pL) = physics.get_conv_flux_projected(UqL, n_hat)

		# Right flux
		FqR, (u2R, v2R, rhoR, pR) = physics.get_conv_flux_projected(UqR, n_hat)

		# Jump
		dUq = UqR - UqL

		# Max wave speeds at each point
		# TODO: Flag for non-physical
		aL = np.empty_like(n_mag)
		aR = np.empty_like(n_mag)
		aL[:,:,0] = np.sqrt(u2L + v2L) + np.sqrt(physics.gamma * pL / rhoL)
		aR[:,:,0] = np.sqrt(u2R + v2R) + np.sqrt(physics.gamma * pR / rhoR)
		idx = aR > aL
		aL[idx] = aR[idx]

		# Put together
		return .5 * n_mag * (FqL + FqR - aL*dUq)
