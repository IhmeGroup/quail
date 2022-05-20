# ------------------------------------------------------------------------ #
#
#       quail: A lightweight discontinuous Galerkin code for
#              teaching and prototyping
#		<https://github.com/IhmeGroup/quail>
#       
#		Copyright (C) 2020-2021
#
#       This program is distributed under the terms of the GNU
#		General Public License v3.0. You should have received a copy
#       of the GNU General Public License along with this program.  
#		If not, see <https://www.gnu.org/licenses/>.
#
# ------------------------------------------------------------------------ #

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
		ConvNumFluxBase, DiffNumFluxBase


class FcnType(Enum):
	'''
	Enum class that stores the types of analytical functions for initial
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

class DiffNumFluxType(Enum):
	'''
	Enum class that stores the types of diffusive numerical fluxes. These
	numerical fluxes are generalizable to different kinds of physics.
	'''
	SIP = auto()


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
		Uq = np.tile(state, x.shape[:2] + (1,))

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
These classes inherit from the ConvNumFluxBase or DiffNumFluxBase class. 
See ConvNumFluxBase/DiffNumFluxBase for detailed comments of attributes 
and methods. Information specific to the corresponding child classes can 
be found below. These classes should correspond to the ConvNumFluxType 
or DiffNumFluxType enum members above.
'''
class LaxFriedrichs(ConvNumFluxBase):
	'''
	This class corresponds to the local Lax-Friedrichs flux function.
	'''
	def compute_flux(self, physics, UqL, UqR, gUqL, gUqR, normals, x, t):
		# Normalize the normal vectors
		n_mag = np.linalg.norm(normals, axis=2, keepdims=True)
		n_hat = normals/n_mag

		# Left flux
		FqL,_ = physics.get_conv_flux_projected(UqL, gUqL, n_hat, x, t)

		# Right flux
		FqR,_ = physics.get_conv_flux_projected(UqR, gUqR, n_hat, x, t)

		# Jump
		dUq = UqR - UqL

		# Calculate max wave speeds at each point
		a = physics.compute_variable("MaxWaveSpeed", UqL, gUqL, x, t,
				flag_non_physical=True)
		aR = physics.compute_variable("MaxWaveSpeed", UqR, gUqR, x, t,
				flag_non_physical=True)

		idx = aR > a
		a[idx] = aR[idx]
		
		FL = n_mag*(0.5*(FqL+FqR) - 0.5*a*dUq)
		FR = n_mag*(0.5*(FqL+FqR) - 0.5*a*dUq)

		# Put together
		return FL, FR


class SIP(DiffNumFluxBase):
	'''
	This class corresponds to the Symmetric Interior Penalty Method (SIP)
	for the NavierStokes class. It is a diffusion flux method.
	'''
	def compute_iface_helpers(self, solver):
		'''
		Helper function that computes additional terms for the diff flux
		These include the penalty terms and vol/area ratios for the 
		left and right states

		Inputs:
		-------
			solver: solver object

		Outputs:
		--------
			self.eta: penalty term
			self.hL: volume to face length ratio [nfL]
			self.hR: volume to face length ratio [nfR]
		'''
		# Unpack
		elem_helpers = solver.elem_helpers
		int_face_helpers = solver.int_face_helpers

		face_lengths = int_face_helpers.face_lengths
		vol_elems = elem_helpers.vol_elems

		# Calculate the penalty term
		self.eta = self.get_ip_eta(solver.mesh, solver.order)

		# Calculate ratio of volume/area for each L/R face
		self.hL = vol_elems[int_face_helpers.elemL_IDs] / \
				face_lengths[int_face_helpers.elemL_IDs, -1]

		self.hR = vol_elems[int_face_helpers.elemR_IDs] / \
				face_lengths[int_face_helpers.elemR_IDs - 1, -1]

	def compute_bface_helpers(self, solver, bgroup_num):
		'''
		Helper function that computes additional terms for the diff flux
		These include the penalty terms and vol/area ratios for the 
		boundary states

		Inputs:
		-------
			solver: solver object
			bgroup_num: number of corresponding boundary group

		Outputs:
		--------
			self.eta: penalty term
			self.h: volume to face length ratio [nf]
		'''
		# Unpack
		bface_helpers = solver.bface_helpers
		elem_helpers = solver.elem_helpers
		elem_IDs = bface_helpers.elem_IDs[bgroup_num]

		vol_elems = elem_helpers.vol_elems
		face_lengths_bgroups = bface_helpers.face_lengths_bgroups
		face_lengths = face_lengths_bgroups[bgroup_num]

		# Calculate the penalty term
		self.eta = self.get_ip_eta(solver.mesh, solver.order)

		# Calculate ratio of volume/area for each L/R face
		self.h = vol_elems[elem_IDs] / face_lengths[-1]

	def get_ip_eta(self, mesh, order):
		'''
		Calculate the interior penalty constant based on solution 
		order and number of faces for the geometric basis

		Inputs:
		-------
			mesh: mesh object
			order: solution order

		Outputs:
		--------
			eta: interior penalty constant
		'''
		i = order
		if i > 8:
			i = 8;
		etas = np.array([1., 4., 12., 12., 20., 30., 35., 45., 50.])

		return etas[i] * mesh.gbasis.NFACES

	def compute_flux(self, physics, UqL, UqR, gUqL, gUqR, normals, x, t, epsilon):
		'''
		See definition of compute_flux in physics/data.py. Additional 
		comments are below:

		Nomenclature for the additional SIP terms:
			Fv_dir: directional diffusion flux [nf, nq, ns, ndims]
			Fv_dir_jump: diffusion flux in the direction of the jump cond
				[nf, nq, ns, ndims]
			Fv: diffusion flux [nf, nq, ns]
			FvL: left diffusion flux [nf, nq, ns, ndims]
			FvR: right diffusion flux [nf, nq, ns, ndims]
		'''
		#Unpack
		hL = self.hL
		hR = self.hR
		eta = self.eta

		# Calculate jump condition
		dU = UqL - UqR

		# Normalize the normal vectors
		n_mag = np.linalg.norm(normals, axis=2, keepdims=True)
		n_hat = normals/n_mag

		# Tensor product of normal vector with jump
		dUxn = np.einsum('ijk, ijl -> ijlk', n_hat, dU)

		# Left State
		Fv_dir = 0.5 * physics.get_diff_flux_interior(UqL, gUqL, x, t, epsilon)
		Fv_dir_jump = physics.get_diff_flux_interior(UqL, dUxn, x, t, epsilon)

		C4 = 0.5 * eta / hL
		C5 = 0.5 * n_mag

		Fv_dir += -1. * np.einsum('i, ijkl -> ijkl', C4, Fv_dir_jump)
		FvL = np.einsum('ijv, ijkl -> ijkl', C5, Fv_dir_jump)

		# Right State
		Fv_dir += 0.5 * physics.get_diff_flux_interior(UqR, gUqR, x, t, epsilon)
		Fv_dir_jump = physics.get_diff_flux_interior(UqR, dUxn, x, t, epsilon)
		
		C4 = 0.5 * eta / hR
		C5 = 0.5 * n_mag

		Fv_dir += -1. * np.einsum('i, ijkl -> ijkl', C4, Fv_dir_jump)
		FvR = np.einsum('ijv, ijkl -> ijkl', C5, Fv_dir_jump)

		Fv = np.einsum('ijl, ijkl -> ijk', normals, Fv_dir)

		return Fv, FvL, FvR # [nf, nq, ns], [nf, nq, ns, ndims] 
			# [nf, nq, ns, ndims]

	def compute_boundary_flux(self, physics, UqI, UqB, gUq, normals, x, t, epsilon):
		'''
		Flux computation for the diffusion terms on the boundary faces.
		See SIP's class definition of compute_flux for nomenclature 
		definitions.

		Inputs:
		-------
			physics: physics object
			UqI: solution state evaluated at the quadrature points of the
				interior elements face [nf, nq, ns]
			UqB: solution state evaluated at the quadrature points of the
				boundary elements face [nf, nq, ns]
			gUq: gradient of the solution state evaluated at the quadrature
				points of the interior elements face [nf, nq, ns, ndims]
			normals: normal vector at the boundary faces [nf, nq, ndims]
		
		Outputs:
		--------
			Fv: diffusion flux [nf, nq, ns]
			FvB: directional diffusion flux evaluated on the boundary 
				[nf, nq, ns, ndims]
		'''	
		#Unpack
		h = self.h
		eta = self.eta

		# Calculate jump condition
		dU = UqI - UqB

		# Normalize the normal vectors
		n_mag = np.linalg.norm(normals, axis=2, keepdims=True)
		n_hat = normals/n_mag

		# Tensor product of normal vector with jump
		dUxn = np.einsum('ijk, ijl -> ijlk', n_hat, dU)

		# Boundary State
		Fv_dir = physics.get_diff_flux_interior(UqB, gUq, x, t, epsilon)

		# Right State
		Fv_dir_jump = physics.get_diff_flux_interior(UqB, dUxn, x, t, epsilon)

		C4 = - eta / h
		C5 = n_mag

		Fv_dir += np.einsum('i, ijkl -> ijkl', C4, Fv_dir_jump)
		FvB = np.einsum('ijv, ijkl -> ijkl', C5, Fv_dir_jump)

		Fv = np.einsum('ijl, ijkl -> ijk', normals, Fv_dir)

		return Fv, FvB # [nf, nq, ns], [nf, nq, ns, ndims]
