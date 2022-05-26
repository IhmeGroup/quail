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
#		File : src/physics/base/data.py
#
#		Contains definitions for base classes for Functions (for initial
#		conditions and exact solutions), boundary conditions, and source
#		terms.
#
# ------------------------------------------------------------------------ #
from abc import ABC, abstractmethod
import numpy as np


class FcnBase(ABC):
	'''
	This is an abstract base class for evaluating a given analytical
	function for initial conditions, exact solutions, and/or
	boundary conditions.

	Abstract Methods:
	-----------------
	get_state
		computes the state variables
	'''
	@abstractmethod
	def get_state(self, physics, x, t):
		'''
		This method computes the state variables.

		Inputs:
		-------
			physics: physics object
			x: coordinates in physical space (typically the quadrature
				points) [nq, ndims]
			t: time

		Outputs:
		--------
			Uq: values of the state variables at x [nq, ns]
		'''
		pass


class BCBase(ABC):
	'''
	This is an abstract base class for imposing boundary conditions.

	Abstract Methods:
	-----------------
	get_boundary_state
		computes the exterior state at a boundary face
	get_boundary_flux
		computes the flux at a boundary face
	'''
	@abstractmethod
	def get_boundary_state(self, physics, UqI, normals, x, t):
		'''
		This method computes the exterior state at a boundary face.

		Inputs:
		-------
			physics: physics object
			UqI: interior values of the state variables (typically
				at the quadrature points) [nq, ns]
			normals: outward-pointing normals [nq, ndims]
			x: coordinates in physical space [nq, ndims]
			t: time

		Outputs:
		--------
			UqB: exterior values of the state variables [nq, ns]
		'''
		pass

	@abstractmethod
	def get_boundary_flux(self, physics, UqI, normals, x, t, epsilon):
		'''
		This method computes the flux at a boundary face.

		Inputs:
		-------
			physics: physics object
			UqI: interior values of the state variables (typically
				at the quadrature points) [nq, ns]
			normals: outward-pointing normals [nq, ndims]
			x: coordinates in physical space [nq, ndims]
			t: time
			gUq: Gradient of the state [nq, ndims, ns]

		Outputs:
		--------
			Fq: values of the flux dotted with the normals [nq, ns]
		'''
		pass


class BCWeakRiemann(BCBase):
	'''
	BCWeakRiemann inherits attributes and methods from the BCBase class.
	See BCBase for detailed comments of attributes and methods.
	Child classes define their own get_boundary_state.

	This class computes the boundary flux via the numerical flux, which
	depends on the interior and exterior states, i.e. Fnum(UqI, UqB, n).
	'''
	def get_boundary_flux(self, physics, UqI, normals, x, t, gUq, epsilon):
		UqB = self.get_boundary_state(physics, UqI, normals, x, t)
		FL, FR = physics.get_conv_flux_numerical(UqI, UqB, gUq, gUq, normals, x, t)
		# Compute diffusive boundary fluxes if needed
		if physics.diff_flux_fcn:
			Fv, FvB = physics.get_diff_boundary_flux_numerical(UqI, UqB, 
					gUq, normals, x, t, epsilon) # [nf, nq, ns]
			FL -= Fv
			return FL, FvB # [nf, nq, ns], [nf, nq, ns, ndims]
		else:
			return FL, None


class BCWeakPrescribed(BCBase):
	'''
	BCWeakRiemann inherits attributes and methods from the BCBase class.
	See BCBase for detailed comments of attributes and methods.
	Child classes define their own get_boundary_state.

	This class computes the boundary flux via the analytical flux based on
	only the exterior state, i.e. F(UqB, n).
	'''
	def get_boundary_flux(self, physics, UqI, normals, x, t, gUq, epsilon):
		UqB = self.get_boundary_state(physics, UqI, normals, x, t)
		F,_ = physics.get_conv_flux_projected(UqB, gUq, normals, x, t)
		
		# Compute diffusive boundary fluxes if needed
		if physics.diff_flux_fcn:
			Fv, FvB = physics.get_diff_boundary_flux_numerical(UqI, UqB, 
					gUq, normals, x, t, epsilon) # [nf, nq, ns]
			F -= Fv
			return F, FvB # [nf, nq, ns], [nf, nq, ns, ndims]
		else:
			return F, None


class SourceBase(ABC):
	'''
	This is an abstract base class for evaluating source terms.

	Abstract Methods:
	-----------------
	get_source
		computes the source term

	Methods:
	--------
	get_jacobian
		computes the Jacobian of the source term
	'''
	def __init__(self, kwargs=None):
		# By default set the source treatment to implicit but
		# allow users to specify explicit if desired.
		if kwargs:
			self.source_treatment = kwargs['source_treatment']
		else:
			self.source_treatment = 'Implicit'

	@abstractmethod
	def get_source(self, physics, Uq, gUq, x, t):
		'''
		This method evaluates the source term.

		Inputs:
		-------
			physics: physics object
			Uq: values of the state variables (typically at the
				quadrature points) [nq, ns]
			x: coordinates in physical space [nq, ndims]
			t: time

		Outputs:
		--------
			Sq: values of source term [nq, ns]
		'''
		pass

	def get_jacobian(self, physics, Uq, x, t):
		'''
		This method evaluates the Jacobian of the source term.

		Inputs:
		-------
			physics: physics object
			Uq: values of the state variables (typically at the
				quadrature points) [nq, ns]
			x: coordinates in physical space [nq, ndims]
			t: time

		Outputs:
		--------
			jac: values of source term Jacobian [nq, ns, ns]
		'''
		raise NotImplementedError


class ConvNumFluxBase(ABC):
	'''
	This is an abstract base class for evaluating the convective
	numerical flux. Attributes depend on the type of numerical flux.

	Abstract Methods:
	-----------------
	compute_flux
		computes the numerical flux

	Methods:
	--------
	alloc_helpers
		allocates helper arrays
	'''
	def __init__(self, Uq=None):
		'''
		This method initializes the attributes, which depend on the
		type of numerical flux.

		Inputs:
		-------
			Uq: values of the state variables (typically at the
				quadrature points) [nq, ns]

		Outputs:
		--------
			self: attributes initialized
		'''
		pass

	def alloc_helpers(self, Uq):
		'''
		This method is a wrapper for __init__, in which helper arrays
		should be allocated.

		Inputs:
		-------
			Uq: values of the state variables (typically at the
				quadrature points) [nq, ns]

		Notes:
		------
			Outputs depend on the specific numerical flux.
		'''
		self.__init__(Uq)

	@abstractmethod
	def compute_flux(self, physics, UqL, UqR, gUqL, gUqR, normals, x=None, t=None):
		'''
		This method computes the numerical flux.

		Inputs:
		-------
			physics: physics object
			UqL: left values of the state variables (typically at the
				quadrature points) [nq, ns]
			UqR: right values of the state variables (typically at the
				quadrature points) [nq, ns]
			normals: directions from left to right [nq, ndims]

		Outputs:
		--------
			numerical flux values [nq, ns]
		'''
		pass

class DiffNumFluxBase(ABC):
	'''
	This is an abstract base class for evaluating the diffusiv
	numerical flux. Attributes depend on the type of numerical flux.

	Abstract Methods:
	-----------------
	compute_flux
		computes the numerical flux

	Methods:
	--------
	alloc_helpers
		allocates helper arrays
	'''
	def __init__(self, Uq=None):
		'''
		This method initializes the attributes, which depend on the
		type of numerical flux.

		Inputs:
		-------
			Uq: values of the state variables (typically at the
				quadrature points) [nq, ns]

		Outputs:
		--------
			self: attributes initialized
		'''
		pass

	def alloc_helpers(self, Uq):
		'''
		This method is a wrapper for __init__, in which helper arrays
		should be allocated.

		Inputs:
		-------
			Uq: values of the state variables (typically at the
				quadrature points) [nq, ns]

		Notes:
		------
			Outputs depend on the specific numerical flux.
		'''
		self.__init__(Uq)

	@abstractmethod
	def compute_flux(self, physics, UqL, UqR, gUqL, gUqR, normals, x=None, t=None):
		'''
		This method computes the numerical flux.

		Inputs:
		-------
			physics: physics object
			UqL: left values of the state variables (typically at the
				quadrature points) [nq, ns]
			UqR: right values of the state variables (typically at the
				quadrature points) [nq, ns]
			normals: directions from left to right [nq, ndims]

		Outputs:
		--------
			numerical flux values [nq, ns]
		'''
		pass
