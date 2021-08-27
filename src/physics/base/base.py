# ------------------------------------------------------------------------ #
#
#       File : src/physics/base/base.py
#
#       Contains definition and helper functions for base physics class.
#
# ------------------------------------------------------------------------ #
from abc import ABC, abstractmethod
from enum import Enum, auto
import numpy as np

import errors

import physics.base.functions as base_fcns
from physics.base.functions import BCType as base_BC_type
from physics.base.functions import FcnType as base_fcn_type
from physics.base.functions import ConvNumFluxType as base_conv_num_flux_type


def process_map(fcn_type, fcn_map):
	'''
	This function extracts the desired Function class from fcn_map.

	Inputs:
	-------
	    fcn_type: Function type (member of corresponding Function enum)
	    fcn_map: Function map; dict whose keys are the types of Functions
	    	and whose values are the corresponding Function classes

	Outputs:
	--------
	    fcn_class: desired Function class (not instantiated)
	'''
	fcn_class = None
	for fcn_keys in fcn_map.keys():
		if fcn_keys.name == fcn_type:
			fcn_class = fcn_map[fcn_keys]
			break

	if fcn_class is None:
		raise ValueError("Function not found in corresponding map")

	return fcn_class


def set_state_indices_slices(physics):
	'''
	This function sets indices and slices corresponding to each state
	variable.

	Inputs:
	-------
	    physics: physics object

	Outputs:
	--------
	    physics: physics object (modified)
	'''
	index = 0
	for key in physics.StateVariables:
		physics.state_indices[key.name] = index
		physics.state_slices[key.name] = slice(index, index+1)
		index += 1


class PhysicsBase(ABC):
	'''
	This is a base class for any physics type.

	Abstract Constants:
	-------------------
	NUM_STATE_VARS
	    number of state variables
	NDIMS
	    number of dimensions
	PHYSICS_TYPE
		physics type (general.PhysicsType enum member)

	Attributes:
	-----------
	state_indices: dict
	    keys are the names of the state variables; values are the
	    corresponding indices
	state_slices: dict
	    keys are the names of the state variables; values are the
	    corresponding slices
	IC_fcn_map: dict
		keys are the types of initial conditions (members of FcnType enum);
		values are the corresponding classes
	exact_fcn_map: dict
		keys are the types of exact solutions (members of FcnType enum);
		values are the corresponding classes
	BC_map: dict
		keys are the types of boundary conditions (members of BCType enum);
		values are the corresponding classes
	BC_fcn_map: dict
		keys are the types of functions for use with the StateAll BC
		(members of FcnType enum); values are the corresponding classes
	source_map: dict
		keys are the types of source terms (members of SourceType enum);
		values are the corresponding classes
	conv_num_flux_map: dict
		keys are the types of convective numerical fluxes (members of
		ConvNumFluxType enum); values are the corresponding classes
	IC: Function object
	    holds information about the initial condition
	exact_soln: Function object
	    holds information about the (optional) exact solution
	BCs: dict
	    keys are the names of the boundary groups; values are the
	    corresponding Function objects
	source_terms: list
	    list of Function objects corresponding to each source term
	conv_flux_fcn: Function object
	    holds information about the convective flux function
	diff_flux_fcn: Function object
		holds information about the diffusive flux function
		
	Inner Classes:
	--------------
	StateVariables: enum
		state variables
	AdditionalVariables: enum
		additional variables (other than state variables)

	Abstract Methods:
	-----------------
	get_conv_flux_interior
		computes the convective analytic flux for element interiors

	Methods:
	--------
	set_physical_params
		sets physical parameters
	set_maps
		sets maps for the initial condition, exact solution, boundary
		conditions, source terms, and convective flux function. Each of
		these maps the members of the corresponding enum to the
		associated class.
	set_IC
		instantiates and stores the initial condition object
	set_exact
		instantiates and stores the exact solution object
	set_BC
		instantiates and stores the boundary condition objects
	set_source
		instantiates and stores the source term objects
	set_conv_num_flux
		instantiates and stores the convective numerical flux object
	get_state_index
		gets the index corresponding to a given state variable
	get_state_slice
		gets the slice corresponding to a given state variable
	get_quadrature_order
		gets the recommended quadrature order associated with the given
		physics class
	get_conv_flux_projected
		computes the analytic convective flux projected in a given
		direction
	get_conv_flux_numerical
		computes the convective numerical flux
	get_diff_flux_numerical
		computes the diffusive numerical flux
	eval_source_terms
		evaluates the source term(s)
	eval_source_term_jacobians
		evaluates the source term Jacobian(s)
	compute_variable
		wrapper to compute a given variable (state or additional)
	compute_additional_variable
		computes a given additional variable
	'''
	@property
	@abstractmethod
	def NUM_STATE_VARS(self):
		'''
		Number of state variables
		'''
		pass

	@property
	@abstractmethod
	def NDIMS(self):
		'''
		Number of dimensions
		'''
		pass

	@property
	@abstractmethod
	def PHYSICS_TYPE(self):
		'''
		Physics type (general.PhysicsType enum member)
		'''
		pass

	def __init__(self, mesh):
		'''
		This method initializes the attributes (see above for attribute
		details).

		Inputs:
		-------
		    mesh: mesh object

		Outputs:
		--------
			self: attributes initialized
		'''
		self.state_indices = {}
		self.state_slices = {}
		self.IC_fcn_map = {}
		self.exact_fcn_map = {}
		self.BC_map = {}
		self.BC_fcn_map = {}
		self.source_map = {}
		self.conv_num_flux_map = {}
		self.diff_num_flux_map = {}
		self.IC = None
		self.exact_soln = None
		self.BCs = dict.fromkeys(mesh.boundary_groups.keys())
		self.source_terms = []
		self.conv_flux_fcn = None
		self.diff_flux_fcn = None

		# Compatibility check
		if mesh.ndims != self.NDIMS:
			raise errors.IncompatibleError

		# Set indices and slices corresponding to the state variables
		set_state_indices_slices(self)

		# Set function maps
		self.set_maps()

	def __repr__(self):
		return '{self.__class__.__name__}'.format(self=self)

	@abstractmethod
	class StateVariables(Enum):
		'''
		Enum class that stores the state variable names. The value of
		a given member is a string consisting of the symbol used to
		denote the corresponding variable (in TeX format), e.g., "u"
		for velocity.
		'''
		pass

	class AdditionalVariables(Enum):
		'''
		Enum class that stores additional variable names. The value of
		a given member is a string consisting of the symbol used to
		denote the corresponding variable (in TeX format), e.g., "u"
		for velocity.
		'''
		pass

	def set_physical_params(self):
		'''
		This method sets physical parameters.

		Notes:
		------
			Inputs and outputs are specific to the physics class.
		'''
		pass

	def set_maps(self):
		'''
		This method sets the maps for the initial condition, exact solution,
		boundary conditions, source terms, and convective flux function.

		Outputs:
		--------
		    self.IC_fcn_map: dict whose keys are the types of initial
		    	conditions (members of FcnType enum); values are the
		    	corresponding classes
		    self.exact_fcn_map: dict whose keys are the types of exact
		    	solutions (members of FcnType enum); values are the
		    	corresponding classes
		    self.BC_map: dict whose keys are the types of boundary
		    	conditions (members of BCType enum); values are the
		    	corresponding classes
		    self.BC_fcn_map: dict whose keys are the types of functions for
		    	use with the StateAll BC (members of FcnType enum); values
		    	are to the corresponding classes
		    self.source_map: dict whose keys are the types of source
		    	terms (members of SourceType enum); values are the
		    	corresponding classes
		    self.conv_num_flux_map: dict whose keys are the types of
		    	convective numerical fluxes (members of ConvNumFluxType
		    	enum); values are the corresponding classes
		    self.diff_num_flux_map: dict whose keys are the types of 
		    	diffusive numerical fluxes (members of DiffNumFluxType
		    	enum); values are the corresponding classes

		Notes:
		------
			In general, the maps set here should be applicable to all
			child classes. In child classes, these maps should be updated
			with only the functions/BCs/source terms/numerical fluxes
			compatible with said classes.
		'''
		self.IC_fcn_map = {
			base_fcn_type.Uniform : base_fcns.Uniform,
		}

		self.exact_fcn_map = {
			base_fcn_type.Uniform : base_fcns.Uniform,
		}

		self.BC_map = {
			base_BC_type.StateAll : base_fcns.StateAll,
			base_BC_type.Extrapolate : base_fcns.Extrapolate,
		}

		self.BC_fcn_map = {
			base_fcn_type.Uniform : base_fcns.Uniform,
		}

		self.source_map = {}

		self.conv_num_flux_map = {}
		if "MaxWaveSpeed" in self.AdditionalVariables.__members__:
			self.conv_num_flux_map.update({
				base_conv_num_flux_type.LaxFriedrichs :
					base_fcns.LaxFriedrichs,
			})

		self.diff_num_flux_map = {}

	def set_IC(self, IC_type, **kwargs):
		'''
		This method sets the initial condition.

		Inputs:
		-------
			IC_type: type of initial condition (member of FcnType enum)
			kwargs: keyword arguments; depends on specific initial
				condition

		Outputs:
		--------
			self.IC: stores IC object
		'''
		# Get specified initial condition class
		fcn_class = process_map(IC_type, self.IC_fcn_map)
		# Instantiate class and store
		self.IC = fcn_class(**kwargs)

	def set_exact(self, exact_type, **kwargs):
		'''
		This method sets the exact solution.

		Inputs:
		-------
			exact_type: type of exact condition (member of FcnType enum)
			kwargs: keyword arguments; depends on specific exact
				solution

		Outputs:
		--------
			self.exact_soln: stores exact solution object
		'''
		# Get specified exact solution class
		fcn_class = process_map(exact_type, self.exact_fcn_map)
		# Instantiate class and store
		self.exact_soln = fcn_class(**kwargs)

	def set_BC(self, bname, BC_type, fcn_type=None, **kwargs):
		'''
		This method sets a given boundary condition.

		Inputs:
		-------
			bname: boundary name
			BC_type: type of boundary condition (member of BCType enum)
			fcn_type: Function type (member of FcnType enum);
				only for StateAll BC
			kwargs: keyword arguments; depends on specific BC

		Outputs:
		--------
			self.BCs: stores BC object
		'''
		if self.BCs[bname] is not None:
			raise ValueError
		else:
			# Process Function type
			if fcn_type is not None:
				fcn_class = process_map(fcn_type, self.BC_fcn_map)
				kwargs.update(function=fcn_class)
			# Get specified boundary condition class
			BC_class = process_map(BC_type, self.BC_map)
			# Instantiate class
			BC = BC_class(**kwargs)
			# Store
			self.BCs[bname] = BC

	def set_source(self, source_type, **kwargs):
		'''
		This method sets a given source term.

		Inputs:
		-------
			source_type: type of source term (member of SourceType enum)
			kwargs: keyword arguments; depends on specific source term

		Outputs:
		--------
			self.source_terms: stores source term object
		'''
		# Get specified source term class
		source_class = process_map(source_type, self.source_map)
		# Instantiate class
		source = source_class(**kwargs)
		# Store
		self.source_terms.append(source)

	def set_conv_num_flux(self, conv_num_flux_type, **kwargs):
		'''
		This method sets the convective numerical flux.

		Inputs:
		-------
			conv_num_flux_type: type of convective numerical flux
				(member of ConvNumFluxType enum)
			kwargs: keyword arguments; depends on specific convective
				numerical flux

		Outputs:
		--------
			self.conv_flux_fcn: stores convective numerical flux object
		'''
		# Get specified source term class
		conv_num_flux_class = process_map(conv_num_flux_type,
				self.conv_num_flux_map)
		# Instantiate class and store
		self.conv_flux_fcn = conv_num_flux_class(**kwargs)


	def set_diff_num_flux(self, diff_num_flux_type, **kwargs):
		'''
		This method sets the diffusive numerical flux

		Inputs:
		-------
			diff_num_flux_type: type of diffusive numerical flux
				(member of DiffNumFluxType enum)
			kwargs: keyword arguments; depends on specific diffusive 
				numerical flux

		Outputs:
		--------
			self.diff_flux_fcn : stores diffusive numerical flux object
		'''
		diff_num_flux_class = process_map(diff_num_flux_type,
				self.diff_num_flux_map)
		# Instantiate class and store
		self.diff_flux_fcn = diff_num_flux_class(**kwargs)

		
	def get_state_index(self, var_name):
		'''
		This method gets the index corresponding to a given state variable.

		Inputs:
		-------
			var_name: name of state variable

		Outputs:
		--------
			idx: index of state variable (int)
		'''
		idx = self.state_indices[var_name]

		return idx

	def get_state_slice(self, var_name):
		'''
		This method gets the slice corresponding to a given state variable.

		Inputs:
		-------
			var_name: name of state variable

		Outputs:
		--------
			slc: slice corresponding to state variable
		'''
		slc = self.state_slices[var_name]

		return slc

	def get_quadrature_order(self, order):
		'''
		This method gets recommended quadrature order for the given physics.

		Inputs:
		-------
			order: order of solution approximation

		Outputs:
		--------
			recommended quadrature order
		'''
		return 2*order+1

	@abstractmethod
	def get_conv_flux_interior(self, Uq):
		'''
		This method computes the convective analytic flux for element
		interiors.

		Inputs:
		-------
			Uq: values of the state variables (typically at the quadrature
				points) [nq, ns]

		Outputs:
		--------
			Fq: flux values [nq, ns, ndims]
		'''
		pass

	def get_conv_flux_projected(self, Uq, normals):
		'''
		This method computes the convective analytic flux projected in a
		given direction.

		Inputs:
		-------
			Uq: values of the state variables (typically at the quadrature
				points) [nf, nq, ns]
			normals: directions in which to project flux [nf, nq, ndims]

		Outputs:
		--------
			projected flux values [nf, nq, ns]
			tuple of extra variables computed by interior flux
		'''
		Fq, vars = self.get_conv_flux_interior(Uq) # [nf, nq, ns, ndims]
		
		# Check needed for ADER shapes to be consistent. This appears to 
		# be a minimally invasive approach.
		if normals.shape[1] < Fq.shape[1]:
			normals = np.tile(normals, (normals.shape[1], 1))

		return np.einsum('ijkl, ijl -> ijk', Fq, normals), vars

	def get_conv_flux_numerical(self, UqL, UqR, normals):
		'''
		This method computes the convective numerical flux.

		Inputs:
		-------
			UqL: left values of the state variables (typically at the
				quadrature points) [nf, nq, ns]
			UqR: right values of the state variables (typically at the
				quadrature points) [nf, nq, ns]
			normals: directions from left to right [nf, nq, ndims]

		Outputs:
		--------
			Fnum: numerical flux values [nf, nq, ns]
		'''
		Fnum = self.conv_flux_fcn.compute_flux(self, UqL, UqR, normals)

		return Fnum

	def get_diff_flux_numerical(self, UqL, UqR, gUqL, gUqR, normals,
			hL, hR, eta=50.):
		'''
		This method computes the diffusive numerical flux.

		Inputs:
		-------
			UqL: left values of the state variables (typically at the
				quadrature points) [nf, nq, ns]
			UqR: right values of the state variables (typically at the
				quadrature points) [nf, nq, ns]
			gUqL: left values of the gradient of the state variables
				(typically at the quadrature points) [nf, nq, ns, ndims]
			gUqR: right values of the gradient of the state variables
				(typically at the quadrature points) [nf, nq, ns, ndims]
			normals: directions from left to right [nf, nq, ndims]
		
		Outputs:
		--------
			Fnum: numerical flux values[nf, nq, ns]
		'''
		Fnum = self.diff_flux_fcn.compute_flux(self, UqL, UqR, gUqL, gUqR,
				normals, hL, hR, eta)

		return Fnum

	def eval_source_terms(self, Uq, xphys, time, Sq):
		'''
		This method computes the source term(s).

		Inputs:
		-------
			Uq: values of the state variables (typically at the quadrature
				points) [ne, nq, ns]
			xphys: coordinates in physical space [ne, nq, ndims]
			time: time
			Sq: initial values of the sum of the source term(s) (typically
				initialized to zero) [ne, nq, ns]

		Outputs:
		--------
			Sq: sum of the values of source term(s) [ne, nq, ns]
		'''
		for source in self.source_terms:
			Sq += source.get_source(self, Uq, xphys, time)

		return Sq

	def eval_source_term_jacobians(self, Uq, xphys, time, jac):
		'''
		This method computes the source term Jacobian(s).

		Inputs:
		-------
			Uq: values of the state variables (typically at the quadrature
				points) [nq, ns]
			xphys: coordinates in physical space [nq, ndims]
			time: time
			jac: initial values of the sum of the Jacobian(s) (typically
				initialized to zero) [nq, ns, ns]

		Outputs:
		--------
			jac: sum of the values of Jacobian(s) [nq, ns]
		'''
		for source in self.source_terms:
			jac += source.get_jacobian(self, Uq, xphys, time)

		return jac

	def compute_variable(self, var_name, Uq, flag_non_physical=False):
		'''
		This method computes a given variable.

		Inputs:
		-------
			var_name: name of variable to compute
			Uq: values of the state variables (typically at the quadrature
				points) [nq, ns]
			flag_non_physical: if True, will raise an error if a
				non-physical quantity is computed, e.g., negative pressure
				for the Euler equations

		Outputs:
		--------
			varq: values of the given variable [nq, 1]
		'''
		try:
			# First try state variables
			sidx = self.get_state_index(var_name)
			varq = Uq[:, :, sidx:sidx+1].copy()
		except KeyError:
			# Now try additional
			varq = self.compute_additional_variable(var_name, Uq,
					flag_non_physical)

		return varq

	def compute_additional_variable(self, var_name, Uq, flag_non_physical):
		'''
		This method computes a variable that is not a state variable.

		Inputs:
		-------
			var_name: name of variable to compute
			Uq: values of the state variables (typically at the quadrature
				points) [nq, ns]
			flag_non_physical: if True, will raise an error if a
				non-physical quantity is computed, e.g., negative pressure
				for the Euler equations

		Outputs:
		--------
			varq: values of the given variable [nq, 1]
		'''
		pass
