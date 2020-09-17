from abc import ABC, abstractmethod
from enum import Enum, auto
import numpy as np

import errors

import numerics.basis.tools as basis_tools
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
	This class stores information about the physics.

	Abstract Constants:
	-------------------
	NUM_STATE_VARS
	    number of state variables
	DIM
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
	conv_num_flux_map
		keys are the types of convective numerical fluxes (members of 
		ConvNumFluxType enum); values are the corresponding classes
	IC: function object
	    holds information about the initial condition
	exact_soln: function object
	    holds information about the (optional) exact solution
	BCs: dict
	    keys are the names of the boundary groups; values are the
	    corresponding function objects
	source_terms: list
	    list of function objects corresponding to each source term
	conv_flux_fcn: function object
	    holds information about the convective flux function
	order: int
	    order of solution approximation
	U: numpy array
		coefficients of polynomial approximation of global solution

	Inner Classes:
	--------------
	StateVariables: enum
		state variables
	AdditionalVariables: enum
		additional variables (other than state variables)

	Abstract Methods:
	-----------------
	get_conv_flux_interior
		computes the analytic convective flux for element interiors

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
	eval_source_terms
		evaluates the source terms
	eval_source_term_jacobians
		evaluates the source term Jacobians
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
	def DIM(self):
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

	def __init__(self, order, basis_type, mesh):
		self.state_indices = {}
		self.state_slices = {}
		self.IC_fcn_map = {}
		self.exact_fcn_map = {}
		self.BC_map = {}
		self.BC_fcn_map = {}
		self.source_map = {}
		self.conv_num_flux_map = {}
		self.IC = None
		self.exact_soln = None
		self.BCs = dict.fromkeys(mesh.boundary_groups.keys())
		self.source_terms = []
		self.conv_flux_fcn = None

		# Coefficients of polynomial approximation of global solution
		self.order = order
		basis = basis_tools.set_basis(self.order, basis_type)
		self.U = np.zeros([mesh.num_elems, basis.get_num_basis_coeff(
				self.order), self.NUM_STATE_VARS])

		# Compatibility check
		if mesh.dim != self.DIM:
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
		Enum class that stores the state variables.
		'''
		pass

	class AdditionalVariables(Enum):
		'''
		Enum class that stores additional variables.
		'''
		pass

	def set_physical_params(self):
		'''
		This method sets physical parameters.

		Notes:
		------
			Inputs are specific to the physics class.
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

	def set_IC(self, IC_type, **kwargs):
		'''
		This method sets the initial condition.

		Inputs:
		-------
			IC_type: type of initial condition (member of ICType enum)
			kwargs: keyword arguments; depends on specific initial 
				condition
		'''
		# Get specified initial condition class
		fcn_class = process_map(IC_type, self.IC_fcn_map)
		# Instantiate class and store
		self.IC = fcn_class(**kwargs)

	def set_exact(self, exact_type, **kwargs):
		fcn_class = process_map(exact_type, self.exact_fcn_map)
		self.exact_soln = fcn_class(**kwargs)

	def set_BC(self, bname, BC_type, fcn_type=None, **kwargs):
		if self.BCs[bname] is not None:
			raise ValueError
		else:
			if fcn_type is not None:
				fcn_class = process_map(fcn_type, self.BC_fcn_map)
				kwargs.update(function=fcn_class)
			BC_class = process_map(BC_type, self.BC_map)
			BC = BC_class(**kwargs)
			self.BCs[bname] = BC

		# for i in range(len(self.BCs)):
		# 	BC = self.BCs[i]
		# 	if BC is None:
		# 		if fcn_type is not None:
		# 			fcn_class = process_map(fcn_type, self.BC_fcn_map)
		# 			kwargs.update(function=fcn_class)
		# 		BC_class = process_map(BC_type, self.BC_map)
		# 		BC = BC_class(**kwargs)
		# 		self.BCs[i] = BC
		# 		break

	# def SetBC(self, BCName, **kwargs):
	# 	found = False
	# 	code.interact(local=locals())
	# 	for BC in self.BCs:
	# 		if BC.Name == BCName:
	# 			BC.Set(**kwargs)
	# 			found = True
	# 			break

	# 	if not found:
	# 		raise NameError

	def set_source(self, source_type, **kwargs):
		source_class = process_map(source_type, self.source_map)
		source = source_class(**kwargs)
		self.source_terms.append(source)

	def set_conv_num_flux(self, conv_num_flux_type, **kwargs):
		conv_num_flux_class = process_map(conv_num_flux_type, 
				self.conv_num_flux_map)
		self.conv_flux_fcn = conv_num_flux_class(**kwargs)

	def get_state_index(self, var_name):
		# idx = self.VariableType[VariableName]
		idx = self.state_indices[var_name]
		# idx = self.StateVariables.__members__.keys().index(VariableName)
		return idx

	def get_state_slice(self, var_name):
		# idx = self.VariableType[VariableName]
		slc = self.state_slices[var_name]
		# idx = self.StateVariables.__members__.keys().index(VariableName)
		return slc

	# @abstractmethod
	# class BCType(IntEnum):
	# 	pass

	# @abstractmethod
	# class BCTreatment(IntEnum):
	# 	pass

	# def SetBCTreatment(self):
	# 	# default is Prescribed
	# 	self.BCTreatments = {n:self.BCTreatment.Prescribed for n in range(len(self.BCType))}
	# 	self.BCTreatments[self.BCType.StateAll] = self.BCTreatment.Riemann

	# @abstractmethod
	# class ConvFluxType(IntEnum):
	# 	pass

	# def SetSource(self, **kwargs):
	# 	#append src data to source_terms list 
	# 	Source = SourceData()
	# 	self.source_terms.append(Source)
	# 	Source.Set(**kwargs)

	def get_quadrature_order(self, order):
		return 2*order+1

	@abstractmethod
	def get_conv_flux_interior(self, u):
		pass
		
	def get_conv_flux_projected(self, Uq, normals):

		F = self.get_conv_flux_interior(Uq)
		return np.sum(F.transpose(1,0,2)*normals, axis=2).transpose()

	def get_conv_flux_numerical(self, UqL, UqR, normals):
		# self.conv_flux_fcn.AllocHelperArrays(uL)
		F = self.conv_flux_fcn.compute_flux(self, UqL, UqR, normals)

		return F

	# @abstractmethod
	# def BoundaryState(self, BC, nq, xphys, Time, normals, uI):
	# 	pass

	#Source state takes multiple source terms (if needed) and sums them together. 
	def eval_source_terms(self, nq, xphys, time, Uq, s=None):
		for source in self.source_terms:

			#loop through available source terms
			# source.x = xphys
			# source.nq = nq
			# source.time = Time
			# source.U = Uq
			# s += self.CallSourceFunction(source,source.x,source.time)
			s += source.get_source(self, Uq, xphys, time)

		return s

	def eval_source_term_jacobians(self, nq, xphys, time, Uq, jac=None):
		for source in self.source_terms:
			#loop through available source terms
			# source.x = xphys
			# source.nq = nq
			# source.time = Time
			# source.U = Uq
			# jac += self.CallSourceJacobianFunction(source,source.x,
			# 		source.time)
			jac += source.get_jacobian(self, Uq, xphys, time)

		return jac

	# def ConvFluxBoundary(self, BC, uI, uB, normals, nq, data):
	# 	bctreatment = self.BCTreatments[BC.BCType]
	# 	if bctreatment == self.BCTreatment.Riemann:
	# 		F = self.ConvFluxNumerical(uI, uB, normals, nq, data)
	# 	else:
	# 		# Prescribe analytic flux
	# 		try:
	# 			Fa = data.Fa
	# 		except AttributeError:
	# 			data.Fa = Fa = np.zeros([nq, self.NUM_STATE_VARS, self.DIM])
	# 		# Fa = self.get_conv_flux_interior(uB, Fa)
	# 		# # Take dot product with n
	# 		try: 
	# 			F = data.F
	# 		except AttributeError:
	# 			data.F = F = np.zeros_like(uI)
	# 		F[:] = self.get_conv_flux_projected(uB, normals)

	# 	return F

	def compute_variable(self, scalar_name, Uq, flag_non_physical=False):
		# if type(var_names) is list:
		# 	nscalar = len(var_names)
		# elif type(var_names) is str:
		# 	nscalar = 1
		# 	var_names = [var_names]
		# else:
		# 	raise TypeError

		# nq = U.shape[0]
		# if scalar is None or scalar.shape != (nq, nscalar):
		# 	scalar = np.zeros([nq, nscalar])
		# scalar = np.zeros([Uq.shape[0], 1])

		# for iscalar in range(nscalar):
		# 	sname = var_names[iscalar]
		# 	try:
		# 		sidx = self.get_state_index(sname)
		# 		scalar[:,iscalar] = U[:,sidx]
		# 	# if sidx < self.NUM_STATE_VARS:
		# 	# 	# State variable
		# 	# 	scalar[:,iscalar] = U[:,sidx]
		# 	# else:
		# 	except KeyError:
		# 		scalar[:,iscalar:iscalar+1] = self.compute_additional_variable(sname, U, scalar[:,iscalar:iscalar+1],
		# 			flag_non_physical)

		try:
			sidx = self.get_state_index(scalar_name)
			# scalar[:,iscalar] = Uq[:,sidx]
			scalar = Uq[:, sidx:sidx+1].copy()
		# if sidx < self.NUM_STATE_VARS:
		# 	# State variable
		# 	scalar[:,iscalar] = U[:,sidx]
		# else:
		except KeyError:
			scalar = self.compute_additional_variable(scalar_name, Uq, 
					flag_non_physical)

		return scalar

	def compute_additional_variable(self, var_name, Uq, flag_non_physical):
		pass