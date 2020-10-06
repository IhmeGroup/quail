# ------------------------------------------------------------------------ #
#
#       File : src/numerics/limiting/base.py
#
#       Contains class definition for the limiter abstract base class.
#      
# ------------------------------------------------------------------------ #
from abc import ABC, abstractmethod
import numpy as np 

import errors
import general

import numerics.limiting.positivitypreserving as pp


class LimiterBase(ABC):
	'''
	This is a base class for any limiter type.

	Abstract Constants:
	-------------------
	COMPATIBLE_PHYSICS_TYPES
	    physics types compatible with the limiter; either 
	    general.PhysicsType enum member or iterable of enum members

	Methods:
	--------
	check_compatibility
		checks for compatibility between physics and limiter
	precompute_helpers
		precomputes helper arrays
	limit_element
		applies limiter to individual element
	limit_solution
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
	def COMPATIBLE_PHYSICS_TYPES(self):
		'''
		physics types compatible with the limiter; either 
	    general.PhysicsType enum member or iterable of enum members
	    '''
		pass

	def __init__(self, physics_type):
		self.check_compatibility(physics_type)

	def check_compatibility(self, physics_type):
		'''
		This method checks for compatibility with the given physics type.

		Inputs:
		-------
			physics_type: physics type (general.PhysicsType enum member)
		'''
		try:
			if physics_type not in self.COMPATIBLE_PHYSICS_TYPES:
				raise errors.IncompatibleError
		except TypeError:
			if physics_type != self.COMPATIBLE_PHYSICS_TYPES:
				raise errors.IncompatibleError

	@abstractmethod
	def precompute_helpers(self, solver):
		'''
		This method precomputes helper arrays

		Inputs:
		-------
			solver: solver object
		'''
		pass

	@abstractmethod
	def limit_element(self, solver, elem_id, Uc):
		'''
		This method element-local solution

		Inputs:
		-------
			solver: solver object
			elem_id: element ID
			Uc: state coefficients on element [nb, ns]
		'''
		pass

	def limit_solution(self, solver, U):
		'''
		This method limits the global solution

		Inputs:
		-------
			solver: solver object
			Uc: state coefficients of global solution 
				[num_elems, nb, ns]

		Outputs:
		--------
			Uc: state coefficients of global solution 
				[num_elems, nb, ns] (modified)
		'''
		for elem in range(solver.mesh.num_elems):
			U[elem] = self.limit_element(solver, elem, U[elem])