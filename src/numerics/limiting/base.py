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
		applies limiter to global solution
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
	def limit_element(self, solver, Uc):
		'''
		This method element-local solution

		Inputs:
		-------
			solver: solver object
			elem_ID: element ID
			Uc: state coefficients on element [num_elems, nb, ns]
		'''
		pass

	def limit_solution(self, solver, Uc):
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
		Uc = self.limit_element(solver, Uc)