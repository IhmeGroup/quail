from abc import ABC, abstractmethod
import code
import numpy as np 

import errors
import general

import numerics.limiting.positivitypreserving as pp


# def set_limiter(limiter_type, physics_type):
# 	'''
#     Method: set_limiter
#     ----------------------------
# 	selects limiter bases on input deck

#     INPUTS:
# 		limiterType: type of limiter selected (Default: None)
# 	'''
# 	if limiter_type is None:
# 		return None
# 	elif general.LimiterType[limiter_type] is general.LimiterType.PositivityPreserving:
# 		limiter_ref = pp.PositivityPreserving
# 	elif general.LimiterType[limiter_type] is general.LimiterType.ScalarPositivityPreserving:
# 		limiter_ref = pp.ScalarPositivityPreserving
# 	else:
# 		raise NotImplementedError

# 	limiter = limiter_ref(physics_type)

# 	return limiter


class LimiterBase(ABC):
	@property
	@abstractmethod
	def COMPATIBLE_PHYSICS_TYPES(self):
		pass

	def __init__(self, physics_type):
		self.check_compatibility(physics_type)

	def check_compatibility(self, physics_type):
		try:
			if physics_type not in self.COMPATIBLE_PHYSICS_TYPES:
				raise errors.IncompatibleError
		except TypeError:
			if physics_type != self.COMPATIBLE_PHYSICS_TYPES:
				raise errors.IncompatibleError

	@abstractmethod
	def precompute_operators(self, solver):
		pass

	@abstractmethod
	def limit_element(self, solver, elem, U):
		pass

	def limit_solution(self, solver, U):
		'''
		Method: limit_solution
		------------------------
		Calls the limiter function for each element
		INPUTS:
			solver: type of solver (e.g., DG, ADER-DG, etc...)

		OUTPUTS:
			U: solution array
		'''
		for elem in range(solver.mesh.num_elems):
			U[elem] = self.limit_element(solver, elem, U[elem])