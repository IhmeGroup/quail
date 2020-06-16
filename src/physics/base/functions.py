from enum import Enum, auto
import numpy as np
from physics.base.data import FcnBase, BCWeakRiemann, BCWeakPrescribed, SourceData


class FcnType(Enum):
    Uniform = auto()


class BCType(Enum):
	StateAll = auto()
	Extrapolate = auto()


class SourceType(Enum):
	pass


'''
State functions
'''
class Uniform(FcnBase):
	def __init__(self, state=None):
		if state is None:
			raise ValueError
		self.state = np.array(state)

	def get_state(self, physics, x, t):
		state = self.state
		Up = np.tile(state, [x.shape[0], 1])
		# for s in range(len(self.state)):
		# 	self.Up[:,s] = self.state[s]

		return Up


'''
Boundary conditions
'''

class StateAll(BCWeakRiemann):
    def __init__(self, **kwargs):
    	if "function" not in kwargs.keys():
    		raise Exception("function must be specified for StateAll BC")
    	fcn_ref = kwargs["function"]
    	kwargs.pop("function")
    	self.function = fcn_ref(**kwargs)

    def get_boundary_state(self, physics, x, t, normals, UpI):
    	UpB = self.function.get_state(physics, x, t)

    	return UpB


class Extrapolate(BCWeakPrescribed):

    def get_boundary_state(self, physics, x, t, normals, UpI):

        return UpI.copy()


# def uniform(physics, fcn_data):
# 	Data = fcn_data.Data
# 	U = fcn_data.U
# 	ns = physics.StateRank

# 	for k in range(ns):
# 		U[:,k] = Data.State[k]

# 	return U


# def extrapolate(physics, fcn_data):
# 	Data = fcn_data.Data
# 	U = fcn_data.U
# 	ns = physics.StateRank

# 	for k in range(ns):
# 		U[:,k] = Data.State[k]

# 	return U