from enum import Enum, auto
import code
import numpy as np
from physics.base.data import FcnBase, BCWeakRiemann, BCWeakPrescribed, ConvNumFluxBase 


class FcnType(Enum):
    Uniform = auto()


class BCType(Enum):
	StateAll = auto()
	Extrapolate = auto()


class SourceType(Enum):
	pass


class ConvNumFluxType(Enum):
	LaxFriedrichs = auto()


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
	def __init__(self, **kwargs):
		pass
	def get_boundary_state(self, physics, x, t, normals, UpI):
		return UpI.copy()

'''
Numerical flux functions
'''

class LaxFriedrichs(ConvNumFluxBase):
	# def __init__(self, u=None):
	# 	if u is not None:
	# 		n = u.shape[0]
	# 	else:
	# 		n = 0
	# 	# self.FL = np.zeros_like(u)
	# 	# self.FR = np.zeros_like(u)
	# 	# self.du = np.zeros_like(u)
	# 	# self.a = np.zeros([n,1])
	# 	# self.aR = np.zeros([n,1])
	# 	# self.idx = np.empty([n,1], dtype=bool) 

	# def AllocHelperArrays(self, u):
	# 	self.__init__(u)

	def compute_flux(self, physics, UpL, UpR, normals):
		'''
		Function: ConvFluxLaxFriedrichs
		-------------------
		This function computes the numerical flux (dotted with the normal)
		using the Lax-Friedrichs flux function

		INPUTS:
		    gam: specific heat ratio
		    UpL: Left state
		    UpR: Right state
		    normals: Normal vector (assumed left to right)

		OUTPUTS:
		    F: Numerical flux dotted with the normal, i.e., F_hat dot n
		'''

		# Extract helper arrays
		# FL = self.FL
		# FR = self.FR 
		# du = self.du 
		# a = self.a 
		# aR = self.aR 
		# idx = self.idx 

		n_mag = np.linalg.norm(normals, axis=1, keepdims=True)
		n_hat = normals/n_mag

		# Left State
		FL = physics.ConvFluxProjected(UpL, n_hat)

		# Right State
		FR = physics.ConvFluxProjected(UpR, n_hat)

		dU = UpR - UpL

		# max characteristic speed
		# code.interact(local=locals())
		a = physics.ComputeScalars("MaxWaveSpeed", UpL, flag_non_physical=True)
		aR = physics.ComputeScalars("MaxWaveSpeed", UpR, flag_non_physical=True)

		idx = aR > a
		a[idx] = aR[idx]

		# flux assembly 
		return n_mag*(0.5*(FL+FR) - 0.5*a*dU)


# def uniform(physics, fcn_data):
# 	Data = fcn_data.Data
# 	U = fcn_data.U
# 	ns = physics.NUM_STATE_VARS

# 	for k in range(ns):
# 		U[:,k] = Data.State[k]

# 	return U


# def extrapolate(physics, fcn_data):
# 	Data = fcn_data.Data
# 	U = fcn_data.U
# 	ns = physics.NUM_STATE_VARS

# 	for k in range(ns):
# 		U[:,k] = Data.State[k]

# 	return U