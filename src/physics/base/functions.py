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
		Uq = np.tile(state, [x.shape[0], 1])
		# for s in range(len(self.state)):
		# 	self.Uq[:,s] = self.state[s]

		return Uq


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

    def get_boundary_state(self, physics, UqI, normals, x, t):
    	UqB = self.function.get_state(physics, x, t)

    	return UqB


class Extrapolate(BCWeakPrescribed):
	def __init__(self, **kwargs):
		pass
	def get_boundary_state(self, physics, UqI, normals, x, t):
		return UqI.copy()


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

	def compute_flux(self, physics, UqL, UqR, normals):
		'''
		Function: ConvFluxLaxFriedrichs
		-------------------
		This function computes the numerical flux (dotted with the normal)
		using the Lax-Friedrichs flux function

		Inputs:
		    gam: specific heat ratio
		    UqL: Left state
		    UqR: Right state
		    normals: Normal vector (assumed left to right)

		Outputs:
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
		FqL = physics.get_conv_flux_projected(UqL, n_hat)

		# Right State
		FqR = physics.get_conv_flux_projected(UqR, n_hat)

		dUq = UqR - UqL

		# max characteristic speed
		# code.interact(local=locals())
		a = physics.compute_variable("MaxWaveSpeed", UqL, flag_non_physical=True)
		aR = physics.compute_variable("MaxWaveSpeed", UqR, flag_non_physical=True)

		idx = aR > a
		a[idx] = aR[idx]

		# flux assembly 
		return n_mag*(0.5*(FqL+FqR) - 0.5*a*dUq)


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