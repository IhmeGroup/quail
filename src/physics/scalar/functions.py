import code
from enum import Enum, auto
import numpy as np
from scipy.optimize import root

from physics.base.data import FcnBase, BCWeakRiemann, BCWeakPrescribed, SourceBase, ConvNumFluxBase



class FcnType(Enum):
    Sine = auto()
    DampingSine = auto()
    # ShiftedCosine = auto()
    # Exponential = auto()
    Gaussian = auto()
    Paraboloid = auto()
    ShockBurgers = auto()
    SineBurgers = auto()
    LinearBurgers = auto()


class BCType(Enum):
	pass


class SourceType(Enum):
    SimpleSource = auto()
    StiffSource = auto()


'''
State functions
'''

class Sine(FcnBase):
	def __init__(self, omega=2*np.pi):
		self.omega = omega

	def get_state(self, physics, x, t):
		c = physics.c
		Uq = np.sin(self.omega*(x-c*t))

		return Uq


class DampingSine(FcnBase):
	def __init__(self, omega=2*np.pi, nu=1.):
		self.omega = omega
		self.nu = nu

	def get_state(self, physics, x, t):
		c = physics.c
		Uq = np.sin(self.omega*(x-c*t))*np.exp(self.nu*t)

		return Uq


# class shifted_cosine(FcnBase):
# 	def __init__(self, omega=2*np.pi):
# 		self.omega = omega

# 	def get_state(self, physics, x, t):
# 		c = physics.c
# 		Uq = 1. - np.cos(self.omega*x)

# 		return Uq


# class exponential(FcnBase):
# 	def __init__(self, theta=1.):
# 		self.theta = theta

# 	def get_state(self, physics, x, t):
# 		Uq = np.exp(self.theta*x)

# 		return Uq


class Gaussian(FcnBase):
	def __init__(self, sig=1., x0=0.):
		self.sig = sig # standard deviation
		self.x0 = x0 # center

	def get_state(self, physics, x, t):
		r = np.linalg.norm(x-self.x0-physics.c*t, axis=1, keepdims=True)
		Uq = 1./(self.sig*np.sqrt(2.*np.pi))**float(physics.dim) * np.exp(-r**2./(2.*self.sig**2.))

		return Uq


class Paraboloid(FcnBase):
	def __init__(self):
		pass

	def get_state(self, physics, x, t):
		r2 = x[:,0:1]**2. + x[:,1:2]**2.
		Uq = r2

		return Uq


class ShockBurgers(FcnBase):
	def __init__(self, uL=1., uR=0., xshock=0.3):
		self.uL = uL 
		self.uR = uR
		self.xshock = xshock

	def get_state(self, physics, x, t):
		uL = self.uL
		uR = self.uR
		xshock = self.xshock

		us = uR + uL
		xshock = xshock + us*t
		ileft = (x <= xshock).reshape(-1)
		iright = (x > xshock).reshape(-1)

		Uq = np.zeros([x.shape[0], physics.NUM_STATE_VARS])

		Uq[ileft] = uL
		Uq[iright] = uR

		return Uq


class SineBurgers(FcnBase):
	def __init__(self, omega=2*np.pi):
		self.omega = omega

	def get_state(self, physics, x, t):

		def F(u):
			x1 = np.reshape(x, (len(x)))
			F = u - np.sin(self.omega*(x1-u*t)) 
			return F
			
		u = np.sin(self.omega*x)
		u1 = np.reshape(u, (len(u)))
		sol = root(F, u1, tol=1e-12)
		
		Uq = sol.x.reshape(-1,1)

		return Uq


class LinearBurgers(FcnBase):
	def __init__(self):
		pass

	def get_state(self, physics, x, t):
		a = -1.
		b = 1.
		Uq = (a*x+b)/(a*t+1.)

		return Uq


'''
Source term functions
'''
class SimpleSource(SourceBase):
	def __init__(self, nu=-1):
		self.nu = nu

	def get_source(self, physics, x, t):
		nu = self.nu
		U = self.U
		S = nu*U

		return S
	def get_jacobian(self, physics, x, t):
		return self.nu

class StiffSource(SourceBase):
	def __init__(self, nu=-1., beta =0.5):
		self.nu = nu
		self.beta = beta

	def get_source(self, physics, x, t):
		nu = self.nu
		beta = self.beta
		U = self.U

		S = -nu*U*(U-1.)*(U-beta)
		return S

	def get_jacobian(self, physics, x, t):
		U = self.U
		jac = np.zeros([U.shape[0], U.shape[-1], U.shape[-1]])
		nu = self.nu
		beta = self.beta
		jac[:,0,0] = -nu*(3.*U[:,0]**2 - 2.*U[:,0] - 2.*beta*U[:,0] + beta)
		return jac




