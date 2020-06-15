import code
from enum import Enum, auto
import numpy as np
from scipy.optimize import root

from physics.base.data import FcnBase, BCWeakRiemann, BCWeakPrescribed, SourceBase



class FcnType(Enum):
    Sine = auto()
    DampingSine = auto()
    ShiftedCosine = auto()
    Exponential = auto()
    Gaussian = auto()
    ScalarShock = auto()
    Paraboloid = auto()
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

class sine(FcnBase):
	def __init__(self, omega=2*np.pi):
		self.omega = omega

	def get_state(self, physics, x, t):
		c = physics._c
		Up = np.sin(self.omega*(x-c*t))

		return Up


class damping_sine(FcnBase):
	def __init__(self, omega=2*np.pi, nu=1.):
		self.omega = omega
		self.nu = nu

	def get_state(self, physics, x, t):
		c = physics._c
		Up = np.sin(self.omega*(x-c*t))*np.exp(self.nu*t)

		return Up


class shifted_cosine(FcnBase):
	def __init__(self, omega=2*np.pi):
		self.omega = omega

	def get_state(self, physics, x, t):
		c = physics._c
		Up = 1. - np.cos(self.omega*x)

		return Up


class exponential(FcnBase):
	def __init__(self, theta=1.):
		self.theta = theta

	def get_state(self, physics, x, t):
		Up = np.exp(self.theta*x)

		return Up


class scalar_shock(FcnBase):
	def __init__(self, uL=1., uR=0., xshock=-0.5):
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

		Up = np.zeros([x.shape[0], physics.StateRank])

		Up[ileft] = uL
		Up[iright] = uR

		return Up


class gaussian(FcnBase):
	def __init__(self, sig=1., x0=0.):
		self.sig = sig # standard deviation
		self.x0 = x0 # center

	def get_state(self, physics, x, t):
		r = np.linalg.norm(x-self.x0-physics._c*t, axis=1, keepdims=True)
		Up = 1./(self.sig*np.sqrt(2.*np.pi))**float(physics.Dim) * np.exp(-r**2./(2.*self.sig**2.))

		return Up


class paraboloid(FcnBase):
	def __init__(self):
		pass

	def get_state(self, physics, x, t):
		r2 = x[:,0:1]**2. + x[:,1:2]**2.
		Up = r2

		return Up


class sine_burgers(FcnBase):
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
		
		Up = sol.x.reshape(-1,1)

		return Up


class linear_burgers(FcnBase):
	def __init__(self):
		pass

	def get_state(self, physics, x, t):
		a = -1.
		b = 1.
		Up = (a*x+b)/(a*t+1.)

		return Up


'''
Source term functions
'''
class simple_source(SourceBase):
	def __init__(self, nu=-1):
		self.nu = nu

	def get_source(self, physics, FcnData, x, t):
		nu = self.nu
		U = FcnData.U
		S = nu*U[:]

		return S
	def get_jacobian(self):
		return self.nu


# def stiff_source(physics, fcn_data):
# 	x = fcn_data.x
# 	t = fcn_data.Time
# 	U = fcn_data.U
# 	S = fcn_data.S
# 	Data = fcn_data.Data

# 	try:
# 		beta = Data.beta
# 	except AttributeError:
# 		beta = 0.5
# 	try: 
# 		stiffness = Data.stiffness
# 	except AttributeError:
# 		stiffness = 1.

# 	S[:] = (1./stiffness)*(1.-U[:])*(U[:]-beta)*U[:]

# 	return S





