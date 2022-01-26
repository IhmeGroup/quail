# ------------------------------------------------------------------------ #
#
#       quail: A lightweight discontinuous Galerkin code for
#              teaching and prototyping
#		<https://github.com/IhmeGroup/quail>
#       
#		Copyright (C) 2020-2021
#
#       This program is distributed under the terms of the GNU
#		General Public License v3.0. You should have received a copy
#       of the GNU General Public License along with this program.  
#		If not, see <https://www.gnu.org/licenses/>.
#
# ------------------------------------------------------------------------ #

# ------------------------------------------------------------------------ #
#
#       File : src/physics/scalar/functions.py
#
#       Contains definitions of Functions, boundary conditions, and source
#       terms for scalar equations.
#
# ------------------------------------------------------------------------ #
from enum import Enum, auto
import numpy as np
from scipy.optimize import root

from physics.base.data import FcnBase, BCWeakRiemann, BCWeakPrescribed, \
		SourceBase, ConvNumFluxBase


class FcnType(Enum):
	'''
	Enum class that stores the types of analytical functions for initial
	conditions, exact solutions, and/or boundary conditions. These
	functions are specific to the available scalar equation sets.
	'''
	Sine = auto()
	DampingSine = auto()
	Gaussian = auto()
	Paraboloid = auto()
	ShockBurgers = auto()
	SineBurgers = auto()
	LinearBurgers = auto()
	DiffGaussian = auto()
	DiffGaussian2D = auto()
	Heaviside = auto()


class BCType(Enum):
	'''
	Enum class that stores the types of boundary conditions. These
	boundary conditions are specific to the available scalar equation sets.
	'''
	pass


class SourceType(Enum):
	'''
	Enum class that stores the types of source terms. These
	source terms are specific to the available scalar equation sets.
	'''
	SimpleSource = auto()
	

class ConvNumFluxType(Enum):
	'''
	Enum class that stores the types of convective numerical fluxes. These
	numerical fluxes are specific to the available Euler equation sets.
	'''
	ExactLinearFlux = auto()

'''
---------------
State functions
---------------
These classes inherit from the FcnBase class. See FcnBase for detailed
comments of attributes and methods. Information specific to the
corresponding child classes can be found below. These classes should
correspond to the FcnType enum members above.
'''

class Sine(FcnBase):
	'''
	Sinusoidal profile.

	Attributes:
	-----------
	omega: float
		frequency
	'''
	def __init__(self, omega=2*np.pi):
		'''
		This method initializes the attributes.

		Inputs:
		-------
		    omega: frequency

		Outputs:
		--------
		    self: attributes initialized
		'''
		self.omega = omega

	def get_state(self, physics, x, t):
		c = physics.c
		Uq = np.sin(self.omega*(x-c*t))

		return Uq


class DampingSine(FcnBase):
	'''
	Sinusoidal profile with damping.

	Attributes:
	-----------
	omega: float
		frequency
	nu: float
		damping parameter
	'''
	def __init__(self, omega=2*np.pi, nu=1.):
		'''
		This method initializes the attributes.

		Inputs:
		-------
		    omega: frequency
		    nu: damping parameter

		Outputs:
		--------
		    self: attributes initialized
		'''
		self.omega = omega
		self.nu = nu

	def get_state(self, physics, x, t):
		c = physics.c
		Uq = np.sin(self.omega*(x-c*t)) * np.exp(self.nu*t)

		return Uq


class Gaussian(FcnBase):
	'''
	Gaussian profile.

	Attributes:
	-----------
	sig: float
		standard deviation
	x0: float
		center
	'''
	def __init__(self, sig=1., x0=0.):
		'''
		This method initializes the attributes.

		Inputs:
		-------
		    sig: standard deviation
		    x0: center

		Outputs:
		--------
		    self: attributes initialized
		'''
		self.sig = sig
		self.x0 = x0

	def get_state(self, physics, x, t):

		r = np.linalg.norm(x[:] - self.x0 - physics.c*t, axis=2,
				keepdims=True)
		Uq = 1./(self.sig*np.sqrt(2.*np.pi))**float(physics.NDIMS) * \
				np.exp(-r**2./(2.*self.sig**2.))

		return Uq


class Paraboloid(FcnBase):
	'''
	Paraboloid profile. Does not take into account time dependence, so
	should not necessarily be used as an exact solution.
	'''
	def __init__(self):
		pass

	def get_state(self, physics, x, t):
		r2 = x[:, 0:1]**2. + x[:, 1:2]**2.
		Uq = r2

		return Uq


class ShockBurgers(FcnBase):
	'''
	Burgers problem with a shock.

	Attributes:
	-----------
	uL: float
		left state
	uL: float
		right state
	xshock: float
		initial shock location
	'''
	def __init__(self, uL=1., uR=0., xshock=0.3):
		'''
		This method initializes the attributes.

		Inputs:
		-------
		    uL: left state
		    uR: right state
		    xshock: initial shock location

		Outputs:
		--------
		    self: attributes initialized
		'''
		self.uL = uL
		self.uR = uR
		self.xshock = xshock

	def get_state(self, physics, x, t):
		# Unpack
		uL = self.uL
		uR = self.uR
		xshock = self.xshock

		# Shock
		us = uR + uL
		xshock = xshock + us*t
		ileft = (x <= xshock).reshape(-1)
		iright = (x > xshock).reshape(-1)

		Uq = np.zeros([x.shape[0], physics.NUM_STATE_VARS])

		Uq[ileft] = uL
		Uq[iright] = uR

		return Uq


class SineBurgers(FcnBase):
	'''
	Burgers sinusoidal profile.

	Attributes:
	-----------
	omega: float
		frequency
	'''
	def __init__(self, omega=2*np.pi):
		'''
		This method initializes the attributes.

		Inputs:
		-------
			omega: frequency


		Outputs:
		--------
			self: attributes initialized
		'''
		self.omega = omega

	def get_state(self, physics, x, t):

		def F(u):
			x1 = x.reshape(x.shape[0]*x.shape[1])
			F = u - np.sin(self.omega*(x1-u*t))
			return F

		u = np.sin(self.omega*x)
		u1 = u.reshape(u.shape[0]*u.shape[1])
		sol = root(F, u1, tol=1e-12)

		Uq = sol.x.reshape(u.shape[0], u.shape[1], 1)

		return Uq


class LinearBurgers(FcnBase):
	'''
	Burgers linear profile.
	'''
	def __init__(self):
		pass

	def get_state(self, physics, x, t):
		a = -1.
		b = 1.
		Uq = (a*x + b) / (a*t + 1.)

		return Uq


class DiffGaussian(FcnBase):
	'''
	Advecting/Diffusing Gaussian wave
	'''
	def __init__(self, xo):
		self.xo = xo # Center of wave

	def get_state(self, physics, x, t):
		# unpack
		c = physics.c
		al = physics.al

		xo = self.xo

		C1 = 1. / np.sqrt(4.*t + 1.)
		C2 = (x - xo - c*t) * (x - xo - c*t)
		C3 = al * (4*t + 1)

		Uq = C1 * np.exp(-C2 / C3)

		return Uq

class DiffGaussian2D(FcnBase):
	'''
	Advecting/Diffusing Gaussian wave
	'''
	def __init__(self, xo, yo):
		self.xo = xo # Center of wave (x-coordinate)
		self.yo = yo # Center of wave (y-coordinate)

	def get_state(self, physics, x, t):
		# unpack
		c = physics.c
		al = physics.al

		x1 = x[:, :, 0]
		x2 = x[:, :, 1]

		xo = self.xo
		yo = self.yo

		Uq = np.zeros([x.shape[0], x.shape[1], physics.NUM_STATE_VARS])

		C1 = 1. / (4.*t + 1.)
		C2x = (x1 - xo - c[0]*t)**2
		C2y = (x2 - yo - c[1]*t)**2
		C3x = al[0] * (4*t + 1)
		C3y = al[1] * (4*t + 1)

		Uq[:, :, 0] = C1 * np.exp(-1.*(C2x / C3x) - (C2y / C3y))

		return Uq

class Heaviside(FcnBase):
	def __init__(self, xc=-0.75):
		self.xc = xc

	def get_state(self, physics, x, t):
		c = physics.c
		Uq = np.heaviside(((x-self.xc)-c*t),0.5) * \
			 (1.0-np.heaviside(((x-(self.xc+0.5))-c*t),0.5))

		return Uq

'''
---------------------
Source term functions
---------------------
These classes inherit from the SourceBase class. See SourceBase for detailed
comments of attributes and methods. Information specific to the
corresponding child classes can be found below. These classes should
correspond to the SourceType enum members above.
'''

class SimpleSource(SourceBase):
	'''
	Simple source term of the form S = nu*U

	Attributes:
	-----------
	nu: float
		source term parameter
	'''
	def __init__(self, nu=-1, **kwargs):
		super().__init__(kwargs)
		'''
		This method initializes the attributes.

		Inputs:
		-------
		    nu: source term parameter

		Outputs:
		--------
		    self: attributes initialized
		'''
		self.nu = nu

	def get_source(self, physics, Uq, x, t):
		nu = self.nu
		S = nu*Uq

		return S

	def get_jacobian(self, physics, Uq, x, t):
		return self.nu


'''
------------------------
Exact flux functions
------------------------
These classes inherit from the ConvNumFluxBase or DiffNumFluxBase class. 
See ConvNumFluxBase/DiffNumFluxBase for detailed comments of attributes 
and methods. Information specific to the corresponding child classes can 
be found below. These classes should correspond to the ConvNumFluxType 
or DiffNumFluxType enum members above.
'''
class ExactLinearFlux(ConvNumFluxBase):
	'''
	This class corresponds to the exact flux for linear advection.
	'''
	def compute_flux(self, physics, UqL, UqR, normals):
		# Normalize the normal vectors
		n_mag = np.linalg.norm(normals, axis=2, keepdims=True)
		n_hat = normals/n_mag

		Uq_upwind = UqR.copy()
		iL = (np.einsum('ijl, l -> ij', n_hat, physics.c) >= 0.) 
		Uq_upwind[iL, :] = UqL[iL, :]

		# Flux
		Fq,_ = physics.get_conv_flux_projected(Uq_upwind, n_hat)

		# Put together
		return n_mag*Fq
