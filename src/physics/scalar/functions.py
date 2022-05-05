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
	Zalesak = auto()
	Rider = auto()
	Droplet_translation = auto()


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
	HeavisideSource = auto()
	SharpeningSource = auto()
	ZalesakSource = auto()
	RiderSource = auto()
	

class ConvNumFluxType(Enum):
	'''
	Enum class that stores the types of convective numerical fluxes. These
	numerical fluxes are specific to the available Euler equation sets.
	'''
	ExactLinearFlux = auto()
	LaxFriedrichs_THINC = auto()

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
	def __init__(self, xc=0., thick=0., wide=0.0):
		'''
		This method initializes the attributes.

		Inputs:
		-------
		    xc    : center
		    thick : thickness

		Outputs:
		--------
		    self: attributes initialized
		'''
		self.xc = xc
		self.thick = thick
		self.wide = wide

	def get_state(self, physics, x, t):
		c = physics.c
		
		tol = 1e-14
		Uq = np.zeros([x.shape[0], x.shape[1], physics.NUM_STATE_VARS])
		
		xx = np.zeros(x.shape)
		
		xx[:,:,0] = x[:,:,0] - 0.0*t
		
		Hx1 	  = 0.5*(1.0+np.tanh(self.thick*(xx[:,:,0]-self.xc)))
		Hx2 	  = 0.5*(1.0+np.tanh(self.thick*(xx[:,:,0]-self.xc-self.wide)))
		
		Uq[:,:,0] = (Hx1-Hx2)
		
		# Gradient
		par = 0.1
		
		dHx1dx    = 0.5*par*self.thick/(np.cosh(par*self.thick*(xx[:,:,0]-self.xc))**2)
		dHx2dx    = 0.5*par*self.thick/(np.cosh(par*self.thick*(xx[:,:,0]-self.xc-self.wide))**2)
			 
		Uq[:,:,1] = dHx1dx - dHx2dx
		
#		Hx3	  = 0.5*(1.0+np.tanh(par*self.thick*(xx[:,:,0])))
#		Uq[:,:,1] = (2.0*(1.0-Hx3)-1.0)
		
		return Uq
		
	def get_advection(self, physics, x, t):
	
		cc = np.zeros(x.shape)
		#cc[:,:,0] = physics.c #- 0.5*np.sin(np.pi*(x[:,:,0] + 1.0))
		T = 0.5
		cc[:,:,0] = np.sin(np.pi*x[:,:,0])*np.cos(np.pi*t/T)
					
		return cc
		
class Zalesak(FcnBase):
	'''
	Zalesak's disk.

	Attributes:
	-----------
	x0: float
		center
	radius: float
		radius
	thick: float
		Heaviside thickness
	'''
	def __init__(self, x0=0., radius=0., thick=0.):
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
		self.x0 = x0
		self.radius = radius
		self.thick = thick

	def get_state(self, physics, x, t):

		Uq = np.zeros([x.shape[0], x.shape[1], physics.NUM_STATE_VARS])

		tol = 1e-6
		
		omega = 1.0
		theta = -omega*t
		
		R = np.zeros((2,2))
		
		R[0,0] = np.cos(theta)
		R[0,1] = -np.sin(theta)
		R[1,0] = np.sin(theta)
		R[1,1] = np.cos(theta)
		
		xnew = np.zeros(x.shape)
		
		for ii in range(len(x[:,0,0])):
			for jj in range(len(x[0,:,0])):
				xnew[ii,jj,0] = R[0,0]*x[ii,jj,0] + R[0,1]*x[ii,jj,1]
				xnew[ii,jj,1] = R[1,0]*x[ii,jj,0] + R[1,1]*x[ii,jj,1]
		
		r = np.linalg.norm(xnew[:] - self.x0, axis=2,
				keepdims=True)
		
		#xabs = (1.0/(20.5*self.thick))*np.log(np.cosh(20.5*self.thick*x[:,:,0]))
		xabs = np.abs(xnew[:,:,0])

		Hr    = 0.5*(1.0+np.tanh(self.thick*(r[:,:,0]-self.radius)))
		Hy    = 0.5*(1.0+np.tanh(self.thick*(xnew[:,:,1]-0.35)))
		Habsx = 0.5*(1.0+np.tanh(self.thick*(xabs-0.025)))

		#########################
		# Phase field
		#########################
		Uq[:,:,0] = tol + (1.0-2.0*tol)*(1.0-Hr)*(1.0-(1.0-Hy)*(1.0-Habsx))
		#########################
		# Level-set
		#########################
		Uq[:,:,1] = physics.al[0]*np.log(Uq[:,:,0]/(1.0-Uq[:,:,0]))
		
		return Uq
		
	def get_advection(self, physics, x, t):
	
		cc = np.zeros(x.shape)

		switch = physics.switch

		cc[:,:,0] = -x[:,:,1]*switch
		cc[:,:,1] = x[:,:,0]*switch
					
		return cc
		
class Rider(FcnBase):
	'''
	Rider-Korther vortex

	Attributes:
	-----------
	x0: float
		center
	radius: float
		radius
	thick: float
		Heaviside thickness
	'''
	def __init__(self, x0=0., radius=0., thick=0.):
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
		self.x0 = x0
		self.radius = radius
		self.thick = thick

	def get_state(self, physics, x, t):
		
		Uq = np.zeros([x.shape[0], x.shape[1], physics.NUM_STATE_VARS])
	
		tol = 1e-6

		r = np.linalg.norm(x[:] - self.x0, axis=2,
				keepdims=True)

		#########################
		# Phase field
		#########################
		Hr    = 0.5*(1.0+np.tanh(self.thick*(r[:,:,0]-self.radius)))
		Uq[:,:,0] = tol + (1.0-2.0*tol)*(1.0-Hr)
		
		#########################
		# Level-set
		#########################
		Uq[:,:,1] = -(r[:,:,0]-self.radius) #LS
		
		return Uq
		
	def get_advection(self, physics, x, t):
	
		cc = np.zeros(x.shape)
		T = 4.0
		
		switch = physics.switch
		
		cc[:,:,0] = -np.sin(np.pi*(x[:,:,0]+0.5))**2.0* \
					np.sin(2.0*np.pi*(x[:,:,1]+0.5))*np.cos(np.pi*t/T)*switch
		cc[:,:,1] = np.sin(2.0*np.pi*(x[:,:,0]+0.5))* \
					np.sin(np.pi*(x[:,:,1]+0.5))**2.0*np.cos(np.pi*t/T)*switch
	
		return cc
		
		
class Droplet_translation(FcnBase):
	'''
	Droplet translation test case

	Attributes:
	-----------
	x0: float
		center
	radius: float
		radius
	thick: float
		Heaviside thickness
	'''
	def __init__(self, x0=0., radius=0., thick=0.):
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
		self.x0 = x0
		self.radius = radius
		self.thick = thick

	def get_state(self, physics, x, t):
		
		Uq = np.zeros([x.shape[0], x.shape[1], physics.NUM_STATE_VARS])
	
		tol = 1e-6
		
		xx = np.zeros(x.shape)
		xx[:,:,0] = x[:,:,0] - 1.0*t
		xx[:,:,1] = x[:,:,1]

		r = np.linalg.norm(xx[:] - self.x0, axis=2,
				keepdims=True)
		
		#########################
		# Phase field
		#########################
		Hr    = 0.5*(1.0+np.tanh(self.thick*(r[:,:,0]-self.radius)))
		Uq[:,:,0] = tol + (1.0-2.0*tol)*(1.0-Hr)
		#########################
		# Level-set
		#########################
		Uq[:,:,1] = physics.al[0]*np.log(Uq[:,:,0]/(1.0-Uq[:,:,0]))

		return Uq
		
	def get_advection(self, physics, x, t):
	
		cc = np.zeros(x.shape)
		switch = physics.switch
		cc[:,:,0] = 1.0*switch
		cc[:,:,1] = 0.0*switch

		return cc
	

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
		
class SharpeningSource(SourceBase):
	'''
	Non-conservative sharperning source term

	Attributes:
	-----------
	nu: float
		source term parameter
	'''
	def __init__(self, eta=-1, **kwargs):
		super().__init__(kwargs)
		'''
		This method initializes the attributes.

		Inputs:
		-------
		    nu: source term parameter
		    eta: second source term parameter

		Outputs:
		--------
		    self: attributes initialized
		'''
		self.nu  = nu
		self.eta = eta #0.1 standard value

	def get_source(self, physics, Uq, x, t):
		nu  = self.nu
		eta = self.eta
		
		S = np.zeros(Uq.shape)

		S[:,:,0] = nu*Uq[:,:,0] + 0.5*eta*Uq[:,:,0]*(1.0-Uq[:,:,0])*(Uq[:,:,0]-0.5)
		S[:,:,1] = 0.0
		S[:,:,2] = 0.0

		return S
		
		
class HeavisideSource(SourceBase):
	'''
	1D Heaviside source term
	'''

	def get_source(self, physics, Uq, x, t):
		al = physics.al
		S = np.zeros(Uq.shape)

		T=0.5
		#S[:,:,0] = -0.5*Uq[:,:,0]*np.cos(np.pi*(x[:,:,0] + 1.0))*np.pi
		S[:,:,0] = np.pi*np.cos(np.pi*x[:,:,0])*np.cos(np.pi*t/T)*Uq[:,:,0]
		S[:,:,1] = 0.0

		return S

class ZalesakSource(SourceBase):
	'''
	Source for the Zalesak case of the type -\nabla u \cdot \psi
	'''

	def get_source(self, physics, Uq, gUq, ggUqx, ggUqy, x, t):
		
		S = np.zeros(Uq.shape)
		
		# Zalesak
#		S[:,:,0] = 0.0
#		S[:,:,1] = -Uq[:,:,2]
#		S[:,:,2] = Uq[:,:,1]

		eps = physics.al[0]
		switch = physics.switch
		
		psi0 = Uq[:,:,0]-0.5
		Uqx = gUq[:,:,0,0]
		Uqy = gUq[:,:,0,1]
		mag3 = np.sqrt(Uqx**2 + Uqy**2 + 1e-32)
		sgn = np.tanh(0.5*(psi0)/eps/mag3)
		mag = np.sqrt(gUq[:, :, 1, 0]**2 + gUq[:, :, 1, 1]**2 + 1e-32)

		l = 1.0
		S[:,:,1] = (1.0-mag)*sgn*l*(1.0-switch)

		return S
		
class RiderSource(SourceBase):
	'''
	Source for the Rider-Korthe case of the type -\nabla u \cdot \psi
	'''
	
	def get_source(self, physics, Uq, gUq, ggUqx, ggUqy, x, t):
		
		S = np.zeros(Uq.shape)
		switch = physics.switch
		
		if switch == 0:
			eps = physics.al[0]
			psi0 = Uq[:,:,0]-0.5
			Uqx = gUq[:,:,0,0]
			Uqy = gUq[:,:,0,1]
			mag3 = np.sqrt(Uqx**2 + Uqy**2 + 1e-32)
			sgn = np.tanh(0.5*(psi0)/eps/mag3)
			mag = np.sqrt(gUq[:, :, 1, 0]**2 + gUq[:, :, 1, 1]**2 + 1e-32)
			l = 1.0
			S[:,:,1] = (1.0-mag)*sgn*l*(1.0-switch)

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
	def compute_flux(self, physics, UqL, UqR, normals, x=None, t=None):
		# Normalize the normal vectors
		n_mag = np.linalg.norm(normals, axis=2, keepdims=True)
		n_hat = normals/n_mag

		Uq_upwind = UqR.copy()
		iL = (np.einsum('ijl, l -> ij', n_hat, physics.c) >= 0.) 
		Uq_upwind[iL, :] = UqL[iL, :]

		# Flux
		Fq,_ = physics.get_conv_flux_projected(Uq_upwind, n_hat, x=None, t=None)

		# Put together
		return n_mag*Fq
		
		
class LaxFriedrichs_THINC(ConvNumFluxBase):
	'''
	This class corresponds to the local Lax-Friedrichs flux function
	coupled with a THINC reconstruction (implemented only for p=0).
	'''
	def compute_flux(self, physics, UqL, UqR, normals, x):
		# Normalize the normal vectors
		n_mag = np.linalg.norm(normals, axis=2, keepdims=True)
		n_hat = normals/n_mag
		
		# THINC ALGORITHM
		varepsilon = 1e-3
		for ll in range(0,len(UqL[:,0,0])):
			beta = 2.3
			Ull = UqL[ll,0,0]
			Urr = UqR[ll,0,0]
			Ull = max(0.0,UqL[ll,0,0])
			Ull = min(1.0,Ull)
			Urr = max(0.0,UqR[ll,0,0])
			Urr = min(1.0,Urr)
			if (np.abs(Ull) > varepsilon) and (np.abs(Ull-1.0) > varepsilon):
				sigma = (Urr-Ull)/(np.abs(Urr-Ull)+1e-15)
				A = np.exp(2.0*beta*sigma)
				B = np.exp(2.0*beta*Ull*sigma)
				xx = 1.0/(2.0*beta)*np.log((B-1.0+1e-15)/(A-B+1e-15))
				UqL[ll,0,0] = 0.5*(1.0 + np.tanh(beta*(sigma+xx)))
			if (np.abs(Urr) > varepsilon) and (np.abs(Urr-1.0) > varepsilon):
				sigma = (Urr-Ull)/(np.abs(Urr-Ull)+1e-15)
				A = np.exp(2.0*beta*sigma)
				B = np.exp(2.0*beta*Urr*sigma)
				xx = 1.0/(2.0*beta)*np.log((B-1.0+1e-15)/(A-B+1e-15))
				UqR[ll,0,0] = 0.5*(1.0 + np.tanh(beta*xx))

		# Left flux
		FqL,_ = physics.get_conv_flux_projected(UqL, n_hat, x)

		# Right flux
		FqR,_ = physics.get_conv_flux_projected(UqR, n_hat, x)

		# Jump
		dUq = UqR - UqL

		# Calculate max wave speeds at each point
		a = physics.compute_variable("MaxWaveSpeed", UqL, x,
				flag_non_physical=True)
		aR = physics.compute_variable("MaxWaveSpeed", UqR, x,
				flag_non_physical=True)

		idx = aR > a
		a[idx] = aR[idx]

		# Put together
		return n_mag*(0.5*(FqL+FqR) - 0.5*a*dUq)
