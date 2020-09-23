import code
from enum import Enum, auto
import numpy as np
from scipy.optimize import fsolve, root

import general

from physics.base.data import FcnBase, BCWeakRiemann, BCWeakPrescribed, SourceBase, ConvNumFluxBase


class FcnType(Enum):
	'''
	Enum class that stores the types of analytic functions for initial 
	conditions, exact solutions, and/or boundary conditions. These
	functions are specific to the available Euler equation sets.
	'''
	SmoothIsentropicFlow = auto()
	MovingShock = auto()
	IsentropicVortex = auto()
	DensityWave = auto()
	RiemannProblem = auto()
	ExactRiemannSolution = auto()
	TaylorGreenVortex = auto()


class BCType(Enum):
	'''
	Enum class that stores the types of boundary conditions. These
	boundary conditions are specific to the available Euler equation sets.
	'''
	SlipWall = auto()
	PressureOutlet = auto()


class SourceType(Enum):
	'''
	Enum class that stores the types of source terms. These
	source terms are specific to the available Euler equation sets.
	'''
	StiffFriction = auto()
	TaylorGreenSource = auto()


class ConvNumFluxType(Enum):
	'''
	Enum class that stores the types of convective numerical fluxes. These
	numerical fluxes are specific to the available Euler equation sets.
	'''
	Roe = auto()
	HLLC = auto()


'''
---------------
State functions
---------------
These classes inherit from the FcnBase class. See FcnBase for detailed 
comments of attributes and methods. Information specific to the 
corresponding child classes can be found below. These classes should 
correspond to the FcnType enum members above.
'''

class SmoothIsentropicFlow(FcnBase):
	'''
	Smooth isentropic flow problem from the following references:
		[1] J. Cheng, C.-W. Shu, "Positivity-preserving Lagrangian 
		scheme for multi-material compressible flow," Journal of 
		Computational Physics. 257:143-168, 2014.
		[2] R. Abgrall, P. Bacigaluppi, S. Tokareva, "High-order residual 
		distribution scheme for the time-dependent Euler equations of fluid 
		dynamics," Computers and Mathematics with Applications, 78:274-297,
		2019.

	Attributes:
	-----------
	a: float
		parameter that controls magnitude of sinusoidal profile
	'''
	def __init__(self, a=0.9):
		'''
		This method initializes the attributes. 

		Inputs:
		-------
		    a: parameter that controls magnitude of sinusoidal profile

		Outputs:
		--------
		    self: attributes initialized
		'''
		if a > 1:
			raise ValueError
		self.a = a

	def get_state(self, physics, x, t):
		a = self.a
		gamma = physics.gamma
		irho, irhou, irhoE = physics.get_state_indices()

		# Lambda functions
		rho0 = lambda x, a: 1. + a*np.sin(np.pi*x)
		pressure = lambda rho, gamma: rho**gamma
		rho = lambda x1, x2, a: 0.5*(rho0(x1, a) + rho0(x2, a))
		vel = lambda x1, x2, a: np.sqrt(3)*(rho(x1, x2, a) - rho0(x1, a))

		# Nonlinear equations to be solved
		f1 = lambda x1, x, t, a: x + np.sqrt(3)*rho0(x1, a)*t - x1
		f2 = lambda x2, x, t, a: x - np.sqrt(3)*rho0(x2, a)*t - x2

		xr = x.reshape(-1)

		Uq = np.zeros([x.shape[0], physics.NUM_STATE_VARS])

		# Solve above nonlinear equations for x1 and x2
		x1 = fsolve(f1, 0.*xr, (xr, t, a))
		if np.abs(x1.any()) > 1.: raise Exception("x1 = %g out of range" % 
				(x1))
		x2 = fsolve(f2, 0.*xr, (xr, t, a))
		if np.abs(x2.any()) > 1.: raise Exception("x2 = %g out of range" % 
				(x2))
		
		# State
		den = rho(x1, x2, a)
		u = vel(x1, x2, a)
		p = pressure(den, gamma)
		rhoE = p/(gamma - 1.) + 0.5*den*u*u

		# Store
		Uq[:, irho] = den
		Uq[:, irhou] = den*u
		Uq[:, irhoE] = rhoE

		return Uq


class MovingShock(FcnBase):
	'''
	Moving shock problem.

	Attributes:
	-----------
	M: float
		Mach number
	xshock: float
		initial location of shock
	'''
	def __init__(self, M=5.0, xshock=0.2):
		'''
		This method initializes the attributes. 

		Inputs:
		-------
		    M: Mach number
		    xshock: initial location of shock

		Outputs:
		--------
		    self: attributes initialized
		'''
		self.M = M
		self.xshock = xshock

	def get_state(self, physics, x, t):
		# Unpack
		M = self.M
		xshock = self.xshock

		srho, srhou, srhoE = physics.get_state_slices()

		gamma = physics.gamma

		''' Pre-shock state '''
		rho1 = 1.
		p1 = 1.e5
		u1 = 0.

		''' Update xshock based on shock speed '''
		a1 = np.sqrt(gamma*p1/rho1)
		W = M*a1 
		us = u1 + W # shock speed in lab frame
		xshock = xshock + us*t

		''' Post-shock state '''
		rho2 = (gamma + 1.)*M**2./((gamma - 1.)*M**2. + 2.)*rho1
		p2 = (2.*gamma*M**2. - (gamma - 1.))/(gamma + 1.)*p1
		# To get velocity, first work in reference frame fixed to shock
		ux = W
		uy = ux*rho1/rho2
		# Convert back to lab frame
		u2 = W + u1 - uy

		''' Fill state '''
		ileft = (x <= xshock).reshape(-1)
		iright = (x > xshock).reshape(-1)
		Uq = np.zeros([x.shape[0], physics.NUM_STATE_VARS])

		# Density
		Uq[iright, srho] = rho1
		Uq[ileft, srho] = rho2
		# Momentum
		Uq[iright, srhou] = rho1*u1
		Uq[ileft, srhou] = rho2*u2
		# Energy
		Uq[iright, srhoE] = p1/(gamma - 1.) + 0.5*rho1*u1*u1
		Uq[ileft, srhoE] = p2/(gamma - 1.) + 0.5*rho2*u2*u2

		return Uq


class IsentropicVortex(FcnBase):
	'''
	Isentropic vortex problem from the following reference:
		[1] C.-W. Shu, "Essentially non-oscillatory and weighted essentially 
		non-oscillatory schemes for hyperbolic conservation laws," in: 
		Advanced Numerical Approximation of Nonlinear Hyperbolic Equations,
		Springer-Verlag, Berlin/New York, 1998, pp. 325–432.

	Attributes:
	-----------
	rhob: float
		base density
	ub: float
		base x-velocity
	vb: float
		base y-velocity
	pb: float
		base pressure
	vs: float
		vortex strength
	'''
	def __init__(self, rhob=1., ub=1., vb=1., pb=1., vs=5.):
		'''
		This method initializes the attributes. 

		Inputs:
		-------
			rhob: base density
			ub: base x-velocity
			vb: base y-velocity
			pb: base pressure
			vs: vortex strength

		Outputs:
		--------
		    self: attributes initialized
		'''
		self.rhob = rhob
		self.ub = ub
		self.vb = vb
		self.pb = pb
		self.vs = vs

	def get_state(self, physics, x, t):		
		Uq = np.zeros([x.shape[0], physics.NUM_STATE_VARS])
		gamma = physics.gamma
		Rg = physics.R

		''' Base flow '''
		# Density
		rhob = self.rhob
		# x-velocity
		ub = self.ub
		# y-velocity
		vb = self.vb
		# Pressure
		pb = self.pb
		# Vortex strength
		vs = self.vs
		# Make sure Rg is 1
		if Rg != 1.:
			raise ValueError

		# Base temperature
		Tb = pb/(rhob*Rg)

		# Entropy
		s = pb/rhob**gamma

		# Track center of vortex
		xr = x[:,0] - ub*t
		yr = x[:,1] - vb*t
		r = np.sqrt(xr**2. + yr**2.)

		# Perturbations
		dU = vs/(2.*np.pi)*np.exp(0.5*(1-r**2.))
		du = dU*-yr
		dv = dU*xr

		dT = -(gamma - 1.)*vs**2./(8.*gamma*np.pi**2.)*np.exp(1. - r**2.)

		u = ub + du 
		v = vb + dv 
		T = Tb + dT

		# Convert to conservative variables
		rho = np.power(T/s, 1./(gamma - 1.))
		rhou = rho*u
		rhov = rho*v
		rhoE = rho*Rg/(gamma - 1.)*T + 0.5*(rhou*rhou + rhov*rhov)/rho

		Uq[:, 0] = rho
		Uq[:, 1] = rhou
		Uq[:, 2] = rhov
		Uq[:, 3] = rhoE

		return Uq


class DensityWave(FcnBase):
	'''
	Simple smooth density wave.

	Attributes:
	-----------
	p: float
		pressure
	'''
	def __init__(self, p=1.0):
		'''
		This method initializes the attributes. 

		Inputs:
		-------
			p: pressure

		Outputs:
		--------
		    self: attributes initialized
		'''
		self.p = p

	def get_state(self, physics, x, t):
		p = self.p
		srho, srhou, srhoE = physics.get_state_slices()
		gamma = physics.gamma

		Uq = np.zeros([x.shape[0], physics.NUM_STATE_VARS])
		
		rho = 1.0 + 0.1*np.sin(2.*np.pi*x)
		rhou = rho*1.0
		rhoE = p/(gamma - 1.) + 0.5*rhou**2/rho

		Uq[:, srho] = rho
		Uq[:, srhou] = rhou
		Uq[:, srhoE] = rhoE

		return Uq


class RiemannProblem(FcnBase):
	'''
	Riemann problem. Initial condition only.

	Attributes:
	-----------
	rhoL: float
		left density
	uL: float
		left velocity
	pL: float
		left pressure
	rhoR: float
		right density
	uR: float
		right velocity
	pR: float
		right pressure
	xd: float
		location of initial discontinuity
	w: float
		parameter that controls smearing of initial discontinuity; larger w
		results in more smearing.
	'''
	def __init__(self, rhoL=1., uL=0., pL=1., rhoR=0.125, uR=0., pR=0.1,
				xd=0., w=1.e-30):
		'''
		This method initializes the attributes. 

		Inputs:
		-------
			rhoL: left density
			uL: left velocity
			pL: left pressure
			rhoR: right density
			uR: right velocity
			pR: right pressure
			xd: location of initial discontinuity
			w: parameter that controls smearing of initial discontinuity; 
				larger w results in more smearing

		Outputs:
		--------
		    self: attributes initialized

		Notes:
		------
			Default values set up for Sod problem (without smearing).
		'''
		self.rhoL = rhoL
		self.uL = uL
		self.pL = pL
		self.rhoR = rhoR
		self.uR = uR
		self.pR = pR
		self.xd = xd
		if w <= 0:
			# For zero smearing, make w close to zero but not equal to zero
			raise ValueError
		self.w = w

	def get_state(self, physics, x, t):
		# Unpack
		rhoL = self.rhoL
		uL = self.uL
		pL = self.pL
		rhoR = self.rhoR
		uR = self.uR
		pR = self.pR
		xd = self.xd
		w = self.w

		srho, srhou, srhoE = physics.get_state_slices()

		gamma = physics.gamma
		
		Uq = np.zeros([x.shape[0], physics.NUM_STATE_VARS])

		def set_tanh(a, b, w, x0):
			'''
			This function prescribes a tanh profile. 

			Inputs:
			-------
				a: left value
				b: right value
				w: characteristic width
				x0: center

			Outputs:
			--------
			    tanh profile
			'''
			return 0.5*((a+b) + (b-a)*np.tanh((x-x0)/w))

		# Density
		Uq[:, srho] =  set_tanh(rhoL, rhoR, w, xd)

		# Momentum
		Uq[:, srhou] = set_tanh(rhoL*uL, rhoR*uR, w, xd)

		# Energy
		rhoEL = pL/(gamma-1.) + 0.5*rhoL*uL*uL
		rhoER = pR/(gamma-1.) + 0.5*rhoR*uR*uR
		Uq[:, srhoE] = set_tanh(rhoEL, rhoER, w, xd)

		return Uq


class ExactRiemannSolution(FcnBase):
	'''
	Riemann problem. Exact solution included (with time dependence), 
	obtained using the method of characteristics. Detailed derivation not 
	discussed here. Region 1 is to the right of the shock, region 2 between 
	the shock and the contact discontinuity, region 3 is between the contact
	discontinuity and the expansion fan, and region 4 is to the left of the 
	expansion fan.

	Attributes:
	-----------
	rhoL: float
		left density
	uL: float
		left velocity
	pL: float
		left pressure
	rhoR: float
		right density
	uR: float
		right velocity
	pR: float
		right pressure
	xd: float
		location of initial discontinuity
	'''
	def __init__(self, rhoL=1., uL=0., pL=1., rhoR=0.125, uR=0., pR=0.1,
				xd=0.):
		'''
		This method initializes the attributes. 

		Inputs:
		-------
			rhoL: left density
			uL: left velocity
			pL: left pressure
			rhoR: right density
			uR: right velocity
			pR: right pressure
			xd: location of initial discontinuity

		Outputs:
		--------
		    self: attributes initialized

		Notes:
		------
			Default values set up for Sod problem.
		'''
		self.rhoL = rhoL
		self.uL = uL
		self.pL = pL
		self.rhoR = rhoR
		self.uR = uR
		self.pR = pR
		self.xd = xd

	def get_state(self, physics, x, t):
		''' Unpack '''
		xd = self.xd
		gamma = physics.gamma
		srho, srhou, srhoE = physics.get_state_slices()

		rho4 = self.rhoL; p4 = self.pL; u4 = self.uL
		rho1 = self.rhoR; p1 = self.pR; u1 = self.uR

		# Speeds of sound in regions 1 and 4
		c4 = np.sqrt(gamma*p4/rho4)
		c1 = np.sqrt(gamma*p1/rho1)

		def F(y):
			# Nonlinear equation to get y = p2/p1
			F = y * (1. + (gamma-1.)/(2.*c4) * (u4 - u1 - c1/gamma*(y-1.)/ \
					np.sqrt((gamma+1.)/(2.*gamma)*(y-1.) + 1)))**(-2. \
					*gamma/(gamma-1)) - p4/p1
			return F			

		y0 = 0.5*p4/p1 # initial guess
		Y = fsolve(F, y0)

		''' Region 2 '''
		# Pressure
		p2 = Y*p1
		# Velocity
		u2 = u1 + c1/gamma*(p2/p1-1)/np.sqrt((gamma+1)/(2*gamma)*(p2/p1-1) \
				+ 1)
		# Speed of sound
		num = (gamma+1)/(gamma-1) + p2/p1
		den = 1 + (gamma+1)/(gamma-1)*(p2/p1)
		c2 = c1*np.sqrt(p2/p1*num/den)
		# Shock speed
		V = u1 + c1*np.sqrt((gamma+1)/(2*gamma)*(p2/p1-1) + 1)
		# Density
		rho2 = gamma*p2/c2**2

		''' Region 3 '''
		# Pressure
		p3 = p2 
		# Velocity
		u3 = u2
		# Speed of sound
		c3 = (gamma-1)/2*(u4-u3+2/(gamma-1)*c4)
		# Density
		rho3 = gamma*p3/c3**2

		# Expansion fan
		xe1 = (u4-c4)*t + xd; # "start" of expansion fan
		xe2 = (t*((gamma+1)/2*u3 - (gamma-1)/2*u4 - c4)+xd) # end

		# Location of shock
		xs = V*t + xd
		# Location of contact
		xc = u2*t + xd

		u = np.zeros_like(x); p = np.zeros_like(x); rho = np.zeros_like(x);

		for i in range(x.shape[0]):
		    if x[i] <= xe1:
		    	# Left of expansion fan (region 4)
		        u[i] = u4; p[i] = p4; rho[i] = rho4
		    elif x[i] > xe1 and x[i] <= xe2:
		    	# Expansion fan
		        u[i] = (2/(gamma+1)*((x[i]-xd)/t + (gamma-1)/2*u4 + c4)) 
		        c = u[i] - (x[i]-xd)/t
		        p[i] = p4*(c/c4)**(2*gamma/(gamma-1))
		        rho[i] = gamma*p[i]/c**2
		    elif x[i] > xe2 and x[i] <= xc:
		    	# Between expansion fan and and contact discontinuity
		    	# (region 3)
		        u[i] = u3; p[i] = p3; rho[i] = rho3
		    elif x[i] > xc and x[i] <= xs:
		    	# Between the contact discontinuity and the shock (region 2)
		        u[i] = u2; p[i] = p2; rho[i] = rho2
		    else:
		    	# Right of the shock (region 1)
		        u[i] = u1; p[i] = p1; rho[i] = rho1

		Uq = np.zeros([x.shape[0], physics.NUM_STATE_VARS])
		Uq[:, srho] = rho
		Uq[:, srhou] = rho*u
		Uq[:, srhoE] = p/(gamma-1.) + 0.5*rho*u*u

		return Uq


class TaylorGreenVortex(FcnBase):
	'''
	2D steady-state Taylor-Green vortex problem. Source term required to
	account for incompressibility and ensure steady state. Reference:
		[1] C. Wang, "Reconstructed discontinous Galerkin method for the 
		compressible Navier-Stokes equations in arbitrary Langrangian and 
		Eulerian formulation", PhD Thesis, North Carolina State University,
		2017.
	'''
	def get_state(self, physics, x, t):
		# Unpack		
		Uq = np.zeros([x.shape[0], physics.NUM_STATE_VARS])
		gamma = physics.gamma
		Rg = physics.R

		irho, irhou, irhov, irhoE = physics.get_state_indices()

		# State
		rho = 1.
		u = np.sin(np.pi*x[:, 0])*np.cos(np.pi*x[:, 1])
		v = -np.cos(np.pi*x[:, 0])*np.sin(np.pi*x[:, 1])
		p = 0.25*(np.cos(2.*np.pi*x[:, 0]) + np.cos(2*np.pi*x[:, 1])) + 1.
		E = p/(rho*(gamma - 1.)) + 0.5*(u**2. + v**2.)

		# Store
		Uq[:, irho] = rho
		Uq[:, irhou] = rho*u
		Uq[:, irhov] = rho*v
		Uq[:, irhoE] = rho*E

		return Uq


'''
-------------------
Boundary conditions
-------------------
These classes inherit from either the BCWeakRiemann or BCWeakPrescribed
classes. See those parent classes for detailed comments of attributes 
and methods. Information specific to the corresponding child classes can be
found below. These classes should correspond to the BCType enum members 
above.
'''

class SlipWall(BCWeakPrescribed):
	'''
	This class corresponds to a slip wall. See documentation for more
	details.
	'''
	def get_boundary_state(self, physics, UqI, normals, x, t):
		smom = physics.get_momentum_slice()

		# Unit normals
		n_hat = normals/np.linalg.norm(normals, axis=1, keepdims=True)

		# Remove momentum contribution in normal direction from boundary
		# state
		rhoveln = np.sum(UqI[:, smom] * n_hat, axis=1, keepdims=True)
		UqB = UqI.copy()
		UqB[:, smom] -= rhoveln * n_hat

		return UqB


class PressureOutlet(BCWeakPrescribed):
	'''
	This class corresponds to an outflow boundary condition with static
	pressure prescribed.

	Attributes:
	-----------
	p: float
		pressure
	'''
	def __init__(self, p):
		'''
		This method initializes the attributes. 

		Inputs:
		-------
			p: pressure

		Outputs:
		--------
		    self: attributes initialized
		'''
		self.p = p

	def get_boundary_state(self, physics, UqI, normals, x, t):
		# Unpack
		srho = physics.get_state_slice("Density")
		srhoE = physics.get_state_slice("Energy")
		smom = physics.get_momentum_slice()

		# Pressure
		pB = self.p

		gamma = physics.gamma

		UqB = UqI.copy()

		# Unit normals
		n_hat = normals/np.linalg.norm(normals, axis=1, keepdims=True)

		# Interior velocity in normal direction
		rhoI = UqI[:, srho]
		velI = UqI[:, smom]/rhoI
		velnI = np.sum(velI*n_hat, axis=1, keepdims=True)

		if np.any(velnI < 0.):
			print("Incoming flow at outlet")

		# Interior pressure
		pI = physics.compute_variable("Pressure", UqI)

		if np.any(pI < 0.):
			raise errors.NotPhysicalError

		# Interior speed of sound
		cI = physics.compute_variable("SoundSpeed", UqI)
		JI = velnI + 2.*cI/(gamma - 1.)
		# Interior velocity in tangential direction
		veltI = velI - velnI*n_hat

		# Normal Mach number
		Mn = velnI/cI
		if np.any(Mn >= 1.):
			# If supersonic, then extrapolate interior to exterior
			return UqB

		# Boundary density from interior entropy
		rhoB = rhoI*np.power(pB/pI, 1./gamma)
		UqB[:, srho] = rhoB

		# Boundary speed of sound
		cB = np.sqrt(gamma*pB/rhoB)
		# Boundary velocity
		velB = (JI - 2.*cB/(gamma-1.))*n_hat + veltI
		UqB[:, smom] = rhoB*velB

		# Boundary energy
		rhovel2B = rhoB*np.sum(velB**2., axis=1, keepdims=True)
		UqB[:, srhoE] = pB/(gamma - 1.) + 0.5*rhovel2B

		return UqB


'''
---------------------
Source term functions
---------------------
These classes inherit from the SourceBase class. See SourceBase for detailed 
comments of attributes and methods. Information specific to the 
corresponding child classes can be found below. These classes should 
correspond to the SourceType enum members above.
'''

class StiffFriction(SourceBase):
	'''
	Stiff source term (1D) of the form:
	S = [0, nu*rho*u, nu*rho*u^2]

	Attributes:
	-----------
	nu: float
		stiffness parameter
	'''
	def __init__(self, nu=-1):
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

		irho, irhou, irhoE = physics.get_state_indices()

		S = np.zeros_like(Uq)

		eps = general.eps
		S[:, irho] = 0.0
		S[:, irhou] = nu*(Uq[:, irhou])
		S[:, irhoE] = nu*((Uq[:, irhou])**2/(eps + Uq[:, irho]))
		
		return S

	def get_jacobian(self, physics, Uq, x, t):
		nu = self.nu

		irho, irhou, irhoE = physics.get_state_indices()

		jac = np.zeros([Uq.shape[0], Uq.shape[-1], Uq.shape[-1]])
		vel = Uq[:, 1]/(general.eps + Uq[:, 0])

		jac[:, irhou, irhou] = nu
		jac[:, irhoE, irho] = -nu*vel**2
		jac[:, irhoE, irhou] = 2.0*nu*vel

		return jac


class TaylorGreenSource(SourceBase):
	'''
	Source term for 2D Taylor-Green vortex (see above). Reference:
		[1] C. Wang, "Reconstructed discontinous Galerkin method for the 
		compressible Navier-Stokes equations in arbitrary Langrangian and 
		Eulerian formulation", PhD Thesis, North Carolina State University,
		2017.
	'''
	def get_source(self, physics, Uq, x, t):
		gamma = physics.gamma

		irho, irhou, irhov, irhoE = physics.get_state_indices()
				
		S = np.zeros_like(Uq)

		S[:, irhoE] = np.pi/(4.*(gamma - 1.))*(np.cos(3.*np.pi*x[:, 0])* \
				np.cos(np.pi*x[:, 1]) - np.cos(np.pi*x[:, 0])*np.cos(3.* \
				np.pi*x[:, 1]))
		
		return S


'''
------------------------
Numerical flux functions
------------------------
These classes inherit from the ConvNumFluxBase class. See 
ConvNumFluxBase for detailed comments of attributes and methods. 
Information specific to the corresponding child classes can be found below.
These classes should correspond to the ConvNumFluxType enum members above.
'''

class Roe1D(ConvNumFluxBase):
	'''
	1D Roe numerical flux. References:
		[1] P. L. Roe, "Approximate Riemann solvers, parameter vectors, and 
		difference schemes," Journal of Computational Physics, 
		43(2):357–372, 1981.
		[2] J. S. Hesthaven, T. Warburton, "Nodal discontinuous Galerkin 
		methods: algorithms, analysis, and applications," Springer Science 
		& Business Media, 2007.

	Attributes:
	-----------
	UqL: numpy array
		helper array for left state [nq, ns]
	UqR: numpy array
		helper array for right state [nq, ns]
	vel: numpy array
		helper array for velocity [nq, dim]
	alphas: numpy array
		helper array: left eigenvectors multipled by dU [nq, ns]
	evals: numpy array
		helper array for eigenvalues [nq, ns]
	R: numpy array
		helper array for right eigenvectors [nq, ns, ns]
	'''
	def __init__(self, Uq=None):
		'''
		This method initializes the attributes. 

		Inputs:
		-------
			Uq: values of the state variables (typically at the quadrature
				points) [nq, ns]; used to allocate helper arrays; if None,
				then empty arrays allocated

		Outputs:
		--------
		    self: attributes initialized
		'''
		if Uq is not None:
			n = Uq.shape[0]
			ns = Uq.shape[1]
			dim = ns - 2
		else:
			n = 0; ns = 0; dim = 0

		self.UqL = np.zeros_like(Uq)
		self.UqR = np.zeros_like(Uq)
		self.vel = np.zeros([n, dim])
		self.alphas = np.zeros_like(Uq)
		self.evals = np.zeros_like(Uq)
		self.R = np.zeros([n, ns, ns])

	def rotate_coord_sys(self, smom, Uq, n):
		'''
		This method expresses the momentum vector in the rotated coordinate 
		system, which is aligned with the face normal and tangent.

		Inputs:
		-------
			smom: momentum slice
			Uq: values of the state variable (typically at the quadrature 
				points) [nq, ns]
			n: normals (typically at the quadrature points) [nq, dim]

		Outputs:
		--------
		    Uq: momentum terms modified
		'''
		Uq[:, smom] *= n

		return Uq

	def undo_rotate_coord_sys(self, smom, Uq, n):
		'''
		This method expresses the momentum vector in the standard coordinate 
		system. It "undoes" the rotation above.

		Inputs:
		-------
			smom: momentum slice
			Uq: values of the state variable (typically at the quadrature 
				points) [nq, ns]
			n: normals (typically at the quadrature points) [nq, dim]

		Outputs:
		--------
		    Uq: momentum terms modified
		'''
		Uq[:, smom] /= n

		return Uq

	def roe_average_state(self, physics, srho, velL, velR, UqL, UqR):
		'''
		This method computes the Roe-averaged variables.

		Inputs:
		-------
			physics: physics object
			srho: density slice
			velL: left velocity (typically evaluated at the quadrature 
				points) [nq, dim]
			velR: right velocity (typically evaluated at the quadrature 
				points) [nq, dim]
			UqL: left state (typically evaluated at the quadrature 
				points) [nq, ns]
			UqR: right state (typically evaluated at the quadrature 
				points) [nq, ns]

		Outputs:
		--------
		    rhoRoe: Roe-averaged density [nq, 1]
		    velRoe: Roe-averaged velocity [nq, dim]
		    HRoe: Roe-averaged total enthalpy [nq, 1]
		'''
		rhoL_sqrt = np.sqrt(UqL[:, srho])
		rhoR_sqrt = np.sqrt(UqR[:, srho])
		HL = physics.compute_variable("TotalEnthalpy", UqL)
		HR = physics.compute_variable("TotalEnthalpy", UqR)

		velRoe = (rhoL_sqrt*velL + rhoR_sqrt*velR)/(rhoL_sqrt+rhoR_sqrt)
		HRoe = (rhoL_sqrt*HL + rhoR_sqrt*HR)/(rhoL_sqrt+rhoR_sqrt)
		rhoRoe = rhoL_sqrt*rhoR_sqrt

		return rhoRoe, velRoe, HRoe

	def get_differences(self, physics, srho, velL, velR, UqL, UqR):
		'''
		This method computes velocity, density, and pressure jumps.

		Inputs:
		-------
			physics: physics object
			srho: density slice
			velL: left velocity (typically evaluated at the quadrature 
				points) [nq, dim]
			velR: right velocity (typically evaluated at the quadrature 
				points) [nq, dim]
			UqL: left state (typically evaluated at the quadrature 
				points) [nq, ns]
			UqR: right state (typically evaluated at the quadrature 
				points) [nq, ns]

		Outputs:
		--------
		    drho: density jump [nq, 1]
		    dvel: velocity jump [nq, dim]
		    dp: pressure jump [nq, 1]
		'''
		dvel = velR - velL
		drho = UqR[:, srho] - UqL[:, srho]
		dp = physics.compute_variable("Pressure", UqR) - \
			physics.compute_variable("Pressure", UqL)

		return drho, dvel, dp

	def get_alphas(self, c, c2, dp, dvel, drho, rhoRoe):
		'''
		This method computes alpha_i = ith left eigenvector * dU.

		Inputs:
		-------
			c: speed of sound [nq, 1]
			c2: speed of sound squared [nq, 1]
			dp: pressure jump [nq, 1]
			dvel: velocity jump [nq, dim]
			drho: density jump [nq, 1]
			rhoRoe: Roe-averaged density [nq, 1]

		Outputs:
		--------
		    alphas: left eigenvectors multipled by dU [nq, ns]
		'''
		alphas = self.alphas 

		alphas[:, 0:1] = 0.5/c2*(dp - c*rhoRoe*dvel[:, 0:1])
		alphas[:, 1:2] = drho - dp/c2 
		alphas[:, -1:] = 0.5/c2*(dp + c*rhoRoe*dvel[:, 0:1])

		return alphas 

	def get_eigenvalues(self, velRoe, c):
		'''
		This method computes the eigenvalues.

		Inputs:
		-------
			velRoe: Roe-averaged velocity [nq, 1]
			c: speed of sound [nq, 1]

		Outputs:
		--------
		    evals: eigenvalues [nq, ns]
		'''
		evals = self.evals 

		evals[:,0:1] = velRoe[:,0:1] - c
		evals[:,1:2] = velRoe[:,0:1]
		evals[:,-1:] = velRoe[:,0:1] + c
		
		# entropy fix
		# ep = 1e-2
		# for iq in range(evals.shape[0]):
		# 	eps = ep*c[iq]
		# 	for i in range(evals.shape[-1]):
		# 		if evals[iq,i] < eps and evals[iq,i] > -eps:
		# 			eps1 = 1./eps
		# 			evals[iq,i] = 0.5*(eps+evals[iq,i]*evals[iq,i]*eps1)
		return evals 

	def get_right_eigenvectors(self, c, evals, velRoe, HRoe):
		R = self.R

		# first row
		# R[:,0,[0,1,-1]] = 1.
		R[:,0,0:2] = 1.; R[:,0,-1] = 1.
		# second row
		R[:,1,0] = evals[:,0]; R[:,1,1] = velRoe[:,0]; R[:,1,-1] = evals[:,-1]
		# last row
		R[:,-1,0:1] = HRoe - velRoe[:,0:1]*c; R[:,-1,1:2] = 0.5*np.sum(velRoe*velRoe, axis=1, keepdims=True)
		R[:,-1,-1:] = HRoe + velRoe[:,0:1]*c

		return R 


	def compute_flux(self, physics, UqL_std, UqR_std, n):
		'''
		Function: ConvFluxLaxFriedrichs
		-------------------
		This function computes the numerical flux (dotted with the normal)
		using the Lax-Friedrichs flux function

		Inputs:
		    gamma: specific heat ratio
		    UqL: Left state
		    UqR: Right state
		    n: Normal vector (assumed left to right)

		Outputs:
		    F: Numerical flux dotted with the normal, i.e., F_hat dot n
		'''

		# Unpack
		UqL = self.UqL 
		UqR = self.UqR
		srho = physics.get_state_slice("Density")
		smom = physics.get_momentum_slice()
		gamma = physics.gamma

		# Unit normals
		nmag = np.linalg.norm(n, axis=1, keepdims=True)
		n_unit = n/nmag

		# Copy values before rotating
		UqL[:] = UqL_std
		UqR[:] = UqR_std

		# Rotated coordinate system
		UqL = self.rotate_coord_sys(smom, UqL, n_unit)
		UqR = self.rotate_coord_sys(smom, UqR, n_unit)

		# Velocities
		velL = UqL[:, smom]/UqL[:, srho]
		velR = UqR[:, smom]/UqR[:, srho]

		# Roe-averaged state
		rhoRoe, velRoe, HRoe = self.roe_average_state(physics, srho, velL, 
				velR, UqL, UqR)

		# Speed of sound from Roe-averaged state
		c2 = (gamma - 1.)*(HRoe - 0.5*np.sum(velRoe*velRoe, axis=1, 
				keepdims=True))
		if np.any(c2 <= 0.):
			# Non-physical state
			raise errors.NotPhysicalError
		c = np.sqrt(c2)

		# Jumps
		drho, dvel, dp = self.get_differences(physics, srho, velL, velR, 
				UqL, UqR)

		# alphas (left eigenvectors multipled by dU)
		# alphas[:,[0]] = 0.5/c2*(dp - c*rhoRoe*dvel[:,[0]])
		# alphas[:,[1]] = drho - dp/c2 
		# alphas[:,ydim] = rhoRoe*dvel[:,[-1]]
		# alphas[:,[-1]] = 0.5/c2*(dp + c*rhoRoe*dvel[:,[0]])
		alphas = self.get_alphas(c, c2, dp, dvel, drho, rhoRoe)

		# Eigenvalues
		# evals[:,[0]] = velRoe[:,[0]] - c
		# evals[:,1:-1] = velRoe[:,[0]]
		# evals[:,[-1]] = velRoe[:,[0]] + c
		evals = self.get_eigenvalues(velRoe, c)

		# Right eigenvector matrix
		# first row
		# R[:,0,[0,1,-1]] = 1.; R[:,0,ydim] = 0.
		# # second row
		# R[:,1,0] = evals[:,0]; R[:,1,1] = velRoe[:,0]; R[:,1,ydim] = 0.; R[:,1,-1] = evals[:,-1]
		# # last row
		# R[:,-1,[0]] = HRoe - velRoe[:,[0]]*c; R[:,-1,[1]] = 0.5*np.sum(velRoe*velRoe, axis=1, keepdims=True)
		# R[:,-1,[-1]] = HRoe + velRoe[:,[0]]*c; R[:,-1,ydim] = velRoe[:,[-1]]
		# # [third] row
		# R[:,ydim,0] = velRoe[:,[-1]];  R[:,ydim,1] = velRoe[:,[-1]]; 
		# R[:,ydim,-1] = velRoe[:,[-1]]; R[:,ydim,ydim] = 1.
		R = self.get_right_eigenvectors(c, evals, velRoe, HRoe)

		# Form flux Jacobian matrix multiplied by dU
		FRoe = np.matmul(R, np.expand_dims(np.abs(evals)*alphas, 
				axis=2)).squeeze(axis=2)

		FRoe = self.undo_rotate_coord_sys(smom, FRoe, n_unit)

		# Left flux
		FL = physics.get_conv_flux_projected(UqL_std, n_unit)

		# Right flux
		FR = physics.get_conv_flux_projected(UqR_std, n_unit)
		
		return nmag*(0.5*(FL+FR) - 0.5*FRoe)


class Roe2D(Roe1D):

	def rotate_coord_sys(self, smom, Uq, n):
		vel = self.vel
		vel[:] = Uq[:,smom]

		vel[:, 0] = np.sum(Uq[:, smom]*n, axis=1)
		vel[:, 1] = np.sum(Uq[:, smom]*n[:, ::-1]*np.array([[-1.,1.]]), 
				axis=1)
		
		Uq[:,smom] = vel

		return Uq

	def undo_rotate_coord_sys(self, smom, Uq, n):
		vel = self.vel
		vel[:] = Uq[:,smom]

		vel[:, 0] = np.sum(Uq[:, smom]*n*np.array([[1., -1.]]), axis=1)
		vel[:, 1] = np.sum(Uq[:, smom]*n[:, ::-1], axis=1)

		Uq[:,smom] = vel

		return Uq

	def get_alphas(self, c, c2, dp, dvel, drho, rhoRoe):
		alphas = self.alphas 

		alphas = super().get_alphas(c, c2, dp, dvel, drho, rhoRoe)

		alphas[:, 2:3] = rhoRoe*dvel[:, -1:]

		return alphas 

	def get_eigenvalues(self, velRoe, c):
		evals = self.evals 

		evals = super().get_eigenvalues(velRoe, c)

		evals[:, 2:3] = velRoe[:, 0:1]

		return evals 

	def get_right_eigenvectors(self, c, evals, velRoe, HRoe):
		R = self.R

		R = super().get_right_eigenvectors(c, evals, velRoe, HRoe)

		i = 2

		# first row
		R[:, 0, i] = 0.
		# second row
		R[:, 1, i] = 0.
		# last row
		R[:, -1, i] = velRoe[:, -1]
		# [third] row
		R[:, i, 0] = velRoe[:, -1];  R[:, i, 1] = velRoe[:, -1]; 
		R[:, i, -1] = velRoe[:, -1]; R[:, i, i] = 1.

		return R 


class HLLC1D(ConvNumFluxBase):
	def __init__(self, Uq=None):
		if Uq is not None:
			n = Uq.shape[0]
			ns = Uq.shape[1]
			dim = ns - 2
		else:
			n = 0; ns = 0; dim = 0

	def compute_flux(self, physics, UqL, UqR, n):

		# Indices
		srho = physics.get_state_slice("Density")
		smom = physics.get_momentum_slice()
		srhoE = physics.get_state_slice("Energy")

		NN = np.linalg.norm(n, axis=1, keepdims=True)
		n1 = n/NN

		gamma = physics.gamma

		# unpack left hand state
		rhoL = UqL[:, srho]
		uL = UqL[:, smom]/rhoL
		unL = uL * n1
		pL = physics.compute_variable("Pressure", UqL)
		cL = physics.compute_variable("SoundSpeed", UqL)
		# unpack right hand state
		rhoR = UqR[:, srho]
		uR = UqR[:, smom]/rhoR
		unR = uR * n1
		pR = physics.compute_variable("Pressure", UqR)
		cR = physics.compute_variable("SoundSpeed", UqR)	

		# calculate averages
		rho_avg = 0.5 * (rhoL + rhoR)
		c_avg = 0.5 * (cL + cR)

		# Step 1: Get pressure estimate in the star region
		pvrs = 0.5 * ((pL + pR) - (unR - unL)*rho_avg*c_avg)
		p_star = max(0., pvrs)

		pspl = p_star / pL
		pspr = p_star / pR

		# Step 2: Get SL and SR
		qL = 1.
		if pspl > 1.:
			qL = np.sqrt(1. + (gamma + 1.) / (2.*gamma) * (pspl - 1.)) 
		SL = unL - cL*qL

		qR = 1.
		if pspr > 1.:
			qR = np.sqrt(1. + (gamma + 1.) / (2.*gamma) * (pspr - 1.))
		SR = unR + cR*qR

		# Step 3: Get shear wave speed
		raa1 = 1./(rho_avg*c_avg)
		sss = 0.5*(unL+unR) + 0.5*(pL-pR)*raa1

		# flux assembly 

		# Left State
		FL = physics.get_conv_flux_projected(UqL, n1)
		# Right State
		FR = physics.get_conv_flux_projected(UqR, n1)

		Fhllc = np.zeros_like(FL)

		if SL >= 0.:
			Fhllc = FL
		elif SR <= 0.:
			Fhllc = FR
		elif (SL <= 0.) and (sss >= 0.):
			slul = SL - unL
			cl = slul/(SL - sss)
			sssul = sss - unL
			sssel = pL / (rhoL*slul)
			ssstl = sss+sssel
			c1l = rhoL*cl*sssul
			c2l = rhoL*cl*sssul*ssstl

			Fhllc[:, srho] = FL[:, srho] + SL*(UqL[:, srho]*(cl-1.))
			Fhllc[:, smom] = FL[:, smom] + SL*(UqL[:, smom]*(cl-1.)+c1l*n1)
			Fhllc[:, srhoE] = FL[:, srhoE] + SL*(UqL[:, srhoE]*(cl-1.)+c2l)

		elif (sss <= 0.) and (SR >= 0.):
			slur = SR - unR
			cr = slur/(SR - sss)
			sssur = sss - unR
			ssser = pR / (rhoR*slur)
			ssstr = sss+ssser
			c1r = rhoR*cr*sssur
			c2r = rhoR*cr*sssur*ssstr

			Fhllc[:, srho] = FR[:, srho] + SR*(UqR[:, srho]*(cr-1.))
			Fhllc[:, smom] = FR[:, smom] + SR*(UqR[:, smom]*(cr-1.)+c1r*n1)
			Fhllc[:, srhoE] = FR[:, srhoE] + SR*(UqR[:, srhoE]*(cr-1.)+c2r)
							  
		return Fhllc