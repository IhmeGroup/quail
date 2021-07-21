# ------------------------------------------------------------------------ #
#
#       File : src/physics/scalar/functions.py
#
#       Contains definitions of Functions, boundary conditions, and source
#       terms for scalar equations.
#
# ------------------------------------------------------------------------ #
import sys
sys.path.append('/Users/brettbornhoft/utilities/pyJac')
import pyjacob
import cantera as ct
from enum import Enum, auto
import numpy as np
from scipy.optimize import root

from physics.base.data import FcnBase, BCWeakRiemann, BCWeakPrescribed, \
		SourceBase, ConvNumFluxBase


class FcnType(Enum):
	'''
	Enum class that stores the types of analytic functions for initial
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
	PendulumExact = auto()
	ModelPSR = auto()

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
	ScalarArrhenius = auto()
	ScalarMixing = auto()
	Pendulum = auto()
	Mixing = auto()
	Reacting = auto()

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

class PendulumExact(FcnBase):
	def __init__(self):
		pass

	def get_state(self, physics, x, t):

		g = physics.g
		l = physics.l

		Uq = np.zeros([x.shape[0], x.shape[1], physics.NUM_STATE_VARS])

		Uq[:, :, 0] = 0.1745 * np.cos(np.sqrt(g/l)*t)
		Uq[:, :, 1] = -0.1745 * np.sqrt(g/l) * np.sin(np.sqrt(g/l)*t)

		return Uq

class ModelPSR(FcnBase):
	def __init__(self):
		pass

	def get_state(self, physics, x, t):
		# unpack

		P = physics.P
		Tu = physics.Tu
		phi = physics.phi
		tau = physics.tau

		# filename = '/Users/brettbornhoft/utilities/pyJac/data/h2o2.cti'
		# gas = ct.Solution(filename)
		gas = ct.Solution('h2o2.yaml')
		n2_ind = gas.species_index('Ar')
		specs = gas.species()[:]
		gas = ct.Solution(thermo='IdealGas', kinetics='GasKinetics',
    		species=specs[:n2_ind] + specs[n2_ind + 1:] + [specs[n2_ind]],
    		reactions=gas.reactions())
		gas.TPX = Tu, P, "H2:{},O2:{},N2:{}".format(phi, 0.5, 0.5*3.76)
		y0 = np.hstack((gas.T, gas.Y))

		Uq = np.zeros([x.shape[0], x.shape[1], physics.NUM_STATE_VARS])
		Uq[:, :] = y0

		# Save the gas object for passing to source terms
		physics.gas = gas

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


class ScalarMixing(SourceBase):
	'''
	Mixing source term for scalar PSR model problem

	Attributes:
	-----------
	Da: float
		Dahmkohler number
	'''
	def __init__(self, Da=15.89, **kwargs):
		super().__init__(kwargs)
		'''
		This method initializes the attributes.

		Inputs:
		-------
		    Da: Dahmkohler number

		Outputs:
		--------
		    self: attributes initialized
		'''
		self.Da = Da

	def get_source(self, physics, Uq, x, t):
		Da = self.Da
		T_in = physics.T_in

		S = (1./Da) * (T_in - Uq) 

		return S

	def get_jacobian(self, physics, Uq, x, t):
		Da = self.Da
		# jac = get_numerical_jacobian(self, physics, Uq, x, t)

		return -1./Da


class ScalarArrhenius(SourceBase):
	'''
	Arrhenius source term for scalar PSR model problem
	'''
	def __init__(self, **kwargs):
		super().__init__(kwargs)

	def get_source(self, physics, Uq, x, t):
		T_ad = physics.T_ad
		T_a = physics.T_a

		S = (T_ad - Uq) * np.exp(-T_a/Uq) 
		return S

	def get_jacobian(self, physics, Uq, x, t):
		T_ad = physics.T_ad
		T_a = physics.T_a

		jac = -np.exp(-T_a/Uq) * (Uq**2 - T_a*T_ad + T_a*Uq)/Uq**2
		# jac = get_numerical_jacobian(self, physics, Uq, x, t)

		return np.expand_dims(jac, axis=-1)

class Pendulum(SourceBase):
	'''
	Arrhenius source term for scalar PSR model problem
	'''
	def __init__(self, **kwargs):
		super().__init__(kwargs)

	def get_source(self, physics, Uq, x, t):
		g = physics.g
		l = physics.l

		S = np.zeros_like(Uq)
		S[:, :, 0] = Uq[:, :, 1]
		S[:, :, 1] = (-g/l) * Uq[:, :, 0]

		return S

	def get_jacobian(self, physics, Uq, x, t):
		g = physics.g
		l = physics.l

		ns = physics.NUM_STATE_VARS

		jac = np.zeros([Uq.shape[0], Uq.shape[1], ns, ns])

		jac[:, :, 0, 0] = 1.
		jac[:, :, 0, 1] = 0.
		jac[:, :, 1, 0] = 0.
		jac[:, :, 1, 1] = -g/l
		
		return jac

class Mixing(SourceBase):
	'''
	Mixing source term for PSR model problem

	Attributes:
	-----------
	'''
	def __init__(self, **kwargs):
		super().__init__(kwargs)

	def get_source(self, physics, Uq, x, t):
		# Unpack T and Y
		S = np.zeros([Uq.shape[0], Uq.shape[1], physics.NUM_STATE_VARS])
		for i in range(Uq.shape[1]):
			T = Uq[0,i,0]
			y = Uq[0,i,1:]

			tau = physics.tau
			gas = physics.gas
			gas.set_unnormalized_mass_fractions(y)

			gas.TPY = T, physics.P, y

			rho = gas.density
			wdot = gas.net_production_rates
			h_hat = gas.partial_molar_enthalpies
			cp = gas.cp_mass

			mw = gas.molecular_weights

			dTdt = (1./(tau*cp)) * np.dot(physics.yin[1:] ,(physics.hin/mw - h_hat/mw)) 
			dYdt = (1./tau) * (physics.yin[1:] - y)

			S[:, i] = np.hstack((dTdt, dYdt))

		return S

	def get_jacobian(self, physics, Uq, x, t):
		
		jac = get_numerical_jacobian(self, physics, Uq, x, t)

		return jac


class Reacting(SourceBase):
	'''
	Arrhenius source term for scalar PSR model problem
	'''
	def __init__(self, **kwargs):
		super().__init__(kwargs)

	def get_source(self, physics, Uq, x, t):
		# Unpack T and Y
		S = np.zeros([Uq.shape[0], Uq.shape[1], physics.NUM_STATE_VARS])
		for i in range(Uq.shape[1]):
			T = Uq[0,i,0]
			y = Uq[0,i,1:]

			tau = physics.tau
			gas = physics.gas
			gas.set_unnormalized_mass_fractions(y)

			gas.TPY = T, physics.P, y

			rho = gas.density
			wdot = gas.net_production_rates
			h_hat = gas.partial_molar_enthalpies
			cp = gas.cp_mass

			mw = gas.molecular_weights

			dTdt = -1.*np.dot(h_hat, wdot) * (1./ (rho*cp))
			dYdt = wdot * mw / rho

			S[:, i] = np.hstack((dTdt, dYdt))

		return S

	def get_jacobian(self, physics, Uq, x, t):
		ns = physics.NUM_STATE_VARS
		jac_ = np.zeros([(ns-1)*(ns-1)])
		jac2 = np.zeros([Uq.shape[0], Uq.shape[1], ns, ns])

		for ie in range(Uq.shape[0]):
			for iq in range(Uq.shape[1]):
			    pyjacob.py_eval_jacobian(0, physics.P, Uq[ie, iq, :(ns-1)], jac_)
			    jac2[ie, iq, :(ns-1), :(ns-1)] = jac_.reshape([(ns-1), (ns-1)])

		# jac = get_numerical_jacobian(self, physics, Uq, x, t)
		# physics.jac2 = jac2.transpose(0,1,3,2)
		return jac2.transpose(0,1,3,2)


# def get_numerical_jacobian(source, physics, Uq, x, t):

# 	ns = physics.NUM_STATE_VARS
# 	eps = 1.e-6

# 	S = source.get_source(physics, Uq, x, t)
# 	Sperturb = np.zeros([ns, S.shape[0], S.shape[1], S.shape[2]])
# 	Sperturb2 = np.zeros_like(Sperturb)
# 	eps_ = Uq*eps

# 	jac = np.zeros([Uq.shape[0], Uq.shape[1], ns, ns])
# 	for i in range(ns):
# 		Uq_per = Uq.copy()
# 		Uq_per2 = Uq.copy()
# 		Uq_per[:, :, i] += eps_[:, :, i] 
# 		Uq_per2[:, :, i] -= eps_[:, :, i]
# 		Sperturb[i] = source.get_source(physics, Uq_per, x, t)
# 		# Sperturb2[i] = source.get_source(physics, Uq_per2, x, t)
# 	for i in range(ns):
# 		for j in range(ns):
# 				if eps_[:, :, j] == 0.0:
# 					jac[:, :, i, j] = 0
# 				else:
# 					# Second-order central difference
# 					# jac[:, :, i, j] = (Sperturb[j, :, :, i] - Sperturb2[j, :, :, i]) / (2*eps_[:, :, j])
# 					# First-order finite difference
# 					jac[:, :, i, j] = (Sperturb[j, :, :, i] - S[:, :, i]) / (eps_[:, :, j])

# 	return jac	


def get_numerical_jacobian(source, physics, Uq, x, t):


	nelem = Uq.shape[0]
	ns = physics.NUM_STATE_VARS
	eps = 1.e-6

	S = np.zeros([nelem, 1, ns])
	Sp = np.zeros([nelem, ns, 1, ns])
	Sp2 = np.zeros([nelem, ns, 1, ns])

	jac = np.zeros([Uq.shape[0], Uq.shape[1], ns, ns])

	for ielem in range(nelem):

		S[ielem] = source.get_source(physics, Uq[ielem].reshape([1,1,ns]), x, t)

		for i in range(ns):
			Uq_per = Uq[ielem].copy()
			# Uq2_per = Uq[ielem].copy()
			Uq_per[0, i] += eps# eps_[ielem, 0] 
			# Uq2_per[0, i] -= eps# eps_[ielem, 0] 

			Sp[ielem, i] = source.get_source(physics, Uq_per.reshape([1,1,ns]), x, t)
			# Sp2[ielem, i] = source.get_source(source, physics, Uq2_per.reshape([1,1,ns]), x, t)
		
			# import code; code.interact(local=locals())
	for ielem in range(nelem):
		for i in range(ns):
			for j in range(ns):
					# if eps_[ielem, 0, j] == 0.0:
						# jac[ielem, 0, i, j] = 0
					# else:
						# Second-order central difference
					# jac[ielem, 0, i, j] = (Sp[ielem, i, 0, j] - Sp2[ielem, i, 0, j]) / (2*eps)#eps_[:, :, j])
						# First-order finite difference
					jac[ielem, 0, i, j] = (Sp[ielem, i, 0, j] - S[ielem, 0, j]) / (eps)#eps_[ielem, 0])
			# import code; code.interact(local=locals())
	return jac.transpose(0, 1, 3, 2)











