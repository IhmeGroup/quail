# ------------------------------------------------------------------------ #
#
#       File : src/physics/zerodimensional/functions.py
#
#       Contains definitions of Functions, boundary conditions, and source
#       terms for zero dimensional equations.
#
# ------------------------------------------------------------------------ #
import importlib
from enum import Enum, auto
import numpy as np
from scipy.optimize import root

from external.optional_cantera import ct

from physics.base.data import FcnBase, BCWeakRiemann, BCWeakPrescribed, \
		SourceBase, ConvNumFluxBase


class FcnType(Enum):
	'''
	Enum class that stores the types of analytic functions for initial
	conditions, exact solutions, and/or boundary conditions. These
	functions are specific to the available scalar equation sets.
	'''
	PendulumExact = auto()
	MultispeicesPSR_IC = auto()

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

class PendulumExact(FcnBase):
	'''
	Exact solution to the 2nd order pendulum problem. 
	See zerodimensional.Pendulum for further details.
	'''
	def __init__(self):
		pass

	def get_state(self, physics, x, t):
		# unpack
		g = physics.g
		l = physics.l

		Uq = np.zeros([x.shape[0], x.shape[1], physics.NUM_STATE_VARS])

		Uq[:, :, 0] = 0.1745 * np.cos(np.sqrt(g/l)*t)
		Uq[:, :, 1] = -0.1745 * np.sqrt(g/l) * np.sin(np.sqrt(g/l)*t)

		return Uq

class MultispeicesPSR_IC(FcnBase):
	'''
	Constructs the initial condition for the multispecies 
	perfectly stirred reactor test case. See zerodimensional.MultispeciesPSR
	for more details.
	'''
	def __init__(self):
		pass

	def get_state(self, physics, x, t):
		# unpack
		P = physics.P
		Tu = physics.Tu
		phi = physics.phi
		tau = physics.tau
		physics.gas.TPX = Tu, P, "H2:{},O2:{},N2:{}".format(phi, 0.5, 0.5*3.76)
		y0 = np.hstack((physics.gas.T, physics.gas.Y))

		Uq = np.zeros([x.shape[0], x.shape[1], physics.NUM_STATE_VARS])
		Uq[:, :] = y0

		return Uq # [1, nq, ns]
'''
---------------------
Source term functions
---------------------
These classes inherit from the SourceBase class. See SourceBase for detailed
comments of attributes and methods. Information specific to the
corresponding child classes can be found below. These classes should
correspond to the SourceType enum members above.
'''

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
		return -1./self.Da


class ScalarArrhenius(SourceBase):
	'''
	Arrhenius-like source term for scalar PSR model problem
	'''
	def __init__(self, **kwargs):
		super().__init__(kwargs)

	def get_source(self, physics, Uq, x, t):
		# Unpack
		T_ad = physics.T_ad
		T_a = physics.T_a

		S = (T_ad - Uq) * np.exp(-T_a/Uq) 
		return S

	def get_jacobian(self, physics, Uq, x, t):
		# Unpack
		T_ad = physics.T_ad
		T_a = physics.T_a

		jac = -np.exp(-T_a/Uq) * (Uq**2 - T_a*T_ad + T_a*Uq)/Uq**2

		return np.expand_dims(jac, axis=-1)

class Pendulum(SourceBase):
	'''
	Source term for the second-order pendulum problem
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
		
		return jac # [1, nq, 2, 2]

class Mixing(SourceBase):
	'''
	Mixing source term for PSR model problem with H2/Air chemistry
	'''
	def __init__(self, **kwargs):
		super().__init__(kwargs)

	def get_source(self, physics, Uq, x, t):
		# Unpack T and Y
		S = np.zeros([Uq.shape[0], Uq.shape[1], physics.NUM_STATE_VARS])
		# Loop over quadrature points
		for i in range(Uq.shape[1]):
			T = Uq[0, i, 0]
			y = Uq[0, i, 1:]

			tau = physics.tau
			gas = physics.gas
			gas.set_unnormalized_mass_fractions(y)

			gas.TPY = T, physics.P, y

			rho = gas.density
			wdot = gas.net_production_rates
			h_hat = gas.partial_molar_enthalpies
			cp = gas.cp_mass

			mw = gas.molecular_weights

			dTdt = (1./(tau*cp)) * np.dot(physics.yin[1:], 
					(physics.hin/mw - h_hat/mw))
			dYdt = (1./tau) * (physics.yin[1:] - y)

			S[:, i] = np.hstack((dTdt, dYdt))

		return S # [1, nq, ns]

	def get_jacobian(self, physics, Uq, x, t):
		
		jac = get_numerical_jacobian(self, physics, Uq, x, t)

		return jac # [1, nq, ns, ns]


class Reacting(SourceBase):
	'''
	Arrhenius source term for PSR model problem with H2/Air chemistry
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

		return S # [1, nq, ns]

	def get_jacobian(self, physics, Uq, x, t):
		jac = get_numerical_jacobian(self, physics, Uq, x, t)

		return jac # [ne, nq, ns, ns]


def get_numerical_jacobian(source, physics, Uq, x, t):
	'''
	Calculates the numerical jacobian of a given source term. 
	Note: Needs vectorizing and testing for dimensional cases.

	Inputs:
	-------
		source: source object
		physics: physics object
		Uq: solution state at quadrature points [ne, nq, ns]
		x: coordinates in physical space [nq, ndims]
		t: time

	Outputs:
	--------
		jac: numerical jacobian (dS/dU) [ne, nq, ns, ns]
	'''
	nelem = Uq.shape[0]
	ns = physics.NUM_STATE_VARS
	eps = 1.e-6 #Note: Can be adjusted depending on the problem

	# Initialize source and perturbed source
	S = np.zeros([nelem, 1, ns])
	Sp = np.zeros([nelem, ns, 1, ns])

	jac = np.zeros([Uq.shape[0], Uq.shape[1], ns, ns])

	for ielem in range(nelem):
		# Get source term in each element
		S[ielem] = source.get_source(physics, 
				Uq[ielem].reshape([1, 1, ns]), x, t)
		# Construct the perturbed sources
		for i in range(ns):
			Uq_per = Uq[ielem].copy()
			Uq_per[0, i] += eps
			Sp[ielem, i] = source.get_source(physics, 
					Uq_per.reshape([1, 1, ns]), x, t)
	# First-order finite difference
	for ielem in range(nelem):
		for i in range(ns):
			for j in range(ns):
					jac[ielem, 0, i, j] = (Sp[ielem, i, 0, j] - \
						S[ielem, 0, j]) / (eps)

	return jac.transpose(0, 1, 3, 2)