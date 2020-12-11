# ------------------------------------------------------------------------ #
#
#       File : src/physics/chemistry/functions.py
#
#       Contains definitions of Functions, boundary conditions, and source
#       terms for the 1D Euler equations with a simple transport equation
#		for mass fraction.
#
# ------------------------------------------------------------------------ #
from enum import Enum, auto
import numpy as np
from scipy.optimize import fsolve, root

from physics.base.data import (FcnBase, BCWeakRiemann, BCWeakPrescribed,
        SourceBase, ConvNumFluxBase)


class FcnType(Enum):
    DensityWave = auto()
    SimpleDetonation1 = auto()
    OverdrivenDetonation = auto()


class SourceType(Enum):
    Arrhenius = auto()
    # Heaviside = auto()


class ConvNumFluxType(Enum):
	Roe = auto()
	# HLLC = auto()


'''
State functions
'''
class DensityWave(FcnBase):
	def __init__(self, p = 1.0):
		self.p = p

	def get_state(self, physics, x, t):
		p = self.p
		srho, srhou, srhoE, srhoz = physics.get_state_slices()
		gam = physics.gamma
		qo = physics.qo
		Uq = np.zeros([x.shape[0], x.shape[1], physics.NUM_STATE_VARS])

		rho = 1.0+0.1*np.sin(2.*np.pi*x)
		rhou = rho*1.0
		rhoz = rho*1.0
		rhoE = (p/(gam-1.))+0.5*rhou**2/rho + qo*rhoz

		Uq[:, :, srho] = rho
		Uq[:, :, srhou] = rhou
		Uq[:, :, srhoE] = rhoE
		Uq[:, :, srhoz] = rhoz

		return Uq # [ne, nq, ns]


class SimpleDetonation1(FcnBase):
	def __init__(self, rho_u=1., u_u=0., p_u=1., Y_u=1., xshock = 10.):
		# These values represent the unburned state.
		self.rho_u = rho_u
		self.u_u = u_u
		self.p_u = p_u
		self.Y_u = Y_u
		self.xshock = xshock

	def get_state(self, physics, x, t):
		# Unpack the unburned state.
		rho_u = self.rho_u
		u_u = self.u_u
		p_u = self.p_u
		Y_u = self.Y_u
		xshock = self.xshock

		# Unpack relevant constants from physics class.
		srho, srhou, srhoE, srhoz = physics.get_state_slices()
		gam = physics.gamma
		qo = physics.qo

		b = -p_u - rho_u*qo * (gam-1.)
		c = p_u**2 + (2.*(gam-1.) * rho_u*p_u*qo) / (gam+1.)

		p_b = -b + (b**2 - c)**0.5 # burned gas pressure
		rho_b = rho_u * (p_b * (gam+1.) - p_u) \
				/ (gam*p_b) # burned gas density
		scj = (rho_u*u_u + (gam*p_b*rho_b)**0.5) / rho_u # detonation speed
		u_b = scj - (gam*p_b / rho_b)**0.5 # burned gas velocity
		Y_b = 0. # burned gas mixture fraction

		xshock += scj*t

		''' Fill state '''
		Uq = np.zeros([x.shape[0], x.shape[1], physics.NUM_STATE_VARS])

		for elem_ID in range(Uq.shape[0]):
			ileft = (x[elem_ID] <= xshock).reshape(-1)
			iright = (x[elem_ID] > xshock).reshape(-1)
			# Density
			Uq[elem_ID, iright, srho] = rho_u
			Uq[elem_ID, ileft, srho] = rho_b
			# Momentum
			Uq[elem_ID, iright, srhou] = rho_u*u_u
			Uq[elem_ID, ileft, srhou] = rho_b*u_b
			# Energy
			Uq[elem_ID, iright, srhoE] = (p_u/(gam-1.) + 0.5*rho_u*u_u*u_u +
					qo*rho_u*Y_u)
			Uq[elem_ID, ileft, srhoE] = (p_b/(gam-1.) + 0.5*rho_b*u_b*u_b +
					qo*rho_b*Y_b)
			# MixtureFraction
			Uq[elem_ID, iright, srhoz] = rho_u*Y_u
			Uq[elem_ID, ileft, srhoz] = rho_b*Y_b

		return Uq # [ne, nq, ns]

class OverdrivenDetonation(FcnBase):
	'''
	The one dimensional overdriven detonation case is a temporally evolving
	unsteady wave. The gas mixture is considered calorically perfect.

	The chemical kinetics are governed by a single-step irreversible 
	reaction (see Arrhenius in source terms below).

	A single species mass fraction characterizes the conversion from the
	unburnt to burnt state for the single step chemistry. The initial 
	conditions are determined from the Zeldovich, Neumann, and Doering (ZND) 
	profiles by specifying an overdrive factor of 1.6. Details of solving 
	for the ZND initial condition can be found in Fickett and Davis:

		[1] Fickett, W., and Davis, W., Detonation - Theory and Experiment, 
			Dover Publications, 2000.

	Further details of this case can be found in the following references:

		[2] Hwang, P., Fedkiw, R., Merriman, B., Aslam, T., Karagozian, A., 
			and Osher, S., “Numerical resolution of pulsating detonation
			waves,” Combustion Theory and Modeling, Vol. 4, 2000, pp. 
			217–240.

		[3] Lv, Y., and Ihme, M., “Discontinuous Galerkin method for 
			multicomponent chemically reacting flows and combustion,” 
			Journal of Computational Physics, Vol. 270, 2014, pp. 105–137.

		[4] Bornhoft, B., Ching, E., and Ihme, M., "Time integration 
			considerations for the solution of reacting flows using 
			discontinuous Galerkin methods". AIAA SciTech, 2021.

	Attributes:
	-----------
	xshock: float
		initial location of shock
	'''
	def __init__(self, xshock=75.0):
		'''
		This method initializes the attributes.

		Inputs:
		-------
		    xshock: initial location of shock

		Outputs:
		--------
		    self: attributes initialized
		'''
		self.xshock = xshock
		self.lam = None

	def get_state(self, physics, x, t):
		# Unpack
		xshock = self.xshock

		srho, srhou, srhoE, srhoY = physics.get_state_slices()

		gamma = physics.gamma
		qo = physics.qo
		R = physics.R

		# Source term constants
		Ta = physics.source_terms[0].Tign
		A = physics.source_terms[0].A

		# Normalized Pre-shock state 
		rho1 = 1.
		p1 = 1.
		u1 = 0.
		y1 = 1.

		# Sound speed and M calc
		a1 = np.sqrt(gamma*p1/rho1)

		# Specify overdriven shock speed from Ref. [2]
		W = 8.6134

		# Calculate Mach number
		M = W / a1

		dt = 0.5
		tfinal = 2.0 * A # Two times A is enough to get to steady region.
		NumTimeSteps = int(tfinal/dt)
		t = 0.
		if self.lam is None: # Only compute on initialization.
			# Allocate arrays
			lam = np.zeros([NumTimeSteps])
			xref = np.zeros([NumTimeSteps])
			p = np.zeros([NumTimeSteps])
			rho = np.zeros([NumTimeSteps])
			u = np.zeros([NumTimeSteps])
			T = np.zeros([NumTimeSteps])

			# Use Forward Euler integration to solve for reaction layer 
			# properties.
			for i in range(NumTimeSteps-1):
				alpha = np.sqrt(((M-(1./M))**2 - 2.*(gamma**2-1.)* \
				(lam[i]*qo/gamma))/((gamma+1.)**2*M**2))
				p[i] = (gamma*M**2 + 1.)/(gamma+1.) + gamma*M**2 * alpha
				v = (gamma + (1./M**2))/(gamma+1.) - alpha
				rho[i] = 1./v
				u[i] = ((M - (1/M))/(gamma+1.) + M*alpha)*np.sqrt(gamma)
				T[i] = p[i]/(rho[i]*R)
				
				rhs = (1 - lam[i]) * np.exp(-Ta/(R*T[i]))

				lam[i+1] = lam[i] + dt * rhs
				t = t + dt
				xref[i+1] = xref[i] + (W-u[i])*dt/A

			# Store properties
			self.lam = lam
			self.p = p
			self.rho = rho
			self.u = u
			self.T = T
			self.xref = xref

		else:

			# Unpack
			lam = self.lam
			p = self.p
			rho = self.rho
			u = self.u
			T = self.T
			xref = self.xref

		Uq = np.zeros([x.shape[0], x.shape[1], physics.NUM_STATE_VARS])
		for elem_ID in range(Uq.shape[0]):

			# Determine left, right, and middle (reacting layer) locations
			ileft = (x[elem_ID] <= xshock - 3.0).reshape(-1)
			iright = (x[elem_ID] > xshock).reshape(-1)
			imiddle = np.logical_and(x[elem_ID] <= xshock,
					x[elem_ID] > xshock- 3.0).reshape(-1)
			if np.any(imiddle) == True:
				
				for j in range(Uq.shape[1]):
					if imiddle[j] == True:
						xdist = xshock - xref
						index = (np.abs(xdist - x[elem_ID,j])).argmin()

						# Density
						Uq[elem_ID, j, srho] = rho[index]
						# Momentum
						Uq[elem_ID, j, srhou] = rho[index]*(u[index])
						# Energy
						Uq[elem_ID, j, srhoE] = p[index]/(gamma - 1.) + \
								0.5*rho[index]*(u[index])**2 + \
								qo*rho[index]*(1.-lam[index])
						# Mass fraction
						Uq[elem_ID, j, srhoY] = rho[index]*(1.-lam[index])

			else:
					# Density
				Uq[elem_ID, iright, srho] = rho1
				# Momentum
				Uq[elem_ID, iright, srhou] = rho1*u1
				# Energy (Mass Fraction is one at iright state)
				Uq[elem_ID, iright, srhoE] = p1/(gamma - 1.) + \
						0.5*rho1*u1*u1 + qo*rho1*1.
				# Mass fraction (Mass Fraction is one at iright state)
				Uq[elem_ID, iright, srhoY] = rho1*1.


				# Calculate post-detonation properties.
				alpha = np.sqrt(((M-(1./M))**2 - 2*(gamma**2-1.)* \
				(1.*qo/gamma))/((gamma+1.)**2*M**2))
				p3 = (gamma*M**2 + 1.)/(gamma+1.) + gamma*M**2 * alpha
				v3 = (gamma + (1./M**2))/(gamma+1.) - alpha
				rho3 = 1./v3
				u3 = ((M - (1/M))/(gamma+1.) + M*alpha)*np.sqrt(gamma)

				# Density
				Uq[elem_ID, ileft, srho] = rho3
				# Momentum
				Uq[elem_ID, ileft, srhou] = rho3*(u3)
				# Energy (Mass Fraction is zero at ileft state)
				Uq[elem_ID, ileft, srhoE] = p3/(gamma - 1.) + \
						0.5*rho3*(u3)**2 + qo*rho3*0.
				# Mass fraction (Mass Fraction is zero at ileft state)
				Uq[elem_ID, ileft, srhoY] = rho3*0.						

		return Uq # [ne, nq, ns]


'''
Source term functions
'''
class Arrhenius(SourceBase):
	def __init__(self, A=16418., b=0.1, Tign=15.):
		self.A = A
		self.b = b
		self.Tign = Tign

	def get_source(self, physics, Uq, x, t):
		# Unpack
		A = self.A
		b = self.b
		Tign = self.Tign

		irho, irhou, irhoE, irhoY = physics.get_state_indices()

		# Get temperature and calculate arrhenius rate constant
		T = physics.compute_variable("Temperature", Uq)
		K = A * T**b * np.exp(-Tign / T)

		# Calculate source term
		S = np.zeros_like(Uq)
		S[:, :, irhoY] = -K[:, :, 0] * Uq[:, :, irhoY]

		return S # [ne, nq, ns, ns]

	def get_jacobian(self, physics, Uq, x, t):
		# Unpack
		A = self.A
		Tign = self.Tign
		ne = Uq.shape[0]
		nq = Uq.shape[1]

		irho, irhou, irhoE, irhoY = physics.get_state_indices()

		# Allocate jacobian matrix
		jac = np.zeros([ne, nq, Uq.shape[-1], Uq.shape[-1]])

		T = physics.compute_variable("Temperature", Uq)
		K = A * np.exp(-Tign / T)

		elem_IDs = np.where(T<0)[0]
		K[elem_IDs] = 0.0
		# Calculate the temperature jacobian
		dTdU = get_temperature_jacobian(physics, Uq)

		# Get dKdU
		dKdrho =  (A * Tign * np.exp(-Tign / T) * \
				dTdU[:, :, irhoY, irho].reshape([ne, nq, 1]))  / T**2
		dKdrhou = (A * Tign * np.exp(-Tign / T) * \
				dTdU[:, :, irhoY, irhou].reshape([ne, nq, 1])) / T**2
		dKdrhoE = (A * Tign * np.exp(-Tign / T) * \
				dTdU[:, :, irhoY, irhoE].reshape([ne, nq, 1])) / T**2
		dKdrhoY = (A * Tign * np.exp(-Tign / T) * \
				dTdU[:, :, irhoY, irhoY].reshape([ne, nq, 1])) / T**2

		# Calculate jacobian of the source term
		jac[:, :, irhoY, irho] =  (-1.*dKdrho[:, :, 0] * Uq[:, :, irhoY])
		jac[:, :, irhoY, irhou] = (-1.*dKdrhou[:, :, 0] * Uq[:, :, irhoY])
		jac[:, :, irhoY, irhoE] = (-1.*dKdrhoE[:, :, 0] * Uq[:, :, irhoY])
		jac[:, :, irhoY, irhoY] = (-1.*dKdrhoY[:, :, 0] * Uq[:, :, irhoY]
				- K[:, :, 0] )

		return jac # [ne, nq, ns, ns]

def get_temperature_jacobian(physics, Uq):
		'''
		This function calculates the jacobian of the temperature (dT/dU).

		Inputs:
		-------
			physics: physics object instantiated for Chemistry
			Uq: state coefficients [ne, nq, ns]

		Outputs:
		--------
			dTdU: Jacobian of the temperature [ne, nq, ns, ns]
		'''
		irho, irhou, irhoE, irhoY = physics.get_state_indices()

		gam = physics.gamma
		qo = physics.qo
		R = physics.R

		dTdU = np.zeros([Uq.shape[0], Uq.shape[1], Uq.shape[-1],
				Uq.shape[-1]])

		rho = Uq[:, :, irho]
		rhou = Uq[:, :, irhou]
		rhoE = Uq[:, :, irhoE]
		rhoY = Uq[:, :, irhoY]

		E = rhoE/rho
		Y = rhoY/rho
		u = rhou/rho

		gamR = (gam - 1.) / R
		dTdU[:, :, irhoY, irho] = (gamR / rho) * (-1.*E + u**2 + qo*Y)
		dTdU[:, :, irhoY, irhou] = (gamR / rho) * (-1.*u)
		dTdU[:, :, irhoY, irhoE] = gamR / rho
		dTdU[:, :, irhoY, irhoY] = -1.*qo * (gamR / rho)

		return dTdU # [ne, nq, ns, ns]

# class Heaviside(SourceBase):
# 	def __init__(self, Da=1000., Tign=15.):
# 		self.Da = Da
# 		self.Tign = Tign

# 	def get_source(self, physics, Uq, x, t):

# 		# Unpack source term constants
# 		Da = self.Da
# 		Tign = self.Tign

# 		irho, irhou, irhoE, irhoz = physics.get_state_indices()

# 		# Uq = self.Uq
# 		T = physics.compute_variable("Temperature", Uq)
# 		K = np.zeros([Uq.shape[0]])

# 		for i in range(len(T)):
# 			if T[i] >= Tign:
# 				K[i] = Da
# 			else:
# 				K[i] = 0.

# 		S = np.zeros_like(Uq)
# 		S[:,irhoz] = -K[:] * Uq[:,irhoz]

# 		return S

# 	def get_jacobian(self, physics, Uq, x, t):

# 		# Unpack source term constants
# 		Da = self.Da
# 		Tign = self.Tign

# 		# Uq = self.Uq

# 		irho, irhou, irhoE, irhoY = physics.get_state_indices()

# 		T = physics.compute_variable("Temperature", Uq)
# 		K = np.zeros([Uq.shape[0]])

# 		for i in range(len(T)):
# 			if T[i] >= Tign:
# 				K[i] = Da
# 			else:
# 				K[i] = 0.

# 		jac = np.zeros([Uq.shape[0], Uq.shape[-1], Uq.shape[-1]])
# 		jac[:, irhoY, irhoY] = -K

# 		return jac

'''
Numerical flux functions
'''
