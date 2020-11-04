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
    # SimpleDetonation2 = auto()
    # SimpleDetonation3 = auto()

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

# class SimpleDetonation2(FcnBase):
# 	def __init__(self, uL=np.array([2.,4.,40.,0.]), xshock=0.):
# 		# These values represent the unburned state.
# 		self.uL = uL
# 		self.xshock = xshock

# 	def get_state(self, physics, x, t):

# 		uL = self.uL
# 		xshock = self.xshock

# 		rhoL = uL[0]
# 		vL = uL[1]
# 		pL = uL[2]
# 		yL = uL[3]

# 		# Unpack relevant constants from physics class.
# 		srho, srhou, srhoE, srhoz = physics.get_state_slices()
# 		gam = physics.gamma
# 		qo = physics.qo
# 		Uq = np.zeros([x.shape[0], physics.NUM_STATE_VARS])

# 		delta = np.sqrt((2.*(gam - 1.)) / (gam + 1.))

# 		rhoR = gam / (1. + delta)
# 		vR = -1.*delta
# 		pR = 1. - gam * delta
# 		yR = 1.

# 		ileft = (x <= xshock).reshape(-1)
# 		iright = (x > xshock).reshape(-1)

# 		Uq[iright, srho] = rhoR
# 		Uq[ileft, srho] = rhoL
# 		# Momentum
# 		Uq[iright, srhou] = rhoR*vR
# 		Uq[ileft, srhou] = rhoL*vL
# 		# Energy
# 		Uq[iright, srhoE] = pR/(gam-1.) + 0.5*rhoR*vR*vR + qo*rhoR*yR
# 		Uq[ileft, srhoE] = pL/(gam-1.) + 0.5*rhoL*vL*vL + qo*rhoL*yL
# 		# MixtureFraction
# 		Uq[iright, srhoz] = rhoR*yR
# 		Uq[ileft, srhoz] = rhoL*yL

# 		return Uq

# class SimpleDetonation3(FcnBase):
# 	def __init__(self, uL=np.array([2.,4.,40.,0.]), uR=np.array([0.,0.,0.,0.]), xshock=0.):
# 		# These values represent the unburned state.
# 		self.uL = uL
# 		self.uR = uR
# 		self.xshock = xshock

# 	def get_state(self, physics, x, t):

# 		uL = self.uL
# 		uR = self.uR
# 		xshock = self.xshock

# 		rhoL = uL[0]
# 		vL = uL[1]
# 		pL = uL[2]
# 		yL = uL[3]

# 		rhoR = uR[0]
# 		vR = uR[1]
# 		pR = uR[2]
# 		yR = uR[3]

# 		# Unpack relevant constants from physics class.
# 		srho, srhou, srhoE, srhoz = physics.get_state_slices()
# 		gam = physics.gamma
# 		qo = physics.qo
# 		Uq = np.zeros([x.shape[0], physics.NUM_STATE_VARS])

# 		ileft = (x <= xshock).reshape(-1)
# 		iright = (x > xshock).reshape(-1)
# 		# Density
# 		rhoR = 1. + 0.5*np.sin(2.*x[iright])
# 		Uq[iright, srho] = rhoR
# 		Uq[ileft, srho] = rhoL
# 		# Momentum
# 		Uq[iright, srhou] = rhoR*vR
# 		Uq[ileft, srhou] = rhoL*vLA
# 		# Energy
# 		Uq[iright, srhoE] = pR/(gam-1.) + 0.5*rhoR*vR*vR + qo*rhoR*yR
# 		Uq[ileft, srhoE] = pL/(gam-1.) + 0.5*rhoL*vL*vL + qo*rhoL*yL
# 		# MixtureFraction
# 		Uq[iright, srhoz] = rhoR*yR
# 		Uq[ileft, srhoz] = rhoL*yL

# 		return Uq

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
# class HLLC1D(ConvNumFluxBase):
# 	def __init__(self, Uq=None):
# 		if Uq is not None:
# 			n = Uq.shape[0]
# 			ns = Uq.shape[1]
# 			dim = ns - 2
# 		else:
# 			n = 0; ns = 0; dim = 0

# 	def compute_flux(self, physics, UqL, UqR, n):

# 		# Indices
# 		srho = physics.get_state_slice("Density")
# 		smom = physics.get_momentum_slice()
# 		srhoE = physics.get_state_slice("Energy")
# 		srhoY = physics.get_state_slice("Mixture")


# 		NN = np.linalg.norm(n, axis=1, keepdims=True)
# 		n1 = n/NN

# 		gam = physics.gamma

# 		# unpack left hand state
# 		rhoL = UqL[:, srho]
# 		uL = UqL[:, smom]/rhoL
# 		unL = uL * n1
# 		pL = physics.compute_variable("Pressure", UqL)
# 		cL = physics.compute_variable("SoundSpeed", UqL)
# 		# unpack right hand state
# 		rhoR = UqR[:, srho]
# 		uR = UqR[:, smom]/rhoR
# 		unR = uR * n1
# 		pR = physics.compute_variable("Pressure", UqR)
# 		cR = physics.compute_variable("SoundSpeed", UqR)

# 		# calculate averages
# 		rho_avg = 0.5 * (rhoL + rhoR)
# 		c_avg = 0.5 * (cL + cR)

# 		# Step 1: Get pressure estimate in the star region
# 		pvrs = 0.5 * ((pL + pR) - (unR - unL)*rho_avg*c_avg)
# 		p_star = max(0., pvrs)

# 		pspl = p_star / pL
# 		pspr = p_star / pR

# 		# Step 2: Get SL and SR
# 		qL = 1.
# 		if pspl > 1.:
# 			qL = np.sqrt(1. + (gam + 1.) / (2.*gam) * (pspl - 1.))
# 		SL = unL - cL*qL

# 		qR = 1.
# 		if pspr > 1.:
# 			qR = np.sqrt(1. + (gam + 1.) / (2.*gam) * (pspr - 1.))
# 		SR = unR + cR*qR

# 		# Step 3: Get shear wave speed
# 		raa1 = 1./(rho_avg*c_avg)
# 		sss = 0.5*(unL+unR) + 0.5*(pL-pR)*raa1

# 		# flux assembly

# 		# Left State
# 		FL = physics.get_conv_flux_projected(UqL, n1)
# 		# Right State
# 		FR = physics.get_conv_flux_projected(UqR, n1)

# 		Fhllc = np.zeros_like(FL)

# 		if SL >= 0.:
# 			Fhllc = FL
# 		elif SR <= 0.:
# 			Fhllc = FR
# 		elif (SL <= 0.) and (sss >= 0.):
# 			slul = SL - unL
# 			cl = slul/(SL - sss)
# 			sssul = sss - unL
# 			sssel = pL / (rhoL*slul)
# 			ssstl = sss+sssel
# 			c1l = rhoL*cl*sssul
# 			c2l = rhoL*cl*sssul*ssstl

# 			Fhllc[:, srho] = FL[:, srho] + SL*(UqL[:, srho]*(cl-1.))
# 			Fhllc[:, smom] = FL[:, smom] + SL*(UqL[:, smom]*(cl-1.)+c1l*n1)
# 			Fhllc[:, srhoE] = FL[:, srhoE] + SL*(UqL[:, srhoE]*(cl-1.)+c2l)
# 			Fhllc[:, srhoY] = FL[:, srhoY] + SL*(UqL[:, srhoY]*(cl-1.))

# 		elif (sss <= 0.) and (SR >= 0.):
# 			slur = SR - unR
# 			cr = slur/(SR - sss)
# 			sssur = sss - unR
# 			ssser = pR / (rhoR*slur)
# 			ssstr = sss+ssser
# 			c1r = rhoR*cr*sssur
# 			c2r = rhoR*cr*sssur*ssstr

# 			Fhllc[:, srho] = FR[:, srho] + SR*(UqR[:, srho]*(cr-1.))
# 			Fhllc[:, smom] = FR[:, smom] + SR*(UqR[:, smom]*(cr-1.)+c1r*n1)
# 			Fhllc[:, srhoE] = FR[:, srhoE] + SR*(UqR[:, srhoE]*(cr-1.)+c2r)
# 			Fhllc[:, srhoY] = FR[:, srhoY] + SR*(UqR[:, srhoY]*(cr-1.))

# 		return Fhllc
