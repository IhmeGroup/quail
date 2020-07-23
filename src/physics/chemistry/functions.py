import code
from enum import Enum, auto
import numpy as np
from scipy.optimize import fsolve, root

from physics.base.data import FcnBase, BCWeakRiemann, BCWeakPrescribed, SourceBase, ConvNumFluxBase

class FcnType(Enum):
    DensityWave = auto()
    SimpleDetonation1 = auto()
    SimpleDetonation2 = auto()
    SimpleDetonation3 = auto()

# class BCType(Enum):
	# SlipWall = auto()
	# PressureOutlet = auto()


class SourceType(Enum):
    Arrhenius = auto()
    Heaviside = auto()

class ConvNumFluxType(Enum):
	Roe = auto()


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
		Up = np.zeros([x.shape[0], physics.NUM_STATE_VARS])
		
		rho = 1.0+0.1*np.sin(2.*np.pi*x)
		rhou = rho*1.0
		rhoz = rho*1.0
		rhoE = (p/(gam-1.))+0.5*rhou**2/rho + qo*rhoz
		# rhoE = (p/(gam-1.))+0.5*rhou**2/rho


		Up[:,srho] = rho
		Up[:,srhou] = rhou
		Up[:,srhoE] = rhoE
		Up[:,srhoz] = rhoz

		return Up

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
		Up = np.zeros([x.shape[0], physics.NUM_STATE_VARS])

		b = -p_u - rho_u*qo * (gam-1.)
		c = p_u**2 + (2.*(gam-1.) * rho_u*p_u*qo) / (gam+1.)

		p_b = -b + (b**2 - c)**0.5 # burned gas pressure
		rho_b = rho_u * (p_b * (gam+1.) - p_u) / (gam*p_b) # burned gas density
		scj = (rho_u*u_u + (gam*p_b*rho_b)**0.5) / rho_u # detonation speed
		u_b = scj - (gam*p_b / rho_b)**0.5 # burned gas velocity
		Y_b = 0. # burned gas mixture fraction

		xshock += scj*t

		''' Fill state '''
		ileft = (x <= xshock).reshape(-1)
		iright = (x > xshock).reshape(-1)

		# Density
		Up[iright, srho] = rho_u
		Up[ileft, srho] = rho_b
		# Momentum
		Up[iright, srhou] = rho_u*u_u
		Up[ileft, srhou] = rho_b*u_b
		# Energy
		Up[iright, srhoE] = p_u/(gam-1.) + 0.5*rho_u*u_u*u_u + qo*rho_u*Y_u
		Up[ileft, srhoE] = p_b/(gam-1.) + 0.5*rho_b*u_b*u_b + qo*rho_b*Y_b
		# MixtureFraction
		Up[iright, srhoz] = rho_u*Y_u
		Up[ileft, srhoz] = rho_b*Y_b

		return Up

class SimpleDetonation2(FcnBase):
	def __init__(self, uL=np.array([2.,4.,40.,0.]), xshock=0.):
		# These values represent the unburned state.
		self.uL = uL
		self.xshock = xshock

	def get_state(self, physics, x, t):

		uL = self.uL
		xshock = self.xshock

		rhoL = uL[0]
		vL = uL[1]
		pL = uL[2]
		yL = uL[3]

		# Unpack relevant constants from physics class.
		srho, srhou, srhoE, srhoz = physics.get_state_slices()
		gam = physics.gamma
		qo = physics.qo
		Up = np.zeros([x.shape[0], physics.NUM_STATE_VARS])		

		delta = np.sqrt((2.*(gam - 1.)) / (gam + 1.))

		rhoR = gam / (1. + delta)
		vR = -1.*delta
		pR = 1. - gam * delta
		yR = 1.

		ileft = (x <= xshock).reshape(-1)
		iright = (x > xshock).reshape(-1)
		
		Up[iright, srho] = rhoR
		Up[ileft, srho] = rhoL
		# Momentum
		Up[iright, srhou] = rhoR*vR
		Up[ileft, srhou] = rhoL*vL
		# Energy
		Up[iright, srhoE] = pR/(gam-1.) + 0.5*rhoR*vR*vR + qo*rhoR*yR
		Up[ileft, srhoE] = pL/(gam-1.) + 0.5*rhoL*vL*vL + qo*rhoL*yL
		# MixtureFraction
		Up[iright, srhoz] = rhoR*yR
		Up[ileft, srhoz] = rhoL*yL

		return Up

class SimpleDetonation3(FcnBase):
	def __init__(self, uL=np.array([2.,4.,40.,0.]), uR=np.array([0.,0.,0.,0.]), xshock=0.):
		# These values represent the unburned state.
		self.uL = uL
		self.uR = uR
		self.xshock = xshock

	def get_state(self, physics, x, t):

		uL = self.uL
		uR = self.uR
		xshock = self.xshock

		rhoL = uL[0]
		vL = uL[1]
		pL = uL[2]
		yL = uL[3]
		
		rhoR = uR[0]
		vR = uR[1]
		pR = uR[2]
		yR = uR[3]

		# Unpack relevant constants from physics class.
		srho, srhou, srhoE, srhoz = physics.get_state_slices()
		gam = physics.gamma
		qo = physics.qo
		Up = np.zeros([x.shape[0], physics.NUM_STATE_VARS])
		
		ileft = (x <= xshock).reshape(-1)
		iright = (x > xshock).reshape(-1)
		# Density
		rhoR = 1. + 0.5*np.sin(2.*x[iright])
		Up[iright, srho] = rhoR
		Up[ileft, srho] = rhoL
		# Momentum
		Up[iright, srhou] = rhoR*vR
		Up[ileft, srhou] = rhoL*vL
		# Energy
		Up[iright, srhoE] = pR/(gam-1.) + 0.5*rhoR*vR*vR + qo*rhoR*yR
		Up[ileft, srhoE] = pL/(gam-1.) + 0.5*rhoL*vL*vL + qo*rhoL*yL
		# MixtureFraction
		Up[iright, srhoz] = rhoR*yR
		Up[ileft, srhoz] = rhoL*yL

		return Up

'''
Boundary conditions
'''


'''
Source term functions
'''
class Arrhenius(SourceBase):
	def __init__(self, A=16418., b=0.1, Tign=15.):
		self.A = A
		self.b = b
		self.Tign = Tign

	def get_source(self, physics, FcnData, x, t):
		
		# Unpack source term constants
		A = self.A
		b = self.b
		Tign = self.Tign


		irho, irhou, irhoE, irhoz = physics.GetStateIndices()

		U = FcnData.U
		T = physics.ComputeScalars("Temperature", U)
		# K = np.zeros_like(T)
		# for i in range(len(T)):
		# 	if T[i]<0.:
		# 		K[i] = 0.
		# 	else:
		# 		K[i] = A * T[i]**b * np.exp(-Tign / T[i])
		K = A * T**b * np.exp(-Tign / T)

		S = np.zeros_like(U)

		S[:,irhoz] = -K[:].reshape(-1) * U[:,irhoz]
		
		return S

	def get_jacobian(self, physics, FcnData, x, t):
		
		# Note: This assumes b = 0 for now.

		# Unpack source term constants
		A = self.A
		Tign = self.Tign
		
		irho, irhou, irhoE, irhoY = physics.GetStateIndices()

		U = FcnData.U
		jac = np.zeros([U.shape[0], U.shape[-1], U.shape[-1]])

		T = physics.ComputeScalars("Temperature", U)
		K = A * np.exp(-Tign / T)

		dTdU = get_temperature_jacobian(physics, U)

		dKdrho =  (A * Tign * np.exp(-Tign / T) * dTdU[:, irhoY, irho].reshape([U.shape[0],1]))  / T**2 
		dKdrhou = (A * Tign * np.exp(-Tign / T) * dTdU[:, irhoY, irhou].reshape([U.shape[0],1])) / T**2
		dKdrhoE = (A * Tign * np.exp(-Tign / T) * dTdU[:, irhoY, irhoE].reshape([U.shape[0],1])) / T**2
		dKdrhoY = (A * Tign * np.exp(-Tign / T) * dTdU[:, irhoY, irhoY].reshape([U.shape[0],1])) / T**2

		jac[:, irhoY, irho] =  (-1.*dKdrho * U[:, irhoY].reshape([U.shape[0],1])).reshape(-1)
		jac[:, irhoY, irhou] = (-1.*dKdrho * U[:, irhoY].reshape([U.shape[0],1])).reshape(-1)
		jac[:, irhoY, irhoE] = (-1.*dKdrho * U[:, irhoY].reshape([U.shape[0],1])).reshape(-1)
		jac[:, irhoY, irhoY] = (-1.*dKdrho * U[:, irhoY].reshape([U.shape[0],1]) - K ).reshape(-1)

		# return jac.transpose(0,2,1)
		return jac
		
def get_temperature_jacobian(physics, U):
		
		irho, irhou, irhoE, irhoY = physics.GetStateIndices()

		gam = physics.gamma
		qo = physics.qo
		R = physics.R
		
		dTdU = np.zeros([U.shape[0], U.shape[-1], U.shape[-1]])

		rho = U[:, irho]
		rhou = U[:, irhou]
		rhoE = U[:, irhoE]
		rhoY = U[:, irhoY]

		E = rhoE/rho
		Y = rhoY/rho
		u = rhou/rho

		gamR = (gam - 1.) / R
		dTdU[:, irhoY, irho] = (gamR / rho) * (-1.*E + u**2 - qo*Y)
		dTdU[:, irhoY, irhou] = (gamR / rho) * (-1.*u)
		dTdU[:, irhoY, irhoE] = gamR / rho
		dTdU[:, irhoY, irhoY] = -1.*qo * (gamR / rho)

		return dTdU # [nq, ns, ns]

class Heaviside(SourceBase):
	def __init__(self, Da=1000., Tign=15.):
		self.Da = Da
		self.Tign = Tign

	def get_source(self, physics, FcnData, x, t):
		
		# Unpack source term constants
		Da = self.Da
		Tign = self.Tign

		irho, irhou, irhoE, irhoz = physics.GetStateIndices()

		U = FcnData.U
		T = physics.ComputeScalars("Temperature", U)
		K = np.zeros([U.shape[0]])

		for i in range(len(T)):
			if T[i] >= Tign:
				K[i] = Da
			else:
				K[i] = 0.

		S = np.zeros_like(U)
		S[:,irhoz] = -K[:] * U[:,irhoz]
		
		return S

	def get_jacobian(self, physics, FcnData, x, t):

		# Unpack source term constants
		Da = self.Da
		Tign = self.Tign

		U = FcnData.U
		
		irho, irhou, irhoE, irhoY = physics.GetStateIndices()

		T = physics.ComputeScalars("Temperature", U)
		K = np.zeros([U.shape[0]])

		for i in range(len(T)):
			if T[i] >= Tign:
				K[i] = Da
			else:
				K[i] = 0.

		jac = np.zeros([U.shape[0], U.shape[-1], U.shape[-1]])
		jac[:, irhoY, irhoY] = -K

		return jac

'''
Numerical flux functions
'''