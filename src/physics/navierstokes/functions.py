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
#       File : src/physics/navierstokes/functions.py
#
#       Contains definitions of Functions, boundary conditions, and source
#       terms for the Euler equations.
#
# ------------------------------------------------------------------------ #
from enum import Enum, auto
import numpy as np
from scipy.optimize import fsolve, root

import errors
import general

from physics.base.data import (FcnBase, BCWeakRiemann, BCWeakPrescribed,
        SourceBase, ConvNumFluxBase, DiffNumFluxBase)

class FcnType(Enum):
	'''
	Enum class that stores the types of analytical functions for initial
	conditions, exact solutions, and/or boundary conditions. These
	functions are specific to the available Euler equation sets.
	'''
	TaylorGreenVortexNS = auto()
	ManufacturedSolution = auto()
	Bubble = auto()
	Bubble2 = auto()
	RayleighTaylor = auto()
	Channel = auto()
	Couette = auto()
	Rising_bubble = auto()
	Rider_Korthe = auto()

class BCType(Enum):
	'''
	Enum class that stores the types of boundary conditions. These
	boundary conditions are specific to the available Euler equation sets.
	'''
	NoSlipWall = auto()
	SlipWall = auto()
	Subsonic_Outlet = auto()
	Subsonic_Inlet = auto()
#	PressureOutletNS = auto()
	
	pass



class SourceType(Enum):
	'''
	Enum class that stores the types of source terms. These
	source terms are specific to the available Euler equation sets.
	'''
	ManufacturedSource = auto()
	BubbleSource = auto()
	BubbleSource2 = auto()

'''
---------------
State functions
---------------
These classes inherit from the FcnBase class. See FcnBase for detailed
comments of attributes and methods. Information specific to the
corresponding child classes can be found below. These classes should
correspond to the FcnType enum members above.
'''

class ManufacturedSolution(FcnBase):
	'''
	Manufactured solution to the Navier-Stokes equations used for 
	verifying the order of accuracy of a given scheme.

	Script to generate sources is located in examples/navierstokes/
	2D/manufacturedNS2D
	'''
	def __init__(self):
		pass
	def get_state(self, physics, x, t):
		# Unpack
		gamma = physics.gamma
		
		irho, irhou, irhov, irhoE = physics.get_state_indices()	
		
		x1 = x[:, :, 0]
		x2 = x[:, :, 1]

		''' Fill state '''
		Uq = np.zeros([x.shape[0], x.shape[1], physics.NUM_STATE_VARS])

		# Generated initial condition from sympy

		Uq[:, :, irho] = 0.1*np.sin(np.pi*x1) + 0.1* \
			np.cos(np.pi*x1)*np.cos(np.pi*x2) -  \
			0.2*np.cos(np.pi*x2) + 1.0

		Uq[:, :, irhou] = (0.1*np.sin(np.pi*x1) + 0.1* \
			np.cos(np.pi*x1)*np.cos(np.pi*x2) - 0.2* \
			np.cos(np.pi*x2) + 1.0)*(0.3*np.sin(3* \
			np.pi*x1) + 0.3*np.cos(np.pi*x1)* \
			np.cos(np.pi*x2) + 0.3*np.cos(np.pi*x2) + 2.0)

		Uq[:, :, irhov] = (0.1*np.sin(np.pi*x1) + 0.1* \
			np.cos(np.pi*x1)*np.cos(np.pi*x2) - 0.2* \
			np.cos(np.pi*x2) + 1.0)*(0.3*np.sin(np.pi*x2) \
			+ 0.3*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.3* \
			np.cos(np.pi*x1) + 2.0)

		Uq[:, :, irhoE] = (4.0*(0.15*np.sin(3*np.pi*x1) + 0.15* \
			np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.15* \
			np.cos(np.pi*x2) + 1)**2 + 4.0*(0.15* \
			np.sin(np.pi*x2) + 0.15*np.cos(np.pi*x1)* \
			np.cos(np.pi*x2) + 0.15*np.cos(np.pi*x1) + 1)**2) \
			*(0.05*np.sin(np.pi*x1) + 0.05*np.cos(np.pi*x1)* \
			np.cos(np.pi*x2) - 0.1*np.cos(np.pi*x2) + 0.5) + \
			(1.0*np.sin(np.pi*x2) + 0.5*np.cos(np.pi*x1)* \
			np.cos(np.pi*x2) + 1.0*np.cos(2*np.pi*x1) + \
			10.0)/(gamma - 1)
		# End generated code

		return Uq # [ne, nq, ns]

class TaylorGreenVortexNS(FcnBase):
	'''
	2D Taylor Green Vortex Case
	'''
	def __init__(self):
		pass
	def get_state(self, physics, x, t):
		# Unpack
		x1 = x[:, :, 0]
		x2 = x[:, :, 1]
		
		Uq = np.zeros([x.shape[0], x.shape[1], physics.NUM_STATE_VARS])

		gamma = physics.gamma
		
		irho, irhou, irhov, irhoE = physics.get_state_indices()	

		Ma = 0.01
		P0 = 1. / (gamma*Ma*Ma)
		mu, _ = physics.get_transport(physics, Uq)

		nu = mu/1. # Rho = 1

		F = np.exp(-8. * np.pi*np.pi * nu * t)
		P = P0 + 0.25 * (np.cos(4.*np.pi * x1) + \
			np.cos(4.*np.pi * x2)) * F * F

		''' Fill state '''
		Uq[:, :, irho] = 1.0
		Uq[:, :, irhou] = np.sin(2.*np.pi * x1) * \
			np.cos(2.*np.pi * x2) * F
		Uq[:, :, irhov] = -np.cos(2.*np.pi * x1) * \
			np.sin(2.*np.pi * x2) * F
		Uq[:, :, irhoE] = P / (gamma-1.) + 0.5 * \
			(Uq[:, :, irhou]*Uq[:, :, irhou] +
			Uq[:, :, irhov]*Uq[:, :, irhov]) / Uq[:, :, irho]

		return Uq

class Bubble(FcnBase):
	'''
	2D advection of air bubble
	'''
	def __init__(self, x0=0., radius=0., thick=0., u=0., v=0., pressure=1., \
			rho1_in=1., rho2_in=1.):
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
		self.u = u
		self.v = v
		self.pressure = pressure
		self.rho1_in = rho1_in
		self.rho2_in = rho2_in
		
	def get_state(self, physics, x, t):
	
		r = np.linalg.norm(x[:] - self.x0, axis=2,
				keepdims=True)
				
		r = np.sqrt((x[:,:,0])**2+(x[:,:,1])**2 + 1e-10)
		
		Uq = np.zeros([x.shape[0], x.shape[1], physics.NUM_STATE_VARS])

		irho1phi1, irho2phi2, irhou, irhov, irhoE, iPF, iLS = physics.get_state_indices()
		
		tol = 1e-6
		
		Hr  = 0.5*(1.0+np.tanh(self.thick*(r-self.radius)))
		
		# Phase-field and Level-set
		Uq[:,:,iPF] = tol + (1.0-2.0*tol)*(1.0-Hr)
#		Uq[:,:,iLS] = -(r-self.radius)
		
		Hr  = 0.5*(1.0+np.tanh(0.5*self.thick*(r-self.radius)))
		Uq[:,:,iLS] = tol + (1.0-2.0*tol)*(1.0-Hr) - 0.5
		
		Uq[:,:,irho1phi1] = self.rho1_in*Uq[:,:,iPF]
		Uq[:,:,irho2phi2] = self.rho2_in*(1.0-Uq[:,:,iPF])
		
		rho = Uq[:,:,irho1phi1] + Uq[:,:,irho2phi2]
		
		Uq[:,:,irhou] = self.u*rho
		Uq[:,:,irhov] = self.v*rho
		if physics.sigma == 0:
			p = self.pressure
		else:
			p = self.pressure*Uq[:,:,iPF] + (self.pressure + physics.sigma/self.radius)*(1.0-Uq[:,:,iPF])
		
		gamma1 = physics.gamma1
		gamma2 = physics.gamma2
		pinf1  = physics.pinf1
		pinf2  = physics.pinf2
		q1     = physics.q1
		q2     = physics.q2
		
		rhoe1 = (p + gamma1*pinf1)/(gamma1-1.0)
		rhoe2 = (p + gamma2*pinf2)/(gamma2-1.0)
		
		rhoq = Uq[:,:,irho1phi1]*q1 +Uq[:,:,irho2phi2]*q2
		
		rhoe = rhoe1*Uq[:,:,iPF] + rhoe2*(1.-Uq[:,:,iPF]) + rhoq
		
		Uq[:,:,irhoE] = rhoe + 0.5*(Uq[:,:,irhou]**2 + Uq[:,:,irhov]**2)/rho
		
		return Uq
		
class Bubble2(FcnBase):
	'''
	2D advection of air bubble
	'''
	def __init__(self, x0=0., radius=0., thick=0., u=0., v=0., pressure=1., \
			rho1=1., rho2=1.):
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
		self.u = u
		self.v = v
		self.pressure = pressure
		self.rho1 = rho1
		self.rho2 = rho2
		
	def get_state(self, physics, x, t):
	
		r = np.linalg.norm(x[:] - self.x0, axis=2,
				keepdims=True)
				
		Uq = np.zeros([x.shape[0], x.shape[1], physics.NUM_STATE_VARS])

		irhou, irhov, ip, iPF, iLS = physics.get_state_indices()
		
		tol = 1e-3
		
		Hr  = 0.5*(1.0+np.tanh(self.thick*(r[:,:,0]-self.radius)))
		
		# Phase-field and Level-set
		Uq[:,:,iPF] = tol + (1.0-2.0*tol)*(1.0-Hr)
		Uq[:,:,iLS] = -(r[:,:,0]-self.radius)
		
		rho1 = self.rho1
		rho2 = self.rho2
		rho = rho1*Uq[:,:,iPF] + rho2*(1.0-Uq[:,:,iPF])
		
		Uq[:,:,irhou] = self.u*rho
		Uq[:,:,irhov] = self.v*rho
		
		p = self.pressure
		
		Uq[:,:,ip] = p
		
		return Uq
		
class RayleighTaylor(FcnBase):
	'''
	2D advection of air bubble
	'''
	def __init__(self, d=0., thick=0., u=0., v=0., pressure=1., \
			rho1_in=1., rho2_in=1.):
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
		self.d = d
		self.thick = thick
		self.u = u
		self.v = v
		self.pressure = pressure
		self.rho1_in = rho1_in
		self.rho2_in = rho2_in
		
	def get_state(self, physics, x, t):
	
		Uq = np.zeros([x.shape[0], x.shape[1], physics.NUM_STATE_VARS])

		irho1phi1, irho2phi2, irhou, irhov, irhoE, iPF, iLS = physics.get_state_indices()
		
		tol = 1e-6
		
		d = self.d
		h = 2.*d + 0.1*d*np.cos(2.*np.pi*x[:,:,0]/d)
		Hd = 0.5*(1.0+np.tanh(self.thick*(h-x[:,:,1])))
		Uq[:,:,iPF] = tol + (1.0-2.0*tol)*(1.0-Hd)
		Hd = 0.5*(1.0+np.tanh(0.5*self.thick*(h-x[:,:,1])))
		Uq[:,:,iLS] = tol + (1.0-2.0*tol)*(1.0-Hd) - 0.5
		
		Uq[:,:,irho1phi1] = self.rho1_in*Uq[:,:,iPF]
		Uq[:,:,irho2phi2] = self.rho2_in*(1.0-Uq[:,:,iPF])
		
		rho = Uq[:,:,irho1phi1] + Uq[:,:,irho2phi2]
		
		Uq[:,:,irhou] = self.u*rho
		Uq[:,:,irhov] = self.v*rho
		
		p = self.pressure
		
		gamma1 = physics.gamma1
		gamma2 = physics.gamma2
		pinf1  = physics.pinf1
		pinf2  = physics.pinf2
		
		one_over_gamma = Uq[:,:,iPF]/(gamma1-1.0) + (1.0-Uq[:,:,iPF])/(gamma2-1.0)
		gamma = (one_over_gamma+1.0)/one_over_gamma
		pinf = (gamma-1.0)/gamma*(Uq[:,:,iPF]*gamma1*pinf1/(gamma1-1.0) + (1.0-Uq[:,:,iPF])*gamma2*pinf2/(gamma2-1.0))
		
		rhoe = (p + gamma*pinf)*one_over_gamma
		
		Uq[:,:,irhoE] = rhoe + 0.5*rho*(self.u**2+self.v**2)
		
		return Uq
		
class Channel(FcnBase):
	'''
	2D advection of air bubble
	'''
	def __init__(self, d=0., thick=0., uavg=0., pressure=1., \
			rho1_in=1., rho2_in=1., x0=0., y0=0., r0=0.):
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
		self.d = d
		self.thick = thick
		self.uavg = uavg
		self.pressure = pressure
		self.rho1_in = rho1_in
		self.rho2_in = rho2_in
		self.x0 = x0
		self.y0 = y0
		self.r0 = r0
		
	def get_state(self, physics, x, t):
	
		Uq = np.zeros([x.shape[0], x.shape[1], physics.NUM_STATE_VARS])

		irho1phi1, irho2phi2, irhou, irhov, irhoE, iPF, iLS = physics.get_state_indices()
		
		tol = 1e-6
		
		r = np.sqrt((x[:,:,0]-self.x0)**2+(x[:,:,1]-self.y0)**2)
		
		Hr = 0.5*(1.0+np.tanh(self.thick*(r-self.r0)))
		Uq[:,:,iPF] = tol + (1.0-2.0*tol)*Hr
		Hr = 0.5*(1.0+np.tanh(0.25*self.thick*(r-self.r0))) - 0.5
		Uq[:,:,iLS] = (tol + (1.0-2.0*tol)*Hr)
		
#		Uq[:,:,iLS] = 0.5/self.thick*np.log(Uq[:,:,iPF]/(1.0-Uq[:,:,iPF]))
		
#		Uq[:,:,iPF] = 1.
#		Uq[:,:,iLS] = 1.
		
		Uq[:,:,irho1phi1] = self.rho1_in*Uq[:,:,iPF]
		Uq[:,:,irho2phi2] = self.rho2_in*(1.0-Uq[:,:,iPF])
		
		rho = Uq[:,:,irho1phi1] + Uq[:,:,irho2phi2]
		
		A = self.uavg*6.0/self.d**2
		
		Uq[:,:,irhou] = -A*(x[:,:,1]**2-0.25*self.d**2)*rho
		Uq[:,:,irhov] = 0.
		
		p = self.pressure*Uq[:,:,iPF] + (self.pressure + physics.sigma/self.r0)*(1.0-Uq[:,:,iPF])
		
		gamma1 = physics.gamma1
		gamma2 = physics.gamma2
		pinf1  = physics.pinf1
		pinf2  = physics.pinf2
		
		one_over_gamma_m1 = Uq[:,:,iPF]/(gamma1-1.0) + (1.0-Uq[:,:,iPF])/(gamma2-1.0)
		gamma = (one_over_gamma_m1+1.0)/one_over_gamma_m1
		pinf = (gamma-1.0)/gamma*(Uq[:,:,iPF]*gamma1*pinf1/(gamma1-1.0) + (1.0-Uq[:,:,iPF])*gamma2*pinf2/(gamma2-1.0))
		
		rhoq = Uq[:,:,irho1phi1]*physics.q1 + Uq[:,:,irho2phi2]*physics.q2
		rhoe = (p + gamma*pinf)*one_over_gamma_m1 + rhoq
		
		Uq[:,:,irhoE] = rhoe + 0.5*(Uq[:,:,irhou]**2+Uq[:,:,irhov]**2)/rho
		
		return Uq

class Couette(FcnBase):
	'''
	2D advection of air bubble
	'''
	def __init__(self, d=0., thick=0., uavg=0., pressure=1., \
			rho1_in=1., rho2_in=1., x0=0., y0=0., r0=0., shear = 0.):
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
		self.d = d
		self.thick = thick
		self.shear = shear
		self.pressure = pressure
		self.rho1_in = rho1_in
		self.rho2_in = rho2_in
		self.x0 = x0
		self.y0 = y0
		self.r0 = r0
		
	def get_state(self, physics, x, t):
	
		Uq = np.zeros([x.shape[0], x.shape[1], physics.NUM_STATE_VARS])

		irho1phi1, irho2phi2, irhou, irhov, irhoE, iPF, iLS = physics.get_state_indices()
		
		tol = 1e-6
		
		r = np.sqrt((x[:,:,0]-self.x0)**2+(x[:,:,1]-self.y0)**2)
		
		Hr = 1.0 - 0.5*(1.0+np.tanh(self.thick*(r-self.r0)))
		Uq[:,:,iPF] = tol + (1.0-2.0*tol)*Hr
		Hr = 1.0 - (0.5*(1.0+np.tanh(0.75*self.thick*(r-self.r0)))) - 0.5
		Uq[:,:,iLS] = (tol + (1.0-2.0*tol)*Hr)
		
		Uq[:,:,irho1phi1] = self.rho1_in*Uq[:,:,iPF]
		Uq[:,:,irho2phi2] = self.rho2_in*(1.0-Uq[:,:,iPF])
		
		rho = Uq[:,:,irho1phi1] + Uq[:,:,irho2phi2]
		
		Uq[:,:,irhou] = self.shear*rho*x[:,:,1]
		Uq[:,:,irhov] = 0.
		
		p = self.pressure*Uq[:,:,iPF] + (self.pressure + physics.sigma/self.r0)*(1.0-Uq[:,:,iPF])
		
		gamma1 = physics.gamma1
		gamma2 = physics.gamma2
		pinf1  = physics.pinf1
		pinf2  = physics.pinf2
		
		one_over_gamma_m1 = Uq[:,:,iPF]/(gamma1-1.0) + (1.0-Uq[:,:,iPF])/(gamma2-1.0)
		gamma = (one_over_gamma_m1+1.0)/one_over_gamma_m1
		pinf = (gamma-1.0)/gamma*(Uq[:,:,iPF]*gamma1*pinf1/(gamma1-1.0) + (1.0-Uq[:,:,iPF])*gamma2*pinf2/(gamma2-1.0))
		
		rhoq = Uq[:,:,irho1phi1]*physics.q1 + Uq[:,:,irho2phi2]*physics.q2
		rhoe = (p + gamma*pinf)*one_over_gamma_m1 + rhoq
		
		Uq[:,:,irhoE] = rhoe + 0.5*(Uq[:,:,irhou]**2+Uq[:,:,irhov]**2)/rho
		
		return Uq
		
class Rising_bubble(FcnBase):
	'''
	2D advection of air bubble
	'''
	def __init__(self, d=0., thick=0., uavg=0., pressure=1., \
			rho1_in=1., rho2_in=1., x0=0., y0=0., r0=0.):
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
		self.d = d
		self.thick = thick
		self.uavg = uavg
		self.pressure = pressure
		self.rho1_in = rho1_in
		self.rho2_in = rho2_in
		self.x0 = x0
		self.y0 = y0
		self.r0 = r0
		
	def get_state(self, physics, x, t):
	
		Uq = np.zeros([x.shape[0], x.shape[1], physics.NUM_STATE_VARS])

		irho1phi1, irho2phi2, irhou, irhov, irhoE, iPF, iLS = physics.get_state_indices()
		
		tol = 1e-10
		
		r = np.sqrt((x[:,:,0]-self.x0)**2+(x[:,:,1]-self.y0)**2)
		
		Hr = 0.5*(1.0+np.tanh(self.thick*(r-self.r0)))
		Uq[:,:,iPF] = tol + (1.0-2.0*tol)*Hr
#		Uq[:,:,iPF] = 1. - Uq[:,:,iPF]
		Hr = 0.5*(1.0+np.tanh(0.75*self.thick*(r-self.r0))) - 0.5
		Uq[:,:,iLS] = (tol + (1.0-2.0*tol)*Hr)
#		Uq[:,:,iLS] = - Uq[:,:,iLS]
		
		Uq[:,:,irho1phi1] = self.rho1_in*Uq[:,:,iPF]
		Uq[:,:,irho2phi2] = self.rho2_in*(1.0-Uq[:,:,iPF])
		
		rho = Uq[:,:,irho1phi1] + Uq[:,:,irho2phi2]
		
		Uq[:,:,irhou] = 0.
		Uq[:,:,irhov] = 0.
		
		p = self.pressure*Uq[:,:,iPF] + (self.pressure + physics.sigma/self.r0)*(1.0-Uq[:,:,iPF])
		
		gamma1 = physics.gamma1
		gamma2 = physics.gamma2
		pinf1  = physics.pinf1
		pinf2  = physics.pinf2
		q1     = physics.q1
		q2     = physics.q2
		
		rhoe1 = (p + gamma1*pinf1)/(gamma1-1.0)
		rhoe2 = (p + gamma2*pinf2)/(gamma2-1.0)
		
		rhoq = Uq[:,:,irho1phi1]*q1 +Uq[:,:,irho2phi2]*q2
		
		rhoe = rhoe1*Uq[:,:,iPF] + rhoe2*(1.-Uq[:,:,iPF]) + rhoq
		
		Uq[:,:,irhoE] = rhoe
		
		return Uq

class Rider_Korthe(FcnBase):
	'''
	2D advection of air bubble
	'''
	def __init__(self, d=0., thick=0., uavg=0., pressure=1., \
			rho1_in=1., rho2_in=1., x0=0., y0=0., r0=0.):
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
		self.d = d
		self.thick = thick
		self.uavg = uavg
		self.pressure = pressure
		self.rho1_in = rho1_in
		self.rho2_in = rho2_in
		self.x0 = x0
		self.y0 = y0
		self.r0 = r0
		
	def get_state(self, physics, x, t):
	
		Uq = np.zeros([x.shape[0], x.shape[1], physics.NUM_STATE_VARS])

		irho1phi1, irho2phi2, irhou, irhov, irhoE, iPF, iLS = physics.get_state_indices()
		
		tol = 1e-6

		r = np.sqrt((x[:,:,0]-self.x0)**2+(x[:,:,1]-self.y0)**2)

		Hr = 1.0 - 0.5*(1.0+np.tanh(self.thick*(r-self.r0)))
		Uq[:,:,iPF] = tol + (1.0-2.0*tol)*Hr
		Hr = 1.0 - 0.5*(1.0+np.tanh(0.25*self.thick*(r-self.r0))) - 0.5
		Uq[:,:,iLS] = (tol + (1.0-2.0*tol)*Hr)
		
		Uq[:,:,irho1phi1] = self.rho1_in*Uq[:,:,iPF]
		Uq[:,:,irho2phi2] = self.rho2_in*(1.0-Uq[:,:,iPF])
		
		rho = Uq[:,:,irho1phi1] + Uq[:,:,irho2phi2]
		
		Uq[:,:,irhou] = 0.
		Uq[:,:,irhov] = 0.
		
		p = self.pressure*Uq[:,:,iPF] + (self.pressure + physics.sigma/self.r0)*(1.0-Uq[:,:,iPF])
		
		gamma1 = physics.gamma1
		gamma2 = physics.gamma2
		pinf1  = physics.pinf1
		pinf2  = physics.pinf2
		
		one_over_gamma = Uq[:,:,iPF]/(gamma1-1.0) + (1.0-Uq[:,:,iPF])/(gamma2-1.0)
		gamma = (one_over_gamma+1.0)/one_over_gamma
		pinf = (gamma-1.0)/gamma*(Uq[:,:,iPF]*gamma1*pinf1/(gamma1-1.0) + (1.0-Uq[:,:,iPF])*gamma2*pinf2/(gamma2-1.0))
		
		rhoe = (p + gamma*pinf)*one_over_gamma
		
		Uq[:,:,irhoE] = rhoe
		
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



'''
---------------------
Source term functions
---------------------
These classes inherit from the SourceBase class. See SourceBase for detailed
comments of attributes and methods. Information specific to the
corresponding child classes can be found below. These classes should
correspond to the SourceType enum members above.
'''
class ManufacturedSource(SourceBase):
	'''
	Generated source term for the manufactured solution of the 
	Navier-Stokes equations. Generated using script in 
	examples/navierstokes/2D/manufactured_solution. Exact solution can 
	be found in the following paper.
		[1] Dumbser, M. (2010)
	'''
	def get_source(self, physics, Uq, x, t):
		# Unpack
		gamma = physics.gamma
		R = physics.R

		irho, irhou, irhov, irhoE = physics.get_state_indices()
		x1 = x[:, :, 0]
		x2 = x[:, :, 1]

		mu, kappa = physics.get_transport(physics, Uq, 
			flag_non_physical=False)
		# import code; code.interact(local=locals())
		Sq = np.zeros_like(Uq)
		Sq[:, :, irho], Sq[:, :, irhou], Sq[:, :, irhov], \
			Sq[:, :, irhoE] = self.manufactured_source(x1, x2, 
			t, gamma, kappa, 
			mu, R)

		
		return Sq # [ne, nq, ns]

	def manufactured_source(self, x1, x2, t, gamma, kappa, mu, R):
	
		
		# The following lines of code are generated using sympy
		S_rho = (-0.3*np.pi*np.sin(np.pi*x1)*np.cos(np.pi*x2) + \
			0.9*np.pi*np.cos(3*np.pi*x1))*(0.1*np.sin(np.pi*x1) \
			+ 0.1*np.cos(np.pi*x1)*np.cos(np.pi*x2) - 0.2* \
			np.cos(np.pi*x2) + 1.0) + (-0.1*np.pi* \
			np.sin(np.pi*x1)*np.cos(np.pi*x2) + 0.1*np.pi* \
			np.cos(np.pi*x1))*(0.3*np.sin(3*np.pi*x1) + \
			0.3*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.3* \
			np.cos(np.pi*x2) + 2.0) + (-0.3*np.pi* \
			np.sin(np.pi*x2)*np.cos(np.pi*x1) + 0.3*np.pi* \
			np.cos(np.pi*x2))*(0.1*np.sin(np.pi*x1) + 0.1* \
			np.cos(np.pi*x1)*np.cos(np.pi*x2) - 0.2* \
			np.cos(np.pi*x2) + 1.0) + (-0.1*np.pi* \
			np.sin(np.pi*x2)*np.cos(np.pi*x1) + 0.2*np.pi* \
			np.sin(np.pi*x2))*(0.3*np.sin(np.pi*x2) + \
			0.3*np.cos(np.pi*x1)*np.cos(np.pi*x2) + \
			0.3*np.cos(np.pi*x1) + 2.0)

		S_rhou = -mu*(-0.2*np.pi**2*np.sin(np.pi*x1)* \
			np.sin(np.pi*x2) - 3.6*np.pi**2*np.sin(3*np.pi*x1) \
			 - 0.4*np.pi**2*np.cos(np.pi*x1)*np.cos(np.pi*x2)) \
			- mu*(0.3*np.pi**2*np.sin(np.pi*x1)*np.sin(np.pi*x2) \
			- 0.3*np.pi**2*np.cos(np.pi*x1)*np.cos(np.pi*x2) - \
			0.3*np.pi**2*np.cos(np.pi*x2)) + 4.0*(-0.3*np.pi* \
			np.sin(np.pi*x1)*np.cos(np.pi*x2) + 0.9*np.pi* \
			np.cos(3*np.pi*x1))*(0.1*np.sin(np.pi*x1) + \
			0.1*np.cos(np.pi*x1)*np.cos(np.pi*x2) - 0.2* \
			np.cos(np.pi*x2) + 1.0)*(0.15*np.sin(3*np.pi*x1) \
			+ 0.15*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.15* \
			np.cos(np.pi*x2) + 1) + 4.0*(-0.1*np.pi* \
			np.sin(np.pi*x1)*np.cos(np.pi*x2) + 0.1*np.pi* \
			np.cos(np.pi*x1))*(0.15*np.sin(3*np.pi*x1) + 0.15* \
			np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.15* \
			np.cos(np.pi*x2) + 1)**2 + (-0.3*np.pi* \
			np.sin(np.pi*x2)*np.cos(np.pi*x1) - 0.3* \
			np.pi*np.sin(np.pi*x2))*(0.1*np.sin(np.pi*x1) + \
			0.1*np.cos(np.pi*x1)*np.cos(np.pi*x2) - 0.2* \
			np.cos(np.pi*x2) + 1.0)*(0.3*np.sin(np.pi*x2) + \
			0.3*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.3* \
			np.cos(np.pi*x1) + 2.0) + (-0.3*np.pi* \
			np.sin(np.pi*x2)*np.cos(np.pi*x1) + 0.3*np.pi* \
			np.cos(np.pi*x2))*(0.1*np.sin(np.pi*x1) + 0.1* \
			np.cos(np.pi*x1)*np.cos(np.pi*x2) - 0.2* \
			np.cos(np.pi*x2) + 1.0)*(0.3*np.sin(3*np.pi*x1) \
			+ 0.3*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.3* \
			np.cos(np.pi*x2) + 2.0) + (-0.1*np.pi* \
			np.sin(np.pi*x2)*np.cos(np.pi*x1) + 0.2*np.pi* \
			np.sin(np.pi*x2))*(0.3*np.sin(3*np.pi*x1) + 0.3* \
			np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.3* \
			np.cos(np.pi*x2) + 2.0)*(0.3*np.sin(np.pi*x2) + \
			0.3*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.3* \
			np.cos(np.pi*x1) + 2.0) - 0.5*np.pi* \
			np.sin(np.pi*x1)*np.cos(np.pi*x2) - \
			2.0*np.pi*np.sin(2*np.pi*x1)

		S_rhov = -mu*(-0.2*np.pi**2*np.sin(np.pi*x1)* \
			np.sin(np.pi*x2) - 0.4*np.pi**2*np.sin(np.pi*x2) \
			- 0.4*np.pi**2*np.cos(np.pi*x1)*np.cos(np.pi*x2)) \
			 - mu*(0.3*np.pi**2*np.sin(np.pi*x1)* \
		 	np.sin(np.pi*x2) - 0.3*np.pi**2* \
		 	np.cos(np.pi*x1)*np.cos(np.pi*x2) - 0.3*np.pi**2* \
		 	np.cos(np.pi*x1)) + (-0.3*np.pi*np.sin(np.pi*x1)* \
		 	np.cos(np.pi*x2) - 0.3*np.pi*np.sin(np.pi*x1))* \
		 	(0.1*np.sin(np.pi*x1) + 0.1*np.cos(np.pi*x1)* \
	 		np.cos(np.pi*x2) - 0.2*np.cos(np.pi*x2) + 1.0)* \
	 		(0.3*np.sin(3*np.pi*x1) + 0.3*np.cos(np.pi*x1)* \
 			np.cos(np.pi*x2) + 0.3*np.cos(np.pi*x2) + 2.0) + \
 			(-0.3*np.pi*np.sin(np.pi*x1)*np.cos(np.pi*x2) + \
			0.9*np.pi*np.cos(3*np.pi*x1))*(0.1*np.sin(np.pi*x1) \
			+ 0.1*np.cos(np.pi*x1)*np.cos(np.pi*x2) - \
			0.2*np.cos(np.pi*x2) + 1.0)*(0.3*np.sin(np.pi*x2) + \
			0.3*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.3* \
			np.cos(np.pi*x1) + 2.0) + (-0.1*np.pi* \
			np.sin(np.pi*x1)*np.cos(np.pi*x2) + 0.1*np.pi* \
			np.cos(np.pi*x1))*(0.3*np.sin(3*np.pi*x1) + 0.3* \
			np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.3* \
			np.cos(np.pi*x2) + 2.0)*(0.3*np.sin(np.pi*x2) + 0.3 \
			*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.3* \
			np.cos(np.pi*x1) + 2.0) + 4.0*(-0.3*np.pi* \
			np.sin(np.pi*x2)*np.cos(np.pi*x1) + 0.3*np.pi* \
			np.cos(np.pi*x2))*(0.1*np.sin(np.pi*x1) + 0.1* \
			np.cos(np.pi*x1)*np.cos(np.pi*x2) - 0.2* \
			np.cos(np.pi*x2) + 1.0)*(0.15*np.sin(np.pi*x2) + \
			0.15*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.15* \
			np.cos(np.pi*x1) + 1) + 4.0*(-0.1*np.pi* \
			np.sin(np.pi*x2)*np.cos(np.pi*x1) + 0.2*np.pi* \
			np.sin(np.pi*x2))*(0.15*np.sin(np.pi*x2) + 0.15* \
			np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.15* \
			np.cos(np.pi*x1) + 1)**2 - 0.5*np.pi* \
			np.sin(np.pi*x2)*np.cos(np.pi*x1) + \
			1.0*np.pi*np.cos(np.pi*x2)

		S_rhoE = -mu*(-0.3*np.pi*np.sin(np.pi*x1)* \
			np.cos(np.pi*x2) - 0.3*np.pi*np.sin(np.pi*x1)) \
			*(-0.3*np.pi*np.sin(np.pi*x1)*np.cos(np.pi*x2) \
			- 0.3*np.pi*np.sin(np.pi*x1) - 0.3*np.pi* \
			np.sin(np.pi*x2)*np.cos(np.pi*x1) - 0.3*np.pi* \
			np.sin(np.pi*x2)) - mu*(-0.3*np.pi* \
			np.sin(np.pi*x1)*np.cos(np.pi*x2) + 0.9*np.pi* \
			np.cos(3*np.pi*x1))*(-0.4*np.pi*np.sin(np.pi*x1)* \
			np.cos(np.pi*x2) + 0.2*np.pi*np.sin(np.pi*x2)* \
			np.cos(np.pi*x1) + 1.2*np.pi*np.cos(3*np.pi*x1) - \
			0.2*np.pi*np.cos(np.pi*x2)) - mu*(-0.3*np.pi* \
			np.sin(np.pi*x2)*np.cos(np.pi*x1) - 0.3*np.pi* \
			np.sin(np.pi*x2))*(-0.3*np.pi*np.sin(np.pi*x1)* \
			np.cos(np.pi*x2) - 0.3*np.pi*np.sin(np.pi*x1) - \
			0.3*np.pi*np.sin(np.pi*x2)*np.cos(np.pi*x1) - 0.3* \
			np.pi*np.sin(np.pi*x2)) - mu*(-0.3*np.pi* \
			np.sin(np.pi*x2)*np.cos(np.pi*x1) + 0.3*np.pi* \
			np.cos(np.pi*x2))*(0.2*np.pi*np.sin(np.pi*x1)* \
			np.cos(np.pi*x2) - 0.4*np.pi*np.sin(np.pi*x2)* \
			np.cos(np.pi*x1) - 0.6*np.pi*np.cos(3*np.pi*x1) \
			+ 0.4*np.pi*np.cos(np.pi*x2)) - mu*(-0.2*np.pi**2* \
			np.sin(np.pi*x1)*np.sin(np.pi*x2) - 3.6*np.pi**2* \
			np.sin(3*np.pi*x1) - 0.4*np.pi**2*np.cos(np.pi*x1)* \
			np.cos(np.pi*x2))*(0.3*np.sin(3*np.pi*x1) + 0.3* \
			np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.3* \
			np.cos(np.pi*x2) + 2.0) - mu*(-0.2*np.pi**2* \
			np.sin(np.pi*x1)*np.sin(np.pi*x2) - 0.4*np.pi**2* \
			np.sin(np.pi*x2) - 0.4*np.pi**2*np.cos(np.pi*x1)* \
			np.cos(np.pi*x2))*(0.3*np.sin(np.pi*x2) + 0.3* \
			np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.3* \
			np.cos(np.pi*x1) + 2.0) - mu*(0.3*np.pi**2* \
			np.sin(np.pi*x1)*np.sin(np.pi*x2) - 0.3*np.pi**2* \
			np.cos(np.pi*x1)*np.cos(np.pi*x2) - 0.3*np.pi**2* \
			np.cos(np.pi*x1))*(0.3*np.sin(np.pi*x2) + 0.3* \
			np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.3* \
			np.cos(np.pi*x1) + 2.0) - mu*(0.3*np.pi**2* \
			np.sin(np.pi*x1)*np.sin(np.pi*x2) - 0.3*np.pi**2* \
			np.cos(np.pi*x1)*np.cos(np.pi*x2) - 0.3*np.pi**2* \
			np.cos(np.pi*x2))*(0.3*np.sin(3*np.pi*x1) + 0.3* \
			np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.3* \
			np.cos(np.pi*x2) + 2.0) + (-0.3*np.pi* \
			np.sin(np.pi*x1)*np.cos(np.pi*x2) + 0.9*np.pi* \
			np.cos(3*np.pi*x1))*((4.0*(0.15* \
			np.sin(3*np.pi*x1) + 0.15*np.cos(np.pi*x1)* \
			np.cos(np.pi*x2) + 0.15*np.cos(np.pi*x2) + 1)**2 \
			+ 4.0*(0.15*np.sin(np.pi*x2) + 0.15* \
			np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.15* \
			np.cos(np.pi*x1) + 1)**2)*(0.05*np.sin(np.pi*x1) \
			+ 0.05*np.cos(np.pi*x1)*np.cos(np.pi*x2) - 0.1* \
			np.cos(np.pi*x2) + 0.5) + 1.0*np.sin(np.pi*x2) + \
			0.5*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 1.0* \
			np.cos(2*np.pi*x1) + 10.0 + (1.0*np.sin(np.pi*x2) \
			+ 0.5*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 1.0* \
			np.cos(2*np.pi*x1) + 10.0)/(gamma - 1)) + (-0.3* \
			np.pi*np.sin(np.pi*x2)*np.cos(np.pi*x1) + 0.3*np.pi* \
			np.cos(np.pi*x2))*((4.0*(0.15*np.sin(3*np.pi*x1) + \
			0.15*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.15* \
			np.cos(np.pi*x2) + 1)**2 + 4.0*(0.15* \
			np.sin(np.pi*x2) + 0.15*np.cos(np.pi*x1)* \
			np.cos(np.pi*x2) + 0.15*np.cos(np.pi*x1) + 1)**2) \
			*(0.05*np.sin(np.pi*x1) + 0.05*np.cos(np.pi*x1)* \
			np.cos(np.pi*x2) - 0.1*np.cos(np.pi*x2) + 0.5) + 1.0 \
			*np.sin(np.pi*x2) + 0.5*np.cos(np.pi*x1)* \
			np.cos(np.pi*x2) + 1.0*np.cos(2*np.pi*x1) + 10.0 \
			+ (1.0*np.sin(np.pi*x2) + 0.5*np.cos(np.pi*x1)* \
			np.cos(np.pi*x2) + 1.0*np.cos(2*np.pi*x1) + 10.0) \
			/(gamma - 1)) + (0.3*np.sin(3*np.pi*x1) + 0.3* \
			np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.3* \
			np.cos(np.pi*x2) + 2.0)*((4.0*(-0.3*np.pi* \
			np.sin(np.pi*x1)*np.cos(np.pi*x2) - 0.3*np.pi* \
			np.sin(np.pi*x1))*(0.15*np.sin(np.pi*x2) + 0.15* \
			np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.15* \
			np.cos(np.pi*x1) + 1) + 4.0*(-0.3*np.pi* \
			np.sin(np.pi*x1)*np.cos(np.pi*x2) + 0.9* \
			np.pi*np.cos(3*np.pi*x1))*(0.15*np.sin(3*np.pi*x1) \
			+ 0.15*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.15* \
			np.cos(np.pi*x2) + 1))*(0.05*np.sin(np.pi*x1) + \
			0.05*np.cos(np.pi*x1)*np.cos(np.pi*x2) - 0.1* \
			np.cos(np.pi*x2) + 0.5) + (-0.05*np.pi* \
			np.sin(np.pi*x1)*np.cos(np.pi*x2) + 0.05*np.pi* \
			np.cos(np.pi*x1))*(4.0*(0.15*np.sin(3*np.pi*x1) + \
			0.15*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.15* \
			np.cos(np.pi*x2) + 1)**2 + 4.0*(0.15* \
			np.sin(np.pi*x2) + 0.15*np.cos(np.pi*x1)* \
			np.cos(np.pi*x2) + 0.15*np.cos(np.pi*x1) + 1)**2) \
			- 0.5*np.pi*np.sin(np.pi*x1)*np.cos(np.pi*x2) - \
			2.0*np.pi*np.sin(2*np.pi*x1) + (-0.5*np.pi* \
			np.sin(np.pi*x1)*np.cos(np.pi*x2) - 2.0*np.pi* \
			np.sin(2*np.pi*x1))/(gamma - 1)) + (0.3* \
			np.sin(np.pi*x2) + 0.3*np.cos(np.pi*x1)* \
			np.cos(np.pi*x2) + 0.3*np.cos(np.pi*x1) + \
			2.0)*((4.0*(-0.3*np.pi*np.sin(np.pi*x2)* \
			np.cos(np.pi*x1) - 0.3*np.pi*np.sin(np.pi*x2))* \
			(0.15*np.sin(3*np.pi*x1) + 0.15*np.cos(np.pi*x1)* \
			np.cos(np.pi*x2) + 0.15*np.cos(np.pi*x2) + 1) + \
			4.0*(-0.3*np.pi*np.sin(np.pi*x2)*np.cos(np.pi*x1) \
			+ 0.3*np.pi*np.cos(np.pi*x2))*(0.15*np.sin(np.pi*x2) \
			+ 0.15*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.15* \
			np.cos(np.pi*x1) + 1))*(0.05*np.sin(np.pi*x1) + \
			0.05*np.cos(np.pi*x1)*np.cos(np.pi*x2) - 0.1* \
			np.cos(np.pi*x2) + 0.5) + (-0.05*np.pi* \
			np.sin(np.pi*x2)*np.cos(np.pi*x1) + 0.1*np.pi* \
			np.sin(np.pi*x2))*(4.0*(0.15*np.sin(3*np.pi*x1) + \
			0.15*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.15* \
			np.cos(np.pi*x2) + 1)**2 + 4.0*(0.15* \
			np.sin(np.pi*x2) + 0.15*np.cos(np.pi*x1)* \
			np.cos(np.pi*x2) + 0.15*np.cos(np.pi*x1) + 1)**2) \
			- 0.5*np.pi*np.sin(np.pi*x2)*np.cos(np.pi*x1) + \
			1.0*np.pi*np.cos(np.pi*x2) + (-0.5*np.pi* \
			np.sin(np.pi*x2)*np.cos(np.pi*x1) + 1.0*np.pi* \
			np.cos(np.pi*x2))/(gamma - 1)) - np.pi**2*kappa* \
			(-0.2*(0.5*np.sin(np.pi*x1)*np.cos(np.pi*x2) + 2.0* \
			np.sin(2*np.pi*x1))*(np.sin(np.pi*x1)* \
			np.cos(np.pi*x2) - np.cos(np.pi*x1))/(0.1* \
			np.sin(np.pi*x1) + 0.1*np.cos(np.pi*x1)* \
			np.cos(np.pi*x2) - 0.2*np.cos(np.pi*x2) + 1.0) \
			+ (0.02*(np.sin(np.pi*x1)*np.cos(np.pi*x2) - \
			np.cos(np.pi*x1))**2/(0.1*np.sin(np.pi*x1) + 0.1 \
			*np.cos(np.pi*x1)*np.cos(np.pi*x2) - 0.2* \
			np.cos(np.pi*x2) + 1.0) + 0.1*np.sin(np.pi*x1) + \
			0.1*np.cos(np.pi*x1)*np.cos(np.pi*x2))*(1.0* \
			np.sin(np.pi*x2) + 0.5*np.cos(np.pi*x1)* \
			np.cos(np.pi*x2) + 1.0*np.cos(2*np.pi*x1) + 10.0) \
			/(0.1*np.sin(np.pi*x1) + 0.1*np.cos(np.pi*x1)* \
			np.cos(np.pi*x2) - 0.2*np.cos(np.pi*x2) + 1.0) - \
			0.5*np.cos(np.pi*x1)*np.cos(np.pi*x2) - 4.0* \
			np.cos(2*np.pi*x1))/(R*(0.1*np.sin(np.pi*x1) + \
			0.1*np.cos(np.pi*x1)*np.cos(np.pi*x2) - 0.2* \
			np.cos(np.pi*x2) + 1.0)) - np.pi**2*kappa*(-2* \
			(0.5*np.sin(np.pi*x2)*np.cos(np.pi*x1) - 1.0* \
			np.cos(np.pi*x2))*(0.1*np.cos(np.pi*x1) - 0.2)* \
			np.sin(np.pi*x2)/(0.1*np.sin(np.pi*x1) + 0.1* \
			np.cos(np.pi*x1)*np.cos(np.pi*x2) - 0.2* \
			np.cos(np.pi*x2) + 1.0) + ((0.2*np.cos(np.pi*x1) \
			- 0.4)*np.sin(np.pi*x2)**2/(0.1*np.sin(np.pi*x1) \
			+ 0.1*np.cos(np.pi*x1)*np.cos(np.pi*x2) - 0.2* \
			np.cos(np.pi*x2) + 1.0) + np.cos(np.pi*x2))*(0.1* \
			np.cos(np.pi*x1) - 0.2)*(1.0*np.sin(np.pi*x2) + 0.5 \
			*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 1.0* \
			np.cos(2*np.pi*x1) + 10.0)/(0.1*np.sin(np.pi*x1) \
			+ 0.1*np.cos(np.pi*x1)*np.cos(np.pi*x2) - 0.2* \
			np.cos(np.pi*x2) + 1.0) - 1.0*np.sin(np.pi*x2) - \
			0.5*np.cos(np.pi*x1)*np.cos(np.pi*x2))/(R*(0.1* \
			np.sin(np.pi*x1) + 0.1*np.cos(np.pi*x1)* \
			np.cos(np.pi*x2) - 0.2*np.cos(np.pi*x2) + 1.0))
		# End of generated code

		
		return S_rho, S_rhou, S_rhov, S_rhoE

class BubbleSource(SourceBase):
	'''
	Source for bubble case
	'''
	
	def get_source(self, physics, Uq, gUq, x, t, kk):
		
		irho1phi1, irho2phi2, irhou, irhov, irhoE, iPF, iLS = physics.get_state_indices()

		rho1phi1  = Uq[:, :, irho1phi1] # [n, nq]
		rho2phi2  = Uq[:, :, irho2phi2] # [n, nq]
		rhou      = Uq[:, :, irhou]     # [n, nq]
		rhov      = Uq[:, :, irhov]     # [n, nq]
		rhoE      = Uq[:, :, irhoE]     # [n, nq]
		phi1      = Uq[:, :, iPF]       # [n, nq]
		LS        = Uq[:, :, iLS]       # [n, nq]

		Sq = np.zeros(Uq.shape)
		
		# Separate x and y gradients
		gUx = gUq[:, :, :, 0] # [ne, nq, ns]
		gUy = gUq[:, :, :, 1] # [ne, nq, ns]

		switch = physics.switch

		if physics.kinetics == 1:
			Sq = np.zeros(Uq.shape)
			T = 4.0
			Uq[:,:,irhou] = -np.sin(np.pi*(x[:,:,0]+0.5))**2.0* \
					np.sin(2.0*np.pi*(x[:,:,1]+0.5))*np.cos(np.pi*t/T)
			Uq[:,:,irhov] = np.sin(2.0*np.pi*(x[:,:,0]+0.5))* \
					np.sin(np.pi*(x[:,:,1]+0.5))**2.0*np.cos(np.pi*t/T)
			
			u = Uq[:,:,irhou]
			v = Uq[:,:,irhov]
			
			Sq[:,:,iPF] = (-u*gUx[:,:,iPF] - v*gUy[:,:,iPF])*switch
			Sq[:,:,iLS] = (-u*gUx[:,:,iLS] - v*gUy[:,:,iLS])*switch
		else:
			# Get velocity in each dimension
			rho   = rho1phi1 + rho2phi2
			u = rhou / rho
			v = rhov / rho
			
			sigma = physics.sigma
			gLS = gUq[:,:,iLS,:]
			n = np.zeros(gLS.shape)
			mag = np.sqrt(gLS[:,:,0]**2+gLS[:,:,1]**2)
			n[:,:,0] = gLS[:,:,0]/(mag+1e-32)
			n[:,:,1] = gLS[:,:,1]/(mag+1e-32)
			magPF = np.sqrt(gUq[:,:,iPF,0]**2+gUq[:,:,iPF,1]**2)
			magLS = np.sqrt(gUq[:,:,iLS,0]**2+gUq[:,:,iLS,1]**2)
			
			Sq[:,:,irhou] =                        + sigma*kk*n[:,:,0]*magPF*switch
			Sq[:,:,irhov] = -rho*physics.g*switch  + sigma*kk*n[:,:,1]*magPF*switch
			
			Sq[:,:,irhoE] = Sq[:,:,irhou]*u + Sq[:,:,irhov]*v

			# phase change + transport
			gamma1 = physics.gamma1
			gamma2 = physics.gamma2
			pinf1  = physics.pinf1
			pinf2  = physics.pinf2
			q1     = physics.q1
			q2     = physics.q2
			cv1    = physics.cv1
			cv2    = physics.cv2
			rho01  = physics.rho01
			rho02  = physics.rho02
			# Stiffened gas EOS
			rhoe = (rhoE - 0.5 * rho * (u**2 + v**2)) # [n, nq]
			one_over_gamma_m1 = phi1/(gamma1-1.) + (1.-phi1)/(gamma2-1.)
			rhoq = rho1phi1*physics.q1 + rho2phi2*physics.q2
			p = (rhoe - rhoq + (-phi1*gamma1*pinf1/(gamma1-1.)-(1.-phi1)*gamma2*pinf2/(gamma2-1.)))/one_over_gamma_m1
			h1     = (p + pinf1)*gamma1/(gamma1-1.)/rho01 + q1
			h2     = (p + pinf2)*gamma2/(gamma2-1.)/rho02 + q2

			mdot = physics.mdot
#			Tsat = 374.0
#			ri = 0.01
#			T1 = (p + pinf1)/(rho01*cv1*(gamma1-1.))
#			mdot = ri*rho1phi1*(Tsat-T1)/Tsat
			rhoI = (physics.rho01*physics.rho02)/rho

			# Transport equations + phase change
			Sq[:,:,iPF] = (-u*gUx[:,:,iPF] - v*gUy[:,:,iPF] + mdot*magPF/rhoI)*switch
			Sq[:,:,iLS] = (-u*gUx[:,:,iLS] - v*gUy[:,:,iLS])*switch
			
			Sq[:,:,irho1phi1] = mdot*magPF*switch
			Sq[:,:,irho2phi2] = -mdot*magPF*switch

			Sq[:,:,irhoE] = (h1-h2)*magPF*mdot*switch

		if switch == 0:
			eps = physics.scl_eps*physics.eps
			psi0 = Uq[:,:,iPF]-0.5
			Uqx = gUq[:,:,iPF,0]
			Uqy = gUq[:,:,iPF,1]
			mag3 = np.sqrt(Uqx**2 + Uqy**2 + 1e-16)
			sgn = np.tanh(0.5*(psi0)/eps/mag3)
			mag = np.sqrt(gUq[:, :, iLS, 0]**2 + gUq[:, :, iLS, 1]**2)
		
			rho   = rho1phi1 + rho2phi2
			u = rhou / rho
			v = rhov / rho
			magu = np.sqrt(u**2+v**2)
			Sq[:,:,iLS] = (1.0-mag)*sgn#*magu/(magu + np.max(magu))
		
		return Sq
	

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

class NoSlipWall(BCWeakRiemann):
	'''
	This class corresponds to a slip wall. See documentation for more
	details.
	'''
	def __init__(self, Twall):
		'''
		This method initializes the attributes.

		Inputs:
		-------
			p: pressure

		Outputs:
		--------
		    self: attributes initialized
		'''
		self.Twall = Twall
		
	def get_boundary_state(self, physics, UqI, normals, x, t):
		#UqB = self.function.get_state(physics, x, t)
		UqB = UqI.copy()
		
		irho1phi1, irho2phi2, irhou, irhov, irhoE, iPF, iLS = physics.get_state_indices()

		rho1phi1  = UqB[:, :, irho1phi1] # [n, nq]
		rho2phi2  = UqB[:, :, irho2phi2] # [n, nq]
		rhou      = UqB[:, :, irhou]     # [n, nq]
		rhov      = UqB[:, :, irhov]     # [n, nq]
		rhoE      = UqB[:, :, irhoE]     # [n, nq]
		phi1      = UqB[:, :, iPF]       # [n, nq]
		LS        = UqB[:, :, iLS]       # [n, nq]
		
		# Classic (adiabatic?)
		Twall = self.Twall
		if self.Twall<0.:
			rho = rho1phi1 + rho2phi2
			UqB[:,:,irhoE] = rhoE - 0.5*(rhou**2 + rhov**2)/rho
		else:
			# Isothermal
			cv1   = physics.cv1
			cv2   = physics.cv2
			pinf1 = physics.pinf1
			pinf2 = physics.pinf2
			gamma1= physics.gamma1
			gamma2= physics.gamma2
			rho = rho1phi1 + rho2phi2
			rhoe = rhoE - 0.5 * (rhou**2+rhov**2)/rho # [n, nq]
			one_over_gamma_m1 = phi1/(gamma1-1.0) + (1.0-phi1)/(gamma2-1.0)
			rhoq = rho1phi1*physics.q1 + rho2phi2*physics.q2
			pI = (rhoe - rhoq + (-phi1*gamma1*pinf1/(gamma1-1.)-(1.0-phi1)*gamma2*pinf2/(gamma2-1.)))/one_over_gamma_m1
#			gamma = (one_over_gamma_m1+1.0)/one_over_gamma_m1
#			pinf = (phi1*gamma1*pinf1/(gamma1-1.0) + (1.0-phi1)*gamma2*pinf2/(gamma2-1.0))/one_over_gamma_m1/gamma

			rhoI1 = (pI + pinf1)/(cv1*Twall*(gamma1-1.))
			rhoI2 = (pI + pinf2)/(cv2*Twall*(gamma2-1.))
			UqB[:,:,irho1phi1] = rhoI1*phi1
			UqB[:,:,irho2phi2] = rhoI2*(1.0-phi1)

			rhoq   = UqB[:,:,irho1phi1]*physics.q1 + UqB[:,:,irho2phi2]*physics.q2
		
			rhoe1 = (pI + gamma1*pinf1)/(gamma1-1.0)
			rhoe2 = (pI + gamma2*pinf2)/(gamma2-1.0)
		
			rhoe = rhoe1*phi1 + rhoe2*(1.0-phi1) + rhoq
		
#			UqB[:,:,irhoE] = rhocp*Twall + pinf + rhoq
			UqB[:,:,irhoE] = rhoe

		UqB[:,:,irhou] = 0.
		UqB[:,:,irhov] = 0.
		
		return UqB

class SlipWall(BCWeakRiemann):
	'''
	This class corresponds to a slip wall. See documentation for more
	details.
	'''
	def __init__(self, Twall):
		'''
		This method initializes the attributes.

		Inputs:
		-------
			p: pressure

		Outputs:
		--------
		    self: attributes initialized
		'''
		self.Twall = Twall
		
	def get_boundary_state(self, physics, UqI, normals, x, t):
		#UqB = self.function.get_state(physics, x, t)
		UqB = UqI.copy()
		
		irho1phi1, irho2phi2, irhou, irhov, irhoE, iPF, iLS = physics.get_state_indices()

		rho1phi1  = UqB[:, :, irho1phi1] # [n, nq]
		rho2phi2  = UqB[:, :, irho2phi2] # [n, nq]
		rhou      = UqB[:, :, irhou]     # [n, nq]
		rhov      = UqB[:, :, irhov]     # [n, nq]
		rhoE      = UqB[:, :, irhoE]     # [n, nq]
		phi1      = UqB[:, :, iPF]       # [n, nq]
		LS        = UqB[:, :, iLS]       # [n, nq]
		
		# Classic (adiabatic?)
		Twall = self.Twall
		if self.Twall<0.:
			rho = rho1phi1 + rho2phi2
			UqB[:,:,irhoE] = rhoE - 0.5*(rhou**2 + rhov**2)/rho
		else:
			# Isothermal
			cv1   = physics.cv1
			cv2   = physics.cv2
			pinf1 = physics.pinf1
			pinf2 = physics.pinf2
			gamma1= physics.gamma1
			gamma2= physics.gamma2
			rho = rho1phi1 + rho2phi2
			rhoe = rhoE - 0.5 * (rhou**2+rhov**2)/rho # [n, nq]
			one_over_gamma_m1 = phi1/(gamma1-1.0) + (1.0-phi1)/(gamma2-1.0)
			rhoq = rho1phi1*physics.q1 + rho2phi2*physics.q2
			pI = (rhoe - rhoq + (-phi1*gamma1*pinf1/(gamma1-1.)-(1.0-phi1)*gamma2*pinf2/(gamma2-1.)))/one_over_gamma_m1
#			gamma = (one_over_gamma_m1+1.0)/one_over_gamma_m1
#			pinf = (phi1*gamma1*pinf1/(gamma1-1.0) + (1.0-phi1)*gamma2*pinf2/(gamma2-1.0))/one_over_gamma_m1/gamma

			rhoI1 = (pI + pinf1)/(cv1*Twall*(gamma1-1.))
			rhoI2 = (pI + pinf2)/(cv2*Twall*(gamma2-1.))
			UqB[:,:,irho1phi1] = rhoI1*phi1
			UqB[:,:,irho2phi2] = rhoI2*(1.0-phi1)

			rhoq   = UqB[:,:,irho1phi1]*physics.q1 + UqB[:,:,irho2phi2]*physics.q2
		
			rhoe1 = (pI + gamma1*pinf1)/(gamma1-1.0)
			rhoe2 = (pI + gamma2*pinf2)/(gamma2-1.0)
		
			rhoe = rhoe1*phi1 + rhoe2*(1.0-phi1) + rhoq
		
#			UqB[:,:,irhoE] = rhocp*Twall + pinf + rhoq
			UqB[:,:,irhoE] = rhoe

		UqB[:,:,irhou] = 0.
		UqB[:,:,irhov] = 0.
		
		return UqB

class Subsonic_Outlet(BCWeakRiemann):
	'''
	This class corresponds to a slip wall. See documentation for more
	details.
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
		#UqB = self.function.get_state(physics, x, t)
		UqB = UqI.copy()
		
		irho1phi1, irho2phi2, irhou, irhov, irhoE, iPF, iLS = physics.get_state_indices()

		rho1phi1  = UqB[:, :, irho1phi1] # [n, nq]
		rho2phi2  = UqB[:, :, irho2phi2] # [n, nq]
		rhou      = UqB[:, :, irhou]     # [n, nq]
		rhov      = UqB[:, :, irhov]     # [n, nq]
		rhoE      = UqB[:, :, irhoE]     # [n, nq]
		phi1      = UqB[:, :, iPF]       # [n, nq]
		LS        = UqB[:, :, iLS]       # [n, nq]
		
		p = self.p
		
		gamma1 = physics.gamma1
		gamma2 = physics.gamma2
		pinf1  = physics.pinf1
		pinf2  = physics.pinf2
		
		rho = rho1phi1 + rho2phi2
		rhoe = rhoE - 0.5 * (rhou**2+rhov**2)/rho # [n, nq]
		one_over_gamma_m1 = phi1/(gamma1-1.0) + (1.0-phi1)/(gamma2-1.0)
		pI = (rhoe + (-phi1*gamma1*pinf1/(gamma1-1.)-(1.0-phi1)*gamma2*pinf2/(gamma2-1.)))/one_over_gamma_m1
		
		rhoe = ((2.0*p-pI) + (phi1*gamma1/(gamma1-1.0)*pinf1+(1.0-phi1)*gamma2/(gamma2-1.0)*pinf2))*one_over_gamma_m1
		
		UqB[:,:,irhoE] = rhoe + 0.5*(rhou**2 + rhov**2)/(rho1phi1+rho2phi2)
		
		return UqB

class Subsonic_Inlet(BCWeakRiemann):
	'''
	This class corresponds to a slip wall. See documentation for more
	details.
	'''
	def __init__(self, d=0., thick=0., uavg=0., \
			rho1_in=1., rho2_in=1., phi0 = 1.0):
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
		self.d = d
		self.thick = thick
		self.uavg = uavg
		self.rho1_in = rho1_in
		self.rho2_in = rho2_in
		self.phi0 = phi0
		
	def get_boundary_state(self, physics, UqI, normals, x, t):
		#UqB = self.function.get_state(physics, x, t)
		UqB = UqI.copy()
		
		irho1phi1, irho2phi2, irhou, irhov, irhoE, iPF, iLS = physics.get_state_indices()
		
		rho1phi1  = UqB[:, :, irho1phi1] # [n, nq]
		rho2phi2  = UqB[:, :, irho2phi2] # [n, nq]
		rhou      = UqB[:, :, irhou]     # [n, nq]
		rhov      = UqB[:, :, irhov]     # [n, nq]
		rhoE      = UqB[:, :, irhoE]     # [n, nq]
		phi1      = UqB[:, :, iPF]       # [n, nq]
		LS        = UqB[:, :, iLS]       # [n, nq]

		UqB[:,:,iPF] = self.phi0
		
		UqB[:,:,irho1phi1] = self.rho1_in*UqB[:,:,iPF]
		UqB[:,:,irho2phi2] = self.rho2_in*(1.0-UqB[:,:,iPF])
		
		rhoB = UqB[:,:,irho1phi1] + UqB[:,:,irho2phi2]
		
		A = self.uavg*6.0/self.d**2
		
		UqB[:,:,irhou] = -A*(x[:,:,1]**2-0.25*self.d**2)*rhoB
		UqB[:,:,irhov] = 0.
		
		rho = rho1phi1 + rho2phi2
		rhoe = (rhoE - 0.5 * (rhou**2+rhov**2)/rho) # [n, nq]
		
		UqB[:,:,irhoE] = rhoe + 0.5*(UqB[:,:,irhou]**2+UqB[:,:,irhov]**2)/rhoB
		
		return UqB

#class PressureOutletNS(BCWeakPrescribed):
#	'''
#	This class corresponds to an outflow boundary condition with static
#	pressure prescribed. See documentation for more details.
#
#	Attributes:
#	-----------
#	p: float
#		pressure
#	'''
#	def __init__(self, p):
#		'''
#		This method initializes the attributes.
#
#		Inputs:
#		-------
#			p: pressure
#
#		Outputs:
#		--------
#		    self: attributes initialized
#		'''
#		self.p = p
#
#	def get_boundary_state(self, physics, UqI, normals, x, t):
#		#UqB = self.function.get_state(physics, x, t)
#		UqB = UqI.copy()
#		irho1phi1, irho2phi2, irhou, irhov, irhoE, iPF, iLS = physics.get_state_indices()
#
#		# Pressure
#		pB = self.p
#
#		# Unit normals
#		n_hat = normals/np.linalg.norm(normals, axis=2, keepdims=True)
#
#		# Interior velocity in normal direction
#		rhoI  = UqI[:, :, irho1phi1] + UqI[:, :, irho2phi2]
#		velxI = UqI[:, :, irhou]/rhoI
#		velyI = UqI[:, :, irhov]/rhoI
#
#		velnI = velxI*n_hat[:,:,0] + velyI*n_hat[:,:,1]
#
#		if np.any(velnI < 0.):
#			print("Incoming flow at outlet")
#
#		# Interior pressure
#		gamma1 = physics.gamma1
#		gamma2 = physics.gamma2
#		pinf1  = physics.pinf1
#		pinf2  = physics.pinf2
#
#		rhoe = UqI[:, :, irhoE] - 0.5 * (UqI[:, :, irhou]**2+UqI[:, :, irhov]**2)/rhoI # [n, nq]
#		one_over_gamma_m1 = UqI[:, :, iPF]/(gamma1-1.0) + (1.0-UqI[:, :, iPF])/(gamma2-1.0)
#		gamma = (one_over_gamma_m1+1.0)/one_over_gamma_m1
#		pinf = (gamma-1.0)/gamma*(UqI[:, :, iPF]*gamma1*pinf1/(gamma1-1.0) + (1.0-UqI[:, :, iPF])*gamma2*pinf2/(gamma2-1.0))
#		pI = rhoe/one_over_gamma_m1 - gamma*pinf
#
#		if np.any(pI < 0.):
#			raise errors.NotPhysicalError
#
#		# Interior speed of sound
#		cI = np.abs(gamma*(pI+pinf)/rhoI)
#		JI = velnI + 2.*cI/(gamma - 1.)
#		# Interior velocity in tangential direction
#		veltxI = velxI - velnI*n_hat[:,:,0]
#		veltyI = velyI - velnI*n_hat[:,:,1]
#
#		# Normal Mach number
#		Mn = velnI/cI
#		if np.any(Mn >= 1.):
#			# If supersonic, then extrapolate interior to exterior
#			return UqB
#
#		# Boundary density from interior entropy
#		rhoB = rhoI*np.power(pB/pI, 1./gamma)
#
#		# Boundary speed of sound
#		cB = np.sqrt(gamma*(pB+pinf)/rhoB)
#		# Boundary velocity
#		velxB = (JI - 2.*cB/(gamma-1.))*n_hat[:,:,0] + veltxI
#		velyB = (JI - 2.*cB/(gamma-1.))*n_hat[:,:,1] + veltyI
#		UqB[:, :, irhou] = rhoB*velxB
#		UqB[:, :, irhov] = rhoB*velyB
#
#		# Boundary energy
#		rhovel2B = rhoB*(velxB**2+velyB**2)
#		UqB[:, :, irhoE] = pB/(gamma - 1.) + 0.5*rhovel2B
#
#		return UqB
'''
------------------------
Numerical flux functions
------------------------
These classes inherit from the ConvNumFluxBase or DiffNumFluxBase class. 
See ConvNumFluxBase/DiffNumFluxBase for detailed comments of attributes 
and methods. Information specific to the corresponding child classes can 
be found below. These classes should correspond to the ConvNumFluxType 
or DiffNumFluxType enum members above.
'''
