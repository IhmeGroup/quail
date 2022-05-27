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
	
class BCType(Enum):
	'''
	Enum class that stores the types of boundary conditions. These
	boundary conditions are specific to the available Euler equation sets.
	'''
	NoSlipWall = auto()
	
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
		
		tol = 1e-10
		
		Hr  = 0.5*(1.0+np.tanh(self.thick*(r-self.radius)))
		
		# Phase-field and Level-set
		Uq[:,:,iPF] = tol + (1.0-2.0*tol)*(1.0-Hr)
		Uq[:,:,iLS] = -(r-self.radius)
		
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
		
		tol = 1e-10
		
		d = self.d
		
		x_I = np.linspace(0,self.d,1000)
		y_I = 2.*d + 0.1*d*np.cos(2.*np.pi*x_I/d)
		
		dist = np.zeros(x[:,:,0].shape)
		for i in range(len(x[:,0,0])):
			for j in range(len(x[0,:,0])):
				dist_min = 1e10
				pt_I = np.zeros(2)
				for ll in range(0,len(x_I)):
					dist_loc = np.sqrt((x[i,j,0]-x_I[ll])**2+(x[i,j,1]-y_I[ll])**2)
					if dist_loc<dist_min:
						dist_min = dist_loc
						pt_I[0] = x_I[ll]
						pt_I[1] = y_I[ll]
		
				if pt_I[1]>x[i,j,1]:
					Uq[i,j,iLS] = -dist_min
				else:
					Uq[i,j,iLS] = dist_min
		
		Hd = 0.5*(1.0+np.tanh(-self.thick*Uq[:,:,iLS]))
		Uq[:,:,iPF] = tol + (1.0-2.0*tol)*(1.0-Hd)
		
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
	
	def get_source(self, physics, Uq, gUq, x, t):
		
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
		
		# Get velocity in each dimension
		rho   = rho1phi1 + rho2phi2
		u = rhou / rho
		v = rhov / rho
		
		switch = physics.switch
		
		# gravity
		Sq[:,:,irhov] = -rho*physics.g*switch
		
		# transport equations
		Sq[:,:,iPF] = -(u*gUx[:,:,iPF] + v*gUy[:,:,iPF])*switch
		Sq[:,:,iLS] = -(u*gUx[:,:,iLS] + v*gUy[:,:,iLS])*switch
		
		if switch == 0:
			eps = physics.scl_eps*physics.eps
			psi0 = Uq[:,:,iPF]-0.5
			Uqx = gUq[:,:,iPF,0]
			Uqy = gUq[:,:,iPF,1]
			mag3 = np.sqrt(Uqx**2 + Uqy**2 + 1e-32)
			sgn = np.tanh(0.5*(psi0)/eps/mag3)
			mag = np.sqrt(gUq[:, :, iLS, 0]**2 + gUq[:, :, iLS, 1]**2 + 1e-32)
			l = 1.0
			Sq[:,:,iLS] = (1.0-mag)*sgn*l
		
		return Sq
		
class BubbleSource2(SourceBase):
	'''
	Source for bubble case
	'''
	
	def get_source(self, physics, Uq, gUq, x, t):
		
		irhou, irhov, ip, iPF, iLS = physics.get_state_indices()

		rhou      = Uq[:, :, irhou]     # [n, nq]
		rhov      = Uq[:, :, irhov]     # [n, nq]
		p         = Uq[:, :, ip]     # [n, nq]
		phi1      = Uq[:, :, iPF]       # [n, nq]
		LS        = Uq[:, :, iLS]       # [n, nq]

		Sq = np.zeros(Uq.shape)
		
		# Separate x and y gradients
		gUx = gUq[:, :, :, 0] # [ne, nq, ns]
		gUy = gUq[:, :, :, 1] # [ne, nq, ns]
		
		# Get velocity in each dimension
		rho1  = physics.rho1
		rho2  = physics.rho2
		cs = physics.cs
		c2 = cs**2
		
		rho   = rho1*phi1 + rho2*(1.0-phi1)
		u = rhou / rho
		v = rhov / rho
		
		Sq[:,:,ip] = c2*(rho1-rho2)*(u*gUx[:,:,iPF]+v*gUy[:,:,iPF])

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
		
		for ii in range(len(x[:,0,0])):
			for jj in range(len(x[0,:,0])):
				if x[ii,jj,1]>2.0:
					UqB[ii,jj,iPF] = 1.0-1e-10
				else:
					UqB[ii,jj,iPF] = 1e-10

		UqB[:,:,irho1phi1] = physics.rho01*UqB[:,:,iPF]
		UqB[:,:,irho2phi2] = physics.rho02*(1.0-UqB[:,:,iPF])

		rho = UqB[:,:,irho1phi1] + UqB[:,:,irho2phi2]
		
		UqB[:,:,irhou] = 0.
		UqB[:,:,irhov] = 0.
		
		rhoe = (0. + physics.gamma1*physics.pinf1)/(physics.gamma1-1.)
		
		UqB[:,:,irhoE] = rhoe
		
		return UqB

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
