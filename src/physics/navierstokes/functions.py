# ------------------------------------------------------------------------ #
#
#       File : src/physics/euler/functions.py
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
	Enum class that stores the types of analytic functions for initial
	conditions, exact solutions, and/or boundary conditions. These
	functions are specific to the available Euler equation sets.
	'''
	ManufacturedSolution = auto()


# class BCType(Enum):
	'''
	Enum class that stores the types of boundary conditions. These
	boundary conditions are specific to the available Euler equation sets.
	'''
	# SlipWall = auto()
	# PressureOutlet = auto()


class SourceType(Enum):
	'''
	Enum class that stores the types of source terms. These
	source terms are specific to the available Euler equation sets.
	'''
	ManufacturedSource = auto()


# class ConvNumFluxType(Enum):
	'''
	Enum class that stores the types of convective numerical fluxes. 
	These numerical fluxes are specific to the available Euler 
	equation sets.
	'''
	# Roe = auto()

class DiffNumFluxType(Enum):
	'''
	Enum class that stores the types of diffusive numerical fluxes. 
	These numerical fluxes are specific to the available NavierStokes 
	equation sets.
	'''
	SIP = auto()

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
	2D/manufactured_solution
	'''
	def __init__(self):
		pass
	def get_state(self, physics, x, t):
		# Unpack
		gamma = physics.gamma
		
		irho, irhou, irhov, irhoE = physics.get_state_indices()

   		# define the constants for the various params
		rhob, rho0, u0, v0, pb, p0, k, om = 1.0, 0.5, 0.25, \
			0.25, 1/gamma, 0.1, 2.0*np.pi/10.0, 2.0*np.pi		
		
		x1 = x[:, :, 0]
		x2 = x[:, :, 1]

		rho = rhob + rho0 * np.cos(k * (x1 + x2) - om * t)
		u = u0 * np.sin(k * (x1 + x2) - om * t)
		v = v0 * np.sin(k * (x1 + x2) - om * t)
		p = pb + p0 * np.sin(k * (x1 + x2) - om * t)

		E = p/(rho*(gamma - 1.)) + 0.5*(u**2. + v**2.)

		''' Fill state '''
		Uq = np.zeros([x.shape[0], x.shape[1], physics.NUM_STATE_VARS])

		Uq[:, :, irho] = rho
		Uq[:, :, irhou] = rho*u
		Uq[:, :, irhov] = rho*v
		Uq[:, :, irhoE] = rho*E

		return Uq # [ne, nq, ns]

# class TaylorGreenVortex(FcnBase):
# 	'''
# 	2D Taylor Green Vortex Case
# 	'''
# 	def __init__(self):
# 		pass
# 	def get_state(self, physics, x, t):
# 		# Unpack
# 		gamma = physics.gamma
		
# 		irho, irhou, irhov, irhoE = physics.get_state_indices()

		
# 		x = x[:, :, 0]
# 		y = x[:, :, 1]
		
# 		u = np.cos(2.*np.pi * x)*np.sin(2.*np.pi * y)* \
# 			np.exp((-8.*np.pi**2 / Re) * t)
# 		v = -np.sin(2.*np.pi * x)*np.cos(2.*np.pi * y)* \
# 			np.exp((-8.*np.pi**2 / Re) * t)
# 		p = -0.25 * (np.cos(4.*np.pi * x) + np.cos(4.*np.pi * y))* \
# 			np.exp((-16.*np.pi**2 / Re) * t)

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

		# Calculate viscosity
		mu = physics.compute_variable("Viscosity", Uq,
			flag_non_physical=True)

		# Calculate thermal conductivity
		kappa = physics.compute_variable("ThermalConductivity",
			Uq, flag_non_physical=True)

		Sq = np.zeros_like(Uq)
		Sq[:, :, irho], Sq[:, :, irhou], Sq[:, :, irhov], \
			Sq[:, :, irhoE] = self.manufactured_source(x1, x2, 
			t, gamma, kappa[:, :, 0], 
			mu[:, :, 0], R)

		
		return Sq # [ne, nq, ns]

	def manufactured_source(self, x1, x2, t, gamma, kappa, mu, R):
	
		# S_rho = -0.4*np.pi*(-0.125*np.cos(2.0*np.pi*t - 0.2*np.pi*( \
		# 	x1 + x2)) - 0.25)*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 \
		# 	+ x2)) - 0.05*np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*( \
		# 	x1 + x2))**2 - 1.0*np.pi*np.sin(2.0*np.pi*t - 0.2* \
		# 	np.pi*(x1 + x2))

		# S_rhou = 0.0266666666666667*np.pi**2*mu*np.sin(2.0*np.pi*t \
		# 	- 0.2*np.pi*(x1 + x2)) + 2.0*np.pi*(-0.125*np.cos( \
		# 	2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.25)*np.cos( \
		# 	2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.05*np.pi*( \
		# 	0.5*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) +  \
		# 	1.0)*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))* \
		# 	np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 0.0125* \
		# 	np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**3 \
		# 	+ 0.25*np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + \
		# 	x2))**2 + 0.02*np.pi*np.cos(2.0*np.pi*t - \
		# 	0.2*np.pi*(x1 + x2))

		# S_rhov = 0.0266666666666667*np.pi**2*mu*np.sin(2.0*np.pi*t \
		# 	- 0.2*np.pi*(x1 + x2)) + 2.0*np.pi*(-0.125*np.cos( \
		# 	2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.25)*np.cos( \
		# 	2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.05*np.pi*( \
		# 	0.5*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + \
		# 	1.0)*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))* \
		# 	np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 0.0125* \
		# 	np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**3 \
		# 	+ 0.25*np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + \
		# 	x2))**2 + 0.02*np.pi*np.cos(2.0*np.pi*t \
		# 	- 0.2*np.pi*(x1 + x2))

		# S_rhoE = -0.0133333333333333*np.pi**2*mu*np.sin(2.0*np.pi*t \
		# 	- 0.2*np.pi*(x1 + x2))**2 + 0.0133333333333333*     \
		# 	np.pi**2*mu*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 +    \
		# 	x2))**2 + 0.5*np.pi*(0.25*np.cos(2.0*np.pi*t - 0.2* \
		# 	np.pi*(x1 + x2)) + 0.5)*np.sin(2.0*np.pi*t - 0.2*   \
		# 	np.pi*(x1 + x2))*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 \
		# 	+ x2)) - 0.4*np.pi*(-0.03125*(0.25*np.cos(2.0*np.pi \
		# 	*t - 0.2*np.pi*(x1 + x2)) + 0.5)*np.sin(2.0*np.pi*t \
		# 	- 0.2*np.pi*(x1 + x2))**2 + 0.025*np.sin(2.0*np.pi* \
		# 	t - 0.2*np.pi*(x1 + x2)) - 0.25*(-0.1*np.sin(2.0*   \
		# 	np.pi*t - 0.2*np.pi*(x1 + x2)) + gamma**(-1))/(     \
		# 	gamma - 1) - 0.25/gamma)*np.cos(2.0*np.pi*t - 0.2*  \
		# 	np.pi*(x1 + x2)) + 2*(0.0125*np.pi*(0.25*np.cos(2.0 \
		# 	*np.pi*t - 0.2*np.pi*(x1 + x2)) + 0.5)*np.sin(2.0*  \
		# 	np.pi*t - 0.2*np.pi*(x1 + x2))*np.cos(2.0*np.pi*t - \
		# 	0.2*np.pi*(x1 + x2)) - 0.0015625*np.pi*np.sin(2.0*  \
		# 	np.pi*t - 0.2*np.pi*(x1 + x2))**3 - 0.005*np.pi*    \
		# 	np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.005*  \
		# 	np.pi*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))/(   \
		# 	gamma - 1))*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 +    \
		# 	x2)) - 0.0625*np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi* \
		# 	(x1 + x2))**3 - 0.2*np.pi*np.cos(2.0*np.pi*t - 0.2* \
		# 	np.pi*(x1 + x2))/(gamma - 1) + 2*np.pi**2*kappa*(   \
		# 	0.02*(0.1*np.sin(np.pi*(2.0*t - 0.2*x1 - 0.2*x2)) - \
		# 	1/gamma)*(np.cos(np.pi*(-2.0*t + 0.2*x1 + 0.2*x2))  \
		# 	+ np.sin(np.pi*(-2.0*t + 0.2*x1 + 0.2*x2))**2/(0.5* \
		# 	np.cos(np.pi*(-2.0*t + 0.2*x1 + 0.2*x2)) + 1.0))/(  \
		# 	0.5*np.cos(np.pi*(-2.0*t + 0.2*x1 + 0.2*x2)) + 1.0  \
		# 	)**2 + 0.004*np.sin(np.pi*(-2.0*t + 0.2*x1 + 0.2*x2 \
		# 	))/(0.5*np.cos(np.pi*(2.0*t - 0.2*x1 - 0.2*x2)) +   \
		# 	1.0) + 0.004*np.sin(np.pi*(2.0*t - 0.2*x1 - 0.2*x2  \
		# 	))*np.cos(np.pi*(2.0*t - 0.2*x1 - 0.2*x2))/(0.5*    \
		# 	np.cos(np.pi*(2.0*t - 0.2*x1 - 0.2*x2)) + 1.0)**2)/R
		
		#---------------------------
		# Currently the baseline

		# S_rho = -0.4*np.pi*(-0.125*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.25)*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.05*np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**2 - 1.0*np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))
		# S_rhou = -0.0266666666666667*np.pi**2*mu*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 2.0*np.pi*(-0.125*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.25)*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.05*np.pi*(0.5*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 1.0)*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 0.0125*np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**3 + 0.25*np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**2 + 0.02*np.pi*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))
		# S_rhov = -0.0266666666666667*np.pi**2*mu*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 2.0*np.pi*(-0.125*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.25)*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.05*np.pi*(0.5*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 1.0)*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 0.0125*np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**3 + 0.25*np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**2 + 0.02*np.pi*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))
		# S_rhoE = 0.0133333333333333*np.pi**2*mu*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**2 - 0.0133333333333333*np.pi**2*mu*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**2 + 0.5*np.pi*(0.25*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 0.5)*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.4*np.pi*(-0.03125*(0.25*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 0.5)*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**2 + 0.025*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.25*(-0.1*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + gamma**(-1))/(gamma - 1) - 0.25/gamma)*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 2*(0.0125*np.pi*(0.25*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 0.5)*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.0015625*np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**3 - 0.005*np.pi*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.005*np.pi*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))/(gamma - 1))*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.0625*np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**3 - 0.2*np.pi*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))/(gamma - 1) + 2*np.pi**2*kappa*(0.02*(0.1*np.sin(np.pi*(2.0*t - 0.2*x1 - 0.2*x2)) - 1/gamma)*(np.cos(np.pi*(-2.0*t + 0.2*x1 + 0.2*x2)) + np.sin(np.pi*(-2.0*t + 0.2*x1 + 0.2*x2))**2/(0.5*np.cos(np.pi*(-2.0*t + 0.2*x1 + 0.2*x2)) + 1.0))/(0.5*np.cos(np.pi*(-2.0*t + 0.2*x1 + 0.2*x2)) + 1.0)**2 + 0.004*np.sin(np.pi*(-2.0*t + 0.2*x1 + 0.2*x2))/(0.5*np.cos(np.pi*(2.0*t - 0.2*x1 - 0.2*x2)) + 1.0) + 0.004*np.sin(np.pi*(2.0*t - 0.2*x1 - 0.2*x2))*np.cos(np.pi*(2.0*t - 0.2*x1 - 0.2*x2))/(0.5*np.cos(np.pi*(2.0*t - 0.2*x1 - 0.2*x2)) + 1.0)**2)/R
		# L2 error -> 0.041730878119526 (text1.mp4)

		#---------------------------
		# Add temporal terms only (doesnt behave properly)
		# S_rho = -0.4*np.pi*(-0.125*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.25)*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.05*np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**2 + 1.0*np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))
		# S_rhou = -0.0266666666666667*np.pi**2*mu*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 2.0*np.pi*(-0.125*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.25)*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.05*np.pi*(0.5*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 1.0)*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 0.0125*np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**3 - 0.25*np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**2 + 0.02*np.pi*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))
		# S_rhov = -0.0266666666666667*np.pi**2*mu*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 2.0*np.pi*(-0.125*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.25)*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.05*np.pi*(0.5*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 1.0)*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 0.0125*np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**3 - 0.25*np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**2 + 0.02*np.pi*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))
		# S_rhoE = 0.0133333333333333*np.pi**2*mu*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**2 - 0.0133333333333333*np.pi**2*mu*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**2 - 0.5*np.pi*(0.25*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 0.5)*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.4*np.pi*(-0.03125*(0.25*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 0.5)*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**2 + 0.025*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.25*(-0.1*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + gamma**(-1))/(gamma - 1) - 0.25/gamma)*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 2*(0.0125*np.pi*(0.25*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 0.5)*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.0015625*np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**3 - 0.005*np.pi*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.005*np.pi*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))/(gamma - 1))*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 0.0625*np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**3 + 0.2*np.pi*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))/(gamma - 1) + 2*np.pi**2*kappa*(0.02*(0.1*np.sin(np.pi*(2.0*t - 0.2*x1 - 0.2*x2)) - 1/gamma)*(np.cos(np.pi*(-2.0*t + 0.2*x1 + 0.2*x2)) + np.sin(np.pi*(-2.0*t + 0.2*x1 + 0.2*x2))**2/(0.5*np.cos(np.pi*(-2.0*t + 0.2*x1 + 0.2*x2)) + 1.0))/(0.5*np.cos(np.pi*(-2.0*t + 0.2*x1 + 0.2*x2)) + 1.0)**2 + 0.004*np.sin(np.pi*(-2.0*t + 0.2*x1 + 0.2*x2))/(0.5*np.cos(np.pi*(2.0*t - 0.2*x1 - 0.2*x2)) + 1.0) + 0.004*np.sin(np.pi*(2.0*t - 0.2*x1 - 0.2*x2))*np.cos(np.pi*(2.0*t - 0.2*x1 - 0.2*x2))/(0.5*np.cos(np.pi*(2.0*t - 0.2*x1 - 0.2*x2)) + 1.0)**2)/R

		#---------------------------
		# Add temporal and convection terms (doesnt behave properly)
		# S_rho = 0.4*np.pi*(-0.125*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.25)*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 0.05*np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**2 + 1.0*np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))

		# S_rhou = -0.0266666666666667*np.pi**2*mu*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 2.0*np.pi*(-0.125*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.25)*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 0.05*np.pi*(0.5*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 1.0)*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.0125*np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**3 - 0.25*np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**2 - 0.02*np.pi*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))

		# S_rhov = -0.0266666666666667*np.pi**2*mu*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 2.0*np.pi*(-0.125*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.25)*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 0.05*np.pi*(0.5*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 1.0)*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.0125*np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**3 - 0.25*np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**2 - 0.02*np.pi*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))

		# S_rhoE = 0.0133333333333333*np.pi**2*mu*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**2 - 0.0133333333333333*np.pi**2*mu*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**2 - 0.5*np.pi*(0.25*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 0.5)*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 0.4*np.pi*(-0.03125*(0.25*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 0.5)*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**2 + 0.025*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.25*(-0.1*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + gamma**(-1))/(gamma - 1) - 0.25/gamma)*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 2*(0.0125*np.pi*(0.25*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 0.5)*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.0015625*np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**3 - 0.005*np.pi*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.005*np.pi*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))/(gamma - 1))*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 0.0625*np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**3 + 0.2*np.pi*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))/(gamma - 1) + 2*np.pi**2*kappa*(0.02*(0.1*np.sin(np.pi*(2.0*t - 0.2*x1 - 0.2*x2)) - 1/gamma)*(np.cos(np.pi*(-2.0*t + 0.2*x1 + 0.2*x2)) + np.sin(np.pi*(-2.0*t + 0.2*x1 + 0.2*x2))**2/(0.5*np.cos(np.pi*(-2.0*t + 0.2*x1 + 0.2*x2)) + 1.0))/(0.5*np.cos(np.pi*(-2.0*t + 0.2*x1 + 0.2*x2)) + 1.0)**2 + 0.004*np.sin(np.pi*(-2.0*t + 0.2*x1 + 0.2*x2))/(0.5*np.cos(np.pi*(2.0*t - 0.2*x1 - 0.2*x2)) + 1.0) + 0.004*np.sin(np.pi*(2.0*t - 0.2*x1 - 0.2*x2))*np.cos(np.pi*(2.0*t - 0.2*x1 - 0.2*x2))/(0.5*np.cos(np.pi*(2.0*t - 0.2*x1 - 0.2*x2)) + 1.0)**2)/R


		#---------------------------
		# Subtract viscous terms
		# L2norm -> 0.097916825525651 (test2.mp4)
		# S_rho = -0.4*np.pi*(-0.125*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.25)*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.05*np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**2 - 1.0*np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))
		# S_rhou = 0.0266666666666667*np.pi**2*mu*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 2.0*np.pi*(-0.125*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.25)*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.05*np.pi*(0.5*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 1.0)*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 0.0125*np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**3 + 0.25*np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**2 + 0.02*np.pi*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))
		# S_rhov = 0.0266666666666667*np.pi**2*mu*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 2.0*np.pi*(-0.125*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.25)*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.05*np.pi*(0.5*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 1.0)*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 0.0125*np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**3 + 0.25*np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**2 + 0.02*np.pi*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))
		# S_rhoE = -0.0133333333333333*np.pi**2*mu*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**2 + 0.0133333333333333*np.pi**2*mu*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**2 + 0.5*np.pi*(0.25*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 0.5)*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.4*np.pi*(-0.03125*(0.25*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 0.5)*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**2 + 0.025*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.25*(-0.1*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + gamma**(-1))/(gamma - 1) - 0.25/gamma)*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 2*(0.0125*np.pi*(0.25*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 0.5)*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.0015625*np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**3 - 0.005*np.pi*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.005*np.pi*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))/(gamma - 1))*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.0625*np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**3 - 0.2*np.pi*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))/(gamma - 1) + 2*np.pi**2*kappa*(0.02*(0.1*np.sin(np.pi*(2.0*t - 0.2*x1 - 0.2*x2)) - 1/gamma)*(np.cos(np.pi*(-2.0*t + 0.2*x1 + 0.2*x2)) + np.sin(np.pi*(-2.0*t + 0.2*x1 + 0.2*x2))**2/(0.5*np.cos(np.pi*(-2.0*t + 0.2*x1 + 0.2*x2)) + 1.0))/(0.5*np.cos(np.pi*(-2.0*t + 0.2*x1 + 0.2*x2)) + 1.0)**2 + 0.004*np.sin(np.pi*(-2.0*t + 0.2*x1 + 0.2*x2))/(0.5*np.cos(np.pi*(2.0*t - 0.2*x1 - 0.2*x2)) + 1.0) + 0.004*np.sin(np.pi*(2.0*t - 0.2*x1 - 0.2*x2))*np.cos(np.pi*(2.0*t - 0.2*x1 - 0.2*x2))/(0.5*np.cos(np.pi*(2.0*t - 0.2*x1 - 0.2*x2)) + 1.0)**2)/R


		#---------------------------
		# Subtract viscous terms and kappa terms
		# L2norm -> 0.086613001287011 (test3.mp4)
		# S_rho = -0.4*np.pi*(-0.125*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.25)*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.05*np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**2 - 1.0*np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))

		# S_rhou = 0.0266666666666667*np.pi**2*mu*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 2.0*np.pi*(-0.125*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.25)*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.05*np.pi*(0.5*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 1.0)*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 0.0125*np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**3 + 0.25*np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**2 + 0.02*np.pi*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))

		# S_rhov = 0.0266666666666667*np.pi**2*mu*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 2.0*np.pi*(-0.125*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.25)*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.05*np.pi*(0.5*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 1.0)*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 0.0125*np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**3 + 0.25*np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**2 + 0.02*np.pi*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))

		# S_rhoE = -0.0133333333333333*np.pi**2*mu*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**2 + 0.0133333333333333*np.pi**2*mu*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**2 + 0.5*np.pi*(0.25*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 0.5)*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.4*np.pi*(-0.03125*(0.25*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 0.5)*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**2 + 0.025*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.25*(-0.1*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + gamma**(-1))/(gamma - 1) - 0.25/gamma)*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 2*(0.0125*np.pi*(0.25*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 0.5)*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.0015625*np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**3 - 0.005*np.pi*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.005*np.pi*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))/(gamma - 1))*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.0625*np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**3 - 0.2*np.pi*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))/(gamma - 1) - 2*np.pi**2*kappa*(0.02*(0.1*np.sin(np.pi*(2.0*t - 0.2*x1 - 0.2*x2)) - 1/gamma)*(np.cos(np.pi*(-2.0*t + 0.2*x1 + 0.2*x2)) + np.sin(np.pi*(-2.0*t + 0.2*x1 + 0.2*x2))**2/(0.5*np.cos(np.pi*(-2.0*t + 0.2*x1 + 0.2*x2)) + 1.0))/(0.5*np.cos(np.pi*(-2.0*t + 0.2*x1 + 0.2*x2)) + 1.0)**2 + 0.004*np.sin(np.pi*(-2.0*t + 0.2*x1 + 0.2*x2))/(0.5*np.cos(np.pi*(2.0*t - 0.2*x1 - 0.2*x2)) + 1.0) + 0.004*np.sin(np.pi*(2.0*t - 0.2*x1 - 0.2*x2))*np.cos(np.pi*(2.0*t - 0.2*x1 - 0.2*x2))/(0.5*np.cos(np.pi*(2.0*t - 0.2*x1 - 0.2*x2)) + 1.0)**2)/R

		#----------------------------
		# # Exactly kihiro's style of implementation (but opposite sign) [most correct so far...]
		S_rho = 0.4*np.pi*(-0.125*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.25)*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 0.05*np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**2 + 1.0*np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))

		S_rhou = 0.0266666666666667*np.pi**2*mu*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 2.0*np.pi*(-0.125*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.25)*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 0.05*np.pi*(0.5*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 1.0)*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.0125*np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**3 - 0.25*np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**2 - 0.02*np.pi*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))

		S_rhov = 0.0266666666666667*np.pi**2*mu*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 2.0*np.pi*(-0.125*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.25)*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 0.05*np.pi*(0.5*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 1.0)*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.0125*np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**3 - 0.25*np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**2 - 0.02*np.pi*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))

		S_rhoE = -0.0133333333333333*np.pi**2*mu*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**2 + 0.0133333333333333*np.pi**2*mu*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**2 - 0.5*np.pi*(0.25*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 0.5)*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 0.4*np.pi*(-0.03125*(0.25*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 0.5)*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**2 + 0.025*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.25*(-0.1*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + gamma**(-1))/(gamma - 1) - 0.25/gamma)*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 2*(0.0125*np.pi*(0.25*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 0.5)*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.0015625*np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**3 - 0.005*np.pi*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.005*np.pi*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))/(gamma - 1))*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 0.0625*np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**3 + 0.2*np.pi*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))/(gamma - 1) - 2*np.pi**2*kappa*(0.02*(0.1*np.sin(np.pi*(2.0*t - 0.2*x1 - 0.2*x2)) - 1/gamma)*(np.cos(np.pi*(-2.0*t + 0.2*x1 + 0.2*x2)) + np.sin(np.pi*(-2.0*t + 0.2*x1 + 0.2*x2))**2/(0.5*np.cos(np.pi*(-2.0*t + 0.2*x1 + 0.2*x2)) + 1.0))/(0.5*np.cos(np.pi*(-2.0*t + 0.2*x1 + 0.2*x2)) + 1.0)**2 + 0.004*np.sin(np.pi*(-2.0*t + 0.2*x1 + 0.2*x2))/(0.5*np.cos(np.pi*(2.0*t - 0.2*x1 - 0.2*x2)) + 1.0) + 0.004*np.sin(np.pi*(2.0*t - 0.2*x1 - 0.2*x2))*np.cos(np.pi*(2.0*t - 0.2*x1 - 0.2*x2))/(0.5*np.cos(np.pi*(2.0*t - 0.2*x1 - 0.2*x2)) + 1.0)**2)/R

		# S_rho = 0.4*np.pi*(-0.125*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.25)*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 0.05*np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**2 + 1.0*np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))

		# S_rhou = 0.0266666666666667*np.pi**2*mu*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 2.0*np.pi*(-0.125*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.25)*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 0.05*np.pi*(0.5*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 1.0)*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.0125*np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**3 - 0.25*np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**2 - 0.02*np.pi*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))

		# S_rhov = 0.0266666666666667*np.pi**2*mu*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 2.0*np.pi*(-0.125*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.25)*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 0.05*np.pi*(0.5*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 1.0)*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.0125*np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**3 - 0.25*np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**2 - 0.02*np.pi*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))

		# S_rhoE = -0.0133333333333333*np.pi**2*mu*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**2 + 0.0133333333333333*np.pi**2*mu*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**2 - 0.5*np.pi*(0.25*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 0.5)*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 0.4*np.pi*(-0.03125*(0.25*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 0.5)*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**2 + 0.025*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.25*(-0.1*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + gamma**(-1))/(gamma - 1) - 0.25/gamma)*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 2*(0.0125*np.pi*(0.25*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 0.5)*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.0015625*np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**3 - 0.005*np.pi*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) - 0.005*np.pi*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))/(gamma - 1))*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2)) + 0.0625*np.pi*np.sin(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))**3 + 0.2*np.pi*np.cos(2.0*np.pi*t - 0.2*np.pi*(x1 + x2))/(gamma - 1) + 2*np.pi**2*kappa*(0.02*(0.1*np.sin(np.pi*(2.0*t - 0.2*x1 - 0.2*x2)) - 1/gamma)*(np.cos(np.pi*(-2.0*t + 0.2*x1 + 0.2*x2)) + np.sin(np.pi*(-2.0*t + 0.2*x1 + 0.2*x2))**2/(0.5*np.cos(np.pi*(-2.0*t + 0.2*x1 + 0.2*x2)) + 1.0))/(0.5*np.cos(np.pi*(-2.0*t + 0.2*x1 + 0.2*x2)) + 1.0)**2 + 0.004*np.sin(np.pi*(-2.0*t + 0.2*x1 + 0.2*x2))/(0.5*np.cos(np.pi*(2.0*t - 0.2*x1 - 0.2*x2)) + 1.0) + 0.004*np.sin(np.pi*(2.0*t - 0.2*x1 - 0.2*x2))*np.cos(np.pi*(2.0*t - 0.2*x1 - 0.2*x2))/(0.5*np.cos(np.pi*(2.0*t - 0.2*x1 - 0.2*x2)) + 1.0)**2)/R

		return -S_rho, -S_rhou, -S_rhov, -S_rhoE




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
class SIP(DiffNumFluxBase):
	'''
	This class corresponds to the Symmetric Interior Penalty Method (SIP)
	for the NavierStokes class.
	'''
	def compute_flux(self, physics, UqL, UqR, gUqL, gUqR, normals, 
		hL, hR, eta=50.):		

		# Calculate jump condition
		dU = UqL - UqR

		# Normalize the normal vectors
		n_mag = np.linalg.norm(normals, axis=2, keepdims=True)
		n_hat = normals/n_mag

		# Tensor product of normal vector with jump
		dUxn = np.einsum('ijk, ijl -> ijlk', n_hat, dU)

		# Left State
		gFloc2 = 0.5 * physics.get_diff_flux_interior(UqL, gUqL)
		gFloc = physics.get_diff_flux_interior(UqL, dUxn)

		C4 = 0.5 * eta / hL
		C5 = 0.5 * n_mag

		gFloc2 += -1. * np.einsum('i, ijkl -> ijkl', C4, gFloc)
		gFL = np.einsum('ijv, ijkl -> ijkl', C5, gFloc)

		# Right State
		gFloc2 += 0.5 * physics.get_diff_flux_interior(UqR, gUqR)
		gFloc = physics.get_diff_flux_interior(UqR, dUxn)
		
		C4 = 0.5 * eta / hR
		C5 = 0.5 * n_mag

		gFloc2 += -1. * np.einsum('i, ijkl -> ijkl', C4, gFloc)
		gFR = np.einsum('ijv, ijkl -> ijkl', C5, gFloc)

		Floc = np.einsum('ijl, ijkl -> ijk', normals, gFloc2)

		return Floc, gFL, gFR
