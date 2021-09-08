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
	ManufacturedSolutionPeriodic = auto()
	TaylorGreenVortexNS = auto()
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
	ManufacturedSourcePeriodic = auto()
	ManufacturedSource = auto()


'''
---------------
State functions
---------------
These classes inherit from the FcnBase class. See FcnBase for detailed
comments of attributes and methods. Information specific to the
corresponding child classes can be found below. These classes should
correspond to the FcnType enum members above.
'''

class ManufacturedSolutionPeriodic(FcnBase):
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

    		# define the constants for the various params
		# rhob, rho0, u0, v0, pb, p0, k, om = 1.0, 0.5, 0.25, \
		# 	0.25, 1/gamma, 0.1, 2.0*np.pi/10.0, 2.0*np.pi		
		
		x1 = x[:, :, 0]
		x2 = x[:, :, 1]

		# rho = rhob + rho0 * np.cos(k * (x1 + x2) - om * t)
		# u = u0 * np.sin(k * (x1 + x2) - om * t)
		# v = v0 * np.sin(k * (x1 + x2) - om * t)
		# p = pb + p0 * np.sin(k * (x1 + x2) - om * t)

		# E = p/(rho*(gamma - 1.)) + 0.5*(u**2. + v**2.)

		''' Fill state '''
		Uq = np.zeros([x.shape[0], x.shape[1], physics.NUM_STATE_VARS])

		# Generated initial condition from 

		Uq[:, :, irho] = 0.1*np.sin(np.pi*x1) + 0.1*np.cos(np.pi*x1)*np.cos(np.pi*x2) - 0.2*np.cos(np.pi*x2) + 1.0

		Uq[:, :, irhou] = (0.1*np.sin(np.pi*x1) + 0.1*np.cos(np.pi*x1)*np.cos(np.pi*x2) - 0.2*np.cos(np.pi*x2) + 1.0)*(0.3*np.sin(3*np.pi*x1) + 0.3*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.3*np.cos(np.pi*x2) + 2.0)

		Uq[:, :, irhov] = (0.1*np.sin(np.pi*x1) + 0.1*np.cos(np.pi*x1)*np.cos(np.pi*x2) - 0.2*np.cos(np.pi*x2) + 1.0)*(0.3*np.sin(np.pi*x2) + 0.3*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.3*np.cos(np.pi*x1) + 2.0)

		Uq[:, :, irhoE] = (4.0*(0.15*np.sin(3*np.pi*x1) + 0.15*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.15*np.cos(np.pi*x2) + 1)**2 + 4.0*(0.15*np.sin(np.pi*x2) + 0.15*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.15*np.cos(np.pi*x1) + 1)**2)*(0.05*np.sin(np.pi*x1) + 0.05*np.cos(np.pi*x1)*np.cos(np.pi*x2) - 0.1*np.cos(np.pi*x2) + 0.5) + (1.0*np.sin(np.pi*x2) + 0.5*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 1.0*np.cos(2*np.pi*x1) + 10.0)/(gamma - 1)

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
class ManufacturedSourcePeriodic(SourceBase):
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
		# mu = physics.compute_variable("Viscosity", Uq,
		# 	flag_non_physical=True)

		# # Calculate thermal conductivity
		# kappa = physics.compute_variable("ThermalConductivity",
		# 	Uq, flag_non_physical=True)

		mu, kappa = physics.get_transport(physics, Uq, 
			flag_non_physical=False)

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
	
		
		#----------------------------
		# # Exactly kihiro's style of implementation (but opposite sign) [most correct so far...]
		S_rho = (-0.3*np.pi*np.sin(np.pi*x1)*np.cos(np.pi*x2) + 0.9*np.pi*np.cos(3*np.pi*x1))*(0.1*np.sin(np.pi*x1) + 0.1*np.cos(np.pi*x1)*np.cos(np.pi*x2) - 0.2*np.cos(np.pi*x2) + 1.0) + (-0.1*np.pi*np.sin(np.pi*x1)*np.cos(np.pi*x2) + 0.1*np.pi*np.cos(np.pi*x1))*(0.3*np.sin(3*np.pi*x1) + 0.3*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.3*np.cos(np.pi*x2) + 2.0) + (-0.3*np.pi*np.sin(np.pi*x2)*np.cos(np.pi*x1) + 0.3*np.pi*np.cos(np.pi*x2))*(0.1*np.sin(np.pi*x1) + 0.1*np.cos(np.pi*x1)*np.cos(np.pi*x2) - 0.2*np.cos(np.pi*x2) + 1.0) + (-0.1*np.pi*np.sin(np.pi*x2)*np.cos(np.pi*x1) + 0.2*np.pi*np.sin(np.pi*x2))*(0.3*np.sin(np.pi*x2) + 0.3*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.3*np.cos(np.pi*x1) + 2.0)

		S_rhou = -mu*(-0.2*np.pi**2*np.sin(np.pi*x1)*np.sin(np.pi*x2) - 3.6*np.pi**2*np.sin(3*np.pi*x1) - 0.4*np.pi**2*np.cos(np.pi*x1)*np.cos(np.pi*x2)) - mu*(0.3*np.pi**2*np.sin(np.pi*x1)*np.sin(np.pi*x2) - 0.3*np.pi**2*np.cos(np.pi*x1)*np.cos(np.pi*x2) - 0.3*np.pi**2*np.cos(np.pi*x2)) + 4.0*(-0.3*np.pi*np.sin(np.pi*x1)*np.cos(np.pi*x2) + 0.9*np.pi*np.cos(3*np.pi*x1))*(0.1*np.sin(np.pi*x1) + 0.1*np.cos(np.pi*x1)*np.cos(np.pi*x2) - 0.2*np.cos(np.pi*x2) + 1.0)*(0.15*np.sin(3*np.pi*x1) + 0.15*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.15*np.cos(np.pi*x2) + 1) + 4.0*(-0.1*np.pi*np.sin(np.pi*x1)*np.cos(np.pi*x2) + 0.1*np.pi*np.cos(np.pi*x1))*(0.15*np.sin(3*np.pi*x1) + 0.15*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.15*np.cos(np.pi*x2) + 1)**2 + (-0.3*np.pi*np.sin(np.pi*x2)*np.cos(np.pi*x1) - 0.3*np.pi*np.sin(np.pi*x2))*(0.1*np.sin(np.pi*x1) + 0.1*np.cos(np.pi*x1)*np.cos(np.pi*x2) - 0.2*np.cos(np.pi*x2) + 1.0)*(0.3*np.sin(np.pi*x2) + 0.3*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.3*np.cos(np.pi*x1) + 2.0) + (-0.3*np.pi*np.sin(np.pi*x2)*np.cos(np.pi*x1) + 0.3*np.pi*np.cos(np.pi*x2))*(0.1*np.sin(np.pi*x1) + 0.1*np.cos(np.pi*x1)*np.cos(np.pi*x2) - 0.2*np.cos(np.pi*x2) + 1.0)*(0.3*np.sin(3*np.pi*x1) + 0.3*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.3*np.cos(np.pi*x2) + 2.0) + (-0.1*np.pi*np.sin(np.pi*x2)*np.cos(np.pi*x1) + 0.2*np.pi*np.sin(np.pi*x2))*(0.3*np.sin(3*np.pi*x1) + 0.3*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.3*np.cos(np.pi*x2) + 2.0)*(0.3*np.sin(np.pi*x2) + 0.3*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.3*np.cos(np.pi*x1) + 2.0) - 0.5*np.pi*np.sin(np.pi*x1)*np.cos(np.pi*x2) - 2.0*np.pi*np.sin(2*np.pi*x1)

		S_rhov = -mu*(-0.2*np.pi**2*np.sin(np.pi*x1)*np.sin(np.pi*x2) - 0.4*np.pi**2*np.sin(np.pi*x2) - 0.4*np.pi**2*np.cos(np.pi*x1)*np.cos(np.pi*x2)) - mu*(0.3*np.pi**2*np.sin(np.pi*x1)*np.sin(np.pi*x2) - 0.3*np.pi**2*np.cos(np.pi*x1)*np.cos(np.pi*x2) - 0.3*np.pi**2*np.cos(np.pi*x1)) + (-0.3*np.pi*np.sin(np.pi*x1)*np.cos(np.pi*x2) - 0.3*np.pi*np.sin(np.pi*x1))*(0.1*np.sin(np.pi*x1) + 0.1*np.cos(np.pi*x1)*np.cos(np.pi*x2) - 0.2*np.cos(np.pi*x2) + 1.0)*(0.3*np.sin(3*np.pi*x1) + 0.3*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.3*np.cos(np.pi*x2) + 2.0) + (-0.3*np.pi*np.sin(np.pi*x1)*np.cos(np.pi*x2) + 0.9*np.pi*np.cos(3*np.pi*x1))*(0.1*np.sin(np.pi*x1) + 0.1*np.cos(np.pi*x1)*np.cos(np.pi*x2) - 0.2*np.cos(np.pi*x2) + 1.0)*(0.3*np.sin(np.pi*x2) + 0.3*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.3*np.cos(np.pi*x1) + 2.0) + (-0.1*np.pi*np.sin(np.pi*x1)*np.cos(np.pi*x2) + 0.1*np.pi*np.cos(np.pi*x1))*(0.3*np.sin(3*np.pi*x1) + 0.3*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.3*np.cos(np.pi*x2) + 2.0)*(0.3*np.sin(np.pi*x2) + 0.3*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.3*np.cos(np.pi*x1) + 2.0) + 4.0*(-0.3*np.pi*np.sin(np.pi*x2)*np.cos(np.pi*x1) + 0.3*np.pi*np.cos(np.pi*x2))*(0.1*np.sin(np.pi*x1) + 0.1*np.cos(np.pi*x1)*np.cos(np.pi*x2) - 0.2*np.cos(np.pi*x2) + 1.0)*(0.15*np.sin(np.pi*x2) + 0.15*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.15*np.cos(np.pi*x1) + 1) + 4.0*(-0.1*np.pi*np.sin(np.pi*x2)*np.cos(np.pi*x1) + 0.2*np.pi*np.sin(np.pi*x2))*(0.15*np.sin(np.pi*x2) + 0.15*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.15*np.cos(np.pi*x1) + 1)**2 - 0.5*np.pi*np.sin(np.pi*x2)*np.cos(np.pi*x1) + 1.0*np.pi*np.cos(np.pi*x2)

		S_rhoE = -mu*(-0.3*np.pi*np.sin(np.pi*x1)*np.cos(np.pi*x2) - 0.3*np.pi*np.sin(np.pi*x1))*(-0.3*np.pi*np.sin(np.pi*x1)*np.cos(np.pi*x2) - 0.3*np.pi*np.sin(np.pi*x1) - 0.3*np.pi*np.sin(np.pi*x2)*np.cos(np.pi*x1) - 0.3*np.pi*np.sin(np.pi*x2)) - mu*(-0.3*np.pi*np.sin(np.pi*x1)*np.cos(np.pi*x2) + 0.9*np.pi*np.cos(3*np.pi*x1))*(-0.4*np.pi*np.sin(np.pi*x1)*np.cos(np.pi*x2) + 0.2*np.pi*np.sin(np.pi*x2)*np.cos(np.pi*x1) + 1.2*np.pi*np.cos(3*np.pi*x1) - 0.2*np.pi*np.cos(np.pi*x2)) - mu*(-0.3*np.pi*np.sin(np.pi*x2)*np.cos(np.pi*x1) - 0.3*np.pi*np.sin(np.pi*x2))*(-0.3*np.pi*np.sin(np.pi*x1)*np.cos(np.pi*x2) - 0.3*np.pi*np.sin(np.pi*x1) - 0.3*np.pi*np.sin(np.pi*x2)*np.cos(np.pi*x1) - 0.3*np.pi*np.sin(np.pi*x2)) - mu*(-0.3*np.pi*np.sin(np.pi*x2)*np.cos(np.pi*x1) + 0.3*np.pi*np.cos(np.pi*x2))*(0.2*np.pi*np.sin(np.pi*x1)*np.cos(np.pi*x2) - 0.4*np.pi*np.sin(np.pi*x2)*np.cos(np.pi*x1) - 0.6*np.pi*np.cos(3*np.pi*x1) + 0.4*np.pi*np.cos(np.pi*x2)) - mu*(-0.2*np.pi**2*np.sin(np.pi*x1)*np.sin(np.pi*x2) - 3.6*np.pi**2*np.sin(3*np.pi*x1) - 0.4*np.pi**2*np.cos(np.pi*x1)*np.cos(np.pi*x2))*(0.3*np.sin(3*np.pi*x1) + 0.3*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.3*np.cos(np.pi*x2) + 2.0) - mu*(-0.2*np.pi**2*np.sin(np.pi*x1)*np.sin(np.pi*x2) - 0.4*np.pi**2*np.sin(np.pi*x2) - 0.4*np.pi**2*np.cos(np.pi*x1)*np.cos(np.pi*x2))*(0.3*np.sin(np.pi*x2) + 0.3*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.3*np.cos(np.pi*x1) + 2.0) - mu*(0.3*np.pi**2*np.sin(np.pi*x1)*np.sin(np.pi*x2) - 0.3*np.pi**2*np.cos(np.pi*x1)*np.cos(np.pi*x2) - 0.3*np.pi**2*np.cos(np.pi*x1))*(0.3*np.sin(np.pi*x2) + 0.3*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.3*np.cos(np.pi*x1) + 2.0) - mu*(0.3*np.pi**2*np.sin(np.pi*x1)*np.sin(np.pi*x2) - 0.3*np.pi**2*np.cos(np.pi*x1)*np.cos(np.pi*x2) - 0.3*np.pi**2*np.cos(np.pi*x2))*(0.3*np.sin(3*np.pi*x1) + 0.3*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.3*np.cos(np.pi*x2) + 2.0) + (-0.3*np.pi*np.sin(np.pi*x1)*np.cos(np.pi*x2) + 0.9*np.pi*np.cos(3*np.pi*x1))*((4.0*(0.15*np.sin(3*np.pi*x1) + 0.15*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.15*np.cos(np.pi*x2) + 1)**2 + 4.0*(0.15*np.sin(np.pi*x2) + 0.15*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.15*np.cos(np.pi*x1) + 1)**2)*(0.05*np.sin(np.pi*x1) + 0.05*np.cos(np.pi*x1)*np.cos(np.pi*x2) - 0.1*np.cos(np.pi*x2) + 0.5) + 1.0*np.sin(np.pi*x2) + 0.5*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 1.0*np.cos(2*np.pi*x1) + 10.0 + (1.0*np.sin(np.pi*x2) + 0.5*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 1.0*np.cos(2*np.pi*x1) + 10.0)/(gamma - 1)) + (-0.3*np.pi*np.sin(np.pi*x2)*np.cos(np.pi*x1) + 0.3*np.pi*np.cos(np.pi*x2))*((4.0*(0.15*np.sin(3*np.pi*x1) + 0.15*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.15*np.cos(np.pi*x2) + 1)**2 + 4.0*(0.15*np.sin(np.pi*x2) + 0.15*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.15*np.cos(np.pi*x1) + 1)**2)*(0.05*np.sin(np.pi*x1) + 0.05*np.cos(np.pi*x1)*np.cos(np.pi*x2) - 0.1*np.cos(np.pi*x2) + 0.5) + 1.0*np.sin(np.pi*x2) + 0.5*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 1.0*np.cos(2*np.pi*x1) + 10.0 + (1.0*np.sin(np.pi*x2) + 0.5*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 1.0*np.cos(2*np.pi*x1) + 10.0)/(gamma - 1)) + (0.3*np.sin(3*np.pi*x1) + 0.3*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.3*np.cos(np.pi*x2) + 2.0)*((4.0*(-0.3*np.pi*np.sin(np.pi*x1)*np.cos(np.pi*x2) - 0.3*np.pi*np.sin(np.pi*x1))*(0.15*np.sin(np.pi*x2) + 0.15*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.15*np.cos(np.pi*x1) + 1) + 4.0*(-0.3*np.pi*np.sin(np.pi*x1)*np.cos(np.pi*x2) + 0.9*np.pi*np.cos(3*np.pi*x1))*(0.15*np.sin(3*np.pi*x1) + 0.15*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.15*np.cos(np.pi*x2) + 1))*(0.05*np.sin(np.pi*x1) + 0.05*np.cos(np.pi*x1)*np.cos(np.pi*x2) - 0.1*np.cos(np.pi*x2) + 0.5) + (-0.05*np.pi*np.sin(np.pi*x1)*np.cos(np.pi*x2) + 0.05*np.pi*np.cos(np.pi*x1))*(4.0*(0.15*np.sin(3*np.pi*x1) + 0.15*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.15*np.cos(np.pi*x2) + 1)**2 + 4.0*(0.15*np.sin(np.pi*x2) + 0.15*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.15*np.cos(np.pi*x1) + 1)**2) - 0.5*np.pi*np.sin(np.pi*x1)*np.cos(np.pi*x2) - 2.0*np.pi*np.sin(2*np.pi*x1) + (-0.5*np.pi*np.sin(np.pi*x1)*np.cos(np.pi*x2) - 2.0*np.pi*np.sin(2*np.pi*x1))/(gamma - 1)) + (0.3*np.sin(np.pi*x2) + 0.3*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.3*np.cos(np.pi*x1) + 2.0)*((4.0*(-0.3*np.pi*np.sin(np.pi*x2)*np.cos(np.pi*x1) - 0.3*np.pi*np.sin(np.pi*x2))*(0.15*np.sin(3*np.pi*x1) + 0.15*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.15*np.cos(np.pi*x2) + 1) + 4.0*(-0.3*np.pi*np.sin(np.pi*x2)*np.cos(np.pi*x1) + 0.3*np.pi*np.cos(np.pi*x2))*(0.15*np.sin(np.pi*x2) + 0.15*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.15*np.cos(np.pi*x1) + 1))*(0.05*np.sin(np.pi*x1) + 0.05*np.cos(np.pi*x1)*np.cos(np.pi*x2) - 0.1*np.cos(np.pi*x2) + 0.5) + (-0.05*np.pi*np.sin(np.pi*x2)*np.cos(np.pi*x1) + 0.1*np.pi*np.sin(np.pi*x2))*(4.0*(0.15*np.sin(3*np.pi*x1) + 0.15*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.15*np.cos(np.pi*x2) + 1)**2 + 4.0*(0.15*np.sin(np.pi*x2) + 0.15*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.15*np.cos(np.pi*x1) + 1)**2) - 0.5*np.pi*np.sin(np.pi*x2)*np.cos(np.pi*x1) + 1.0*np.pi*np.cos(np.pi*x2) + (-0.5*np.pi*np.sin(np.pi*x2)*np.cos(np.pi*x1) + 1.0*np.pi*np.cos(np.pi*x2))/(gamma - 1)) - np.pi**2*kappa*(-0.2*(0.5*np.sin(np.pi*x1)*np.cos(np.pi*x2) + 2.0*np.sin(2*np.pi*x1))*(np.sin(np.pi*x1)*np.cos(np.pi*x2) - np.cos(np.pi*x1))/(0.1*np.sin(np.pi*x1) + 0.1*np.cos(np.pi*x1)*np.cos(np.pi*x2) - 0.2*np.cos(np.pi*x2) + 1.0) + (0.02*(np.sin(np.pi*x1)*np.cos(np.pi*x2) - np.cos(np.pi*x1))**2/(0.1*np.sin(np.pi*x1) + 0.1*np.cos(np.pi*x1)*np.cos(np.pi*x2) - 0.2*np.cos(np.pi*x2) + 1.0) + 0.1*np.sin(np.pi*x1) + 0.1*np.cos(np.pi*x1)*np.cos(np.pi*x2))*(1.0*np.sin(np.pi*x2) + 0.5*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 1.0*np.cos(2*np.pi*x1) + 10.0)/(0.1*np.sin(np.pi*x1) + 0.1*np.cos(np.pi*x1)*np.cos(np.pi*x2) - 0.2*np.cos(np.pi*x2) + 1.0) - 0.5*np.cos(np.pi*x1)*np.cos(np.pi*x2) - 4.0*np.cos(2*np.pi*x1))/(R*(0.1*np.sin(np.pi*x1) + 0.1*np.cos(np.pi*x1)*np.cos(np.pi*x2) - 0.2*np.cos(np.pi*x2) + 1.0)) - np.pi**2*kappa*(-2*(0.5*np.sin(np.pi*x2)*np.cos(np.pi*x1) - 1.0*np.cos(np.pi*x2))*(0.1*np.cos(np.pi*x1) - 0.2)*np.sin(np.pi*x2)/(0.1*np.sin(np.pi*x1) + 0.1*np.cos(np.pi*x1)*np.cos(np.pi*x2) - 0.2*np.cos(np.pi*x2) + 1.0) + ((0.2*np.cos(np.pi*x1) - 0.4)*np.sin(np.pi*x2)**2/(0.1*np.sin(np.pi*x1) + 0.1*np.cos(np.pi*x1)*np.cos(np.pi*x2) - 0.2*np.cos(np.pi*x2) + 1.0) + np.cos(np.pi*x2))*(0.1*np.cos(np.pi*x1) - 0.2)*(1.0*np.sin(np.pi*x2) + 0.5*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 1.0*np.cos(2*np.pi*x1) + 10.0)/(0.1*np.sin(np.pi*x1) + 0.1*np.cos(np.pi*x1)*np.cos(np.pi*x2) - 0.2*np.cos(np.pi*x2) + 1.0) - 1.0*np.sin(np.pi*x2) - 0.5*np.cos(np.pi*x1)*np.cos(np.pi*x2))/(R*(0.1*np.sin(np.pi*x1) + 0.1*np.cos(np.pi*x1)*np.cos(np.pi*x2) - 0.2*np.cos(np.pi*x2) + 1.0))

		
		return S_rho, S_rhou, S_rhov, S_rhoE

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
------------------------
Numerical flux functions
------------------------
These classes inherit from the ConvNumFluxBase or DiffNumFluxBase class. 
See ConvNumFluxBase/DiffNumFluxBase for detailed comments of attributes 
and methods. Information specific to the corresponding child classes can 
be found below. These classes should correspond to the ConvNumFluxType 
or DiffNumFluxType enum members above.
'''