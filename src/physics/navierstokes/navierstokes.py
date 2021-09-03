# ------------------------------------------------------------------------ #
#
#       File : src/physics/navierstokes/navierstokes.py
#
#       Contains class definitions for 1D and 2D Navier-Stokes equations.
#
# ------------------------------------------------------------------------ #
from enum import Enum
import numpy as np

import errors
import general

import physics.base.base as base
import physics.base.functions as base_fcns
from physics.base.functions import BCType as base_BC_type
from physics.base.functions import ConvNumFluxType as base_conv_num_flux_type
from physics.base.functions import FcnType as base_fcn_type

import physics.euler.euler as euler
import physics.euler.functions as euler_fcns
from physics.euler.functions import BCType as euler_BC_type
from physics.euler.functions import ConvNumFluxType as \
		euler_conv_num_flux_type
from physics.euler.functions import FcnType as euler_fcn_type
from physics.euler.functions import SourceType as euler_source_type

import physics.navierstokes.functions as navierstokes_fcns
from physics.navierstokes.functions import DiffNumFluxType as \
		navierstokes_diff_num_flux_type
from physics.navierstokes.functions import FcnType as navierstokes_fcn_type
from physics.navierstokes.functions import SourceType as \
		navierstokes_source_type

class NavierStokes(euler.Euler):
	'''
	This class corresponds to the compressible Navier-Stokes equations. 
	It inherits attributes and methods from the Euler class. See Euler
	for detailed comments of attributes and methods. This class should 
	not be instantiated directly. Instead, the 1D and 2D variants, which
	inherit from this class (see below), should be instantiated.

	Additional methods and attributes are commented below.

	Attributes:
	-----------
	R: float
		mass-specific gas constant
	gamma: float
		specific heat ratio
	'''
	PHYSICS_TYPE = general.PhysicsType.NavierStokes

	def __init__(self, mesh):
		super().__init__(mesh)
		self.R = 0.
		self.gamma = 0.

	def set_maps(self):
		super().set_maps()

		# self.BC_map.update({
		# 	base_BC_type.StateAll : base_fcns.StateAll,
		# 	base_BC_type.Extrapolate : base_fcns.Extrapolate,
		# 	euler_BC_type.SlipWall : euler_fcns.SlipWall,
		# 	euler_BC_type.PressureOutlet : euler_fcns.PressureOutlet,
		# })

	def set_physical_params(self, GasConstant=287., SpecificHeatRatio=1.4, 
			PrandtlNumber=0.7, Viscosity=1.0, s=1.0, T0=1.0, beta=1.5):
		'''
		This method sets physical parameters.

		Inputs:
		-------
			GasConstant: mass-specific gas constant
			SpecificHeatRatio: ratio of specific heats

		Outputs:
		--------
			self: physical parameters set
		'''
		self.R = GasConstant
		self.gamma = SpecificHeatRatio
		self.Pr = PrandtlNumber
		self.mu0 = Viscosity
		self.s = s
		self.T0 = T0
		self.beta = beta

	class AdditionalVariables(Enum):
	    Pressure = "p"
	    Temperature = "T"
	    Viscosity = "\\mu"
	    ThermalConductivity = "\\kappa"
	    Entropy = "s"
	    InternalEnergy = "\\rho e"
	    TotalEnthalpy = "H"
	    SoundSpeed = "c"
	    MaxWaveSpeed = "\\lambda"
	    Velocity = "|u|"

	def compute_additional_variable(self, var_name, Uq, flag_non_physical):
		''' Extract state variables '''
		srho = self.get_state_slice("Density")
		srhoE = self.get_state_slice("Energy")
		smom = self.get_momentum_slice()
		rho = Uq[:, :, srho]
		rhoE = Uq[:, :, srhoE]
		mom = Uq[:, :, smom]

		''' Unpack '''
		gamma = self.gamma
		R = self.R
		Pr = self.Pr

		''' Flag non-physical state '''
		if flag_non_physical:
			if np.any(rho < 0.):
				raise errors.NotPhysicalError

		''' Nested functions for common quantities '''
		def get_pressure():
			varq = (gamma - 1.)*(rhoE - 0.5*np.sum(mom*mom, axis=2,
					keepdims=True)/rho)
			if flag_non_physical:
				if np.any(varq < 0.):
					raise errors.NotPhysicalError
			return varq

		def get_temperature():
			return get_pressure()/(rho*R)

		# def get_viscosity():
		# 	'''
		# 	Compute viscosity from Sutherland's law (needs to get more
		# 	general in the future)
		# 	'''
		# 	# These current constants are for the manufactured solution
		# 	# in Dumbser's 2010 paper.
		# 	T = get_temperature()
		# 	mu0 = 0.1; s = 1.; T0 = 1.; beta = 1.5;
		# 	return mu0 * (T / T0)**beta * ((T0 + s) / (T + s))

		# def get_thermalconductivity():
		# 	'''
		# 	Compute thermal conductivity
		# 	'''
		# 	mu = get_viscosity()
		# 	cv = 1./(gamma - 1) * R
		# 	return mu * cv * gamma / Pr

		''' Compute '''
		vname = self.AdditionalVariables[var_name].name

		if vname is self.AdditionalVariables["Pressure"].name:
			varq = get_pressure()
		elif vname is self.AdditionalVariables["Temperature"].name:
			varq = get_temperature()
		elif vname is self.AdditionalVariables["Entropy"].name:
			varq = np.log(get_pressure()/rho**gamma)
		elif vname is self.AdditionalVariables["InternalEnergy"].name:
			varq = rhoE - 0.5*np.sum(mom*mom, axis=2, keepdims=True)/rho
		elif vname is self.AdditionalVariables["TotalEnthalpy"].name:
			varq = (rhoE + get_pressure())/rho
		elif vname is self.AdditionalVariables["SoundSpeed"].name:
			varq = np.sqrt(gamma*get_pressure()/rho)
		elif vname is self.AdditionalVariables["MaxWaveSpeed"].name:
			# |u| + c
			varq = np.linalg.norm(mom, axis=2, keepdims=True)/rho + np.sqrt(
					gamma*get_pressure()/rho)
		elif vname is self.AdditionalVariables["Velocity"].name:
			varq = np.linalg.norm(mom, axis=2, keepdims=True)/rho 
		else:
			raise NotImplementedError

		return varq


class NavierStokes1D(NavierStokes, euler.Euler1D):
	'''
	This class corresponds to 1D Navier Stokes equations.
	It inherits attributes and methods from the NavierStokes class.
	See NavierStokes for detailed comments of attributes and methods.

	Additional methods and attributes are commented below.
	'''
	NUM_STATE_VARS = 3
	NDIMS = 1

	def set_maps(self):
		super().set_maps()

		# d = {
		# 	navierstokes_fcn_type.ManufacturedSolution : 
		# 			navierstokes_fcns.ManufacturedSolution,
		# }

		# self.IC_fcn_map.update(d)
		# self.exact_fcn_map.update(d)
		# self.BC_fcn_map.update(d)

		# self.source_map.update({
		# 	navierstokes_source_type.StiffFriction : euler_fcns.StiffFriction,
		# })

		# Add diffusive flux map here...
		# self.diff_num_flux_map.update({
			# })


class NavierStokes2D(NavierStokes, euler.Euler2D):
	'''
	This class corresponds to 2D Navier-Stokes equations. It 
	inherits attributes and methods from the Navier-Stokes class.
	See Navier-Stokes for detailed comments of attributes and methods.

	Additional methods and attributes are commented below.
	'''
	NUM_STATE_VARS = 4
	NDIMS = 2

	def __init__(self, mesh):
		super().__init__(mesh)

	def set_maps(self):
		super().set_maps()

		d = {
			navierstokes_fcn_type.ManufacturedSolutionPeriodic : 
					navierstokes_fcns.ManufacturedSolutionPeriodic,
			navierstokes_fcn_type.TaylorGreenVortexNS : 
					navierstokes_fcns.TaylorGreenVortexNS,
			navierstokes_fcn_type.ManufacturedSolution : 
					navierstokes_fcns.ManufacturedSolution,
		}

		self.IC_fcn_map.update(d)
		self.exact_fcn_map.update(d)
		self.BC_fcn_map.update(d)

		self.source_map.update({
			navierstokes_source_type.ManufacturedSourcePeriodic : 
					navierstokes_fcns.ManufacturedSourcePeriodic,
			navierstokes_source_type.ManufacturedSource : 
					navierstokes_fcns.ManufacturedSource,
		})

		# self.conv_num_flux_map.update({
			# base_conv_num_flux_type.LaxFriedrichs :
			# 	euler_fcns.LaxFriedrichs2D,
			# euler_conv_num_flux_type.Roe : euler_fcns.Roe2D,		
		# })

		self.diff_num_flux_map.update({
			navierstokes_diff_num_flux_type.SIP : 
				navierstokes_fcns.SIP,
			})

	def get_diff_flux_interior(self, Uq, gUq):
			# Get indices/slices of state variables
			irho, irhou, irhov, irhoE = self.get_state_indices()
			smom = self.get_momentum_slice()

			# Unpack state coefficients
			rho  = Uq[:, :, irho]  # [n, nq]
			rhou = Uq[:, :, irhou] # [n, nq]
			rhov = Uq[:, :, irhov] # [n, nq]
			rhoE = Uq[:, :, irhoE] # [n, nq]
			mom  = Uq[:, :, smom]  # [n, nq, ndims]

			# Calculate transport
			mu, kappa = self.get_transport(self, Uq, 
				flag_non_physical=False)
			mu = mu.reshape(rho.shape)
			kappa = kappa.reshape(rho.shape)
			nu = mu / rho

			gamma = self.gamma
			R = self.R

			# Set constants for stress tensor
			C1 = 2. / 3.
			C2 = (gamma - 1.) / (R * rho)
			
			# Separate x and y gradients
			gUx = gUq[:, :, :, 0] # [ne, nq, ns]
			gUy = gUq[:, :, :, 1] # [ne, nq, ns]

			# Get velocity in each dimension
			u = rhou / rho
			v = rhov / rho
			
			# Get E
			E = rhoE / rho
			
			# Get squared velocities
			u2 = u**2
			v2 = v**2

			# Store dTdU
			dTdU = np.zeros_like(gUx)
			dTdU[:, :, 0] = C2 * (-E + u2 + v2)
			dTdU[:, :, 1] = C2 * -u
			dTdU[:, :, 2] = C2 * -v
			dTdU[:, :, 3] = C2

			# Get the stress tensor (use product rules to write in 
			# terms of the conservative gradients)
			rhodiv = ((gUx[:, :, 1] - u * gUx[:, :, 0]) + \
				(gUy[:, :, 2] - v * gUy[:, :, 0]))
			tauxx = nu * (2. * (gUx[:, :, 1] - u * gUx[:, :, 0]) - \
				C1 * rhodiv)
			tauxy = nu * ((gUy[:, :, 1] - u * gUy[:, :, 0]) + \
				(gUx[:, :, 2] - v * gUx[:, :, 0]))
			tauyy = nu * (2. * (gUy[:, :, 2] - v * gUy[:, :, 0]) - \
				C1 * rhodiv)

			# Assemble flux matrix
			F = np.empty(Uq.shape + (self.NDIMS,)) # [n, nq, ns, ndims]
			F[:,:,irho,  :] = 0.		   # x,y-flux of rho (zero both dir)

			# x-direction
			F[:,:,irhou, 0] = tauxx 	# x-flux of x-momentum
			F[:,:,irhov, 0] = tauxy     # x-flux of y-momentum
			F[:,:,irhoE, 0] = u * tauxx + v * tauxy + \
				kappa * np.einsum('ijk, ijk -> ij', dTdU, gUx)

			# y-direction
			F[:,:,irhou, 1] = tauxy        # y-flux of x-momentum
			F[:,:,irhov, 1] = tauyy 	   # y-flux of y-momentum
			F[:,:,irhoE, 1] = u * tauxy + v * tauyy + \
				kappa * np.einsum('ijk, ijk -> ij', dTdU, gUy)

			return F