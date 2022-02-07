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
# ------------------------------------------------------------------------ #

# ------------------------------------------------------------------------ #
#
#       File : src/physics/euler/euler.py
#
#       Contains class definitions for 1D and 2D Euler equations.
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

import physics.euler.functions as euler_fcns
from physics.euler.functions import BCType as euler_BC_type
from physics.euler.functions import ConvNumFluxType as \
		euler_conv_num_flux_type
from physics.euler.functions import FcnType as euler_fcn_type
from physics.euler.functions import SourceType as euler_source_type


class Wildfire(base.PhysicsBase):
	'''
	This class corresponds to the compressible Euler equations for a
	calorically perfect gas. It inherits attributes and methods from the
	PhysicsBase class. See PhysicsBase for detailed comments of attributes
	and methods. This class should not be instantiated directly. Instead,
	the 1D and 2D variants, which inherit from this class (see below),
	should be instantiated.

	Additional methods and attributes are commented below.

	Attributes:
	-----------
	Nwood: float
		molar coefficient of wood 
	Cpwood: float
		specific heat of wood [J/kgK]
	'''
	# PHYSICS_TYPE = general.PhysicsType.Euler
	PHYSICS_TYPE = general.PhysicsType.Wildfire

	def __init__(self):
		super().__init__()
		self.Nwood = 0.
		self.Cpwood = 0.

	def set_maps(self):
		super().set_maps()

		self.BC_map.update({
			base_BC_type.StateAll : base_fcns.StateAll,
			base_BC_type.Extrapolate : base_fcns.Extrapolate,
			euler_BC_type.SlipWall : euler_fcns.SlipWall,
			euler_BC_type.PressureOutlet : euler_fcns.PressureOutlet,
		})

	def set_physical_params(self, WoodMolarCoefficient=0.4552., SpecificHeat=1760):
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
		self.Nwood = WoodMolarCoefficient
		self.Cpwood = SpecificHeat

	# class AdditionalVariables(Enum):
	# 	Pressure = "p"
	# 	Temperature = "T"
	# 	Entropy = "s"
	# 	InternalEnergy = "\\rho e"
	# 	TotalEnthalpy = "H"
	# 	SoundSpeed = "c"
	# 	MaxWaveSpeed = "\\lambda"
	# 	Velocity = "|u|"
	# 	XVelocity = "u"
	# 	YVelocity = "v"

	    # Wood Density = "\\rho wood"
		# Water Density = "\\rho water"
		# Temperature = "\\T s"

	# def compute_additional_variable(self, var_name, Uq, flag_non_physical):
	# 	''' Extract state variables ''' 
	# 	srhowood = self.get_state_slice("Wood Density")
	# 	srhowater = self.get_state_slice("Water Density")
	# 	sTs = self.get_momentum_slice()
	# 	rhowood = Uq[:, :, srhowood]
	# 	rhowater = Uq[:, :, srhowater]
	# 	Ts = Uq[:, :, sTs]

	# 	''' Unpack '''
	# 	Cpwood = self.Cpwood
	# 	Nwood = self.Nwood

	# 	''' Flag non-physical state '''
	# 	if flag_non_physical:
	# 		if np.any(rhowood < 0.):
	# 			raise errors.NotPhysicalError

	# 	''' Nested functions for common quantities '''
	# 	def get_pressure():
	# 		varq = (gamma - 1.)*(rhoE - 0.5*np.sum(mom*mom, axis=2,
	# 				keepdims=True)/rho)
	# 		if flag_non_physical:
	# 			if np.any(varq < 0.):
	# 				raise errors.NotPhysicalError
	# 		return varq
	# 	def get_temperature():
	# 		return get_pressure()/(rho*R)

	# 	''' Compute '''
	# 	vname = self.AdditionalVariables[var_name].name

	# 	if vname is self.AdditionalVariables["Pressure"].name:
	# 		varq = get_pressure()
	# 	elif vname is self.AdditionalVariables["Temperature"].name:
	# 		varq = get_temperature()
	# 	elif vname is self.AdditionalVariables["Entropy"].name:
	# 		# varq = R*(gamma/(gamma-1.)*np.log(getT()) - np.log(getP()))
	# 		# Alternate way
	# 		varq = np.log(get_pressure()/rho**gamma)
	# 	elif vname is self.AdditionalVariables["InternalEnergy"].name:
	# 		varq = rhoE - 0.5*np.sum(mom*mom, axis=2, keepdims=True)/rho
	# 	elif vname is self.AdditionalVariables["TotalEnthalpy"].name:
	# 		varq = (rhoE + get_pressure())/rho
	# 	elif vname is self.AdditionalVariables["SoundSpeed"].name:
	# 		varq = np.sqrt(gamma*get_pressure()/rho)
	# 	elif vname is self.AdditionalVariables["MaxWaveSpeed"].name:
	# 		# |u| + c
	# 		varq = np.linalg.norm(mom, axis=2, keepdims=True)/rho + np.sqrt(
	# 				gamma*get_pressure()/rho)
	# 	elif vname is self.AdditionalVariables["Velocity"].name:
	# 		varq = np.linalg.norm(mom, axis=2, keepdims=True)/rho 
	# 	elif vname is self.AdditionalVariables["XVelocity"].name:
	# 		varq = mom[:, :, [0]]/rho
	# 	elif vname is self.AdditionalVariables["YVelocity"].name:
	# 		varq = mom[:, :, [1]]/rho
	# 	else:
	# 		raise NotImplementedError

	# 	return varq

	# def compute_pressure_gradient(self, Uq, grad_Uq):
	# 	'''
	# 	Compute the gradient of pressure with respect to physical space. This is
	# 	needed for pressure-based shock sensors.

	# 	Inputs:
	# 	-------
	# 		Uq: solution in each element evaluated at quadrature points
	# 		[ne, nq, ns]
	# 		grad_Uq: gradient of solution in each element evaluted at quadrature
	# 			points [ne, nq, ns, ndims]

	# 	Outputs:
	# 	--------
	# 		array: gradient of pressure with respected to physical space
	# 			[ne, nq, ndims]
	# 	'''
	# 	srho = self.get_state_slice("Density")
	# 	srhoE = self.get_state_slice("Energy")
	# 	smom = self.get_momentum_slice()
	# 	rho = Uq[:, :, srho]
	# 	rhoE = Uq[:, :, srhoE]
	# 	mom = Uq[:, :, smom]
	# 	gamma = self.gamma

	# 	# Compute dp/dU
	# 	dpdU = np.empty_like(Uq)
	# 	dpdU[:, :, srho] = (.5 * (gamma - 1) * np.sum(mom**2, axis = 2,
	# 		keepdims=True) / rho)
	# 	dpdU[:, :, smom] = (1 - gamma) * mom / rho
	# 	dpdU[:, :, srhoE] = gamma - 1

	# 	# Multiply with dU/dx
	# 	return np.einsum('ijk, ijkl -> ijl', dpdU, grad_Uq)


class Wildfire2D(Wildfire):
	'''
	This class corresponds to 1D Euler equations for a calorically
	perfect gas. It inherits attributes and methods from the Euler class.
	See Euler for detailed comments of attributes and methods.

	Additional methods and attributes are commented below.
	'''
	NUM_STATE_VARS = 3
	NDIMS = 2

	def set_maps(self):
		super().set_maps()

		d = {
			euler_fcn_type.SmoothIsentropicFlow :
					euler_fcns.SmoothIsentropicFlow,
			euler_fcn_type.MovingShock : euler_fcns.MovingShock,
			euler_fcn_type.DensityWave : euler_fcns.DensityWave,
			euler_fcn_type.RiemannProblem : euler_fcns.RiemannProblem,
			euler_fcn_type.ShuOsherProblem :
					euler_fcns.ShuOsherProblem,
		}

		self.IC_fcn_map.update(d)
		self.exact_fcn_map.update(d)
		self.BC_fcn_map.update(d)

		self.source_map.update({
			euler_source_type.StiffFriction : euler_fcns.StiffFriction,
		})

		self.conv_num_flux_map.update({
			base_conv_num_flux_type.LaxFriedrichs :
					euler_fcns.LaxFriedrichs1D,
			euler_conv_num_flux_type.Roe : euler_fcns.Roe1D,
		})

	class StateVariables(Enum):
		Wood Density = "\\rho wood"
		Water Density = "\\rho water"
		Temperature = "\\T s"

	def get_state_indices(self):
		irhowood = self.get_state_index("Wood Density")
		irhowater = self.get_state_index("Water Density")
		iTs = self.get_state_index("Temperature")

		return irhowood, irhowater, iTs

	def get_state_slices(self):
		srhowood = self.get_state_slice("Wood Density")
		srhowater = self.get_state_slice("Water Density")
		sTs = self.get_state_slice("Temperature")

		return srhowood, srhowater, sTs

	# def get_momentum_slice(self):
	# 	irhou = self.get_state_index("XMomentum")
	# 	smom = slice(irhou, irhou+1)

	# 	return smom

	def get_conv_flux_interior(self, Uq):
		# # Get indices of state variables
		# irho, irhou, irhoE = self.get_state_indices()

		# rho  = Uq[:, :, irho]  # [n, nq]
		# rhou = Uq[:, :, irhou] # [n, nq]
		# rhoE = Uq[:, :, irhoE] # [n, nq]

		# # Get velocity
		# u = rhou / rho
		# # Get squared velocitiy
		# u2 = u**2

		# # Calculate pressure using the Ideal Gas Law
		# p = (self.gamma - 1.)*(rhoE - 0.5 * rho * u2) # [n, nq]
		# # Get total enthalpy
		# H = rhoE + p

		# Assemble flux matrix
		F = np.zeros(Uq.shape + (self.NDIMS,)) # [n, nq, ns, ndims]
		# F[:, :, irho, 0] = rhou         # Flux of mass
		# F[:, :, irhou, 0] = rho * u2 + p # Flux of momentum
		# F[:, :, irhoE, 0] = H * u        # Flux of energy

		# return F, (u2, rho, p)
		return F

	# def get_conv_eigenvectors(self, U_bar):
	# 	'''
	# 	This function defines the convective eigenvectors for the
	# 	1D euler equations. This is used with the WENO limiter to
	# 	transform the system of equations from physical space to
	# 	characteristic space.

	# 	Inputs:
	# 	-------
	# 		U_bar: Average state [ne, 1, ns]

	# 	Outputs:
	# 	--------
	# 		right_eigen: Right eigenvector matrix [ne, 1, ns, ns]
	# 		left_eigen: Left eigenvector matrix [ne, 1, ns, ns]
	# 	'''

	# 	# Unpack
	# 	ne = U_bar.shape[0]

	# 	ns = self.NUM_STATE_VARS

	# 	irho, irhou, irhoE = self.get_state_indices()

	# 	rho = U_bar[:, :, irho]
	# 	rhou = U_bar[:, :, irhou]
	# 	rhoE = U_bar[:, :, irhoE]

	# 	# Get velocity
	# 	u = rhou / rho
	# 	# Get squared velocity
	# 	u2 = u**2
	# 	# Calculate pressure using the Ideal Gasd Law
	# 	p = (self.gamma - 1.)*(rhoE - 0.5 * rho * u2) # [n, nq]
	# 	# Get total specific enthalpy
	# 	H = rhoE/rho + p/rho

	# 	# Get sound speed
	# 	a = np.sqrt(self.gamma * p / rho)

	# 	gm1oa2 = (self.gamma - 1.) / (a * a)

	# 	# Allocate the right and left eigenvectors
	# 	right_eigen = np.zeros([ne, 1, ns, ns])
	# 	left_eigen = np.zeros([ne, 1, ns, ns])

	# 	# # Calculate the right and left eigenvectors
	# 	right_eigen[:, :, irho, irho]  = 1.
	# 	right_eigen[:, :, irho, irhou] = 1.
	# 	right_eigen[:, :, irho, irhoE] = 1.

	# 	right_eigen[:, :, irhou, irho]  = u - a
	# 	right_eigen[:, :, irhou, irhou] = u + a
	# 	right_eigen[:, :, irhou, irhoE] = u

	# 	right_eigen[:, :, irhoE, irho]  = H - u*a
	# 	right_eigen[:, :, irhoE, irhou] = H + u*a
	# 	right_eigen[:, :, irhoE, irhoE] = 0.5 * u2

	# 	left_eigen[:, :, irho, irho]  = 0.5 * (0.5*gm1oa2 * u2 + u/a)
	# 	left_eigen[:, :, irho, irhou] = -0.5 * (gm1oa2 * u + 1./a)
	# 	left_eigen[:, :, irho, irhoE] = 0.5 * gm1oa2

	# 	left_eigen[:, :, irhou, irho]  = 0.5 * (0.5*gm1oa2 * u2 - u/a)
	# 	left_eigen[:, :, irhou, irhou] = -0.5 * (gm1oa2 * u - 1./a)
	# 	left_eigen[:, :, irhou, irhoE] = 0.5 * gm1oa2

	# 	left_eigen[:, :, irhoE, irho]  = 1. - 0.5 * gm1oa2 * u2
	# 	left_eigen[:, :, irhoE, irhou] = gm1oa2 * u
	# 	left_eigen[:, :, irhoE, irhoE] = -1.*gm1oa2

	# 	# Can uncomment line below to test l dot r = kronecker delta
	# 	# test = np.einsum('elij,eljk->elik', left_eigen, right_eigen)

	# 	return right_eigen, left_eigen # [ne, 1, ns, ns]


# class Euler2D(Euler):
# 	'''
# 	This class corresponds to 2D Euler equations for a calorically
# 	perfect gas. It inherits attributes and methods from the Euler class.
# 	See Euler for detailed comments of attributes and methods.

# 	Additional methods and attributes are commented below.
# 	'''
# 	NUM_STATE_VARS = 4
# 	NDIMS = 2

# 	def __init__(self):
# 		super().__init__()

# 	def set_maps(self):
# 		super().set_maps()

# 		d = {
# 			euler_fcn_type.IsentropicVortex : euler_fcns.IsentropicVortex,
# 			euler_fcn_type.TaylorGreenVortex : euler_fcns.TaylorGreenVortex,
# 			euler_fcn_type.GravityRiemann : euler_fcns.GravityRiemann,
# 		}

# 		self.IC_fcn_map.update(d)
# 		self.exact_fcn_map.update(d)
# 		self.BC_fcn_map.update(d)

# 		self.source_map.update({
# 			euler_source_type.StiffFriction : euler_fcns.StiffFriction,
# 			euler_source_type.TaylorGreenSource :
# 					euler_fcns.TaylorGreenSource,
# 			euler_source_type.GravitySource : euler_fcns.GravitySource,
# 		})

# 		self.conv_num_flux_map.update({
# 			base_conv_num_flux_type.LaxFriedrichs :
# 				euler_fcns.LaxFriedrichs2D,
# 			euler_conv_num_flux_type.Roe : euler_fcns.Roe2D,
# 		})

# 	class StateVariables(Enum):
# 		Density = "\\rho"
# 		XMomentum = "\\rho u"
# 		YMomentum = "\\rho v"
# 		Energy = "\\rho E"

# 	def get_state_indices(self):
# 		irho = self.get_state_index("Density")
# 		irhou = self.get_state_index("XMomentum")
# 		irhov = self.get_state_index("YMomentum")
# 		irhoE = self.get_state_index("Energy")

# 		return irho, irhou, irhov, irhoE

# 	def get_momentum_slice(self):
# 		irhou = self.get_state_index("XMomentum")
# 		irhov = self.get_state_index("YMomentum")
# 		smom = slice(irhou, irhov + 1)

# 		return smom

# 	def get_conv_flux_interior(self, Uq):
# 		# Get indices/slices of state variables
# 		irho, irhou, irhov, irhoE = self.get_state_indices()
# 		smom = self.get_momentum_slice()

# 		rho  = Uq[:, :, irho]  # [n, nq]
# 		rhou = Uq[:, :, irhou] # [n, nq]
# 		rhov = Uq[:, :, irhov] # [n, nq]
# 		rhoE = Uq[:, :, irhoE] # [n, nq]
# 		mom  = Uq[:, :, smom]  # [n, nq, ndims]

# 		# Get velocity in each dimension
# 		u = rhou / rho
# 		v = rhov / rho
# 		# Get squared velocities
# 		u2 = u**2
# 		v2 = v**2

# 		# Calculate pressure using the Ideal Gas Law
# 		p = (self.gamma - 1.)*(rhoE - 0.5 * rho * (u2 + v2)) # [n, nq]
# 		# Get off-diagonal momentum
# 		rhouv = rho * u * v
# 		# Get total enthalpy
# 		H = rhoE + p

# 		# Assemble flux matrix
# 		F = np.empty(Uq.shape + (self.NDIMS,)) # [n, nq, ns, ndims]
# 		F[:,:,irho,  :] = mom          # Flux of mass in all directions
# 		F[:,:,irhou, 0] = rho * u2 + p # x-flux of x-momentum
# 		F[:,:,irhov, 0] = rhouv        # x-flux of y-momentum
# 		F[:,:,irhou, 1] = rhouv        # y-flux of x-momentum
# 		F[:,:,irhov, 1] = rho * v2 + p # y-flux of y-momentum
# 		F[:,:,irhoE, 0] = H * u        # x-flux of energy
# 		F[:,:,irhoE, 1] = H * v        # y-flux of energy

# 		return F, (u2, v2, rho, p)
