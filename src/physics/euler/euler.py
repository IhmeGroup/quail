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


class Euler(base.PhysicsBase):
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
	R: float
		mass-specific gas constant
	gamma: float
		specific heat ratio
	'''
	PHYSICS_TYPE = general.PhysicsType.Euler

	def __init__(self, mesh):
		super().__init__(mesh)
		self.R = 0.
		self.gamma = 0.

	def set_maps(self):
		super().set_maps()

		self.BC_map.update({
			base_BC_type.StateAll : base_fcns.StateAll,
			base_BC_type.Extrapolate : base_fcns.Extrapolate,
			euler_BC_type.SlipWall : euler_fcns.SlipWall,
			euler_BC_type.PressureOutlet : euler_fcns.PressureOutlet,
		})

	def set_physical_params(self, GasConstant=287., SpecificHeatRatio=1.4):
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

	class AdditionalVariables(Enum):
	    Pressure = "p"
	    Temperature = "T"
	    Entropy = "s"
	    InternalEnergy = "\\rho e"
	    TotalEnthalpy = "H"
	    SoundSpeed = "c"
	    MaxWaveSpeed = "\\lambda"
	    Velocity = "u"

	def get_conv_flux_interior(self, Uq):
		# Unpack
		dim = self.DIM
		irho = self.get_state_index("Density")
		irhoE = self.get_state_index("Energy")
		srho = self.get_state_slice("Density")
		srhoE = self.get_state_slice("Energy")
		smom = self.get_momentum_slice()

		eps = general.eps

		rho = Uq[:, srho]
		rho += eps # prevent rare division-by-zero errors
		rhoE = Uq[:, srhoE]
		mom = Uq[:, smom]

		p = self.compute_variable("Pressure", Uq)
		h = self.compute_variable("TotalEnthalpy", Uq)

		# Pressure term (general to 1D and 2D)
		pmat = np.zeros([Uq.shape[0], dim, dim])
		idx = np.full([dim, dim], False)
		np.fill_diagonal(idx, True)
		pmat[:, idx] = p

		# Put together
		F = np.empty(Uq.shape + (dim,))
		F[:, irho, :] = mom
		F[:, smom, :] = np.einsum('ij,ik->ijk',mom,mom)/np.expand_dims(
				rho, axis=2) + pmat
		F[:, irhoE, :] = mom*h

		rho -= eps

		return F

	def compute_additional_variable(self, var_name, Uq, flag_non_physical):
		''' Extract state variables '''
		srho = self.get_state_slice("Density")
		srhoE = self.get_state_slice("Energy")
		smom = self.get_momentum_slice()
		rho = Uq[:, srho]
		rhoE = Uq[:, srhoE]
		mom = Uq[:, smom]

		''' Unpack '''
		gamma = self.gamma
		R = self.R

		''' Flag non-physical state '''
		if flag_non_physical:
			if np.any(rho < 0.):
				raise errors.NotPhysicalError

		''' Nested functions for common quantities '''
		def get_pressure():
			varq = (gamma - 1.)*(rhoE - 0.5*np.sum(mom*mom, axis=1, 
					keepdims=True)/rho)
			if flag_non_physical:
				if np.any(varq < 0.):
					raise errors.NotPhysicalError
			return varq
		def get_temperature():
			return get_pressure()/(rho*R)

		''' Compute '''
		vname = self.AdditionalVariables[var_name].name

		if vname is self.AdditionalVariables["Pressure"].name:
			varq = get_pressure()
		elif vname is self.AdditionalVariables["Temperature"].name:
			varq = get_temperature()
		elif vname is self.AdditionalVariables["Entropy"].name:
			# varq = R*(gamma/(gamma-1.)*np.log(getT()) - np.log(getP()))
			# Alternate way
			varq = np.log(get_pressure()/rho**gamma)
		elif vname is self.AdditionalVariables["InternalEnergy"].name:
			varq = rhoE - 0.5*np.sum(mom*mom, axis=1, keepdims=True)/rho
		elif vname is self.AdditionalVariables["TotalEnthalpy"].name:
			varq = (rhoE + get_pressure())/rho
		elif vname is self.AdditionalVariables["SoundSpeed"].name:
			varq = np.sqrt(gamma*get_pressure()/rho)
		elif vname is self.AdditionalVariables["MaxWaveSpeed"].name:
			# |u| + c
			varq = np.linalg.norm(mom, axis=1, keepdims=True)/rho + np.sqrt(gamma*get_pressure()/rho)
		elif vname is self.AdditionalVariables["Speed"].name:
			varq = np.linalg.norm(mom, axis=1, keepdims=True)/rho
		else:
			raise NotImplementedError

		return varq


class Euler1D(Euler):
	'''
	This class corresponds to 1D Euler equations for a calorically
	perfect gas. It inherits attributes and methods from the Euler class. 
	See Euler for detailed comments of attributes and methods. 

	Additional methods and attributes are commented below.
	'''
	NUM_STATE_VARS = 3
	DIM = 1

	def set_maps(self):
		super().set_maps()

		d = {
			euler_fcn_type.SmoothIsentropicFlow : 
					euler_fcns.SmoothIsentropicFlow,
			euler_fcn_type.MovingShock : euler_fcns.MovingShock,
			euler_fcn_type.DensityWave : euler_fcns.DensityWave,
			euler_fcn_type.RiemannProblem : euler_fcns.RiemannProblem,
			euler_fcn_type.ExactRiemannSolution : 
					euler_fcns.ExactRiemannSolution,
		}

		self.IC_fcn_map.update(d)
		self.exact_fcn_map.update(d)
		self.BC_fcn_map.update(d)

		self.source_map.update({
			euler_source_type.StiffFriction : euler_fcns.StiffFriction,
		})

		self.conv_num_flux_map.update({
			euler_conv_num_flux_type.Roe : euler_fcns.Roe1D,
			euler_conv_num_flux_type.HLLC : euler_fcns.HLLC1D,
		})

	class StateVariables(Enum):
		Density = "\\rho"
		XMomentum = "\\rho u"
		Energy = "\\rho E"

	def get_state_indices(self):
		irho = self.get_state_index("Density")
		irhou = self.get_state_index("XMomentum")
		irhoE = self.get_state_index("Energy")

		return irho, irhou, irhoE

	def get_state_slices(self):
		srho = self.get_state_slice("Density")
		srhou = self.get_state_slice("XMomentum")
		srhoE = self.get_state_slice("Energy")

		return srho, srhou, srhoE

	def get_momentum_slice(self):
		irhou = self.get_state_index("XMomentum")
		smom = slice(irhou, irhou+1)

		return smom


class Euler2D(Euler):
	'''
	This class corresponds to 2D Euler equations for a calorically
	perfect gas. It inherits attributes and methods from the Euler class. 
	See Euler for detailed comments of attributes and methods. 

	Additional methods and attributes are commented below.
	'''
	NUM_STATE_VARS = 4
	DIM = 2

	def set_maps(self):
		super().set_maps()

		d = {
			euler_fcn_type.IsentropicVortex : euler_fcns.IsentropicVortex,
			euler_fcn_type.TaylorGreenVortex : euler_fcns.TaylorGreenVortex,
		}

		self.IC_fcn_map.update(d)
		self.exact_fcn_map.update(d)
		self.BC_fcn_map.update(d)

		self.source_map.update({
			euler_source_type.StiffFriction : euler_fcns.StiffFriction,
			euler_source_type.TaylorGreenSource : 
					euler_fcns.TaylorGreenSource,
		})

		self.conv_num_flux_map.update({
			euler_conv_num_flux_type.Roe : euler_fcns.Roe2D,
		})

	class StateVariables(Enum):
		Density = "\\rho"
		XMomentum = "\\rho u"
		YMomentum = "\\rho v"
		Energy = "\\rho E"

	def get_state_indices(self):
		irho = self.get_state_index("Density")
		irhou = self.get_state_index("XMomentum")
		irhov = self.get_state_index("YMomentum")
		irhoE = self.get_state_index("Energy")

		return irho, irhou, irhov, irhoE

	def get_momentum_slice(self):
		irhou = self.get_state_index("XMomentum")
		irhov = self.get_state_index("YMomentum")
		imom = slice(irhou, irhov + 1)

		return imom

	