import code
from enum import Enum
import numpy as np
from scipy.optimize import fsolve, root

import errors
import general

import physics.base.base as base
import physics.base.functions as base_fcns
from physics.base.functions import BCType as base_BC_type
from physics.base.functions import ConvNumFluxType as base_conv_num_flux_type
from physics.base.functions import FcnType as base_fcn_type

import physics.euler.functions as euler_fcns
from physics.euler.functions import BCType as euler_BC_type
from physics.euler.functions import ConvNumFluxType as euler_conv_num_flux_type
from physics.euler.functions import FcnType as euler_fcn_type
from physics.euler.functions import SourceType as euler_source_type


class Euler(base.PhysicsBase):

	PHYSICS_TYPE = general.PhysicsType.Euler

	def __init__(self, order, basis, mesh):
		'''
		Method: __init__
		--------------------------------------------------------------------------
		This method initializes the temperature table. The table uses a
		piecewise linear function for the constant pressure specific heat 
		coefficients. The coefficients are selected to retain the exact 
		enthalpies at the table points.
		'''
		super().__init__(order, basis, mesh)
		# Default parameters
		self.Params.update(
			GasConstant = 287., # specific gas constant
			SpecificHeatRatio = 1.4,
		)

	def set_maps(self):
		super().set_maps()

		self.BC_map.update({
			base_BC_type.StateAll : base_fcns.StateAll,
			base_BC_type.Extrapolate : base_fcns.Extrapolate,
			euler_BC_type.SlipWall : euler_fcns.SlipWall,
			euler_BC_type.PressureOutlet : euler_fcns.PressureOutlet,
		})

	def set_physical_params(self, GasConstant=287., SpecificHeatRatio=1.4):
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

	def ConvFluxInterior(self, Up):
		dim = self.dim
		# irho = 0; irhoE = dim + 1
		irho = self.GetStateIndex("Density")
		irhoE = self.GetStateIndex("Energy")
		# imom = self.GetMomentumSlice()
		srho = self.get_state_slice("Density")
		srhoE = self.get_state_slice("Energy")
		smom = self.GetMomentumSlice()

		eps = general.eps

		rho = Up[:, srho]
		rho += eps
		rhoE = Up[:, srhoE]
		mom = Up[:, smom]

		p = self.ComputeScalars("Pressure", Up)
		h = self.ComputeScalars("TotalEnthalpy", Up)

		pmat = np.zeros([Up.shape[0], dim, dim])
		idx = np.full([dim, dim], False)
		np.fill_diagonal(idx, True)
		pmat[:, idx] = p

		# if F is None:
		# 	F = np.empty(Up.shape+(dim,))

		F = np.empty(Up.shape + (dim,))
		F[:, irho, :] = mom
		F[:, smom, :] = np.einsum('ij,ik->ijk',mom,mom)/np.expand_dims(rho, axis=2) + pmat
		F[:, irhoE, :] = mom*h

		rho -= eps

		return F

	def AdditionalScalars(self, ScalarName, Up, flag_non_physical):
		''' Extract state variables '''
		srho = self.get_state_slice("Density")
		srhoE = self.get_state_slice("Energy")
		smom = self.GetMomentumSlice()
		rho = Up[:, srho]
		rhoE = Up[:, srhoE]
		mom = Up[:, smom]

		# scalar = np.zeros([Up.shape[0], 1])

		''' Common scalars '''
		gamma = self.gamma
		R = self.R
		# Pressure
		# P = (gamma - 1.)*(rhoE - 0.5*np.sum(mom*mom, axis=1)/rho)
		# # Temperature
		# T = P/(rho*R)

		if flag_non_physical:
			if np.any(rho < 0.):
				raise errors.NotPhysicalError

		# if np.any(P < 0.) or np.any(rho < 0.):
		# 	raise errors.NotPhysicalError
		def get_pressure():
			scalar = (gamma - 1.)*(rhoE - 0.5*np.sum(mom*mom, axis=1, keepdims=True)/rho) # just use for storage
			# if flag_non_physical:
			# 	if np.any(scalar < 0.):
			# 		raise errors.NotPhysicalError
			return scalar
		def get_temperature():
			return get_pressure()/(rho*R)


		''' Get final scalars '''
		sname = self.AdditionalVariables[ScalarName].name
		if sname is self.AdditionalVariables["Pressure"].name:
			scalar = get_pressure()
		elif sname is self.AdditionalVariables["Temperature"].name:
			# scalar = (gamma - 1.)*(rhoE - 0.5*np.sum(mom*mom, axis=1, keepdims=True)/rho)/(rho*R)
			scalar = get_temperature()
		elif sname is self.AdditionalVariables["Entropy"].name:
			# Pressure
			# P = (gamma - 1.)*(rhoE - 0.5*np.sum(mom*mom, axis=1, keepdims=True)/rho)
			# Temperature
			# T = getP()/(rho*R)

			# scalar = R*(gamma/(gamma-1.)*np.log(getT()) - np.log(getP()))
			scalar = np.log(get_pressure()/rho**gamma)
		elif sname is self.AdditionalVariables["InternalEnergy"].name:
			scalar = rhoE - 0.5*np.sum(mom*mom, axis=1, keepdims=True)/rho
		elif sname is self.AdditionalVariables["TotalEnthalpy"].name:
			scalar = (rhoE + get_pressure())/rho
		elif sname is self.AdditionalVariables["SoundSpeed"].name:
			# Pressure
			# P = (gamma - 1.)*(rhoE - 0.5*np.sum(mom*mom, axis=1, keepdims=True)/rho)
			scalar = np.sqrt(gamma*get_pressure()/rho)
		elif sname is self.AdditionalVariables["MaxWaveSpeed"].name:
			# Pressure
			# P = GetPressure()
			scalar = np.linalg.norm(mom, axis=1, keepdims=True)/rho + np.sqrt(gamma*get_pressure()/rho)
		elif sname is self.AdditionalVariables["Velocity"].name:
			scalar = np.linalg.norm(mom, axis=1, keepdims=True)/rho
		else:
			raise NotImplementedError

		return scalar


class Euler1D(Euler):

	NUM_STATE_VARS = 3
	dim = 1

	def __init__(self, order, basis, mesh):
		'''
		Method: __init__
		--------------------------------------------------------------------------
		This method initializes the temperature table. The table uses a
		piecewise linear function for the constant pressure specific heat 
		coefficients. The coefficients are selected to retain the exact 
		enthalpies at the table points.
		'''
		super().__init__(order, basis, mesh)

	def set_maps(self):
		super().set_maps()

		d = {
			euler_fcn_type.SmoothIsentropicFlow : euler_fcns.SmoothIsentropicFlow,
			euler_fcn_type.MovingShock : euler_fcns.MovingShock,
			euler_fcn_type.DensityWave : euler_fcns.DensityWave,
			euler_fcn_type.RiemannProblem : euler_fcns.RiemannProblem,
			euler_fcn_type.SmoothRiemannProblem : euler_fcns.SmoothRiemannProblem,
			euler_fcn_type.ExactRiemannSolution : euler_fcns.ExactRiemannSolution,
		}

		self.IC_fcn_map.update(d)
		self.exact_fcn_map.update(d)
		# self.exact_fcn_map.update({
		# 	euler_fcn_type.ExactRiemannSolution : euler_fcns.ExactRiemannSolution,
		# })
		self.BC_fcn_map.update(d)

		self.source_map.update({
			euler_source_type.StiffFriction : euler_fcns.StiffFriction,
		})

		self.conv_num_flux_map.update({
			euler_conv_num_flux_type.Roe : euler_fcns.Roe1D,
		})

	class StateVariables(Enum):
		__Order__ = 'Density XMomentum Energy' # only needed in 2.x
		# LaTeX format
		Density = "\\rho"
		XMomentum = "\\rho u"
		Energy = "\\rho E"

	def GetStateIndices(self):
		irho = self.GetStateIndex("Density")
		irhou = self.GetStateIndex("XMomentum")
		irhoE = self.GetStateIndex("Energy")

		return irho, irhou, irhoE

	def get_state_slices(self):
		srho = self.get_state_slice("Density")
		srhou = self.get_state_slice("XMomentum")
		srhoE = self.get_state_slice("Energy")

		return srho, srhou, srhoE

	def GetMomentumSlice(self):
		irhou = self.GetStateIndex("XMomentum")
		smom = slice(irhou, irhou+1)

		return smom


class Euler2D(Euler):

	NUM_STATE_VARS = 4
	dim = 2

	def __init__(self, order, basis, mesh):
		super().__init__(order, basis, mesh) 

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
			euler_source_type.TaylorGreenSource : euler_fcns.TaylorGreenSource,
		})

		self.conv_num_flux_map.update({
			euler_conv_num_flux_type.Roe : euler_fcns.Roe2D,
		})

	class StateVariables(Enum):
		__Order__ = 'Density XMomentum YMomentum Energy' # only needed in 2.x
		# LaTeX format
		Density = "\\rho"
		XMomentum = "\\rho u"
		YMomentum = "\\rho v"
		Energy = "\\rho E"

	def GetStateIndices(self):
		irho = self.GetStateIndex("Density")
		irhou = self.GetStateIndex("XMomentum")
		irhov = self.GetStateIndex("YMomentum")
		irhoE = self.GetStateIndex("Energy")

		return irho, irhou, irhov, irhoE

	def GetMomentumSlice(self):
		irhou = self.GetStateIndex("XMomentum")
		irhov = self.GetStateIndex("YMomentum")
		imom = slice(irhou, irhov+1)

		return imom

	