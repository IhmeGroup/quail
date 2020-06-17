import code
from enum import IntEnum, Enum
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

	def set_physical_params(self, GasConstant=287., SpecificHeatRatio = 1.4):
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

	def ConvFluxInterior(self, u, F=None):
		dim = self.dim
		irho = 0; irhoE = dim + 1
		imom = self.GetMomentumSlice()

		eps = general.eps

		rho = u[:,irho:irho+1]
		rho += eps
		rhoE = u[:,irhoE:irhoE+1]
		mom = u[:,imom]

		p = self.ComputeScalars("Pressure", u)
		h = self.ComputeScalars("TotalEnthalpy", u)

		pmat = np.zeros([u.shape[0], dim, dim])
		idx = np.full([dim,dim],False)
		np.fill_diagonal(idx,True)
		pmat[:,idx] = p

		if F is None:
			F = np.empty(u.shape+(dim,))

		F[:,irho,:] = mom
		F[:,imom,:] = np.einsum('ij,ik->ijk',mom,mom)/np.expand_dims(rho, axis=2) + pmat
		F[:,irhoE,:] = mom*h

		rho -= eps

		return F

	def AdditionalScalars(self, ScalarName, U, scalar, FlagNonPhysical):
		''' Extract state variables '''
		irho = self.GetStateIndex("Density")
		irhoE = self.GetStateIndex("Energy")
		imom = self.GetMomentumSlice()
		rho = U[:,irho:irho+1]
		rhoE = U[:,irhoE:irhoE+1]
		mom = U[:,imom]

		''' Common scalars '''
		gamma = self.gamma
		R = self.R
		# Pressure
		# P = (gamma - 1.)*(rhoE - 0.5*np.sum(mom*mom, axis=1)/rho)
		# # Temperature
		# T = P/(rho*R)

		if FlagNonPhysical:
			if np.any(rho < 0.):
				raise Errors.NotPhysicalError

		# if np.any(P < 0.) or np.any(rho < 0.):
		# 	raise Errors.NotPhysicalError
		def getP():
			scalar[:] = (gamma - 1.)*(rhoE - 0.5*np.sum(mom*mom, axis=1, keepdims=True)/rho) # just use for storage
			if FlagNonPhysical:
				if np.any(scalar < 0.):
					raise Errors.NotPhysicalError
			return scalar
		def getT():
			return getP()/(rho*R)


		''' Get final scalars '''
		sname = self.AdditionalVariables[ScalarName].name
		if sname is self.AdditionalVariables["Pressure"].name:
			scalar[:] = getP()
		elif sname is self.AdditionalVariables["Temperature"].name:
			# scalar = (gamma - 1.)*(rhoE - 0.5*np.sum(mom*mom, axis=1, keepdims=True)/rho)/(rho*R)
			scalar[:] = getT()
		elif sname is self.AdditionalVariables["Entropy"].name:
			# Pressure
			# P = (gamma - 1.)*(rhoE - 0.5*np.sum(mom*mom, axis=1, keepdims=True)/rho)
			# Temperature
			# T = getP()/(rho*R)

			# scalar = R*(gamma/(gamma-1.)*np.log(getT()) - np.log(getP()))
			scalar[:] = np.log(getP()/rho**gamma)
		elif sname is self.AdditionalVariables["InternalEnergy"].name:
			scalar[:] = rhoE - 0.5*np.sum(mom*mom, axis=1, keepdims=True)/rho
		elif sname is self.AdditionalVariables["TotalEnthalpy"].name:
			scalar[:] = (rhoE + getP())/rho
		elif sname is self.AdditionalVariables["SoundSpeed"].name:
			# Pressure
			# P = (gamma - 1.)*(rhoE - 0.5*np.sum(mom*mom, axis=1, keepdims=True)/rho)
			scalar[:] = np.sqrt(gamma*getP()/rho)
		elif sname is self.AdditionalVariables["MaxWaveSpeed"].name:
			# Pressure
			# P = GetPressure()
			scalar[:] = np.linalg.norm(mom, axis=1, keepdims=True)/rho + np.sqrt(gamma*getP()/rho)
		else:
			raise NotImplementedError

		return scalar


class Euler1D(Euler):

	StateRank = 3
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
		}

		self.IC_fcn_map.update(d)
		self.exact_fcn_map.update(d)
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

	def GetMomentumSlice(self):
		irhou = self.GetStateIndex("XMomentum")
		imom = slice(irhou, irhou+1)

		return imom


class Euler2D(Euler):

	StateRank = 4
	dim = 2

	def __init__(self, order, basis, mesh):
		super().__init__(order, basis, mesh) 

	def set_maps(self):
		super().set_maps()

		d = {
			euler_fcn_type.IsentropicVortex : euler_fcns.IsentropicVortex,
		}

		self.IC_fcn_map.update(d)
		self.exact_fcn_map.update(d)
		self.BC_fcn_map.update(d)

		self.source_map.update({
			euler_source_type.StiffFriction : euler_fcns.StiffFriction,
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

	