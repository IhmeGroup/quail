import code
from enum import Enum
import numpy as np
from scipy.optimize import fsolve, root

import errors
import general

import physics.base.base as base
import physics.euler.euler as euler
import physics.base.functions as base_fcns
from physics.base.functions import BCType as base_BC_type
from physics.base.functions import ConvNumFluxType as base_conv_num_flux_type
from physics.base.functions import FcnType as base_fcn_type

import physics.euler.functions as euler_fcns
from physics.euler.functions import BCType as euler_BC_type
from physics.euler.functions import ConvNumFluxType as euler_conv_num_flux_type
from physics.euler.functions import FcnType as euler_fcn_type
from physics.euler.functions import SourceType as euler_source_type

import physics.chemistry.functions as chemistry_fcns
from physics.chemistry.functions import FcnType as chemistry_fcn_type
from physics.chemistry.functions import SourceType as chemistry_source_type
from physics.chemistry.functions import ConvNumFluxType as chemistry_conv_num_flux_type


class Chemistry(base.PhysicsBase):

	PHYSICS_TYPE = general.PhysicsType.Chemistry

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
			HeatRelease = 25. 
		)

	# def set_maps(self):
	# 	super().set_maps()

		self.BC_map.update({
			euler_BC_type.SlipWall : euler_fcns.SlipWall,
		# 	euler_BC_type.PressureOutlet : euler_fcns.PressureOutlet,
		})
	def set_physical_params(self, GasConstant=287., SpecificHeatRatio = 1.4, HeatRelease = 0.):
		self.R = GasConstant
		self.gamma = SpecificHeatRatio
		self.qo = HeatRelease

	class AdditionalVariables(Enum):
	    Pressure = "p"
	    Temperature = "T"
	    Entropy = "s"
	    InternalEnergy = "\\rho e"
	    TotalEnthalpy = "H"
	    SoundSpeed = "c"
	    MaxWaveSpeed = "\\lambda"
	    Velocity = "u"
	    MassFraction = "Y"
	    SourceTerm = "S"
	    Jacobian = "J"

	def ConvFluxInterior(self, Up):
		dim = self.dim
		
		irho = self.GetStateIndex("Density")
		irhoE = self.GetStateIndex("Energy")
		irhoY = self.GetStateIndex("Mixture")
		# imom = self.GetMomentumSlice()
		srho = self.get_state_slice("Density")
		srhoE = self.get_state_slice("Energy")
		smom = self.GetMomentumSlice()
		srhoY = self.get_state_slice("Mixture")

		eps = general.eps

		rho = Up[:, srho]
		rho += eps
		rhoE = Up[:,srhoE]
		mom = Up[:,smom]
		rhoY = Up[:,srhoY]

		p = self.ComputeScalars("Pressure", Up)
		h = self.ComputeScalars("TotalEnthalpy", Up)

		pmat = np.zeros([Up.shape[0], dim, dim])
		idx = np.full([dim,dim],False)
		np.fill_diagonal(idx,True)
		pmat[:, idx] = p

		F = np.empty(Up.shape + (dim,))
		F[:, irho, :] = mom
		F[:, smom, :] = np.einsum('ij,ik->ijk',mom,mom)/np.expand_dims(rho, axis=2) + pmat
		F[:, irhoE, :] = mom*h
		F[:, irhoY, :] = mom*rhoY/rho

		rho -= eps

		return F

	def AdditionalScalars(self, ScalarName, Up, flag_non_physical):
		''' Extract state variables '''
		srho = self.get_state_slice("Density")
		srhoE = self.get_state_slice("Energy")
		srhoY = self.get_state_slice("Mixture")
		smom = self.GetMomentumSlice()
		rho = Up[:, srho]
		rhoE = Up[:, srhoE]
		mom = Up[:, smom]
		rhoY = Up[:, srhoY]

		''' Common scalars '''
		gamma = self.gamma
		R = self.R
		qo = self.qo

		if flag_non_physical:
			if np.any(rho < 0.):
				raise errors.NotPhysicalError

		def get_pressure():
			# scalar = (gamma - 1.)*(rhoE - 0.5*np.sum(mom*mom, axis=1, keepdims=True)/rho) # just use for storage
			scalar = (gamma - 1.)*(rhoE - 0.5*np.sum(mom*mom, axis=1, keepdims=True)/rho - qo*rhoY) # just use for storage

			if flag_non_physical:
				if np.any(scalar < 0.):
					raise errors.NotPhysicalError
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
			scalar = np.log(get_pressure()/rho**gamma)
		elif sname is self.AdditionalVariables["InternalEnergy"].name:
			scalar = rhoE - 0.5*np.sum(mom*mom, axis=1, keepdims=True)/rho
		elif sname is self.AdditionalVariables["TotalEnthalpy"].name:
			scalar = (rhoE + get_pressure())/rho
		elif sname is self.AdditionalVariables["SoundSpeed"].name:
			scalar = np.sqrt(gamma*get_pressure()/rho)
		elif sname is self.AdditionalVariables["MaxWaveSpeed"].name:
			scalar = np.linalg.norm(mom, axis=1, keepdims=True)/rho + np.sqrt(gamma*get_pressure()/rho)
		elif sname is self.AdditionalVariables["Velocity"].name:
			scalar = np.linalg.norm(mom, axis=1, keepdims=True)/rho
		elif sname is self.AdditionalVariables["MassFraction"].name:
			scalar = rhoY/rho
		elif sname is self.AdditionalVariables["SourceTerm"].name:
			nq = Up.shape[0]
			x = np.zeros([nq,1])
			Sp = np.zeros_like(Up) # SourceState is an additive function so source needs to be initialized to zero for each time step
			Sp = self.SourceState(nq, x, 0., Up, Sp) # [nq,ns]
			scalar = Sp[:,3].reshape(7,1)
		elif sname is self.AdditionalVariables["Jacobian"].name:
			nq = Up.shape[0]
			x = np.zeros([nq,1])
			jac = np.zeros([nq,4,4]) # SourceState is an additive function so source needs to be initialized to zero for each time step
			jac = self.SourceJacobianState(nq, x, 0., Up, jac) # [nq,ns]
			scalar = jac[:,3,3].reshape(7,1)
		else:
			raise NotImplementedError

		return scalar

class Chemistry1D(Chemistry):

	NUM_STATE_VARS = 4
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
			# euler_fcn_type.SmoothIsentropicFlow : euler_fcns.SmoothIsentropicFlow,
			# euler_fcn_type.MovingShock : euler_fcns.MovingShock,
			chemistry_fcn_type.DensityWave : chemistry_fcns.DensityWave,
			chemistry_fcn_type.SimpleDetonation1 : chemistry_fcns.SimpleDetonation1,
			chemistry_fcn_type.SimpleDetonation2 : chemistry_fcns.SimpleDetonation2,
			chemistry_fcn_type.SimpleDetonation3 : chemistry_fcns.SimpleDetonation3,

		}

		self.IC_fcn_map.update(d)
		self.exact_fcn_map.update(d)
		self.BC_fcn_map.update(d)

		self.source_map.update({
			chemistry_source_type.Arrhenius : chemistry_fcns.Arrhenius,
			chemistry_source_type.Heaviside : chemistry_fcns.Heaviside,

		})

		self.conv_num_flux_map.update({
			euler_conv_num_flux_type.Roe : euler_fcns.Roe1D,
			chemistry_conv_num_flux_type.HLLC : chemistry_fcns.HLLC1D,
		})

	class StateVariables(Enum):
		__Order__ = 'Density XMomentum Energy Mixture' # only needed in 2.x
		# LaTeX format
		Density = "\\rho"
		XMomentum = "\\rho u"
		Energy = "\\rho E"
		Mixture = "\\rho Y"

	def GetStateIndices(self):
		irho = self.GetStateIndex("Density")
		irhou = self.GetStateIndex("XMomentum")
		irhoE = self.GetStateIndex("Energy")
		irhoY = self.GetStateIndex("Mixture")

		return irho, irhou, irhoE, irhoY

	def get_state_slices(self):
		srho = self.get_state_slice("Density")
		srhou = self.get_state_slice("XMomentum")
		srhoE = self.get_state_slice("Energy")
		srhoY = self.get_state_slice("Mixture")
		return srho, srhou, srhoE, srhoY

	def GetMomentumSlice(self):
		irhou = self.GetStateIndex("XMomentum")
		smom = slice(irhou, irhou+1)

		return smom