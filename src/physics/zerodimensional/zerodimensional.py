# ------------------------------------------------------------------------ #
#
#       File : src/physics/zerodimensional/zerodimensional.py
#
#       Contains class definitions for zero dimensional equations.
#
# ------------------------------------------------------------------------ #
import cantera as ct
from enum import Enum, auto
import numpy as np
import sys

import errors
import general

import physics.base.base as base
import physics.base.functions as base_fcns
from physics.base.functions import BCType as base_BC_type
from physics.base.functions import FcnType as base_fcn_type
from physics.base.functions import ConvNumFluxType as base_conv_num_flux_type

import physics.scalar.functions as scalar_fcns
from physics.scalar.functions import FcnType as scalar_fcn_type
from physics.scalar.functions import SourceType as scalar_source_type

import physics.zerodimensional.functions as zerod_fcns
from physics.zerodimensional.functions import FcnType as zerod_fcn_type
from physics.zerodimensional.functions import SourceType as zerod_source_type


class ZeroDimensional(base.PhysicsBase):
	'''
	This class is a parent class for any zero dimensional problem that
	can be solved in quail. We choose to model 0D cases in quasi-1D where
	we use one element with periodic boundaries to solve systems of ODEs
	in time.
	'''
	def set_maps(self):
		super().set_maps()

		d = {
			base_fcn_type.Uniform : base_fcns.Uniform,
		}
		self.IC_fcn_map.update(d)
		self.exact_fcn_map.update(d)
		self.BC_fcn_map.update(d)

	class StateVariables(Enum):
		Scalar = "T"


	class AdditionalVariables(Enum):
	    MaxWaveSpeed = "\\lambda"

	def get_conv_flux_interior(self, Uq):

		# This can be zero or the mixing function.
		F = np.zeros_like(Uq)
		F = np.expand_dims(F, axis=-1)

		return F, None

	def compute_additional_variable(self, var_name, Uq, flag_non_physical):
		'''
		For the zero dimensional cases we force the "MaxWaveSpeed" in the
		Lax Friedrichs flux to be zero. This is needed for the splitting 
		cases which force fluxes to be evaluated (can be revisited if
		desired). Users can set ConvFluxSwitch to False for cases that 
		do not use splitting for zero dimensional cases.
		'''
		sname = self.AdditionalVariables[var_name].name

		if sname is self.AdditionalVariables["MaxWaveSpeed"].name:
			# Force the "MaxWaveSpeed" to zero
			scalar = np.zeros_like(Uq)
		else:
			raise NotImplementedError
		
		return scalar

class ModelProblem(ZeroDimensional):
	'''
	This class solves the classic model problem:

		dy/dt = nu * y

	This is used for testing order of accuracy in time as it provides
	an exact error.
	'''
	NUM_STATE_VARS = 1
	NDIMS = 1
	PHYSICS_TYPE = general.PhysicsType.ModelProblem

	def set_maps(self):
		super().set_maps()

		self.source_map.update({
			scalar_source_type.SimpleSource : scalar_fcns.SimpleSource,
		})

	class StateVariables(Enum):
		Scalar = "y"

class ModelPSRScalar(ZeroDimensional):
	'''
	This class corresponds to the 0D model of a partially stirred reactor.
	It inherits attributes and methods from the PhysicsBase class. See
	PhysicsBase for detailed comments of attributes and methods.
	'''
	NUM_STATE_VARS = 1
	NDIMS = 1
	PHYSICS_TYPE = general.PhysicsType.ModelPSRScalar

	def set_maps(self):
		super().set_maps()

		self.source_map.update({
			zerod_source_type.ScalarArrhenius : zerod_fcns.ScalarArrhenius,
			zerod_source_type.ScalarMixing : zerod_fcns.ScalarMixing,
		})

	class StateVariables(Enum):
		Scalar = "T"

	def set_physical_params(self, T_ad=1.15, T_in=0.15, T_a=1.8):
		'''
		This method sets physical parameters.

		Inputs:
		-------
			T_in: Inlet temperature of the unburned gas
			T_ad: Adiabatic flame temperature
			T_a:  Activation temperature

		Outputs:
		--------
			self: physical parameters set
		'''
		self.T_ad = T_ad
		self.T_in = T_in
		self.T_a = T_a

class Pendulum(ZeroDimensional):
	'''
	This class is a simple second order ODE where:

	theta'' = (-g/l)*theta

	It represents an oscillating pendulum and is helpful for convergence
	studies in time.
	'''
	NUM_STATE_VARS = 2
	NDIMS = 1
	PHYSICS_TYPE = general.PhysicsType.Pendulum

	def set_maps(self):
		super().set_maps()

		d = {
			zerod_fcn_type.PendulumExact : zerod_fcns.PendulumExact,
		}

		self.IC_fcn_map.update(d)
		self.exact_fcn_map.update(d)
		self.BC_fcn_map.update(d)

		self.source_map.update({
			zerod_source_type.Pendulum : zerod_fcns.Pendulum,
		})

	class StateVariables(Enum):
		Scalar = "$\\theta$"
		Scalar2 = "$\\frac{d\\theta}{dt}$"

	def set_physical_params(self, g=9.81, l=0.6):
		'''
		This method sets physical parameters.

		Inputs:
		-------
			T_in: Inlet temperature of the unburned gas
			T_ad: Adiabatic flame temperature
			T_a:  Activation temperature

		Outputs:
		--------
			self: physical parameters set
		'''
		self.g = g
		self.l = l

class MultispeciesPSR(ZeroDimensional):
	'''
	This class corresponds to the 0D model of a partially stirred reactor.
	It inherits attributes and methods from the PhysicsBase class. See
	PhysicsBase for detailed comments of attributes and methods.
	'''
	NUM_STATE_VARS = 11
	NDIMS = 1
	PHYSICS_TYPE = general.PhysicsType.MultispeciesPSR

	def set_maps(self):
		super().set_maps()

		d = {
			zerod_fcn_type.MultispeicesPSR_IC : zerod_fcns.MultispeicesPSR_IC,
		}

		self.IC_fcn_map.update(d)
		self.exact_fcn_map.update(d)
		self.BC_fcn_map.update(d)

		self.source_map.update({
			zerod_source_type.Reacting : zerod_fcns.Reacting,
			zerod_source_type.Mixing : zerod_fcns.Mixing,
		})

	class StateVariables(Enum):
		Temperature = "T"
		Y_H2 = "$Y_{H_2}$"
		Y_H = "$Y_{H}$"
		Y_O = "$Y_{O}$"
		Y_O2 = "$Y_{O2}$"
		Y_OH = "$Y_{OH}$"
		Y_H2O = "$Y_{H2O}$"
		Y_HO2 = "$Y_{HO2}$"
		Y_H2O2 = "$Y_{H2O2}$"
		Y_AR = "$Y_{AR}$"
		Y_N2 = "$Y_{N2}$"

	def set_physical_params(self, P=80.*ct.one_atm, Tu=875., 
			phi=0.5, tau=2.e-6):
		'''
		This method sets physical parameters for the multispecies
		PSR problem.

		Inputs:
		-------
			P: pressure
			Tu: unburnt gas temperature
			phi: equivalence ratio
			tau: reactor residence time

		Outputs:
		--------
			self: physical parameters set
		'''
		# Unpack
		self.P = P
		self.Tu = Tu
		self.phi = phi
		self.tau = tau		

		# Save object to physics class before calculating inflow props
		gas = ct.Solution('h2o2.yaml')
		self.gas = gas

		# Note: This is hardcoded for the PSR model problem of Wu, 2019
		gas.TPX = Tu, P, "H2:{},O2:{},N2:{},H:{}".format(phi, 
				0.5, 0.5*3.76, 0.095)
		
		# 'Inflow' properties for reactor system
		self.yin = np.hstack((gas.T, gas.Y))
		self.hin = gas.partial_molar_enthalpies
