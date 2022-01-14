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
#      File : src/physics/chemistry/euler_multispecies/euler_multispecies.py
#
#      Contains class definitions for multispecies Euler equations
#
# ------------------------------------------------------------------------ #
from enum import Enum
import numpy as np
from scipy.optimize import fsolve, root
import ctypes

from external.optional_cantera import ct

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
from physics.euler.functions import ConvNumFluxType as \
		euler_conv_num_flux_type
from physics.euler.functions import FcnType as euler_fcn_type
from physics.euler.functions import SourceType as euler_source_type

import physics.chemistry.euler_multispecies.functions as euler_mult_fcns
from physics.chemistry.euler_multispecies.functions import FcnType as \
		euler_mult_fcn_type
from physics.chemistry.euler_multispecies.functions import SourceType as \
		euler_mult_source_type
from external.optional_multispecies import multispecies_tools

# from physics.chemistry.euler_multispecies.functions import SourceType as \
# 		euler_mult_source_type
# from physics.chemistry.euler_multispecies.functions import ConvNumFluxType as \
# 		euler_mult_conv_num_flux_type


class EulerMultispecies(base.PhysicsBase):
	'''
	This class corresponds to the compressible Euler equations with a simple
	transport equation for mass fraction. It is appropriate for testing 
	simple single-species gas mixture chemistry models. It inherits 
	attributes and methods from the PhysicsBase class. See PhysicsBase for 
	detailed comments of attributes and methods. This class should not be 
	instantiated directly. Instead,the 1D and 2D variants, which inherit 
	from this class (see below), should be instantiated.

	Additional methods and attributes are commented below.

	Attributes:
	-----------

	'''
	def __init__(self):
		super().__init__()

	def set_maps(self):
		super().set_maps()

		self.BC_map.update({
			base_BC_type.StateAll : base_fcns.StateAll,
			# base_BC_type.Extrapolate : base_fcns.Extrapolate,
			euler_BC_type.SlipWall : euler_fcns.SlipWall,
			# euler_BC_type.PressureOutlet : euler_fcns.PressureOutlet,
		})
		self.source_map.update({
			euler_mult_source_type.Reacting : euler_mult_fcns.Reacting,
		})

	def set_physical_params(self):
		pass

	def c_cantera_file(self):
		return ctypes.c_char_p(self.CANTERA_FILENAME.encode('utf-8'))

class EulerMultispecies1D(EulerMultispecies):
	'''
	This class corresponds to 1D multispecies Euler classes.
	'''
	NDIMS = 1

	def set_physical_params(self):
		gas = ct.Solution(self.CANTERA_FILENAME)
		# Save object to physics class before calculating inflow props
		self.gas = gas

	def get_momentum_slice(self):
		irhou = self.get_state_index("XMomentum")
		smom = slice(irhou, irhou+1)

		return smom

	def compute_additional_variable(self, var_name, Uq, flag_non_physical):
		''' Extract state variables '''
		srho = self.get_state_slice("Density")
		srhoE = self.get_state_slice("Energy")
		smom = self.get_momentum_slice()

		rho = Uq[:, :, srho]
		rhoE = Uq[:, :, srhoE]
		mom = Uq[:, :, smom]

		''' Flag non-physical state '''
		if flag_non_physical:
			if np.any(rho < 0.):
				raise errors.NotPhysicalError

		''' Get final scalars '''
		vname = self.AdditionalVariables[var_name].name
		if vname is self.AdditionalVariables["Pressure"].name:
			varq = multispecies_tools.get_pressure(self, Uq)
		elif vname is self.AdditionalVariables["Temperature"].name:
			varq = multispecies_tools.get_temperature(self, Uq)
		elif vname is self.AdditionalVariables["SpecificHeatRatio"].name:
			varq = multispecies_tools.get_specificheatratio(self, Uq)
		elif vname is self.AdditionalVariables["MaxWaveSpeed"].name:
			varq = multispecies_tools.get_maxwavespeed(self, Uq)
		#NOTE: 1D only right now.
		elif vname is self.AdditionalVariables["Velocity"].name:
			varq = mom/rho
		else:
			raise NotImplementedError

		return varq

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

	def get_conv_flux_interior(self, Uq):
		rho = Uq[:, :, 0]
		rhou = Uq[:, :, 1]
		rhoE = Uq[:, :, 2]

		# Get velocity
		u = rhou / rho
		# Get squared velocity
		u2 = u**2

		# Calculate pressure
		p = self.compute_variable("Pressure", Uq,
			flag_non_physical=False)[:, :, 0]

		# Get total enthalpy
		H = rhoE + p

		F = np.empty(Uq.shape + (self.NDIMS,))
		F[:, :, 0, 0] = rhou
		F[:, :, 1, 0] = rho * u2 + p
		F[:, :, 2, 0] = H * u
		F[:, :, 3:, 0] = np.expand_dims(rhou/rho, axis=-1)*Uq[:, :, 3:]

		return F, (u2, rho, p)


class EulerMultispecies1D_2sp_air(EulerMultispecies1D):
	'''
	This class corresponds to 1D Euler equations with simple chemistry.
	It inherits attributes and methods from the Chemistry class.
	See Chemistry for detailed comments of attributes and methods.

	Additional methods and attributes are commented below.
	'''
	NUM_STATE_VARS = 4
	NUM_SPECIES  = 2
	PHYSICS_TYPE = general.PhysicsType.EulerMultispecies_2sp_air
	CANTERA_FILENAME = "air_test.xml"

	def set_maps(self):
		super().set_maps()

		d = {
			euler_mult_fcn_type.SodMultispeciesAir : \
				euler_mult_fcns.SodMultispeciesAir,
			# chemistry_fcn_type.DensityWave : chemistry_fcns.DensityWave,
			# chemistry_fcn_type.SimpleDetonation1 : \
			# 		chemistry_fcns.SimpleDetonation1,
			# chemistry_fcn_type.OverdrivenDetonation : \
			# 		chemistry_fcns.OverdrivenDetonation,
		}

		self.IC_fcn_map.update(d)
		self.exact_fcn_map.update(d)
		self.BC_fcn_map.update(d)

		# self.source_map.update({
			# chemistry_source_type.Arrhenius : chemistry_fcns.Arrhenius,
			# chemistry_source_type.Heaviside : chemistry_fcns.Heaviside,

		# })

		self.conv_num_flux_map.update({
		})

	class StateVariables(Enum):
		Density = "\\rho"
		XMomentum = "\\rho u"
		Energy = "\\rho E"
		rhoYO2 = "\\rho Y_{O2}"

	class AdditionalVariables(Enum):
	    Pressure = "p"
	    Temperature = "T"
	    Entropy = "s"
	    InternalEnergy = "\\rho e"
	    TotalEnthalpy = "H"
	    SoundSpeed = "c"
	    MaxWaveSpeed = "\\lambda"
	    MassFractionO2 = "YO2"
	    MassFractionN2 = "YN2"
	    SpecificHeatRatio = "\\gamma"

	def compute_additional_variable(self, var_name, Uq, flag_non_physical):
		''' Extract state variables '''
		srho = self.get_state_slice("Density")
		srhoE = self.get_state_slice("Energy")
		# srhoY = self.get_state_slice("Mixture")
		smom = self.get_momentum_slice()
		# srhoYN2 = self.get_state_slice("rhoYN2")

		rho = Uq[:, :, srho]
		rhoE = Uq[:, :, srhoE]
		mom = Uq[:, :, smom]
		rhoYO2 = Uq[:, :, [3]]
		rhoYN2 = rho * (1.0 - rhoYO2 / rho)

		''' Flag non-physical state '''
		if flag_non_physical:
			if np.any(rho < 0.):
				raise errors.NotPhysicalError

		''' Get final scalars '''
		vname = self.AdditionalVariables[var_name].name
		if vname is self.AdditionalVariables["MassFractionO2"].name:
			varq = rhoYO2/rho
		elif vname is self.AdditionalVariables["MassFractionN2"].name:
			varq = rhoYN2/rho
		else:
			varq = super().compute_additional_variable(var_name, Uq, 
					flag_non_physical)
		return varq


class EulerMultispecies1D_1sp_air(EulerMultispecies1D):
	'''
	This class corresponds to 1D Euler equations with simple chemistry.
	It inherits attributes and methods from the Chemistry class.
	See Chemistry for detailed comments of attributes and methods.

	Additional methods and attributes are commented below.
	'''
	NUM_STATE_VARS = 3
	NUM_SPECIES  = 1
	PHYSICS_TYPE = general.PhysicsType.EulerMultispecies_1sp_air
	CANTERA_FILENAME = "Nitrogen.xml"

	def set_maps(self):
		super().set_maps()

		d = {
			euler_mult_fcn_type.SodMultispeciesAir1sp : \
				euler_mult_fcns.SodMultispeciesAir1sp,
			# chemistry_fcn_type.DensityWave : chemistry_fcns.DensityWave,
			# chemistry_fcn_type.SimpleDetonation1 : \
			# 		chemistry_fcns.SimpleDetonation1,
			# chemistry_fcn_type.OverdrivenDetonation : \
			# 		chemistry_fcns.OverdrivenDetonation,
		}

		self.IC_fcn_map.update(d)
		self.exact_fcn_map.update(d)
		self.BC_fcn_map.update(d)

		# self.source_map.update({
			# chemistry_source_type.Arrhenius : chemistry_fcns.Arrhenius,
			# chemistry_source_type.Heaviside : chemistry_fcns.Heaviside,

		# })

		self.conv_num_flux_map.update({
		})

	class StateVariables(Enum):
		Density = "\\rho"
		XMomentum = "\\rho u"
		Energy = "\\rho E"

	class AdditionalVariables(Enum):
	    Pressure = "p"
	    Temperature = "T"
	    Entropy = "s"
	    InternalEnergy = "\\rho e"
	    TotalEnthalpy = "H"
	    SoundSpeed = "c"
	    MaxWaveSpeed = "\\lambda"
	    MassFractionN2 = "YN2"
	    SpecificHeatRatio = "\\gamma"


class EulerMultispecies1D_3sp_H2O2_inert(EulerMultispecies1D):
	'''
	This class corresponds to 1D Euler equations with simple chemistry.
	It inherits attributes and methods from the Chemistry class.
	See Chemistry for detailed comments of attributes and methods.

	Additional methods and attributes are commented below.
	'''
	NUM_STATE_VARS = 5
	NUM_SPECIES  = 3
	PHYSICS_TYPE = general.PhysicsType.EulerMultispecies1D_3sp_H2O2_inert
	CANTERA_FILENAME = "h2o2_inert.xml"

	def set_maps(self):
		super().set_maps()

		d = {
			euler_mult_fcn_type.InertShockTube : \
				euler_mult_fcns.InertShockTube,
		}

		self.IC_fcn_map.update(d)
		self.exact_fcn_map.update(d)
		self.BC_fcn_map.update(d)
		self.conv_num_flux_map.update({
		})

	class StateVariables(Enum):
		Density = "\\rho"
		XMomentum = "\\rho u"
		Energy = "\\rho E"
		rhoYH2 = "\\rho Y_{H2}"
		rhoYO2 = "\\rho Y_{O2}"

	class AdditionalVariables(Enum):
	    Pressure = "p"
	    Temperature = "T"
	    Entropy = "s"
	    InternalEnergy = "\\rho e"
	    TotalEnthalpy = "H"
	    SoundSpeed = "c"
	    MaxWaveSpeed = "\\lambda"
	    MassFractionO2 = "Y_{O2}"
	    MassFractionH2 = "Y_{H2}"
	    MassFractionAR = "Y_{Ar}"
	    SpecificHeatRatio = "\\gamma"

	def compute_additional_variable(self, var_name, Uq, flag_non_physical):
		''' Extract state variables '''
		srho = self.get_state_slice("Density")
		rho = Uq[:, :, srho]
		rhoYH2 = Uq[:, :, [3]]
		rhoYO2 = Uq[:, :, [4]]
		rhoYAR = rho * (1.0 - (rhoYO2 + rhoYH2) / rho)

		''' Flag non-physical state '''
		if flag_non_physical:
			if np.any(rho < 0.):
				raise errors.NotPhysicalError

		''' Get final scalars '''
		vname = self.AdditionalVariables[var_name].name
		if vname is self.AdditionalVariables["MassFractionH2"].name:
			varq = rhoYH2/rho
		elif vname is self.AdditionalVariables["MassFractionO2"].name:
			varq = rhoYO2/rho
		elif vname is self.AdditionalVariables["MassFractionAR"].name:
			varq = rhoYAR/rho
		else:
			varq = super().compute_additional_variable(var_name, Uq, 
					flag_non_physical)
		return varq

class EulerMultispecies1D_9sp_H2O2(EulerMultispecies1D):
	'''
	This class corresponds to 1D Euler equations with simple chemistry.
	It inherits attributes and methods from the Chemistry class.
	See Chemistry for detailed comments of attributes and methods.

	Additional methods and attributes are commented below.
	'''
	NUM_STATE_VARS = 11
	NUM_SPECIES  = 9
	PHYSICS_TYPE = general.PhysicsType.EulerMultispecies1D_9sp_H2O2
	CANTERA_FILENAME = "h2o2.xml"

	def set_maps(self):
		super().set_maps()

		d = {
			euler_mult_fcn_type.ReactingShockTube : \
				euler_mult_fcns.ReactingShockTube,
		}

		self.IC_fcn_map.update(d)
		self.exact_fcn_map.update(d)
		self.BC_fcn_map.update(d)
		self.conv_num_flux_map.update({
		})

	class StateVariables(Enum):
		Density = "\\rho"
		XMomentum = "\\rho u"
		Energy = "\\rho E"
		rhoYH2 = "\\rho Y_{H2}"
		rhoYH = "\\rho Y_{H}"
		rhoYO = "\\rho Y_{O}"
		rhoYO2 = "\\rho Y_{O2}"
		rhoYOH = "\\rho Y_{OH}"
		rhoYH2O = "\\rho Y_{H2O}"
		rhoYHO2 = "\\rho Y_{HO2}"
		rhoYH2O2 = "\\rho Y_{H2O2}"

	class AdditionalVariables(Enum):
	    Pressure = "p"
	    Temperature = "T"
	    Entropy = "s"
	    InternalEnergy = "\\rho e"
	    TotalEnthalpy = "H"
	    SoundSpeed = "c"
	    MaxWaveSpeed = "\\lambda"
	    Velocity = "u"
	    MassFractionH2 = "Y_{H2}"
	    MassFractionH = "Y_{H}"
	    MassFractionO = "Y_{O}"
	    MassFractionO2 = "Y_{O2}"
	    MassFractionOH = "Y_{OH}"
	    MassFractionH2O = "Y_{H2O}"
	    MassFractionHO2 = "Y_{HO2}"
	    MassFractionH2O2 = "Y_{H2O2}"
	    MassFractionAR = "Y_{Ar}"
	    SpecificHeatRatio = "\\gamma"

	def compute_additional_variable(self, var_name, Uq, flag_non_physical):
		''' Extract state variables '''
		srho = self.get_state_slice("Density")
		rho = Uq[:, :, srho]
		rhoYH2 = Uq[:, :, [3]]
		rhoYH = Uq[:, :, [4]]
		rhoYO = Uq[:, :, [5]]
		rhoYO2 = Uq[:, :, [6]]
		rhoYOH = Uq[:, :, [7]]
		rhoYH2O = Uq[:, :, [8]]
		rhoYHO2 = Uq[:, :, [9]]
		rhoYH2O2 = Uq[:, :, [10]]
		# rhoYAR = rho * (1.0 - (rhoYO2 + rhoYH2) / rho)

		''' Flag non-physical state '''
		if flag_non_physical:
			if np.any(rho < 0.):
				raise errors.NotPhysicalError

		''' Get final scalars '''
		vname = self.AdditionalVariables[var_name].name
		if vname is self.AdditionalVariables["MassFractionH2"].name:
			varq = rhoYH2/rho
		elif vname is self.AdditionalVariables["MassFractionH"].name:
			varq = rhoYH/rho
		elif vname is self.AdditionalVariables["MassFractionO"].name:
			varq = rhoYO/rho
		elif vname is self.AdditionalVariables["MassFractionO2"].name:
			varq = rhoYO2/rho
		elif vname is self.AdditionalVariables["MassFractionOH"].name:
			varq = rhoYOH/rho
		elif vname is self.AdditionalVariables["MassFractionH2O"].name:
			varq = rhoYH2O/rho
		elif vname is self.AdditionalVariables["MassFractionHO2"].name:
			varq = rhoYHO2/rho		
		elif vname is self.AdditionalVariables["MassFractionH2O2"].name:
			varq = rhoYH2O2/rho		
		else:
			varq = super().compute_additional_variable(var_name, Uq, 
					flag_non_physical)
		return varq


class EulerMultispecies1D_10sp_H2O2(EulerMultispecies1D):
	'''
	This class corresponds to 1D Euler equations with simple chemistry.
	It inherits attributes and methods from the Chemistry class.
	See Chemistry for detailed comments of attributes and methods.

	Additional methods and attributes are commented below.
	'''
	NUM_STATE_VARS = 12
	NUM_SPECIES  = 10
	PHYSICS_TYPE = general.PhysicsType.EulerMultispecies1D_10sp_H2O2
	CANTERA_FILENAME = "h2o2_10sp_40rxn.cti"

	def set_maps(self):
		super().set_maps()

		d = {
			# euler_mult_fcn_type.PSR_H2O2 : \
			# 	euler_mult_fcns.PSR_H2O2,
		}

		self.IC_fcn_map.update(d)
		self.exact_fcn_map.update(d)
		self.BC_fcn_map.update(d)
		self.conv_num_flux_map.update({
		})

	class StateVariables(Enum):
		Density = "\\rho"
		XMomentum = "\\rho u"
		Energy = "\\rho E"
		rhoYH = "\\rho Y_{H}"
		rhoYH2 = "\\rho Y_{H2}"
		rhoYO = "\\rho Y_{O}"
		rhoYO2 = "\\rho Y_{O2}"
		rhoYOH = "\\rho Y_{OH}"
		rhoYH2O = "\\rho Y_{H2O}"
		rhoYN2 = "\\rho Y_{N2}"
		rhoYHO2 = "\\rho Y_{HO2}"
		rhoYH2O2 = "\\rho Y_{H2O2}"


	class AdditionalVariables(Enum):
	    Pressure = "p"
	    Temperature = "T"
	    Entropy = "s"
	    InternalEnergy = "\\rho e"
	    TotalEnthalpy = "H"
	    SoundSpeed = "c"
	    MaxWaveSpeed = "\\lambda"
	    Velocity = "u"
	    MassFractionH2 = "Y_{H2}"
	    MassFractionH = "Y_{H}"
	    MassFractionO = "Y_{O}"
	    MassFractionO2 = "Y_{O2}"
	    MassFractionOH = "Y_{OH}"
	    MassFractionH2O = "Y_{H2O}"
	    MassFractionHO2 = "Y_{HO2}"
	    MassFractionH2O2 = "Y_{H2O2}"
	    MassFractionN2 = "Y_{N2}"
	    MassFractionAR = "Y_{AR}"
	    SpecificHeatRatio = "\\gamma"

	def compute_additional_variable(self, var_name, Uq, flag_non_physical):
		''' Extract state variables '''
		srho = self.get_state_slice("Density")
		rho = Uq[:, :, srho]
		rhoYH = Uq[:, :, [3]]
		rhoYH2 = Uq[:, :, [4]]
		rhoYO = Uq[:, :, [5]]
		rhoYO2 = Uq[:, :, [6]]
		rhoYOH = Uq[:, :, [7]]
		rhoYH2O = Uq[:, :, [8]]
		rhoYN2 = Uq[:, :, [9]]
		rhoYHO2 = Uq[:, :, [10]]
		rhoYH2O2 = Uq[:, :, [11]]

		''' Flag non-physical state '''
		if flag_non_physical:
			if np.any(rho < 0.):
				raise errors.NotPhysicalError

		''' Get final scalars '''
		vname = self.AdditionalVariables[var_name].name
		if vname is self.AdditionalVariables["MassFractionH2"].name:
			varq = rhoYH2/rho
		elif vname is self.AdditionalVariables["MassFractionH"].name:
			varq = rhoYH/rho
		elif vname is self.AdditionalVariables["MassFractionO"].name:
			varq = rhoYO/rho
		elif vname is self.AdditionalVariables["MassFractionO2"].name:
			varq = rhoYO2/rho
		elif vname is self.AdditionalVariables["MassFractionOH"].name:
			varq = rhoYOH/rho
		elif vname is self.AdditionalVariables["MassFractionH2O"].name:
			varq = rhoYH2O/rho
		elif vname is self.AdditionalVariables["MassFractionHO2"].name:
			varq = rhoYHO2/rho		
		elif vname is self.AdditionalVariables["MassFractionH2O2"].name:
			varq = rhoYH2O2/rho		
		elif vname is self.AdditionalVariables["MassFractionN2"].name:
			varq = rhoYN2/rho		
	
		else:
			varq = super().compute_additional_variable(var_name, Uq, 
					flag_non_physical)
		return varq