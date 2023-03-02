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
#       File : src/physics/binarynavierstokes/binarynavierstokes.py
#
#       Contains class definitions for 1D and 2D NS equations with a simple
# 		transport equation for mass fraction.
#
# ------------------------------------------------------------------------ #
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
from physics.base.functions import DiffNumFluxType as base_diff_num_flux_type
from physics.base.functions import FcnType as base_fcn_type

import physics.binarynavierstokes.functions as BNS_fcns
from physics.binarynavierstokes.functions import FcnType as BNS_fcn_type
from physics.binarynavierstokes.functions import SourceType as BNS_source_type

class BinaryNavierStokes1D(base.PhysicsBase):
	'''
	This class corresponds to the 1D compressible NS equations with a simple
	transport equation for mass fraction. It is appropriate for testing 
	simple single-species gas mixture chemistry models. It inherits 
	attributes and methods from the PhysicsBase class.

	Additional methods and attributes are commented below.

	Attributes:
	-----------
	R0,1: float
		mass-specific gas constant for species 0,1
	gamma0,1: float
		specific heat ratio for species 0,1
	mu0,1: float
		dynamic viscosity for species 0,1
	cv0,1: float
		constant volume specific heat for species 0,1
	Pr, Sc: float
		Prandtl and Schmidt numbers
	'''
	PHYSICS_TYPE = general.PhysicsType.BinaryNavierStokes
	NUM_STATE_VARS = 4
	NDIMS = 1

	def __init__(self):
		super().__init__()
		self.R0 = 0.
		self.R1 = 0.

		self.gamma0 = 0.
		self.gamma1 = 0.

		self.mu0 = 0.
		self.mu1 = 0.

		self.Pr = 0.
		self.Sc = 0.

		self.cv0 = 0.
		self.cv1 = 0.

	def set_maps(self):
		super().set_maps()

		d = {
			BNS_fcn_type.Waves1D:
				BNS_fcns.Waves1D,
		}
		self.IC_fcn_map.update(d)
		self.exact_fcn_map.update(d)
		self.BC_fcn_map.update(d)

		self.source_map.update({
			BNS_source_type.ManufacturedSourceBinary :
					BNS_fcns.ManufacturedSourceBinary,
		})

		self.diff_num_flux_map.update({
			base_diff_num_flux_type.SIP :
				base_fcns.SIP,
			})

	def set_physical_params(self, R0=287., R1=400., gamma0=1.4, gamma1=1.1, mu0=1., mu1=2., Pr=0.7, Sc=0.7):
		'''
		This method sets physical parameters.

		Inputs:
		-------
			R0,1: mass-specific gas constant for species 0,1
			gamma0,1: specific heat ratio for species 0,1
			mu0,1: dynamic viscosity for species 0,1
			Pr, Sc: Prandtl and Schmidt number

		Outputs:
		--------
			self: physical parameters set
		'''
		self.R0 = R0
		self.R1 = R1
		self.gamma0 = gamma0
		self.gamma1 = gamma1
		self.mu0 = mu0
		self.mu1 = mu1
		self.Pr = Pr
		self.Sc = Sc
		self.cv0 = R0/(gamma0-1.)
		self.cv1 = R1/(gamma1-1.)

	class AdditionalVariables(Enum):
		Pressure = "p"
		Temperature = "T"
		SoundSpeed = "c"
		MaxWaveSpeed = "\\lambda"
		Velocity = "u"
		MassFraction = "Y"
		PartialH = "h"

	class StateVariables(Enum):
		Density = "\\rho"
		XMomentum = "\\rho u"
		Energy = "\\rho E"
		Mixture = "\\rho Y"

	def get_state_indices(self):
		irho = self.get_state_index("Density")
		irhou = self.get_state_index("XMomentum")
		irhoE = self.get_state_index("Energy")
		irhoY = self.get_state_index("Mixture")

		return irho, irhou, irhoE, irhoY

	def get_state_slices(self):
		srho = self.get_state_slice("Density")
		srhou = self.get_state_slice("XMomentum")
		srhoE = self.get_state_slice("Energy")
		srhoY = self.get_state_slice("Mixture")

		return srho, srhou, srhoE, srhoY

	def get_momentum_slice(self):
		irhou = self.get_state_index("XMomentum")
		smom = slice(irhou, irhou+1)

		return smom

	def compute_additional_variable(self, var_name, Uq, flag_non_physical):
		'''
		This method computes a variable that is not a state variable.

		Inputs:
		-------
			var_name: name of variable to compute
			Uq: values of the state variables (typically at the quadrature
				points) [ne, nq, ns]
			flag_non_physical: if True, will raise an error if a
				non-physical quantity is computed, e.g., negative pressure
				for the Euler equations

		Outputs:
		--------
			varq: values of the given variable [ne, nq, 1]
		'''

		''' Extract state variables '''
		srho = self.get_state_slice("Density")
		srhoE = self.get_state_slice("Energy")
		srhoY = self.get_state_slice("Mixture")
		smom = self.get_momentum_slice()
		rho = Uq[:, :, srho]  # [ne,nq,1]
		rhoE = Uq[:, :, srhoE] # [ne,nq,1]
		mom = Uq[:, :, smom] # [ne,nq,1]
		rhoY = Uq[:, :, srhoY] # [ne,nq,1]

		''' Unpack '''
		gamma0 = self.gamma0
		gamma1 = self.gamma1
		R0 = self.R0
		R1 = self.R1
		mu0 = self.mu0
		mu1 = self.mu1
		Pr = self.Pr
		Sc = self.Sc
		cv0 = self.cv0
		cv1 = self.cv1

		''' Flag non-physical state '''
		if flag_non_physical:
			if np.any(rho < 0.):
				raise errors.NotPhysicalError

		# Compute the all of the needed additional variables
		# Your code here
		pressure = None
		temperature = None
		sound_speed = None
		max_wave_speed = None
		mass_fraction = None
		velocity = None
		partialH = None  #  Difference in partial enthalpies h_0 - h_1

		if flag_non_physical:
			if np.any(pressure < 0.):
				raise errors.NotPhysicalError

		''' Get final scalars '''
		vname = self.AdditionalVariables[var_name].name
		if vname is self.AdditionalVariables["Pressure"].name:
			varq = pressure
		elif vname is self.AdditionalVariables["Temperature"].name:
			varq = temperature
		elif vname is self.AdditionalVariables["SoundSpeed"].name:
			varq = sound_speed
		elif vname is self.AdditionalVariables["MaxWaveSpeed"].name:
			varq = max_wave_speed
		elif vname is self.AdditionalVariables["MassFraction"].name:
			varq = mass_fraction
		elif vname is self.AdditionalVariables["Velocity"].name:
			varq = velocity
		elif vname is self.AdditionalVariables["PartialH"].name:
			varq = partialH
		else:
			raise NotImplementedError

		return varq

	def get_conv_flux_interior(self, Uq):
		'''
		This method computes the convective analytical flux for element
		interiors.

		Inputs:
		-------
			Uq: values of the state variables (typically at the quadrature
				points) [ne, nq, ns]

		Outputs:
		--------
			Fq: flux values [ne, nq, ns, ndims]
			None
		'''

		irho, irhou, irhoE, irhoY = self.get_state_indices()

		rho = Uq[:, :, irho]   # [ne, nq]
		rhou = Uq[:, :, irhou]   # [ne, nq]
		rhoE = Uq[:, :, irhoE]   # [ne, nq]
		rhoY = Uq[:, :, irhoY]   # [ne, nq]
		
		# Your code here
		Fq = None

		return Fq, None

	def get_diff_flux_interior(self, Uq, gUq):
		'''
		This method computes the diffusive analytical flux for element
		interiors.

		Inputs:
		-------
			Uq: values of the state variables (typically at the quadrature
				points) [ne, nq, ns]
			gUq: vales of the gradient of the state (typically at the
				quadrature points) [ne, nq, ns, ndims]

		Outputs:
		--------
			Fq: flux values [ne, nq, ns, ndims]
		'''

		# Get indices/slices of state variables
		irho, irhou, irhoE, irhoY = self.get_state_indices()

		# Unpack state coefficients
		rho = Uq[:, :, irho]   # [ne, nq]
		rhou = Uq[:, :, irhou]  # [ne, nq]
		rhoE = Uq[:, :, irhoE]  # [ne, nq]
		rhoY = Uq[:, :, irhoY]  # [ne, nq]

		# Your code here
		F = None

		return F  # [n, nq, ns, ndims]

class BinaryNavierStokes2D(base.PhysicsBase):
	'''
	This class corresponds to the 2D compressible NS equations with a simple
	transport equation for mass fraction. It is appropriate for testing
	simple single-species gas mixture chemistry models. It inherits
	attributes and methods from the PhysicsBase class.

	Additional methods and attributes are commented below.

	Attributes:
	-----------
	R0,1: float
		mass-specific gas constant for species 0,1
	gamma0,1: float
		specific heat ratio for species 0,1
	mu0,1: float
		dynamic viscosity for species 0,1
	cv0,1: float
		constant volume specific heat for species 0,1
	Pr, Sc: float
		Prandtl and Schmidt numbers
	'''
	PHYSICS_TYPE = general.PhysicsType.BinaryNavierStokes
	NUM_STATE_VARS = 5
	NDIMS = 2

	def __init__(self):
		super().__init__()
		self.R0 = 0.
		self.R1 = 0.

		self.gamma0 = 0.
		self.gamma1 = 0.

		self.mu0 = 0.
		self.mu1 = 0.

		self.Pr = 0.
		self.Sc = 0.

		self.cv0 = 0.
		self.cv1 = 0.

	def set_maps(self):
		super().set_maps()

		d = {
			BNS_fcn_type.Waves2D:
				BNS_fcns.Waves2D,
			BNS_fcn_type.Waves2D2D:
				BNS_fcns.Waves2D2D,
		}
		self.IC_fcn_map.update(d)

		self.diff_num_flux_map.update({
			base_diff_num_flux_type.SIP :
				base_fcns.SIP,
			})

	def set_physical_params(self, R0=287., R1=400., gamma0=1.4, gamma1=1.1, mu0=1, mu1=2, Pr=0.7, Sc=0.7):
		'''
		This method sets physical parameters.

		Inputs:
		-------
			R0,1: mass-specific gas constant for species 0,1
			gamma0,1: specific heat ratio for species 0,1
			mu0,1: dynamic viscosity for species 0,1
			Pr, Sc: Prandtl and Schmidt number

		Outputs:
		--------
			self: physical parameters set
		'''
		self.R0 = R0
		self.R1 = R1
		self.gamma0 = gamma0
		self.gamma1 = gamma1
		self.mu0 = mu0
		self.mu1 = mu1
		self.Pr = Pr
		self.Sc = Sc
		self.cv0 = R0/(gamma0-1.)
		self.cv1 = R1/(gamma1-1.)

	class AdditionalVariables(Enum):
		Pressure = "p"
		Temperature = "T"
		SoundSpeed = "c"
		MaxWaveSpeed = "\\lambda"
		XVelocity = "u"
		YVelocity = "v"
		MassFraction = "Y"
		PartialH = "h"

	class StateVariables(Enum):
		Density = "\\rho"
		XMomentum = "\\rho u"
		YMomentum = "\\rho v"
		Energy = "\\rho E"
		Mixture = "\\rho Y"

	def get_state_indices(self):
		irho = self.get_state_index("Density")
		irhou = self.get_state_index("XMomentum")
		irhov = self.get_state_index("YMomentum")
		irhoE = self.get_state_index("Energy")
		irhoY = self.get_state_index("Mixture")

		return irho, irhou, irhov, irhoE, irhoY

	def get_state_slices(self):
		srho = self.get_state_slice("Density")
		srhou = self.get_state_slice("XMomentum")
		srhov = self.get_state_slice("XMomentum")
		srhoE = self.get_state_slice("Energy")
		srhoY = self.get_state_slice("Mixture")

		return srho, srhou, srhov, srhoE, srhoY

	def get_momentum_slice(self):
		irhou = self.get_state_index("XMomentum")
		irhov = self.get_state_index("YMomentum")
		smom = slice(irhou, irhov + 1)

		return smom

	def compute_additional_variable(self, var_name, Uq, flag_non_physical):
		''' Extract state variables '''
		srho = self.get_state_slice("Density")
		srhoE = self.get_state_slice("Energy")
		srhoY = self.get_state_slice("Mixture")
		smom = self.get_momentum_slice()
		rho = Uq[:, :, srho]  # [ne, nq]
		rhoE = Uq[:, :, srhoE]  # [ne, nq]
		rhoY = Uq[:, :, srhoY]  # [ne, nq]
		mom = Uq[:, :, smom]  # [ne, nq, ndims]

		''' Unpack '''
		gamma0 = self.gamma0
		gamma1 = self.gamma1
		R0 = self.R0
		R1 = self.R1
		mu0 = self.mu0
		mu1 = self.mu1
		Pr = self.Pr
		Sc = self.Sc
		cv0 = self.cv0
		cv1 = self.cv1

		''' Flag non-physical state '''
		if flag_non_physical:
			if np.any(rho < 0.):
				raise errors.NotPhysicalError
		# Compute the all of the needed additional variables
		# Your code here
		pressure = None
		temperature = None
		sound_speed = None
		max_wave_speed = None
		mass_fraction = None
		x_velocity = None
		y_velocity = None
		partialH = None  #  Difference in partial enthalpies h_0 - h_1

		if flag_non_physical:
			if np.any(pressure < 0.):
				raise errors.NotPhysicalError

		''' Get final scalars '''
		vname = self.AdditionalVariables[var_name].name
		if vname is self.AdditionalVariables["Pressure"].name:
			varq = pressure
		elif vname is self.AdditionalVariables["Temperature"].name:
			varq = temperature
		elif vname is self.AdditionalVariables["SoundSpeed"].name:
			varq = sound_speed
		elif vname is self.AdditionalVariables["MaxWaveSpeed"].name:
			varq = max_wave_speed
		elif vname is self.AdditionalVariables["MassFraction"].name:
			varq = mass_fraction
		elif vname is self.AdditionalVariables["XVelocity"].name:
			varq = x_velocity
		elif vname is self.AdditionalVariables["YVelocity"].name:
			varq = y_velocity
		elif vname is self.AdditionalVariables["PartialH"].name:
			varq = partialH
		else:
			raise NotImplementedError

		return varq

	def get_conv_flux_interior(self, Uq):
		'''
		This method computes the convective analytical flux for element
		interiors.

		Inputs:
		-------
			Uq: values of the state variables (typically at the quadrature
				points) [ne, nq, ns]

		Outputs:
		--------
			Fq: flux values [ne, nq, ns, ndims]
			None
		'''
		
		irho, irhou, irhov, irhoE, irhoY = self.get_state_indices()

		rho = Uq[:, :, irho]  # [ne, nq]
		rhou = Uq[:, :, irhou]  # [ne, nq]
		rhov = Uq[:, :, irhov]  # [ne, nq]
		rhoE = Uq[:, :, irhoE]  # [ne, nq]
		rhoY = Uq[:, :, irhoY]  # [ne, nq]

		# Your code here
		Fq = None

		return Fq, None

	def get_diff_flux_interior(self, Uq, gUq):
		'''
		This method computes the diffusive analytical flux for element
		interiors.

		Inputs:
		-------
			Uq: values of the state variables (typically at the quadrature
				points) [ne, nq, ns]
			gUq: vales of the gradient of the state (typically at the
				quadrature points) [ne, nq, ns, ndims]

		Outputs:
		--------
			Fq: flux values [ne, nq, ns, ndims]
		'''

		# Get indices/slices of state variables
		irho, irhou, irhov, irhoE, irhoY = self.get_state_indices()

		# Unpack state coefficients
		rho = Uq[:, :, irho]  # [ne, nq]
		rhou = Uq[:, :, irhou]  # [ne, nq]
		rhov = Uq[:, :, irhov]  # [ne, nq]
		rhoE = Uq[:, :, irhoE]  # [ne, nq]
		rhoY = Uq[:, :, irhoY]  # [ne, nq]

		# Your code here
		F = None

		return F  # [n, nq, ns, ndims]
