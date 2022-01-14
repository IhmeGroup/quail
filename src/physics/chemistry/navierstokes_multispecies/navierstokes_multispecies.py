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
#      File : src/physics/chemistry/navierstokes_multispecies
#				/navierstokes_multispecies.py
#
#      Contains class definitions for multispecies Navier-Stokes equations
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
import physics.chemistry.euler_multispecies.euler_multispecies as euler_multispecies
import physics.base.functions as base_fcns
from physics.base.functions import BCType as base_BC_type
from physics.base.functions import ConvNumFluxType as base_conv_num_flux_type
from physics.base.functions import DiffNumFluxType as base_diff_num_flux_type
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

import physics.chemistry.navierstokes_multispecies.functions as navierstokes_mult_fcns
from physics.chemistry.navierstokes_multispecies.functions import FcnType as \
		navierstokes_mult_fcn_type
# from physics.chemistry.navierstokes_multispecies.functions import SourceType as \
		# navierstokes_mult_source_type

from external.optional_multispecies import multispecies_tools


class NavierStokesMultispecies(euler_multispecies.EulerMultispecies):
	'''
	This class corresponds to the multicomponent compressible Navier-Stokes
	equations. It inherits attributes and methods from the EulerMultispecies
	class. See EulerMultispecies for detailed comments of attributes and 
	methods. This class should not be instantiated directly. Instead,the 1D
	and 2D variants, which inherit from this class (see below), should be 
	instantiated.

	Additional methods and attributes are commented below.

	Attributes:
	-----------

	'''
	def __init__(self):
		super().__init__()

	def set_maps(self):
		super().set_maps()

		self.diff_num_flux_map.update({
			base_diff_num_flux_type.SIP : 
				base_fcns.SIP,
			})


class NavierStokesMultispecies1D(NavierStokesMultispecies, 
		euler_multispecies.EulerMultispecies1D):
	'''
	This class corresponds to 1D multispecies Navier-Stokes classes.
	'''
	NDIMS = 1

	def get_diff_flux_interior(self, Uq, gUq):
		breakpoint()


class NavierStokesMultispecies1D_4sp_CH4(NavierStokesMultispecies1D):
	'''
	This class corresponds to 1D Euler equations with simple chemistry.
	It inherits attributes and methods from the Chemistry class.
	See Chemistry for detailed comments of attributes and methods.

	Additional methods and attributes are commented below.
	'''

	NUM_STATE_VARS = 6
	NUM_SPECIES  = 4
	PHYSICS_TYPE = general.PhysicsType.NavierStokesMultispecies1D_4sp_CH4
	CANTERA_FILENAME = "ch4_inert.xml"

	def set_maps(self):
		super().set_maps()

		d = {
			navierstokes_mult_fcn_type.DiffusionMixture : \
				navierstokes_mult_fcns.DiffusionMixture,
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
		rhoYCH4 = "\\rho Y_{CH4}"
		rhoYO2 = "\\rho Y_{O2}"
		rhoYH2O = "\\rho Y_{H2O}"

	class AdditionalVariables(Enum):
	    Pressure = "p"
	    Temperature = "T"
	    Entropy = "s"
	    InternalEnergy = "\\rho e"
	    TotalEnthalpy = "H"
	    SoundSpeed = "c"
	    MaxWaveSpeed = "\\lambda"
	    MassFractionCH4 = "Y_{CH4}"
	    MassFractionO2 = "Y_{O2}"
	    MassFractionH2O = "Y_{H2O}"
	    MassFractionN2 = "Y_{N2}"
	    SpecificHeatRatio = "\\gamma"

	def compute_additional_variable(self, var_name, Uq, flag_non_physical):
		''' Extract state variables '''
		srho = self.get_state_slice("Density")
		rho = Uq[:, :, srho]
		rhoYCH4 = Uq[:, :, [3]]
		rhoYO2 = Uq[:, :, [4]]
		rhoYH2O = Uq[:, :, [5]]
		rhoYN2 = rho * (1.0 - (rhoYO2 + rhoYH2O + rhoYCH4) / rho)

		''' Flag non-physical state '''
		if flag_non_physical:
			if np.any(rho < 0.):
				raise errors.NotPhysicalError

		''' Get final scalars '''
		vname = self.AdditionalVariables[var_name].name
		if vname is self.AdditionalVariables["MassFractionH2O"].name:
			varq = rhoYH2O/rho
		elif vname is self.AdditionalVariables["MassFractionO2"].name:
			varq = rhoYO2/rho
		elif vname is self.AdditionalVariables["MassFractionCH4"].name:
			varq = rhoYCH4/rho
		elif vname is self.AdditionalVariables["MassFractionN2"].name:
			varq = rhoYN2/rho
		else:
			varq = super().compute_additional_variable(var_name, Uq, 
					flag_non_physical)
		return varq

