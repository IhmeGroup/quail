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
#       File : src/physics/chemistry/chemistry.py
#
#       Contains class definitions for 1D Euler equations with a simple
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
from physics.base.functions import FcnType as base_fcn_type

import physics.euler.functions as euler_fcns
from physics.euler.functions import BCType as euler_BC_type
from physics.euler.functions import ConvNumFluxType as \
		euler_conv_num_flux_type
from physics.euler.functions import FcnType as euler_fcn_type
from physics.euler.functions import SourceType as euler_source_type

import physics.chemistry.functions as chemistry_fcns
from physics.chemistry.functions import FcnType as chemistry_fcn_type
from physics.chemistry.functions import SourceType as chemistry_source_type
from physics.chemistry.functions import ConvNumFluxType as \
		chemistry_conv_num_flux_type


class Chemistry(base.PhysicsBase):
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
	R: float
		mass-specific gas constant
	gamma: float
		specific heat ratio
	qo : float
		heat release
	'''
	PHYSICS_TYPE = general.PhysicsType.Chemistry

	def __init__(self, mesh):
		super().__init__(mesh)
		self.R = 0.
		self.gamma = 0.
		self.qo = 0.

	def set_maps(self):
		super().set_maps()

		self.BC_map.update({
			base_BC_type.StateAll : base_fcns.StateAll,
			base_BC_type.Extrapolate : base_fcns.Extrapolate,
			euler_BC_type.SlipWall : euler_fcns.SlipWall,
			euler_BC_type.PressureOutlet : euler_fcns.PressureOutlet,
		})

	def set_physical_params(self, GasConstant=287., SpecificHeatRatio = 1.4,
			HeatRelease = 0.):
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

	def compute_additional_variable(self, var_name, Uq, flag_non_physical):
		''' Extract state variables '''
		srho = self.get_state_slice("Density")
		srhoE = self.get_state_slice("Energy")
		srhoY = self.get_state_slice("Mixture")
		smom = self.get_momentum_slice()
		rho = Uq[:, :, srho]
		rhoE = Uq[:, :, srhoE]
		mom = Uq[:, :, smom]
		rhoY = Uq[:, :, srhoY]

		''' Unpack '''
		gamma = self.gamma
		R = self.R
		qo = self.qo

		''' Flag non-physical state '''
		if flag_non_physical:
			if np.any(rho < 0.):
				raise errors.NotPhysicalError

		''' Nested functions for common quantities '''
		def get_pressure():
			varq = (gamma - 1.)*(rhoE - 0.5*np.sum(mom*mom, axis=2,
					keepdims=True)/rho - qo*rhoY)
			if flag_non_physical:
				if np.any(varq < 0.):
					raise errors.NotPhysicalError
			return varq
		def get_temperature():
			return get_pressure()/(rho*R)

		''' Get final scalars '''
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
			varq = np.linalg.norm(mom, axis=2, keepdims=True)/rho + np.sqrt(
					gamma*get_pressure()/rho)
		elif vname is self.AdditionalVariables["MassFraction"].name:
			varq = rhoY/rho
		#NOTE: 1D only right now.
		elif vname is self.AdditionalVariables["Velocity"].name:
			varq = mom/rho
		else:
			raise NotImplementedError

		return varq


class Chemistry1D(Chemistry):
	'''
	This class corresponds to 1D Euler equations with simple chemistry.
	It inherits attributes and methods from the Chemistry class.
	See Chemistry for detailed comments of attributes and methods.

	Additional methods and attributes are commented below.
	'''
	NUM_STATE_VARS = 4
	NDIMS = 1

	def set_maps(self):
		super().set_maps()

		d = {
			chemistry_fcn_type.DensityWave : chemistry_fcns.DensityWave,
			chemistry_fcn_type.SimpleDetonation1 : \
					chemistry_fcns.SimpleDetonation1,
			chemistry_fcn_type.OverdrivenDetonation : \
					chemistry_fcns.OverdrivenDetonation,
		}

		self.IC_fcn_map.update(d)
		self.exact_fcn_map.update(d)
		self.BC_fcn_map.update(d)

		self.source_map.update({
			chemistry_source_type.Arrhenius : chemistry_fcns.Arrhenius,
			# chemistry_source_type.Heaviside : chemistry_fcns.Heaviside,

		})

		self.conv_num_flux_map.update({
		})

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

	def get_conv_flux_interior(self, Uq, x=None, t=None):

		irho, irhou, irhoE, irhoY = self.get_state_indices()

		rho = Uq[:, :, irho]
		rhou = Uq[:, :, irhou]
		rhoE = Uq[:, :, irhoE]
		rhoY = Uq[:, :, irhoY]

		# Get velocity
		u = rhou / rho
		# Get squared velocity
		u2 = u**2

		# Calculate pressure using the Ideal Gas Law
		p = (self.gamma - 1.)*(rhoE - 0.5 * rho * u2
				- rhoY * self.qo) # [n, nq]
		# Get total enthalpy
		H = rhoE + p

		F = np.empty(Uq.shape + (self.NDIMS,))
		F[:, :, irho, 0] = rhou
		F[:, :, irhou, 0] = rho * u2 + p
		F[:, :, irhoE, 0] = H * u
		F[:, :, irhoY, 0] = rhou*rhoY/rho

		return F, (u2, rho, p)

	def get_conv_eigenvectors(self, U_bar):
		'''
		This function defines the convective eigenvectors for the 
		1D euler equations with single-species chemistry.

		Note: These eigenvectors are defined based on the equation
			  of state chosen (see get_pressure in chemistry.py).

		A more general definition of multicomponent reacting flow 
		eigenvectors can be found in the following thesis by Fedkiw:

		[1] Fedkiw, R., “A survey of chemically reacting, compressible 
			flows,” Ph.D. Thesis, University of California Los Angeles,
			1997. 

		Inputs:
		------- 
			U_bar: Average state [ne, 1, ns]

		Outputs:
		--------
			right_eigen: Right eigenvector matrix [ne, 1, ns, ns]
			left_eigen: Left eigenvector matrix [ne, 1, ns, ns]
		'''

		# Unpack
		ne = U_bar.shape[0]
		
		ns = self.NUM_STATE_VARS
		
		irho, irhou, irhoE, irhoY = self.get_state_indices()

		rho = U_bar[:, :, irho]
		rhou = U_bar[:, :, irhou]
		rhoE = U_bar[:, :, irhoE]
		rhoY = U_bar[:, :, irhoY]
		
		# Get mass fraction
		Y = rhoY / rho

		# Get velocity
		u = rhou / rho
		# Get squared velocity
		u2 = u**2
		# Calculate pressure using the Ideal Gasd Law
		p = (self.gamma - 1.)*(rhoE - 0.5 * rho * u2 \
				- rhoY * self.qo) # [n, nq]
		# Get total enthalpy
		H = rhoE/rho + p/rho # CHECK : do I need to divide by rho here?

		# Get sound speed
		a = np.sqrt(self.gamma * p / rho)

		gm1oa2 = (self.gamma - 1.) / (a * a)

		# Allocate the right and left eigenvectors
		right_eigen = np.zeros([ne, 1, ns, ns])
		left_eigen = np.zeros([ne, 1, ns, ns])

		# Calculate the right and left eigenvectors
		right_eigen[:, :, irho, irho]  = 1.
		right_eigen[:, :, irho, irhou] = 1.
		right_eigen[:, :, irho, irhoE] = 1.
		right_eigen[:, :, irho, irhoY] = 0.

		right_eigen[:, :, irhou, irho]  = u - a
		right_eigen[:, :, irhou,irhou] = u + a
		right_eigen[:, :, irhou, irhoE] = u
		right_eigen[:, :, irhou, irhoY] = 0.

		right_eigen[:, :, irhoE, irho]  = H - u*a
		right_eigen[:, :, irhoE, irhou] = H + u*a
		right_eigen[:, :, irhoE, irhoE] = H - (1./gm1oa2)
		right_eigen[:, :, irhoE, irhoY] = -self.qo

		right_eigen[:, :, irhoY, irho]  = Y
		right_eigen[:, :, irhoY, irhou] = Y
		right_eigen[:, :, irhoY, irhoE] = Y
		right_eigen[:, :, irhoY, irhoY] = 1.

		b1 = gm1oa2
		b2 = 1.+b1*u2-b1*H
		b3 = b1*Y*(-self.qo)

		left_eigen[:, :, irho, irho]  = (b2/2.)+(u/(2.*a))+(b3/2.)
		left_eigen[:, :, irho, irhou] = (b2/2.)-(u/(2.*a))+(b3/2.)
		left_eigen[:, :, irho, irhoE] = 1-b2-b3
		left_eigen[:, :, irho, irhoY] = -Y

		left_eigen[:, :, irhou, irho]  = -b1*u/2. - 1./(2.*a)
		left_eigen[:, :, irhou, irhou] = -b1*u/2. + 1./(2.*a)
		left_eigen[:, :, irhou, irhoE] = b1*u
		left_eigen[:, :, irhou, irhoY] = 0.

		left_eigen[:, :, irhoE, irho]  = b1/2.
		left_eigen[:, :, irhoE, irhou] = b1/2.
		left_eigen[:, :, irhoE, irhoE] = -b1
		left_eigen[:, :, irhoE, irhoY] = 0.

		left_eigen[:, :, irhoY, irho]  = -b1*(-self.qo)/2.
		left_eigen[:, :, irhoY, irhou] = -b1*(-self.qo)/2.
		left_eigen[:, :, irhoY, irhoE] = b1*(-self.qo)
		left_eigen[:, :, irhoY, irhoY] = 1.

		left_eigen = left_eigen.transpose(0,1,3,2)

		# Can uncomment line below to test l dot r = kronecker delta
		# test = np.einsum('elij,eljk->elik', left_eigen, right_eigen)
		# import code; code.interact(local=locals())

		return right_eigen, left_eigen # [ne, 1, ns, ns]
