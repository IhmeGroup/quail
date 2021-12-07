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
#       File : src/physics/navierstokes/navierstokes.py
#
#       Contains class definitions for 1D and 2D Navier-Stokes equations.
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
from physics.base.functions import DiffNumFluxType as base_diff_num_flux_type
from physics.base.functions import FcnType as base_fcn_type

import physics.euler.euler as euler
import physics.euler.functions as euler_fcns
from physics.euler.functions import BCType as euler_BC_type
from physics.euler.functions import ConvNumFluxType as \
		euler_conv_num_flux_type
from physics.euler.functions import FcnType as euler_fcn_type
from physics.euler.functions import SourceType as euler_source_type

import physics.navierstokes.functions as navierstokes_fcns
from physics.navierstokes.functions import FcnType as navierstokes_fcn_type
from physics.navierstokes.functions import SourceType as \
		navierstokes_source_type


class NavierStokes(euler.Euler):
	'''
	This class corresponds to the compressible Navier-Stokes equations. 
	It inherits attributes and methods from the Euler class. See Euler
	for detailed comments of attributes and methods. This class should 
	not be instantiated directly. Instead, the 1D and 2D variants, which
	inherit from this class (see below), should be instantiated.

	Additional methods and attributes are commented below.

	Attributes:
	-----------
	R: float
		mass-specific gas constant
	gamma: float
		specific heat ratio
	'''
	PHYSICS_TYPE = general.PhysicsType.NavierStokes

	def __init__(self, mesh):
		super().__init__(mesh)
		self.R = 0.
		self.gamma = 0.

	def set_maps(self):
		super().set_maps()

		self.diff_num_flux_map.update({
			base_diff_num_flux_type.SIP : 
				base_fcns.SIP,
			})

	def set_physical_params(self, GasConstant=287., SpecificHeatRatio=1.4, 
			PrandtlNumber=0.7, Viscosity=1.0, s=1.0, T0=1.0, beta=1.5):
		'''
		This method sets physical parameters.

		Inputs:
		-------
			GasConstant: mass-specific gas constant
			SpecificHeatRatio: ratio of specific heats
			PrandtlNumber: ratio of kinematic viscosity to thermal diffusivity
			Viscosity: fluid viscosity
			s: Sutherland model constant
			T0: Sutherland model constant
			beta: Sutherland model constant
			
		Outputs:
		--------
			self: physical parameters set
		'''
		self.R = GasConstant
		self.gamma = SpecificHeatRatio
		self.Pr = PrandtlNumber
		self.mu0 = Viscosity
		self.s = s
		self.T0 = T0
		self.beta = beta


class NavierStokes1D(NavierStokes, euler.Euler1D):
	'''
	This class corresponds to 1D Navier Stokes equations.
	It inherits attributes and methods from the NavierStokes class.
	See NavierStokes for detailed comments of attributes and methods.

	Additional methods and attributes are commented below.
	'''
	NUM_STATE_VARS = 3
	NDIMS = 1

	def set_maps(self):
		super().set_maps()

	def get_diff_flux_interior(self, Uq, gUq):
		# Get indices/slices of state variables
		irho, irhou, irhoE = self.get_state_indices()
		smom = self.get_momentum_slice()

		# Unpack state coefficients
		rho  = Uq[:, :, irho]  # [n, nq]
		rhou = Uq[:, :, irhou] # [n, nq]
		rhoE = Uq[:, :, irhoE] # [n, nq]
		mom  = Uq[:, :, smom]  # [n, nq, ndims]

		# Calculate transport
		mu, kappa = self.get_transport(self, Uq, 
			flag_non_physical=False)
		mu = mu.reshape(rho.shape)
		kappa = kappa.reshape(rho.shape)
		nu = mu / rho

		gamma = self.gamma
		R = self.R

		# Set constants for stress tensor
		C1 = 2. / 3.
		C2 = (gamma - 1.) / (R * rho)
		
		# Separate x gradient
		gUx = gUq[:, :, :, 0] # [ne, nq, ns]

		# Get velocity
		u = rhou / rho
		
		# Get E
		E = rhoE / rho
		
		# Get squared velocity
		u2 = u**2

		# Store dTdU
		dTdU = np.zeros_like(gUx)
		dTdU[:, :, 0] = C2 * (-E + u2)
		dTdU[:, :, 1] = C2 * -u
		dTdU[:, :, 2] = C2

		# Get the stress tensor (use product rules to write in 
		# terms of the conservative gradients)
		rhodiv = (gUx[:, :, 1] - u * gUx[:, :, 0])
		tauxx = nu * (2. * (gUx[:, :, 1] - u * gUx[:, :, 0]) - \
			C1 * rhodiv)

		# Assemble flux matrix
		F = np.empty(Uq.shape + (self.NDIMS,)) # [n, nq, ns, ndims]
		F[:,:,irho,  :] = 0.		   # flux of rho 
		F[:,:,irhou, 0] = tauxx 	# flux of momentum
		F[:,:,irhoE, 0] = u * tauxx + \
			kappa * np.einsum('ijk, ijk -> ij', dTdU, gUx)

		return F # [n, nq, ns, ndims]

class NavierStokes2D(NavierStokes, euler.Euler2D):
	'''
	This class corresponds to 2D Navier-Stokes equations. It 
	inherits attributes and methods from the Navier-Stokes class as 
	well as the Euler2D class.
	See Navier-Stokes and Euler2D for detailed comments of 
	attributes and methods.

	Additional methods and attributes are commented below.
	'''
	NUM_STATE_VARS = 4
	NDIMS = 2

	def __init__(self, mesh):
		super().__init__(mesh)

	def set_maps(self):
		super().set_maps()

		d = {
			navierstokes_fcn_type.TaylorGreenVortexNS : 
					navierstokes_fcns.TaylorGreenVortexNS,
			navierstokes_fcn_type.ManufacturedSolution : 
					navierstokes_fcns.ManufacturedSolution,
		}

		self.IC_fcn_map.update(d)
		self.exact_fcn_map.update(d)
		self.BC_fcn_map.update(d)

		self.source_map.update({
			navierstokes_source_type.ManufacturedSource : 
					navierstokes_fcns.ManufacturedSource,
		})

	def get_diff_flux_interior(self, Uq, gUq):
		# Get indices/slices of state variables
		irho, irhou, irhov, irhoE = self.get_state_indices()
		smom = self.get_momentum_slice()

		# Unpack state coefficients
		rho  = Uq[:, :, irho]  # [n, nq]
		rhou = Uq[:, :, irhou] # [n, nq]
		rhov = Uq[:, :, irhov] # [n, nq]
		rhoE = Uq[:, :, irhoE] # [n, nq]
		mom  = Uq[:, :, smom]  # [n, nq, ndims]

		# Calculate transport
		mu, kappa = self.get_transport(self, Uq, 
			flag_non_physical=False)
		mu = mu.reshape(rho.shape)
		kappa = kappa.reshape(rho.shape)
		nu = mu / rho

		gamma = self.gamma
		R = self.R

		# Set constants for stress tensor
		C1 = 2. / 3.
		C2 = (gamma - 1.) / (R * rho)
		
		# Separate x and y gradients
		gUx = gUq[:, :, :, 0] # [ne, nq, ns]
		gUy = gUq[:, :, :, 1] # [ne, nq, ns]

		# Get velocity in each dimension
		u = rhou / rho
		v = rhov / rho
		
		# Get E
		E = rhoE / rho
		
		# Get squared velocities
		u2 = u**2
		v2 = v**2

		# Store dTdU
		dTdU = np.zeros_like(gUx)
		dTdU[:, :, 0] = C2 * (-E + u2 + v2)
		dTdU[:, :, 1] = C2 * -u
		dTdU[:, :, 2] = C2 * -v
		dTdU[:, :, 3] = C2

		# Get the stress tensor (use product rules to write in 
		# terms of the conservative gradients)
		rhodiv = ((gUx[:, :, 1] - u * gUx[:, :, 0]) + \
			(gUy[:, :, 2] - v * gUy[:, :, 0]))
		tauxx = nu * (2. * (gUx[:, :, 1] - u * gUx[:, :, 0]) - \
			C1 * rhodiv)
		tauxy = nu * ((gUy[:, :, 1] - u * gUy[:, :, 0]) + \
			(gUx[:, :, 2] - v * gUx[:, :, 0]))
		tauyy = nu * (2. * (gUy[:, :, 2] - v * gUy[:, :, 0]) - \
			C1 * rhodiv)

		# Assemble flux matrix
		F = np.empty(Uq.shape + (self.NDIMS,)) # [n, nq, ns, ndims]
		F[:,:,irho,  :] = 0.		   # x,y-flux of rho (zero both dir)

		# x-direction
		F[:,:,irhou, 0] = tauxx 	# x-flux of x-momentum
		F[:,:,irhov, 0] = tauxy     # x-flux of y-momentum
		F[:,:,irhoE, 0] = u * tauxx + v * tauxy + \
			kappa * np.einsum('ijk, ijk -> ij', dTdU, gUx)

		# y-direction
		F[:,:,irhou, 1] = tauxy        # y-flux of x-momentum
		F[:,:,irhov, 1] = tauyy 	   # y-flux of y-momentum
		F[:,:,irhoE, 1] = u * tauxy + v * tauyy + \
			kappa * np.einsum('ijk, ijk -> ij', dTdU, gUy)

		return F # [n, nq, ns, ndims]