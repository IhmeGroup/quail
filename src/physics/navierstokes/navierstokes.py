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
from physics.navierstokes.functions import BCType as navierstokes_BC_type
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

	def __init__(self):
		super().__init__()
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

	def __init__(self):
		super().__init__()

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
		
		
		
class Twophase(NavierStokes2D, euler.Euler2D):
	'''
	This class corresponds to 2D Two-phase Navier-Stokes equations. It
	inherits attributes and methods from the Navier-Stokes2D class as
	well as the Euler2D class.

	Additional methods and attributes are commented below.
	'''
	NUM_STATE_VARS = 7
	NDIMS = 2
	PHYSICS_TYPE = general.PhysicsType.Twophase

	def __init__(self):
		super().__init__()
		self.gamma1 = 0.
		self.gamma2 = 0.
		self.pinf1 = 0.
		self.pinf2 = 0.
		self.mu1 = 0.
		self.mu2 = 0.
		self.kappa1 = 0.
		self.kappa2 = 0.
		self.rho01 = 0.
		self.rho02 = 0.
		self.eps = 0.
		self.scl_eps = 0.
		self.gam = 1.
		self.switch = 0.
		self.g = 0.
		self.sigma = 0.
		self.dt_LS = 0.
		self.iter_LS = 0.
		self.cp1 = 0.
		self.cp2 = 0.
		self.mdot = 0.
		
	def set_maps(self):
		super().set_maps()

		d = {
			navierstokes_fcn_type.Bubble :
					navierstokes_fcns.Bubble,
			navierstokes_fcn_type.RayleighTaylor :
					navierstokes_fcns.RayleighTaylor,
			navierstokes_fcn_type.Channel :
					navierstokes_fcns.Channel,
			navierstokes_fcn_type.Rising_bubble :
					navierstokes_fcns.Rising_bubble,
		}

		self.IC_fcn_map.update(d)
		self.exact_fcn_map.update(d)
		self.BC_fcn_map.update(d)

		self.source_map.update({
			navierstokes_source_type.BubbleSource :
					navierstokes_fcns.BubbleSource,
		})
		
		self.BC_map.update({
			base_BC_type.StateAll : base_fcns.StateAll,
			base_BC_type.Extrapolate : base_fcns.Extrapolate,
			navierstokes_BC_type.NoSlipWall : navierstokes_fcns.NoSlipWall,
			navierstokes_BC_type.Subsonic_Inlet : navierstokes_fcns.Subsonic_Inlet,
			navierstokes_BC_type.Subsonic_Outlet : navierstokes_fcns.Subsonic_Outlet,
			#navierstokes_BC_type.PressureOutletNS : navierstokes_fcns.PressureOutletNS
		})
		
	def set_physical_params(self, gamma1=1., gamma2=1., mu1=1., mu2=1., \
			kappa1=1., kappa2=1., pinf1=1., pinf2=1., rho01=1., rho02=1.,\
			eps=0., scl_eps=1.0, gam=1.0, switch=1., g=0., sigma=0., dt_LS=2., iter_LS=200, \
			cp1=1., cp2=1., mdot = 0.):
		'''
		This method sets physical parameters.

		Inputs:
		-------
		Outputs:
		--------
		'''
		self.gamma1 = gamma1
		self.gamma2 = gamma2
		self.pinf1 = pinf1
		self.pinf2 = pinf2
		self.mu1 = mu1
		self.mu2 = mu2
		self.kappa1 = kappa1
		self.kappa2 = kappa2
		self.rho01 = rho01
		self.rho02 = rho02
		self.eps = eps
		self.scl_eps = scl_eps
		self.gam = gam
		self.switch = switch
		self.g = g
		self.sigma = sigma
		self.dt_LS = dt_LS
		self.iter_LS = iter_LS
		self.cp1 = cp1
		self.cp2 = cp2
		self.mdot = mdot

	class StateVariables(Enum):
		Density1 = "\\rho1 \\phi1"
		Density2 = "\\rho2 \\phi2"
		XMomentum = "\\rho u"
		YMomentum = "\\rho v"
		Energy = "\\rho E"
		PhaseField = "\\phi"
		LevelSet = "\\psi"
		
	class AdditionalVariables(Enum):
		Pressure = "p"
		XVelocity = "v_x"
		YVelocity = "v_y"
		Velocity = "v"
		Density = "\\rho"
		Gamma = "\\gamma"
		MaxWaveSpeed = "\\lambda"
		Temperature = "T"
		SoundSpeed = "c"
		Mach = "Mach"
		Divergence = "\\nabla \\cdot u"
		LevelSet2 = "\\psi+0.5"
		
	def get_state_indices(self):
		irho1phi1 = self.get_state_index("Density1")
		irho2phi2 = self.get_state_index("Density2")
		irhou = self.get_state_index("XMomentum")
		irhov = self.get_state_index("YMomentum")
		irhoE = self.get_state_index("Energy")
		iPF = self.get_state_index("PhaseField")
		iLS = self.get_state_index("LevelSet")

		return irho1phi1, irho2phi2, irhou, irhov, irhoE, iPF, iLS


	def get_conv_flux_interior(self, Uq, gUq, x=None, t=None):
		# Get indices/slices of state variables
		irho1phi1, irho2phi2, irhou, irhov, irhoE, iPF, iLS = self.get_state_indices()

		rho1phi1  = Uq[:, :, irho1phi1] # [n, nq]
		rho2phi2  = Uq[:, :, irho2phi2] # [n, nq]
		rhou      = Uq[:, :, irhou]     # [n, nq]
		rhov      = Uq[:, :, irhov]     # [n, nq]
		rhoE      = Uq[:, :, irhoE]     # [n, nq]
		phi1      = Uq[:, :, iPF]       # [n, nq]
		LS        = Uq[:, :, iLS]       # [n, nq]
		
		gLS = gUq[:,:,iLS,:]
		n = np.zeros(gLS.shape)
		mag = np.sqrt(gLS[:,:,0]**2+gLS[:,:,1]**2)
		n[:,:,0] = gLS[:,:,0]/(mag+1e-16)
		n[:,:,1] = gLS[:,:,1]/(mag+1e-16)

		# Get velocity in each dimension
		rho  = rho1phi1 + rho2phi2
		u = rhou / rho
		v = rhov / rho
		# Get squared velocities
		u2 = u**2
		v2 = v**2
		mag = np.sqrt(u2+v2)
		gam = self.gam*np.max(mag)
		#gam = mag
		#gam = 0.
		k = 0.5*(u2 + v2)
		
		# Calculate transport
		gamma1=self.gamma1
		gamma2=self.gamma2
		pinf1=self.pinf1
		pinf2=self.pinf2
		rho01=self.rho01
		rho02=self.rho02
		
		switch = self.switch

		# Stiffened gas EOS
		# Get properties of the fluid: gamma_l, pinf_l
		rhoe = (rhoE - 0.5 * rho * (u2 + v2)) # [n, nq]
		one_over_gamma_m1 = phi1/(gamma1-1.0) + (1.0-phi1)/(gamma2-1.0)
		p = (rhoe + (-phi1*gamma1*pinf1/(gamma1-1.)-(1.0-phi1)*gamma2*pinf2/(gamma2-1.)))/one_over_gamma_m1

		# Get off-diagonal momentum
		rhouv = rho * u * v
		# Get total enthalpy
		H = rhoE + p

		# Correction terms
		a1x = -gam*phi1*(1.0-phi1)*n[:,:,0]
		a1y = -gam*phi1*(1.0-phi1)*n[:,:,1]

		fx = (rho01-rho02)*a1x
		fy = (rho01-rho02)*a1y
		
		h1 = (p + pinf1)*gamma1/(gamma1-1.0)
		h2 = (p + pinf2)*gamma2/(gamma2-1.0)

		# Assemble flux matrix (missing a correction term in energy equation)
		F = np.empty(Uq.shape + (self.NDIMS,)) # [n, nq, ns, ndims]
		F[:,:,irho1phi1,  0] = u * rho1phi1 - rho01*a1x            # x-mass1 flux
		F[:,:,irho1phi1,  1] = v * rho1phi1 - rho01*a1y            # y-mass1 flux
		F[:,:,irho2phi2,  0] = u * rho2phi2 + rho02*a1x            # x-mass2 flux
		F[:,:,irho2phi2,  1] = v * rho2phi2 + rho02*a1y            # y-mass2 flux
		F[:,:,irhou,      0] = rho * u2 + p - fx * u               # x-flux of x-momentum
		F[:,:,irhov,      0] = rhouv        - fx * v               # x-flux of y-momentum
		F[:,:,irhou,      1] = rhouv        - fy * u               # y-flux of x-momentum
		F[:,:,irhov,      1] = rho * v2 + p - fy * v               # y-flux of y-momentum
		F[:,:,irhoE,      0] = H * u        - fx * k - (h1-h2)*a1x # x-flux of energy
		F[:,:,irhoE,      1] = H * v        - fy * k - (h1-h2)*a1y # y-flux of energy
		F[:,:,iPF,        0] = - a1x                               # x-flux of phi1
		F[:,:,iPF,        1] = - a1y                               # y-flux of phi1
		F[:,:,iLS,        0] = 0.                                  # x-flux of Levelset
		F[:,:,iLS,        1] = 0.                                  # y-flux of Levelset
		
		F = F*switch

		return F, (u2, v2, rho, p)


	def get_diff_flux_interior(self, Uq, gUq, x, t, epsilon):
		# Get indices/slices of state variables
		irho1phi1, irho2phi2, irhou, irhov, irhoE, iPF, iLS = self.get_state_indices()

		rho1phi1  = Uq[:, :, irho1phi1] # [n, nq]
		rho2phi2  = Uq[:, :, irho2phi2] # [n, nq]
		rhou      = Uq[:, :, irhou]     # [n, nq]
		rhov      = Uq[:, :, irhov]     # [n, nq]
		rhoE      = Uq[:, :, irhoE]     # [n, nq]
		phi1      = Uq[:, :, iPF]       # [n, nq]
		LS        = Uq[:, :, iLS]       # [n, nq]

		# Calculate transport
		mu1=self.mu1
		mu2=self.mu2
		kappa1=self.kappa1
		kappa2=self.kappa2
		gamma1=self.gamma1
		gamma2=self.gamma2
		pinf1=self.pinf1
		pinf2=self.pinf2
		rho01=self.rho01
		rho02=self.rho02
		
		switch = self.switch
			
		mu    = mu1   *phi1 + mu2   *(1.0-phi1)
		kappa = kappa1*phi1 + kappa2*(1.0-phi1)
		rho   = rho1phi1    + rho2phi2

		# Separate x and y gradients
		gUx = gUq[:, :, :, 0] # [ne, nq, ns]
		gUy = gUq[:, :, :, 1] # [ne, nq, ns]

		# Get velocity in each dimension
		u = rhou / rho
		v = rhov / rho
		
		# Get squared velocities
		u2 = u**2
		v2 = v**2
		mag = np.sqrt(u2+v2)
		gam = self.gam*np.max(mag)
#		gam = mag
#		gam = 0.
		k = 0.5*(u2 + v2)

		# Get the stress tensor (use product rules to write in
		# terms of the conservative gradients)
		dudx = (gUx[:,:,2] - u * (gUx[:,:,0] + gUx[:,:,1]))/rho
		dudy = (gUy[:,:,2] - u * (gUy[:,:,0] + gUy[:,:,1]))/rho
		dvdx = (gUx[:,:,3] - v * (gUx[:,:,0] + gUx[:,:,1]))/rho
		dvdy = (gUy[:,:,3] - v * (gUy[:,:,0] + gUy[:,:,1]))/rho
		
		tauxx = 2.0*mu*(dudx - 1.0/2.0*(dudx + dvdy))
		tauxy = 1.0*mu*(dudy + dvdx)
		tauyy = 2.0*mu*(dvdy - 1.0/2.0*(dudx + dvdy))
		
		# Correction terms
		eps = self.scl_eps*self.eps
		a1x = gam*eps*gUx[:,:,iPF]
		a1y = gam*eps*gUy[:,:,iPF]
		
		# Stiffened gas EOS
		rhoe = (rhoE - 0.5 * rho * (u2 + v2)) # [n, nq]
		one_over_gamma_m1 = phi1/(gamma1-1.0) + (1.0-phi1)/(gamma2-1.0)
		p = (rhoe + (-phi1*gamma1*pinf1/(gamma1-1.)-(1.0-phi1)*gamma2*pinf2/(gamma2-1.)))/one_over_gamma_m1
		
		fx = (rho01-rho02)*a1x
		fy = (rho01-rho02)*a1y
		
		h1 = (p + pinf1)*gamma1/(gamma1-1.0)
		h2 = (p + pinf2)*gamma2/(gamma2-1.0)

		# Assemble flux matrix
		F = np.empty(Uq.shape + (self.NDIMS,)) # [n, nq, ns, ndims]

		# x-direction
		F[:,:,irho1phi1,  0] =  rho01*a1x                # x-mass1 flux
		F[:,:,irho2phi2,  0] = -rho02*a1x                # x-mass2 flux
		F[:,:,irhou,      0] = tauxx + fx * u            # x-flux of x-momentum
		F[:,:,irhov,      0] = tauxy + fx * v            # x-flux of y-momentum
		F[:,:,irhoE,      0] = u * tauxx + v * tauxy  \
			+ fx * k + (h1-h2)*a1x                       # x-flux of energy

		# y-direction
		F[:,:,irho1phi1,  1] =  rho01*a1y                # y-mass1 flux
		F[:,:,irho2phi2,  1] = -rho02*a1y                # y-mass2 flux
		F[:,:,irhou,      1] = tauxy + fy * u            # y-flux of x-momentum
		F[:,:,irhov,      1] = tauyy + fy * v            # y-flux of y-momentum
		F[:,:,irhoE,      1] = u * tauxy + v * tauyy  \
			+ fy * k + (h1-h2)*a1y                       # y-flux of energy
			
		# phase field and level set
		F[:,:,iPF,  0]  = a1x # x-flux of phi1
		F[:,:,iPF,  1]  = a1y # y-flux of phi1
		
		F = F*switch
		
		F[:,:,iLS,  0]  =  0.75*self.eps*gUx[:,:,iLS]*(1.0-switch)
		F[:,:,iLS,  1]  =  0.75*self.eps*gUy[:,:,iLS]*(1.0-switch)

		return F # [n, nq, ns, ndims]

	def compute_additional_variable(self, var_name, Uq, gUq, flag_non_physical, x, t):
		sname = self.AdditionalVariables[var_name].name
		# Get indices/slices of state variables
		irho1phi1, irho2phi2, irhou, irhov, irhoE, iPF, iLS = self.get_state_indices()

		rho1phi1  = Uq[:, :, irho1phi1] # [n, nq]
		rho2phi2  = Uq[:, :, irho2phi2] # [n, nq]
		rhou      = Uq[:, :, irhou]     # [n, nq]
		rhov      = Uq[:, :, irhov]     # [n, nq]
		rhoE      = Uq[:, :, irhoE]     # [n, nq]
		phi1      = Uq[:, :, iPF]       # [n, nq]
		LS        = Uq[:, :, iLS]       # [n, nq]
		

		if sname is self.AdditionalVariables["Pressure"].name:
			# Calculate transport
			gamma1=self.gamma1
			gamma2=self.gamma2
			pinf1=self.pinf1
			pinf2=self.pinf2

			rho   = rho1phi1 + rho2phi2
			# Get velocity in each dimension
			u = rhou / rho
			v = rhov / rho
			u2 = u**2
			v2 = v**2
			rhoe = (rhoE - 0.5 * rho * (u2 + v2)) # [n, nq]
			one_over_gamma_m1 = phi1/(gamma1-1.0) + (1.0-phi1)/(gamma2-1.0)
			gamma = (one_over_gamma_m1+1.0)/one_over_gamma_m1
			pinf = (gamma-1.0)/gamma*(phi1*gamma1*pinf1/(gamma1-1.0) + (1.0-phi1)*gamma2*pinf2/(gamma2-1.0))
			#p = rhoe/one_over_gamma_m1 - gamma*pinf
			p = (rhoe + (-phi1*gamma1*pinf1/(gamma1-1.)-(1.0-phi1)*gamma2*pinf2/(gamma2-1.)))/one_over_gamma_m1
			scalar = p
		elif sname is self.AdditionalVariables["XVelocity"].name:
			rho   = rho1phi1 + rho2phi2
			# Get velocity in each dimension
			scalar = rhou / rho
		elif sname is self.AdditionalVariables["YVelocity"].name:
			rho   = rho1phi1 + rho2phi2
			# Get velocity in each dimension
			scalar = rhov / rho
		elif sname is self.AdditionalVariables["Velocity"].name:
			rho   = rho1phi1 + rho2phi2
			# Get velocity in each dimension
			v = rhov / rho
			u = rhou / rho
			scalar = np.sqrt(u**2+v**2)
		elif sname is self.AdditionalVariables["Density"].name:
			rho   = rho1phi1 + rho2phi2
			# Get velocity in each dimension
			scalar = rho
		elif sname is self.AdditionalVariables["Gamma"].name:
			gamma1=self.gamma1
			gamma2=self.gamma2
			one_over_gamma = phi1/(gamma1-1.0) + (1.0-phi1)/(gamma2-1.0)
			gamma = (one_over_gamma+1.0)/one_over_gamma
			# Get velocity in each dimension
			scalar = gamma
		elif sname is self.AdditionalVariables["MaxWaveSpeed"].name:
			gamma1=self.gamma1
			gamma2=self.gamma2
			pinf1=self.pinf1
			pinf2=self.pinf2

			rho   = rho1phi1 + rho2phi2
			one_over_gamma = phi1/(gamma1-1.0) + (1.0-phi1)/(gamma2-1.0)
			gamma = (one_over_gamma+1.0)/one_over_gamma
			# Get velocity in each dimension
			u = rhou / rho
			v = rhov / rho
			u2 = u**2
			v2 = v**2
			rhoe = (rhoE - 0.5 * rho * (u2 + v2)) # [n, nq]
			one_over_gamma_m1 = phi1/(gamma1-1.0) + (1.0-phi1)/(gamma2-1.0)
			gamma = (one_over_gamma_m1+1.0)/one_over_gamma_m1
			pinf = (gamma-1.0)/gamma*(phi1*gamma1*pinf1/(gamma1-1.0) + (1.0-phi1)*gamma2*pinf2/(gamma2-1.0))
			p = (rhoe + (-phi1*gamma1*pinf1/(gamma1-1.)-(1.0-phi1)*gamma2*pinf2/(gamma2-1.)))/one_over_gamma_m1
			# Wood speed of sound
			csq1rho1 = gamma1*(p + pinf1)
			csq2rho2 = gamma2*(p + pinf2)
			one_over_rho_csq = phi1/csq1rho1 + (1.0-phi1)/csq2rho2
			cw = np.sqrt(1.0/(rho*one_over_rho_csq))
			
			c1 = gamma1*(p + pinf1)/self.rho01
			c2 = gamma2*(p + pinf2)/self.rho02
			cw = np.sqrt(phi1*c1 + (1.-phi1)*c2)
			maxwave = cw + np.sqrt(u2+v2)
			scalar = maxwave
		elif sname is self.AdditionalVariables["SoundSpeed"].name:
			gamma1=self.gamma1
			gamma2=self.gamma2
			pinf1=self.pinf1
			pinf2=self.pinf2

			rho   = rho1phi1 + rho2phi2
			one_over_gamma = phi1/(gamma1-1.0) + (1.0-phi1)/(gamma2-1.0)
			gamma = (one_over_gamma+1.0)/one_over_gamma
			# Get velocity in each dimension
			u = rhou / rho
			v = rhov / rho
			u2 = u**2
			v2 = v**2
			rhoe = (rhoE - 0.5 * rho * (u2 + v2)) # [n, nq]
			one_over_gamma_m1 = phi1/(gamma1-1.0) + (1.0-phi1)/(gamma2-1.0)
			gamma = (one_over_gamma_m1+1.0)/one_over_gamma_m1
			pinf = (gamma-1.0)/gamma*(phi1*gamma1*pinf1/(gamma1-1.0) + (1.0-phi1)*gamma2*pinf2/(gamma2-1.0))
			p = (rhoe + (-phi1*gamma1*pinf1/(gamma1-1.)-(1.0-phi1)*gamma2*pinf2/(gamma2-1.)))/one_over_gamma_m1
			# Wood speed of sound
			csq1rho1 = gamma1*(p + pinf1)
			csq2rho2 = gamma2*(p + pinf2)
			one_over_rho_csq = phi1/csq1rho1 + (1.0-phi1)/csq2rho2
			cw = np.sqrt(1.0/(rho*one_over_rho_csq))
			scalar = cw
		elif sname is self.AdditionalVariables["Temperature"].name:
#			gamma1=self.gamma1
#			gamma2=self.gamma2
#			pinf1=self.pinf1
#			pinf2=self.pinf2
#			cp1=self.cp1
#			cp2=self.cp2
#
#			rho   = rho1phi1 + rho2phi2
#			one_over_gamma = phi1/(gamma1-1.0) + (1.0-phi1)/(gamma2-1.0)
#			cp = cp1*phi1 + cp2*(1.0-phi1)
#			gamma = (one_over_gamma+1.0)/one_over_gamma
#			# Get velocity in each dimension
#			u = rhou / rho
#			v = rhov / rho
#			u2 = u**2
#			v2 = v**2
#			rhoe = (rhoE - 0.5 * rho * (u2 + v2)) # [n, nq]
#			one_over_gamma_m1 = phi1/(gamma1-1.0) + (1.0-phi1)/(gamma2-1.0)
#			gamma = (one_over_gamma_m1+1.0)/one_over_gamma_m1
#			pinf = (gamma-1.0)/gamma*(phi1*gamma1*pinf1/(gamma1-1.0) + (1.0-phi1)*gamma2*pinf2/(gamma2-1.0))
#			p = (rhoe + (-phi1*gamma1*pinf1/(gamma1-1.)-(1.0-phi1)*gamma2*pinf2/(gamma2-1.)))/one_over_gamma_m1
#			T = 1.0/(rho*cp)*((p+pinf)*one_over_gamma_m1)
#			scalar = T
			scalar = 0.
		elif sname is self.AdditionalVariables["Mach"].name:
			gamma1=self.gamma1
			gamma2=self.gamma2
			pinf1=self.pinf1
			pinf2=self.pinf2

			rho   = rho1phi1 + rho2phi2
			one_over_gamma = phi1/(gamma1-1.0) + (1.0-phi1)/(gamma2-1.0)
			gamma = (one_over_gamma+1.0)/one_over_gamma
			# Get velocity in each dimension
			u = rhou / rho
			v = rhov / rho
			u2 = u**2
			v2 = v**2
			rhoe = (rhoE - 0.5 * rho * (u2 + v2)) # [n, nq]
			one_over_gamma_m1 = phi1/(gamma1-1.0) + (1.0-phi1)/(gamma2-1.0)
			gamma = (one_over_gamma_m1+1.0)/one_over_gamma_m1
			pinf = (gamma-1.0)/gamma*(phi1*gamma1*pinf1/(gamma1-1.0) + (1.0-phi1)*gamma2*pinf2/(gamma2-1.0))
			p = (rhoe + (-phi1*gamma1*pinf1/(gamma1-1.)-(1.0-phi1)*gamma2*pinf2/(gamma2-1.)))/one_over_gamma_m1
			# Wood speed of sound
			csq1rho1 = gamma1*(p + pinf1)
			csq2rho2 = gamma2*(p + pinf2)
			one_over_rho_csq = phi1/csq1rho1 + (1.0-phi1)/csq2rho2
			cw = np.sqrt(1.0/(rho*one_over_rho_csq))
			scalar = np.sqrt(u2+v2)/cw
		elif sname is self.AdditionalVariables["Divergence"].name:
			# Separate x and y gradients
			gUx = gUq[:, :, :, 0] # [ne, nq, ns]
			gUy = gUq[:, :, :, 1] # [ne, nq, ns]
		
			# Get velocity in each dimension
			rho   = rho1phi1 + rho2phi2
			u = rhou / rho
			v = rhov / rho
		
			rhodudx = gUx[:,:,irhou] - u * (gUx[:,:,irho1phi1] + gUx[:,:,irho2phi2])
			rhodvdy = gUy[:,:,irhov] - v * (gUy[:,:,irho1phi1] + gUy[:,:,irho2phi2])
		
			scalar = (rhodudx + rhodvdy)/rho
		elif sname is self.AdditionalVariables["LevelSet2"].name:
			scalar = LS+0.5
		else:
			raise NotImplementedError

		return scalar

class EDAC2D(NavierStokes2D, euler.Euler2D):
	'''
	This class corresponds to 2D Two-phase Navier-Stokes equations. It
	inherits attributes and methods from the Navier-Stokes2D class as
	well as the Euler2D class.

	Additional methods and attributes are commented below.
	'''
	NUM_STATE_VARS = 5
	NDIMS = 2
	PHYSICS_TYPE = general.PhysicsType.EDAC2D

	def __init__(self):
		super().__init__()
		self.mu = 0.
		self.rho1 = 0.
		self.rho2 = 0.
		self.cs = 0.
		self.nuP = 0.
		
		self.eps = 0.
		self.switch = 0.
		
	def set_maps(self):
		super().set_maps()

		d = {
			navierstokes_fcn_type.Bubble2 :
					navierstokes_fcns.Bubble2
		}

		self.IC_fcn_map.update(d)
		self.exact_fcn_map.update(d)
		self.BC_fcn_map.update(d)

		self.source_map.update({
			navierstokes_source_type.BubbleSource2 :
					navierstokes_fcns.BubbleSource2,
		})
		
	def set_physical_params(self, mu=1., rho1=1., rho2=1., eps=0., cs=0.,nuP=0.):
		'''
		This method sets physical parameters.

		Inputs:
		-------
		Outputs:
		--------
		'''
		self.mu = mu
		self.rho1 = rho1
		self.rho2 = rho2
		self.eps = eps
		self.cs = cs
		self.nuP = nuP

	class StateVariables(Enum):
		XMomentum = "\\rho u"
		YMomentum = "\\rho v"
		Pressure = "p"
		PhaseField = "\\phi"
		LevelSet = "\\psi"
		
	class AdditionalVariables(Enum):
		XVelocity = "v_x"
		YVelocity = "v_y"
		MaxWaveSpeed = "\\lambda"
		
	def get_state_indices(self):
		irhou = self.get_state_index("XMomentum")
		irhov = self.get_state_index("YMomentum")
		ip    = self.get_state_index("Pressure")
		iPF = self.get_state_index("PhaseField")
		iLS = self.get_state_index("LevelSet")

		return irhou, irhov, ip, iPF, iLS


	def get_conv_flux_interior(self, Uq, gUq, x=None, t=None):
		# Get indices/slices of state variables
		irhou, irhov, ip, iPF, iLS = self.get_state_indices()

		rhou      = Uq[:, :, irhou]     # [n, nq]
		rhov      = Uq[:, :, irhov]     # [n, nq]
		p         = Uq[:, :, ip]        # [n, nq]
		phi1      = Uq[:, :, iPF]       # [n, nq]
		LS        = Uq[:, :, iLS]       # [n, nq]
		
		gLS = gUq[:,:,iPF,:]
		n = np.zeros(gLS.shape)
		mag = np.sqrt(gLS[:,:,0]**2+gLS[:,:,1]**2)
		n[:,:,0] = gLS[:,:,0]/(mag+1e-16)
		n[:,:,1] = gLS[:,:,1]/(mag+1e-16)

		# Get velocity in each dimension
		rho1 = self.rho1
		rho2 = self.rho2
		rho  = rho1*phi1 + rho2*(1.0-phi1)
		u = rhou / rho
		v = rhov / rho
		# Get squared velocities
		u2 = u**2
		v2 = v**2
		mag = np.sqrt(u2+v2)
		gam = np.max(mag)

		# Get off-diagonal momentum
		rhouv = rho * u * v

		# Correction terms
		a1x = -gam*phi1*(1.0-phi1)*n[:,:,0]
		a1y = -gam*phi1*(1.0-phi1)*n[:,:,1]
		
		cs = self.cs
		c2 = cs**2

		# Assemble flux matrix (missing a correction term in energy equation)
		F = np.empty(Uq.shape + (self.NDIMS,)) # [n, nq, ns, ndims]
		F[:,:,irhou, 0] = rho * u2 + p     # x-flux of x-momentum
		F[:,:,irhov, 0] = rhouv            # x-flux of y-momentum
		F[:,:,irhou, 1] = rhouv            # y-flux of x-momentum
		F[:,:,irhov, 1] = rho * v2 + p     # y-flux of y-momentum
		F[:,:,ip,    0] = p * u + c2*rhou  # x-flux of pressure
		F[:,:,ip,    1] = p * v + c2*rhov  # y-flux of pressure
		F[:,:,iPF,   0] = u * phi1 - a1x   # x-flux of phi1
		F[:,:,iPF,   1] = v * phi1 - a1y   # y-flux of phi1
		F[:,:,iLS,   0] = u * LS           # x-flux of Levelset
		F[:,:,iLS,   1] = v * LS           # y-flux of Levelset

		return F, (u2, v2, rho, p)


	def get_diff_flux_interior(self, Uq, gUq, x, t, epsilon):
		# Get indices/slices of state variables
		irhou, irhov, ip, iPF, iLS = self.get_state_indices()

		rhou      = Uq[:, :, irhou]     # [n, nq]
		rhov      = Uq[:, :, irhov]     # [n, nq]
		p         = Uq[:, :, ip]     # [n, nq]
		phi1      = Uq[:, :, iPF]       # [n, nq]
		LS        = Uq[:, :, iLS]       # [n, nq]

		# Separate x and y gradients
		gUx = gUq[:, :, :, 0] # [ne, nq, ns]
		gUy = gUq[:, :, :, 1] # [ne, nq, ns]

		# Get velocity in each dimension
		rho1 = self.rho1
		rho2 = self.rho2
		rho  = rho1*phi1 + rho2*(1.0-phi1)
		u = rhou / rho
		v = rhov / rho
		
		nuP = self.nuP
		
		# Get squared velocities
		u2 = u**2
		v2 = v**2
		mag = np.sqrt(u2+v2)
		gam = np.max(mag)
		
		# Correction terms
		eps = self.eps
		a1x = gam*eps*gUx[:,:,iPF]
		a1y = gam*eps*gUy[:,:,iPF]

		# Assemble flux matrix
		F = np.empty(Uq.shape + (self.NDIMS,)) # [n, nq, ns, ndims]

		# x-direction
		F[:,:,irhou, 0] = 0.0 # x-flux of x-momentum
		F[:,:,irhov, 0] = 0.0 # x-flux of y-momentum
		F[:,:,ip,    0] = nuP*gUx[:,:,ip]#*phi1*(1.-phi1)

		# y-direction
		F[:,:,irhou, 1] = 0.0 # x-flux of x-momentum
		F[:,:,irhov, 1] = 0.0 # x-flux of y-momentum
		F[:,:,ip,    1] = nuP*gUy[:,:,ip]#*phi1*(1.-phi1)
			
		# phase field and level set
		F[:,:,iPF,  0]  = a1x # x-flux of phi1
		F[:,:,iPF,  1]  = a1y # y-flux of phi1
		F[:,:,iLS,  0]  = 0.0       # x-flux of Levelset
		F[:,:,iLS,  1]  = 0.0       # y-flux of Levelset

		return F # [n, nq, ns, ndims]

	def compute_additional_variable(self, var_name, Uq, gUq, flag_non_physical, x, t):
		sname = self.AdditionalVariables[var_name].name
		# Get indices/slices of state variables
		irhou, irhov, ip, iPF, iLS = self.get_state_indices()

		rhou      = Uq[:, :, irhou]     # [n, nq]
		rhov      = Uq[:, :, irhov]     # [n, nq]
		p         = Uq[:, :, ip]     # [n, nq]
		phi1      = Uq[:, :, iPF]       # [n, nq]
		LS        = Uq[:, :, iLS]       # [n, nq]
		

		if sname is self.AdditionalVariables["XVelocity"].name:
			# Get velocity in each dimension
			rho1 = self.rho1
			rho2 = self.rho2
			rho  = rho1*phi1 + rho2*(1.0-phi1)
			u = rhou / rho
			scalar = u
		elif sname is self.AdditionalVariables["YVelocity"].name:
			# Get velocity in each dimension
			rho1 = self.rho1
			rho2 = self.rho2
			rho  = rho1*phi1 + rho2*(1.0-phi1)
			v = rhov / rho
			scalar = v
		elif sname is self.AdditionalVariables["MaxWaveSpeed"].name:
			rho1 = self.rho1
			rho2 = self.rho2
			rho   = rho1*phi1 + rho2*(1.0-phi1)
			# Get velocity in each dimension
			u = rhou / rho
			v = rhov / rho
			u2 = u**2
			v2 = v**2
			cs = self.cs
			maxwave = np.sqrt(cs**2 + u2 + v2 ) + np.sqrt(u2+v2)
			scalar = maxwave
		else:
			raise NotImplementedError

		return scalar
