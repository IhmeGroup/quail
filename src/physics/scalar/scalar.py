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
#       File : src/physics/scalar/scalar.py
#
#       Contains class definitions for scalar equations.
#
# ------------------------------------------------------------------------ #
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
from physics.base.functions import DiffNumFluxType as base_diff_num_flux_type

import physics.base.functions as base_fcns
import physics.scalar.functions as scalar_fcns
from physics.scalar.functions import FcnType as scalar_fcn_type
from physics.scalar.functions import SourceType as scalar_source_type
from physics.scalar.functions import ConvNumFluxType as scalar_conv_num_flux_type



class ConstAdvScalar(base.PhysicsBase):
	'''
	This class corresponds to scalar advection with a constant velocity.
	It inherits attributes and methods from the PhysicsBase class. See
	PhysicsBase for detailed comments of attributes and methods. This
	class should not be instantiated directly. Instead, the 1D and 2D
	variants, which inherit from this class (see below), should be
	instantiated.

	Additional methods and attributes are commented below.

	Attributes:
	-----------
	c: float or numpy array
		advection velocity
	cspeed: float
		advection speed
	'''
	NUM_STATE_VARS = 1
	PHYSICS_TYPE = general.PhysicsType.ConstAdvScalar

	def __init__(self):
		super().__init__()
		self.c = 0.
		self.cspeed = 0.

	def set_maps(self):
		super().set_maps()

		self.source_map.update({
			scalar_source_type.SimpleSource : scalar_fcns.SimpleSource,
		})

		self.conv_num_flux_map.update({
			scalar_conv_num_flux_type.ExactLinearFlux :
					scalar_fcns.ExactLinearFlux,
		})

	class StateVariables(Enum):
		Scalar = "u"

	class AdditionalVariables(Enum):
	    MaxWaveSpeed = "\\lambda"

	def get_conv_flux_interior(self, Uq, x=None, t=None):
		c = self.c
		F = np.expand_dims(c*Uq, axis=-1)

		return F, None

	def compute_additional_variable(self, var_name, Uq, flag_non_physical, x=None, t=None):
		sname = self.AdditionalVariables[var_name].name

		if sname is self.AdditionalVariables["MaxWaveSpeed"].name:
			# Max wave speed is the advection speed
			scalar = np.full([Uq.shape[0], 1, 1], self.cspeed)
		else:
			raise NotImplementedError

		return scalar


class ConstAdvScalar1D(ConstAdvScalar):
	'''
	This class corresponds to 1D scalar advection with a constant velocity.
	It inherits attributes and methods from the ConstAdvScalar class. See
	ConstAdvScalar for detailed comments of attributes and methods.

	Additional methods and attributes are commented below.
	'''
	NDIMS = 1

	def set_maps(self):
		super().set_maps()

		d = {
			base_fcn_type.Uniform : base_fcns.Uniform,
			scalar_fcn_type.Sine : scalar_fcns.Sine,
			scalar_fcn_type.DampingSine : scalar_fcns.DampingSine,
			scalar_fcn_type.ShockBurgers : scalar_fcns.ShockBurgers,
			scalar_fcn_type.Gaussian : scalar_fcns.Gaussian,
		}

		self.IC_fcn_map.update(d)
		self.exact_fcn_map.update(d)
		self.BC_fcn_map.update(d)

	def set_physical_params(self, ConstVelocity=1.):
		'''
		This method sets physical parameters.

		Inputs:
		-------
			ConstVelocity: constant advection velocity

		Outputs:
		--------
			self: physical parameters set
		'''
		self.c = ConstVelocity
		self.cspeed = np.abs(self.c)

class ConstAdvScalar2D(ConstAdvScalar):
	'''
	This class corresponds to 2D scalar advection with a constant velocity.
	It inherits attributes and methods from the ConstAdvScalar class. See
	ConstAdvScalar for detailed comments of attributes and methods.

	Additional methods and attributes are commented below.
	'''
	NDIMS = 2

	def __init__(self):
		super().__init__()
		self.c = np.zeros(2)
		self.cspeed = 0.

	def set_maps(self):
		super().set_maps()

		d = {
			scalar_fcn_type.Gaussian : scalar_fcns.Gaussian,
			scalar_fcn_type.Zalesak : scalar_fcns.Zalesak,
			scalar_fcn_type.Rider : scalar_fcns.Rider
		}

		self.IC_fcn_map.update(d)
		self.IC_fcn_map.update({
			scalar_fcn_type.Paraboloid : scalar_fcns.Paraboloid,
		})

		self.exact_fcn_map.update(d)
		self.BC_fcn_map.update(d)

	def set_physical_params(self, ConstXVelocity=1., ConstYVelocity=1.):
		'''
		This method sets physical parameters.

		Inputs:
		-------
			ConstXVelocity: constant advection velocity in the x-direction
			ConstYVelocity: constant advection velocity in the y-direction

		Outputs:
		--------
			self: physical parameters set
		'''
		self.c = np.array([ConstXVelocity, ConstYVelocity])
		self.cspeed = np.linalg.norm(self.c)

	def get_conv_flux_interior(self, Uq, x=None, t=None):
		c = self.c

		F = np.empty(Uq.shape + (self.NDIMS,)) # [n, nq, ns, ndims]
		F[:, :, :, 0] = c[0] * Uq
		F[:, :, :, 1] = c[1] * Uq

		return F, None

class ConstAdvDiffScalar(base.PhysicsBase):
	'''
	This class corresponds to scalar advection/diffusion with a constant 
	velocity and diffusion coefficient.

	It inherits attributes and methods from the PhysicsBase class. See
	PhysicsBase for detailed comments of attributes and methods. This
	class should not be instantiated directly. Instead, the 1D and 2D
	variants, which inherit from this class (see below), should be
	instantiated.

	Additional methods and attributes are commented below.

	Attributes:
	-----------
	c: float or numpy array
		advection velocity
	al: float or numpy array
		diffusion coefficient
	cspeed: float
		advection speed
	'''
	NUM_STATE_VARS = 1
	PHYSICS_TYPE = general.PhysicsType.ConstAdvDiffScalar

	def __init__(self):
		super().__init__()
		self.c = 0.
		self.cspeed = 0.

	def set_maps(self):
		super().set_maps()

		self.diff_num_flux_map.update({
			base_diff_num_flux_type.SIP : 
				base_fcns.SIP,
			})

	class StateVariables(Enum):
		Scalar = "u"

	class AdditionalVariables(Enum):
	    MaxWaveSpeed = "\\lambda"

	def get_conv_flux_interior(self, Uq, x=None, t=None):
		c = self.c
		F = np.expand_dims(c*Uq, axis=-1)

		return F, None

	def get_diff_flux_interior(self, Uq, gUq):
		al = self.al
		F = al * gUq

		return F

	def compute_additional_variable(self, var_name, Uq, flag_non_physical, x=None, t=None):
		sname = self.AdditionalVariables[var_name].name

		if sname is self.AdditionalVariables["MaxWaveSpeed"].name:
			# Max wave speed is the advection speed
			scalar = np.full([Uq.shape[0], 1, 1], self.cspeed)
		else:
			raise NotImplementedError

		return scalar


class ConstAdvDiffScalar1D(ConstAdvDiffScalar):
	'''
	This class corresponds to 1D scalar advection/diffusion with a constant
	velocity and diffusion coefficient. 

	It inherits attributes and methods from the ConstAdvDiffScalar class. See
	ConstAdvScalar for detailed comments of attributes and methods.

	Additional methods and attributes are commented below.
	'''
	NDIMS = 1

	def set_maps(self):
		super().set_maps()

		d = {
			scalar_fcn_type.DiffGaussian : scalar_fcns.DiffGaussian,
		}

		self.IC_fcn_map.update(d)
		self.exact_fcn_map.update(d)
		self.BC_fcn_map.update(d)

	def set_physical_params(self, ConstVelocity=1., DiffCoefficient=1.):
		'''
		This method sets physical parameters.

		Inputs:
		-------
			ConstVelocity: constant advection velocity
			DiffCoefficient: constant diffusion coefficient

		Outputs:
		--------
			self: physical parameters set
		'''
		self.c = ConstVelocity
		self.cspeed = np.abs(self.c)
		self.al = DiffCoefficient


class ConstAdvDiffScalar2D(ConstAdvDiffScalar):
	'''
	This class corresponds to 2D scalar /diffusion with a constant
	velocity and diffusion coefficient.

	It inherits attributes and methods from the ConstAdvDiffScalar 
	class. See ConstAdvDiffScalar for detailed comments of attributes
	and methods.

	Additional methods and attributes are commented below.
	'''
	NDIMS = 2

	def __init__(self):
		super().__init__()
		self.c = np.zeros(2)
		self.cspeed = 0.
		self.al = np.zeros(2)

	def set_maps(self):
		super().set_maps()

		d = {
			scalar_fcn_type.DiffGaussian2D : scalar_fcns.DiffGaussian2D,
		}

		self.IC_fcn_map.update(d)
		self.exact_fcn_map.update(d)
		self.BC_fcn_map.update(d)
		
		self.source_map.update({
			scalar_source_type.SharpeningSource : scalar_fcns.SharpeningSource,
		})

	def set_physical_params(self, ConstXVelocity=1., ConstYVelocity=1., 
			DiffCoefficientX=1., DiffCoefficientY=1.):
		'''
		This method sets physical parameters.

		Inputs:
		-------
			ConstXVelocity: constant advection velocity in the x-direction
			ConstYVelocity: constant advection velocity in the y-direction
			DiffCoefficientX: constant diffusion coefficient in the 
				x-direction
			DiffCoefficientY: constant diffusion coefficient in the 
				y-direction
		Outputs:
		--------
			self: physical parameters set
		'''
		self.c = np.array([ConstXVelocity, ConstYVelocity])
		self.al = np.array([DiffCoefficientX, DiffCoefficientY])
		self.cspeed = np.linalg.norm(self.c)

	def get_conv_flux_interior(self, Uq, x=None, t=None):
		c = self.c

		F = np.empty(Uq.shape + (self.NDIMS,)) # [n, nq, ns, ndims]
		F[:, :, :, 0] = c[0] * Uq
		F[:, :, :, 1] = c[1] * Uq

		return F, None

	def get_diff_flux_interior(self, Uq, gUq):
		al = self.al
		
		F = np.empty(Uq.shape + (self.NDIMS,)) # [n, nq, ns, ndims]

		F[:, :, :, 0] = al[0] * gUq[:, :, :, 0]
		F[:, :, :, 1] = al[1] * gUq[:, :, :, 1]

		return F


class Burgers1D(base.PhysicsBase):
	'''
	This class corresponds to the 1D Burgers equation.
	It inherits attributes and methods from the PhysicsBase class. See
	PhysicsBase for detailed comments of attributes and methods.
	'''
	NUM_STATE_VARS = 1
	NDIMS = 1
	PHYSICS_TYPE = general.PhysicsType.Burgers

	def set_maps(self):
		super().set_maps()

		d = {
			base_fcn_type.Uniform : base_fcns.Uniform,
			scalar_fcn_type.ShockBurgers : scalar_fcns.ShockBurgers,
			scalar_fcn_type.SineBurgers : scalar_fcns.SineBurgers,
			scalar_fcn_type.LinearBurgers : scalar_fcns.LinearBurgers,
		}

		self.IC_fcn_map.update(d)
		self.exact_fcn_map.update(d)
		self.BC_fcn_map.update(d)

		self.source_map.update({
			scalar_source_type.SimpleSource : scalar_fcns.SimpleSource,
		})

	class StateVariables(Enum):
		Scalar = "u"

	class AdditionalVariables(Enum):
	    MaxWaveSpeed = "\\lambda"

	def get_conv_flux_interior(self, Uq, x=None, t=None):

		F = np.expand_dims(Uq*Uq/2., axis=-1)

		return F, None

	def compute_additional_variable(self, var_name, Uq, flag_non_physical, x=None, t=None):
		sname = self.AdditionalVariables[var_name].name

		if sname is self.AdditionalVariables["MaxWaveSpeed"].name:
			# Max wave speed is u
			scalar = np.abs(Uq)
		else:
			raise NotImplementedError

		return scalar


class NonConstAdvDiffScalar(ConstAdvDiffScalar):
	'''
	This class corresponds to scalar advection/diffusion with a
	non-constant velocity.
	It inherits attributes and methods from the PhysicsBase class. See
	PhysicsBase for detailed comments of attributes and methods. This
	class should not be instantiated directly. Instead, the 1D and 2D
	variants, which inherit from this class (see below), should be
	instantiated.

	Additional methods and attributes are commented below.

	Attributes:
	-----------
	c: float or numpy array
		advection velocity
	cspeed: float
		advection speed
	al: float
		diffusion coefficient
	'''
	NUM_STATE_VARS = 2
	NDIMS = 1
	PHYSICS_TYPE = general.PhysicsType.NonConstAdvScalar

	def __init__(self):
		super().__init__()
		self.c = 0.
		self.cspeed = 0.
		self.al = 0.

	def set_maps(self):
		super().set_maps()

		d = {
			base_fcn_type.Uniform : base_fcns.Uniform,
			scalar_fcn_type.Sine : scalar_fcns.Sine,
			scalar_fcn_type.DampingSine : scalar_fcns.DampingSine,
			scalar_fcn_type.ShockBurgers : scalar_fcns.ShockBurgers,
			scalar_fcn_type.Gaussian : scalar_fcns.Gaussian,
			scalar_fcn_type.Heaviside : scalar_fcns.Heaviside,
		}

		self.IC_fcn_map.update(d)
		self.exact_fcn_map.update(d)
		self.BC_fcn_map.update(d)
		
		self.source_map.update({
			scalar_source_type.SimpleSource : scalar_fcns.SimpleSource,
			scalar_source_type.HeavisideSource : scalar_fcns.HeavisideSource,
		})
		
		self.conv_num_flux_map.update({
			scalar_conv_num_flux_type.LaxFriedrichs_THINC :
					scalar_fcns.LaxFriedrichs_THINC,
		})
		
		self.diff_num_flux_map.update({
			base_diff_num_flux_type.SIP :
				base_fcns.SIP,
			})

		
	def set_physical_params(self, ConstVelocity=1., DiffCoefficient=1.):
		'''
		This method sets physical parameters.

		Inputs:
		-------
			ConstVelocity: constant advection velocity
			DiffCoefficient: constant diffusion coefficient

		Outputs:
		--------
			self: physical parameters set
		'''
		self.c = ConstVelocity
		self.cspeed = np.abs(self.c)
		self.al = DiffCoefficient

	class StateVariables(Enum):
		Scalar = "u"
		Scalarx = "ux"

	class AdditionalVariables(Enum):
		MaxWaveSpeed = "\\lambda"
		Normal_x = "NX"
		adv_x = "adv"

	def get_conv_flux_interior(self, Uq, x, t=None):
		cc = self.IC.get_advection(self,x,t)
		
		n = np.zeros(cc.shape)

		F = np.zeros(Uq.shape + (self.NDIMS,)) # [n, nq, ns, ndims]

		n[:,:,0] = Uq[:,:,1]/np.abs(Uq[:,:,1] + 1e-15)

		F[:, :, 0, 0] = cc[:,:,0] * Uq[:,:,0] + 1.0*Uq[:,:,0]*(1.0-Uq[:,:,0])*n[:,:,0]
		F[:, :, 1, 0] = cc[:,:,0] * Uq[:,:,1]

		return F, None
		
	def get_diff_flux_interior(self, Uq, gUq):
		al = self.al
		
		F = np.zeros(Uq.shape + (self.NDIMS,)) # [n, nq, ns, ndims]
		
		F[:, :, 0, 0] = al * gUq[:, :, 0, 0]

		return F

	def compute_additional_variable(self, var_name, Uq, flag_non_physical, x, t=None):
		sname = self.AdditionalVariables[var_name].name

		if sname is self.AdditionalVariables["MaxWaveSpeed"].name:
			# Max wave speed is the advection speed
			cc = self.IC.get_advection(self,x,t)
			adv = np.zeros(Uq.shape)
			adv[:,:,0] = np.abs(cc[:,:,0] + 1e-15) + np.abs(1.0-2.0*Uq[:,:,0])
			adv[:,:,1] = np.abs(cc[:,:,0] + 1e-15)
			scalar = adv
		elif sname is self.AdditionalVariables["Normal_x"].name:
			scalar = Uq[:,:,1]/np.abs(Uq[:,:,1] + 1e-15)
		elif sname is self.AdditionalVariables["adv_x"].name:
			cc = self.IC.get_advection(self,x,t)
			scalar = cc[:,:,0]
		else:
			raise NotImplementedError

		return scalar

class NonConstAdvDiffScalar2D(NonConstAdvDiffScalar):
	'''
	This class corresponds to the gradient-augmented approach for the
	linear-advection equation involving sharp interfaces.

	It inherits attributes and methods from the ConstAdvDiffScalar
	class. See ConstAdvDiffScalar for detailed comments of attributes
	and methods.

	Additional methods and attributes are commented below.
	'''
	NUM_STATE_VARS = 3
	NDIMS = 2
	PHYSICS_TYPE = general.PhysicsType.NonConstAdvScalar

	def __init__(self):
		super().__init__()
		self.c = np.zeros(2)
		self.cspeed = 0.
		self.al = np.zeros(2)

	def set_maps(self):
		super().set_maps()

		d = {
			scalar_fcn_type.DiffGaussian2D : scalar_fcns.DiffGaussian2D,
			scalar_fcn_type.Zalesak : scalar_fcns.Zalesak,
			scalar_fcn_type.Rider : scalar_fcns.Rider,
		}

		self.IC_fcn_map.update(d)
		self.exact_fcn_map.update(d)
		self.BC_fcn_map.update(d)
		
		self.source_map.update({
			scalar_source_type.SimpleSource : scalar_fcns.SimpleSource,
			scalar_source_type.SharpeningSource : scalar_fcns.SharpeningSource,
			scalar_source_type.ZalesakSource : scalar_fcns.ZalesakSource,
			scalar_source_type.RiderSource : scalar_fcns.RiderSource,
		})


	def set_physical_params(self, ConstXVelocity=1., ConstYVelocity=1.,
			DiffCoefficientX=1., DiffCoefficientY=1.):
		'''
		This method sets physical parameters.

		Inputs:
		-------
			ConstXVelocity: constant advection velocity in the x-direction
			ConstYVelocity: constant advection velocity in the y-direction
			DiffCoefficientX: constant diffusion coefficient in the
				x-direction
			DiffCoefficientY: constant diffusion coefficient in the
				y-direction
		Outputs:
		--------
			self: physical parameters set
		'''
		self.c = np.array([ConstXVelocity, ConstYVelocity])
		self.al = np.array([DiffCoefficientX, DiffCoefficientY])
		self.cspeed = np.linalg.norm(self.c)

	def get_conv_flux_interior(self, Uq, x, t):

		cc = self.IC.get_advection(self,x,t)
		
		n = np.zeros(cc.shape)

		F = np.empty(Uq.shape + (self.NDIMS,)) # [n, nq, ns, ndims]

		n[:,:,0] = Uq[:,:,1]/np.sqrt(Uq[:,:,1]**2 + Uq[:,:,2]**2 + 1e-15)
		n[:,:,1] = Uq[:,:,2]/np.sqrt(Uq[:,:,1]**2 + Uq[:,:,2]**2 + 1e-15)

		F[:, :, 0, 0] = cc[:,:,0] * Uq[:,:,0] + 1.0*Uq[:,:,0]*(1.0-Uq[:,:,0])*n[:,:,0]
		F[:, :, 0, 1] = cc[:,:,1] * Uq[:,:,0] + 1.0*Uq[:,:,0]*(1.0-Uq[:,:,0])*n[:,:,1]
		
		F[:, :, 1, 0] = cc[:,:,0] * Uq[:,:,1]
		F[:, :, 1, 1] = cc[:,:,1] * Uq[:,:,1]
		
		F[:, :, 2, 0] = cc[:,:,0] * Uq[:,:,2]
		F[:, :, 2, 1] = cc[:,:,1] * Uq[:,:,2]
	

		return F, None

	def get_diff_flux_interior(self, Uq, gUq):
		al = self.al
		
		F = np.zeros(Uq.shape + (self.NDIMS,)) # [n, nq, ns, ndims]
		
		F[:, :, 0, 0] = al[0] * gUq[:, :, 0, 0]
		F[:, :, 0, 1] = al[1] * gUq[:, :, 0, 1]
		
		return F
		
	class StateVariables(Enum):
		Scalar = "u"
		Scalarx = "ux"
		Scalary = "uy"
		
	class AdditionalVariables(Enum):
		MaxWaveSpeed = "\\lambda"
		Normal_x = "NX"
		Normal_y = "NY"
		adv_x = "advx"
		adv_y = "advy"
		
	def compute_additional_variable(self, var_name, Uq, flag_non_physical, x, t):
		sname = self.AdditionalVariables[var_name].name

		if sname is self.AdditionalVariables["MaxWaveSpeed"].name:
			# Max wave speed is the advection speed
			cc = self.IC.get_advection(self,x,t)
			adv = np.zeros(Uq.shape)
			adv[:,:,0] = np.sqrt(cc[:,:,0]**2+cc[:,:,1]**2 + 1e-15) +\
				np.abs(1.0-2.0*Uq[:,:,0])
			adv[:,:,1] = np.sqrt(cc[:,:,0]**2+cc[:,:,1]**2 + 1e-15)
			adv[:,:,2] = np.sqrt(cc[:,:,0]**2+cc[:,:,1]**2 + 1e-15)
			scalar = adv
		elif sname is self.AdditionalVariables["Normal_x"].name:
			scalar = Uq[:,:,1]/np.sqrt(Uq[:,:,1]**2 + Uq[:,:,2]**2 + 1e-15)
		elif sname is self.AdditionalVariables["Normal_y"].name:
			scalar = Uq[:,:,2]/np.sqrt(Uq[:,:,1]**2 + Uq[:,:,2]**2 + 1e-15)
		elif sname is self.AdditionalVariables["adv_x"].name:
			cc = self.IC.get_advection(self,x,t)
			scalar = cc[:,:,0]
		elif sname is self.AdditionalVariables["adv_y"].name:
			cc = self.IC.get_advection(self,x,t)
			scalar = cc[:,:,1]
		else:
			raise NotImplementedError

		return scalar
	
