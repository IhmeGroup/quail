import code
from enum import Enum, auto
import numpy as np
import sys

import errors
import general
# from general import *

import physics.base.base as base
import physics.base.functions as base_fcns
from physics.base.functions import BCType as base_BC_type
from physics.base.functions import FcnType as base_fcn_type
from physics.base.functions import ConvNumFluxType as base_conv_num_flux_type

import physics.scalar.functions as scalar_fcns
from physics.scalar.functions import FcnType as scalar_fcn_type
from physics.scalar.functions import SourceType as scalar_source_type


class ConstAdvScalar(base.PhysicsBase):
	'''
	Class: IFace
	--------------------------------------------------------------------------
	This is a class defined to encapsulate the temperature table with the 
	relevant methods
	'''

	NUM_STATE_VARS = 1
	PHYSICS_TYPE = general.PhysicsType.ConstAdvScalar

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
		self.c = 0.
		self.cspeed = 0.

	def set_maps(self):
		super().set_maps()

		self.source_map.update({
			scalar_source_type.SimpleSource : scalar_fcns.SimpleSource,
			scalar_source_type.StiffSource : scalar_fcns.StiffSource,
		})

	class StateVariables(Enum):
		Scalar = "u"

	class AdditionalVariables(Enum):
	    MaxWaveSpeed = "\\lambda"

# <<<<<<< Updated upstream
# 	class BCType(IntEnum):
# 	    StateAll = 0
# 	    Extrapolate = 1

# 	class BCTreatment(IntEnum):
# 		Riemann = 0
# 		Prescribed = 1
# =======
# 	# class BCType(IntEnum):
# 	#     StateAll = 0
# 	#     Extrapolation = 1
# >>>>>>> Stashed changes

	# class BCTreatment(IntEnum):
	# 	Riemann = 0
	# 	Prescribed = 1

	# def getWaveSpeed(self):
	# 	return self.Params["ConstVelocity"]

	# #Calculate velocity based on the advection operator
	# def getAdvOperator(self, u):
	# 	if self.Params["AdvectionOperator"] == self.AdvectionOperatorType.Burgers:
	# 		c = u/2
	# 		return c
	# 	elif self.Params["AdvectionOperator"] == self.AdvectionOperatorType.ConstVel:
	# 		c = self.Params["ConstVelocity"]
	# 		return c

	def ConvFluxInterior(self, u):
		# c = self.getAdvOperator(u)
		c = self.c
		#a = self.Params["Velocity"]
		# if F is None:
		# 	F = np.zeros(u.shape + (self.dim,))
		# for d in range(self.dim):
		# 	F[:,:,d] = c*u
		F = np.expand_dims(c*u, axis=1)
		# F = a*u
		# F.shape = u.shape + (self.dim,) 
		return F

	def AdditionalScalars(self, ScalarName, Up, flag_non_physical):
		sname = self.AdditionalVariables[ScalarName].name
		if sname is self.AdditionalVariables["MaxWaveSpeed"].name:
			scalar = np.full([Up.shape[0], 1], self.cspeed)
			# scalar[:] = self.cspeed
		else:
			raise NotImplementedError

		return scalar

	# def ConvFluxNumerical(self, uL, uR, normals): #, nq, data):
	# 	# nq = NData.nq
	# 	# if nq != uL.shape[0] or nq != uR.shape[0]:
	# 	# 	raise Exception("Wrong nq")	
	# 	# try:
	# 	# 	u = data.u
	# 	# except:
	# 	# 	data.u = u = np.zeros_like(uL)
	# 	# try: 
	# 	# 	F = data.F
	# 	# except AttributeError: 
	# 	# 	data.F = F = np.zeros_like(uL)
	# 	# try:
	# 	# 	c = data.c
	# 	# except:
	# 	# 	data.c = c = np.zeros_like(uL)

	#     #Calculate the max speed and keep its sign.
	# 	# for i in range(nq):

	# 	# 	u[i] = max(abs(uL[i]),abs(uR[i]))

	# 	# 	if u[i] == abs(uL[i]):
	# 	# 		usign = np.sign(uL[i])
	# 	# 	elif u[i] == abs(uR[i]):
	# 	# 		usign = np.sign(uR[i])
	# 	# 	u[i] = usign*u[i]

	# 	# 	c[i] = self.getAdvOperator(u[i])

	# 	self.ConvFluxFcn.AllocHelperArrays(uL)
	# 	F = self.ConvFluxFcn.compute_flux(self, uL, uR, normals)
		
	# 	# ConvFlux = self.Params["ConvFlux"] 
	# 	# if ConvFlux == self.ConvFluxType.LaxFriedrichs:
	# 	# 	F = self.ConvFluxLaxFriedrichs(uL, uR, NData.nvec, F)
		
# <<<<<<< Updated upstream
# 		return F

# 	def BoundaryState(self, BC, nq, xglob, Time, normals, uI, uB=None):
# 		if uB is not None:
# 			BC.U = uB

# 		BC.x = xglob
# 		BC.nq = nq
# 		BC.Time = Time
# 		bctype = BC.BCType
# 		if bctype == self.BCType.StateAll:
# 			uB = self.CallFunction(BC)
# 		elif bctype == self.BCType.Extrapolate:
# 			uB[:] = uI[:]
# 		else:
# 			raise Exception("BC type not supported")

# 		return uB

# 	def ComputeScalars(self, ScalarNames, U, scalar=None, flag_non_physical=False):
# 		if type(ScalarNames) is list:
# 			nscalar = len(ScalarNames)
# 		elif type(ScalarNames) is str:
# 			nscalar = 1
# 			ScalarNames = [ScalarNames]
# 		else:
# 			raise TypeError

# 		nq = U.shape[0]
# 		if scalar is None or scalar.shape != (nq, nscalar):
# 			scalar = np.zeros([nq, nscalar])

# 		for iscalar in range(nscalar):
# 			sname = ScalarNames[iscalar]
# 			try:
# 				sidx = self.GetStateIndex(sname)
# 				scalar[:,iscalar] = U[:,sidx]
# 			# if sidx < self.NUM_STATE_VARS:
# 			# 	# State variable
# 			# 	scalar[:,iscalar] = U[:,sidx]
# 			# else:
# 			except KeyError:
# 				scalar[:,iscalar:iscalar+1] = self.AdditionalScalars(sname, U, scalar[:,iscalar:iscalar+1],
# 					flag_non_physical)

# 		return scalar
# =======
	# 	return F

	# def BoundaryState(self, BC, nq, xglob, Time, normals, uI, uB=None):
	# 	if uB is not None:
	# 		BC.U = uB

	# 	BC.x = xglob
	# 	BC.nq = nq
	# 	BC.Time = Time
	# 	bctype = BC.BCType
	# 	if bctype == self.BCType.StateAll:
	# 		uB = self.CallFunction(BC)
	# 	elif bctype == self.BCType.Extrapolation:
	# 		uB[:] = uI[:]
	# 	else:
	# 		raise Exception("BC type not supported")

	# 	return uB

	# def ComputeScalars(self, ScalarNames, U, scalar=None, flag_non_physical=False):
	# 	if type(ScalarNames) is list:
	# 		nscalar = len(ScalarNames)
	# 	elif type(ScalarNames) is str:
	# 		nscalar = 1
	# 		ScalarNames = [ScalarNames]
	# 	else:
	# 		raise TypeError

	# 	nq = U.shape[0]
	# 	if scalar is None or scalar.shape != (nq, nscalar):
	# 		scalar = np.zeros([nq, nscalar])

	# 	for iscalar in range(nscalar):
	# 		sname = ScalarNames[iscalar]
	# 		try:
	# 			sidx = self.GetStateIndex(sname)
	# 			scalar[:,iscalar] = U[:,sidx]
	# 		# if sidx < self.NUM_STATE_VARS:
	# 		# 	# State variable
	# 		# 	scalar[:,iscalar] = U[:,sidx]
	# 		# else:
	# 		except KeyError:
	# 			scalar[:,iscalar:iscalar+1] = self.AdditionalScalars(sname, U, scalar[:,iscalar:iscalar+1],
	# 				flag_non_physical)

	# 	return scalar
# >>>>>>> Stashed changes


class ConstAdvScalar1D(ConstAdvScalar):
	'''
	Class: IFace
	--------------------------------------------------------------------------
	This is a class defined to encapsulate the temperature table with the 
	relevant methods
	'''

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
		# Default parameters
		self.Params.update(
			ConstVelocity = 1.,
		)
		self.c = 0.
		self.cspeed = 0.

	def set_maps(self):
		super().set_maps()

		d = {
			base_fcn_type.Uniform : base_fcns.Uniform,
			scalar_fcn_type.Sine : scalar_fcns.Sine,
			scalar_fcn_type.DampingSine : scalar_fcns.DampingSine,
			scalar_fcn_type.ShockBurgers : scalar_fcns.ShockBurgers,
			# scalar_fcn_type.ShiftedCosine : scalar_fcns.shifted_cosine,
			# scalar_fcn_type.Exponential : scalar_fcns.exponential,
			scalar_fcn_type.Gaussian : scalar_fcns.Gaussian,
		}

		self.IC_fcn_map.update(d)
		self.exact_fcn_map.update(d)
		self.BC_fcn_map.update(d)

	def set_physical_params(self, ConstVelocity=1.):
		self.c = ConstVelocity
		self.cspeed = np.abs(self.c)

	def SetParams(self,**kwargs):
		super().SetParams(**kwargs)

		self.c = self.Params["ConstVelocity"]
		self.cspeed = np.linalg.norm(self.c)


class ConstAdvScalar2D(ConstAdvScalar):

	dim = 2

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
			ConstXVelocity = 1.,
			ConstYVelocity = 1.,
		)
		self.c = np.zeros(2)
		self.cspeed = 0.

	def set_maps(self):
		super().set_maps()

		d = {
			scalar_fcn_type.Gaussian : scalar_fcns.Gaussian
		}

		self.IC_fcn_map.update(d)
		self.IC_fcn_map.update({
			scalar_fcn_type.Paraboloid : scalar_fcns.Paraboloid,
		})

		self.exact_fcn_map.update(d)
		self.BC_fcn_map.update(d)

	def set_physical_params(self, ConstXVelocity=1., ConstYVelocity=1.):
		self.c = np.array([ConstXVelocity, ConstYVelocity])
		self.cspeed = np.linalg.norm(self.c)

	def SetParams(self,**kwargs):
		super().SetParams(**kwargs)

		self.c = np.array([self.Params["ConstXVelocity"], self.Params["ConstYVelocity"]])
		self.cspeed = np.linalg.norm(self.c)


class Burgers1D(base.PhysicsBase):

	NUM_STATE_VARS = 1
	dim = 1
	PHYSICS_TYPE = general.PhysicsType.Burgers

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
		self.Params = {
		}

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
			scalar_source_type.StiffSource : scalar_fcns.StiffSource,
		})

	class StateVariables(Enum):
		Scalar = "u"

	class AdditionalVariables(Enum):
	    MaxWaveSpeed = "\\lambda"

	def ConvFluxInterior(self, u):
		# c = self.getAdvOperator(u)
		#a = self.Params["Velocity"]
		# if F is None:
		# 	F = np.zeros(u.shape + (self.dim,))
		# for d in range(self.dim):
		# 	F[:,:,d] = c*u
		F = np.expand_dims(u*u/2., axis=2)
		# F = a*u
		# F.shape = u.shape + (self.dim,) 
		return F

	def AdditionalScalars(self, ScalarName, Up, flag_non_physical):
		sname = self.AdditionalVariables[ScalarName].name
		if sname is self.AdditionalVariables["MaxWaveSpeed"].name:
			# Pressure
			# P = GetPressure()
			scalar = np.abs(Up/2.)
		else:
			raise NotImplementedError

		return scalar