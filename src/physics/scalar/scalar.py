import code
import numpy as np
from scipy.optimize import root
import sys


from data import ArrayList, ICData, BCData, ExactData, SourceData, GenericData
import errors
from general import *
from numerics.basis.basis import order_to_num_basis_coeff
import physics.base.base as base


class ConstAdvScalar1D(base.PhysicsBase):
	'''
	Class: IFace
	--------------------------------------------------------------------------
	This is a class defined to encapsulate the temperature table with the 
	relevant methods
	'''

	StateRank = 1

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
			ConvFlux = self.ConvFluxType["LaxFriedrichs"]
		)
		self._c = 0.
		self._cspeed = 0.

	def SetParams(self,**kwargs):
		super().SetParams(**kwargs)

		if self.Params["ConvFlux"] == self.ConvFluxType["LaxFriedrichs"]:
			self.ConvFluxFcn = base.LaxFriedrichsFlux()
		self._c = self.Params["ConstVelocity"]
		self._cspeed = np.linalg.norm(self._c)

	class StateVariables(Enum):
		Scalar = "u"

	class AdditionalVariables(Enum):
	    MaxWaveSpeed = "\\lambda"

	class BCType(IntEnum):
	    FullState = 0
	    Extrapolation = 1

	class BCTreatment(IntEnum):
		Riemann = 0
		Prescribed = 1

	class ConvFluxType(IntEnum):
	    Upwind = 0
	    LaxFriedrichs = 1

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

	def ConvFluxInterior(self, u, F=None):
		# c = self.getAdvOperator(u)
		c = self._c
		#a = self.Params["Velocity"]
		if F is None:
			F = np.zeros(u.shape + (self.Dim,))
		# for d in range(self.Dim):
		# 	F[:,:,d] = c*u
		F[:] = np.expand_dims(c*u, axis=1)
		# F = a*u
		# F.shape = u.shape + (self.Dim,) 
		return F

	def ConvFluxNumerical(self, uL, uR, normals, nq, data):
		# nq = NData.nq
		if nq != uL.shape[0] or nq != uR.shape[0]:
			raise Exception("Wrong nq")	
		try:
			u = data.u
		except:
			data.u = u = np.zeros_like(uL)
		try: 
			F = data.F
		except AttributeError: 
			data.F = F = np.zeros_like(uL)
		try:
			c = data.c
		except:
			data.c = c = np.zeros_like(uL)

	    #Calculate the max speed and keep its sign.
		# for i in range(nq):

		# 	u[i] = max(abs(uL[i]),abs(uR[i]))

		# 	if u[i] == abs(uL[i]):
		# 		usign = np.sign(uL[i])
		# 	elif u[i] == abs(uR[i]):
		# 		usign = np.sign(uR[i])
		# 	u[i] = usign*u[i]

		# 	c[i] = self.getAdvOperator(u[i])

		self.ConvFluxFcn.AllocHelperArrays(uL)
		F = self.ConvFluxFcn.compute_flux(self, uL, uR, normals)
		
		# ConvFlux = self.Params["ConvFlux"] 
		# if ConvFlux == self.ConvFluxType.LaxFriedrichs:
		# 	F = self.ConvFluxLaxFriedrichs(uL, uR, NData.nvec, F)
		
		return F

	def BoundaryState(self, BC, nq, xglob, Time, normals, uI, uB=None):
		if uB is not None:
			BC.U = uB

		BC.x = xglob
		BC.nq = nq
		BC.Time = Time
		bctype = BC.BCType
		if bctype == self.BCType.FullState:
			uB = self.CallFunction(BC)
		elif bctype == self.BCType.Extrapolation:
			uB[:] = uI[:]
		else:
			raise Exception("BC type not supported")

		return uB

	def ComputeScalars(self, ScalarNames, U, scalar=None, FlagNonPhysical=False):
		if type(ScalarNames) is list:
			nscalar = len(ScalarNames)
		elif type(ScalarNames) is str:
			nscalar = 1
			ScalarNames = [ScalarNames]
		else:
			raise TypeError

		nq = U.shape[0]
		if scalar is None or scalar.shape != (nq, nscalar):
			scalar = np.zeros([nq, nscalar])

		for iscalar in range(nscalar):
			sname = ScalarNames[iscalar]
			try:
				sidx = self.GetStateIndex(sname)
				scalar[:,iscalar] = U[:,sidx]
			# if sidx < self.StateRank:
			# 	# State variable
			# 	scalar[:,iscalar] = U[:,sidx]
			# else:
			except KeyError:
				scalar[:,iscalar:iscalar+1] = self.AdditionalScalars(sname, U, scalar[:,iscalar:iscalar+1],
					FlagNonPhysical)

		return scalar

	def AdditionalScalars(self, ScalarName, U, scalar, FlagNonPhysical):
		sname = self.AdditionalVariables[ScalarName].name
		if sname is self.AdditionalVariables["MaxWaveSpeed"].name:
			scalar[:] = self._cspeed
		else:
			raise NotImplementedError

		return scalar

	def FcnUniform(self, FcnData):
		Data = FcnData.Data
		U = FcnData.U
		ns = self.StateRank

		for k in range(ns):
			U[:,k] = Data.State[k]

		return U

	def FcnSine(self, FcnData):
		x = FcnData.x
		t = FcnData.Time
		Data = FcnData.Data
		U = FcnData.U

		try:
			omega = Data.omega
		except AttributeError:
			omega = 2.*np.pi

		# c = self.getAdvOperator(U)
		c = self._c
		U[:] = np.sin(omega*(x-c*t))

		return U

	def FcnDampingSine(self, FcnData):
		x = FcnData.x
		t = FcnData.Time
		Data = FcnData.Data
		U = FcnData.U

		try:
			omega = Data.omega
		except AttributeError:
			omega = 2.*np.pi
		try:
			nu = Data.nu
		except AttributeError:
			nu = 1.0

		# c = self.getAdvOperator(U)
		c = self._c
		U[:] = np.sin(omega*(x-c*t))*np.exp(nu*t)

		return U

	def FcnShiftedCose(self, FcnData):
		x = FcnData.x
		t = FcnData.Time
		U = FcnData.U
		Data = FcnData.Data

		try:
			omega = Data.omega
		except AttributeError:
			omega = 2.*np.pi

		U[:] = 1-np.cos(omega*x)

		return U

	def FcnExponential(self, FcnData):
		x = FcnData.x
		t = FcnData.Time
		Data = FcnData.Data
		U = FcnData.U

		try:
			theta = Data.theta
		except AttributeError:
			theta = 1.

		U[:] = np.exp(theta*x)

		return U

	def FcnGaussian(self, FcnData):
		x = FcnData.x
		t = FcnData.Time
		Data = FcnData.Data
		U = FcnData.U

		# Standard deviation
		try:
			sig = Data.sig
		except AttributeError:
			sig = 1.
		# Center
		try:
			x0 = Data.x0
		except AttributeError:
			x0 = np.zeros(self.Dim)

		r = np.linalg.norm(x-x0-self._c*t, axis=1, keepdims=True)
		U[:] = 1./(sig*np.sqrt(2.*np.pi))**float(self.Dim) * np.exp(-r**2./(2.*sig**2.))

		return U

	def FcnScalarShock(self, FcnData):
		x = FcnData.x
		t = FcnData.Time
		U = FcnData.U
		Data = FcnData.Data

		try:
			uR = Data.uR
		except AttributeError:
			uR = 0.
		try:
			uL = Data.uL
		except AttributeError:
			uL = 1.
		try:
			xshock = Data.xshock
		except AttributeError:
			xshock = -0.5
		''' Fill state '''
		us = (uR+uL)
		xshock = xshock+us*t
		ileft = (x <= xshock).reshape(-1)
		iright = (x > xshock).reshape(-1)

		U[ileft]=uL
		U[iright]=uR

		return U

		# Source Term Functions
	def FcnStiffSource(self,FcnData):
		x = FcnData.x
		t = FcnData.Time
		U = FcnData.U
		S = FcnData.S
		Data = FcnData.Data

		try:
			beta = Data.beta
		except AttributeError:
			beta = 0.5
		try: 
			stiffness = Data.stiffness
		except AttributeError:
			stiffness = 1.

		S[:] = (1./stiffness)*(1.-U[:])*(U[:]-beta)*U[:]


		return S

	def FcnSimpleSource(self,FcnData):
		x = FcnData.x
		t = FcnData.Time
		U = FcnData.U
		S = FcnData.S
		Data = FcnData.Data

		try:
			nu = Data.nu
		except AttributeError:
			nu = -1.0

		S[:] = nu*U[:]

		return S

class ConstAdvScalar2D(ConstAdvScalar1D):

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
			ConvFlux = self.ConvFluxType["LaxFriedrichs"]
		)
		self._c = np.zeros(2)
		self._cspeed = 0.
	
	def SetParams(self,**kwargs):
		super().SetParams(**kwargs)

		self._c = np.array([self.Params["ConstXVelocity"], self.Params["ConstYVelocity"]])
		self._cspeed = np.linalg.norm(self._c)

	def FcnParaboloid(self, FcnData):
		x = FcnData.x
		t = FcnData.Time
		Data = FcnData.Data
		U = FcnData.U

		# Standard deviation
		# try:
		# 	sig = Data.sig
		# except AttributeError:
		# 	sig = 1.
		# # Center
		# try:
		# 	x0 = Data.x0
		# except AttributeError:
		# 	x0 = np.zeros(self.Dim)

		r2 = x[:,0:1]**2. + x[:,1:2]**2.
		U[:] = r2

		return U


class Burgers1D(ConstAdvScalar1D):

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
			"ConvFlux" : self.ConvFluxType["LaxFriedrichs"]
		}

	def ConvFluxInterior(self, u, F=None):
		# c = self.getAdvOperator(u)
		#a = self.Params["Velocity"]
		if F is None:
			F = np.zeros(u.shape + (self.Dim,))
		# for d in range(self.Dim):
		# 	F[:,:,d] = c*u
		F[:] = np.expand_dims(u*u/2., axis=2)
		# F = a*u
		# F.shape = u.shape + (self.Dim,) 
		return F

	def AdditionalScalars(self, ScalarName, U, scalar, FlagNonPhysical):
		sname = self.AdditionalVariables[ScalarName].name
		if sname is self.AdditionalVariables["MaxWaveSpeed"].name:
			# Pressure
			# P = GetPressure()
			scalar[:] = np.abs(U/2.)
		else:
			raise NotImplementedError

		return scalar

	def FcnSineWaveBurgers(self, FcnData):
		x = FcnData.x
		t = FcnData.Time
		U = FcnData.U
		Data = FcnData.Data

		try:
			omega = Data.omega
		except AttributeError:
			omega = 2.*np.pi

		def F(u):
			x1 = np.reshape(x,(len(x)))
			F = u - np.sin(omega*(x1-u*t)) 
			return F
			
		u = np.sin(omega*(x))
		u1 = np.reshape(u,(len(u)))
		sol = root(F, u1, tol=1e-12)
		
		U[:,0] = sol.x

		return U

	def FcnLinearBurgers(self, FcnData):
		x = FcnData.x
		t = FcnData.Time
		U = FcnData.U
		Data = FcnData.Data

		a = -1.
		b = 1.
		U = (a*x+b)/(a*t+1.)

		return U