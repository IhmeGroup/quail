import numpy as np
from Basis import *
from General import *
import code
import Errors
from Data import ArrayList, ICData, BCData, ExactData, SourceData, GenericData
import sys
from scipy.optimize import root


class LaxFriedrichsFlux(object):
	def __init__(self, u=None):
		if u is not None:
			n = u.shape[0]
		else:
			n = 0
		self.FL = np.zeros_like(u)
		self.FR = np.zeros_like(u)
		self.du = np.zeros_like(u)
		self.a = np.zeros([n,1])
		self.aR = np.zeros([n,1])
		self.idx = np.empty([n,1], dtype=bool) 

	def AllocHelperArrays(self, u):
		self.__init__(u)

	def ComputeFlux(self, EqnSet, UL, UR, n):
		'''
		Function: ConvFluxLaxFriedrichs
		-------------------
		This function computes the numerical flux (dotted with the normal)
		using the Lax-Friedrichs flux function

		INPUTS:
		    gam: specific heat ratio
		    UL: Left state
		    UR: Right state
		    n: Normal vector (assumed left to right)

		OUTPUTS:
		    F: Numerical flux dotted with the normal, i.e. F_hat dot n
		'''

		# Extract helper arrays
		FL = self.FL
		FR = self.FR 
		du = self.du 
		a = self.a 
		aR = self.aR 
		idx = self.idx 

		NN = np.linalg.norm(n, axis=1, keepdims=True)
		n1 = n/NN

		# Left State
		FL[:] = EqnSet.ConvFluxProjected(UL, n1)

		# Right State
		FR[:] = EqnSet.ConvFluxProjected(UR, n1)

		du[:] = UR-UL

		# max characteristic speed
		# code.interact(local=locals())
		a[:] = EqnSet.ComputeScalars("MaxWaveSpeed", UL, None, FlagNonPhysical=True)
		aR[:] = EqnSet.ComputeScalars("MaxWaveSpeed", UR, None, FlagNonPhysical=True)
		idx[:] = aR > a
		a[idx] = aR[idx]

		# flux assembly 
		return NN*(0.5*(FL+FR) - 0.5*a*du)


class ConstAdvScalar1D(object):
	'''
	Class: IFace
	--------------------------------------------------------------------------
	This is a class defined to encapsulate the temperature table with the 
	relevant methods
	'''
	def __init__(self,Order,basis,mesh,StateRank=1):
		'''
		Method: __init__
		--------------------------------------------------------------------------
		This method initializes the temperature table. The table uses a
		piecewise linear function for the constant pressure specific heat 
		coefficients. The coefficients are selected to retain the exact 
		enthalpies at the table points.
		'''
		dim = mesh.Dim
		self.Dim = mesh.Dim
		self.StateRank = StateRank
		self.Params = {}
		self.IC = ICData()
		self.ExactSoln = ExactData()
		self.ConvFluxFcn = None
		self.BCTreatments = {}
		self.Sources = []
		# Boundary conditions
		# self.BCs = []
		# for ibfgrp in range(mesh.nBFaceGroup):
		# 	self.BCs.append(BCData(Name=mesh.BFGNames[ibfgrp]))
		self.nBC = mesh.nBFaceGroup
		self.BCs = [BCData() for ibfgrp in range(mesh.nBFaceGroup)]
		for ibfgrp in range(mesh.nBFaceGroup):
			self.BCs[ibfgrp].Name = mesh.BFGNames[ibfgrp]
			# self.BCs[0].Set(Name=mesh.BFGNames[ibfgrp])

		# Basis, Order data for each element group
		# For now, ssume uniform basis and Order for each element group 
		if type(basis) is str:
			basis = BasisType[basis]
		self.Basis = basis
		if type(Order) is int:
			self.Order = Order
		elif type(Order) is list:
			self.Order = Order[0]
		else:
			raise Exception("Input error")

		# State 
		# self.U = ArrayList(nArray=mesh.nElemGroup,nEntriesPerArray=mesh.nElems,FullDim=[mesh.nElemTot,nn,self.StateRank])
		# self.U = ArrayList(nArray=mesh.nElemGroup,ArrayDims=[[mesh.nElemTot,nn,self.StateRank]])
		# ArrayDims = [[mesh.nElems[egrp],order_to_num_basis_coeff(self.Bases[egrp], self.Orders[egrp]), self.StateRank] \
		# 			for egrp in range(mesh.nElemGroup)]
		# self.U = ArrayList(nArray=mesh.nElemGroup,ArrayDims=ArrayDims)
		# self.S = ArrayList(nArray=mesh.nElemGroup,ArrayDims=ArrayDims)
		self.U = np.zeros([mesh.nElem, order_to_num_basis_coeff(self.Basis, self.Order), self.StateRank])
		self.S = np.zeros([mesh.nElem, order_to_num_basis_coeff(self.Basis, self.Order), self.StateRank])

		if dim == 1:
			if basis == BasisType.LegendreSeg:
				basisADER = BasisType.LegendreQuad
			elif basis == BasisType.LagrangeEqSeg:
				basisADER = BasisType.LagrangeEqQuad
			else:
				raise Errors.IncompatibleError
		else:
			basisADER = 0 # dummy	

		self.BasisADER = basisADER # [basisADER for egrp in range(mesh.nElemGroup)]
		# ADERArrayDims = [[mesh.nElems[egrp],order_to_num_basis_coeff(self.BasesADER[egrp],self.Orders[egrp]),self.StateRank] \
		# 			for egrp in range(mesh.nElemGroup)]
		# self.Up = ArrayList(nArray=mesh.nElemGroup,ArrayDims=ADERArrayDims)
		self.Up = np.zeros([mesh.nElem, order_to_num_basis_coeff(self.BasisADER, self.Order), self.StateRank])

		# BC treatments
		self.SetBCTreatment()

		# State indices
		self.StateIndices = {}
		if sys.version_info[0] < 3:
			for key in self.StateVariables.__members__.keys():
				self.StateIndices[key] = self.StateVariables.__members__.keys().index(key)
		else:	
			index = 0
			for key in self.StateVariables:
				self.StateIndices[key.name] = index
				index += 1


		### ConstAdv scalar only
		# Don't make Burgers inherit from this after adding abstract base classes
		self.c = 0.
		self.cspeed = 0.

		# Uarray = np.zeros([mesh.nElemTot, nn, self.StateRank])
		# self.Uarray = Uarray
		# # nElems = [mesh.ElemGroups[i].nElem for i in range(mesh.nElemGroup)]
		# nElems = mesh.nElems

		# self.U = [Uarray[0:nElems[0]]]
		# for i in range(1,mesh.nElemGroup):
		# 	self.U.append(Uarray[nElems[i-1]:nElems[i]])

	def SetParams(self,**kwargs):
		Params = self.Params
		# Default values
		if not Params:
			Params["ConstVelocity"] = 1.
			# Params["AdvectionOperator"] = self.AdvectionOperatorType["ConstVel"]
			Params["ConvFlux"] = self.ConvFluxType["LaxFriedrichs"]
		# Overwrite
		for key in kwargs:
			if key not in Params.keys(): raise Exception("Input error")
			if key is "ConvFlux":
				Params[key] = self.ConvFluxType[kwargs[key]]
			elif key is "AdvectionOperator":
				Params[key] = self.AdvectionOperatorType[kwargs[key]]
			else:
				Params[key] = kwargs[key]

		if Params["ConvFlux"] == self.ConvFluxType["LaxFriedrichs"]:
			self.ConvFluxFcn = LaxFriedrichsFlux()
		self.c = Params["ConstVelocity"]
		self.cspeed = np.linalg.norm(self.c)

	def SetBC(self, BCName, **kwargs):
		found = False
		for BC in self.BCs:
			if BC.Name == BCName:
				BC.Set(**kwargs)
				found = True
				break

		if not found:
			raise NameError

	class StateVariables(Enum):
		Scalar = "u"

	class AdditionalVariables(Enum):
	    MaxWaveSpeed = "\\lambda"

	# class VariableType(IntEnum):
	#     Scalar = 0
	#     # Additional scalars go here

	# class VarLabelType(Enum):
	# 	# LaTeX format
	#     Scalar = "u"
	#     # Additional scalars go here

	def GetStateIndex(self, VariableName):
		# idx = self.VariableType[VariableName]
		idx = self.StateIndices[VariableName]
		# idx = self.StateVariables.__members__.keys().index(VariableName)
		return idx

	class BCType(IntEnum):
	    FullState = 0
	    Extrapolation = 1

	class BCTreatment(IntEnum):
		Riemann = 0
		Prescribed = 1

	def SetBCTreatment(self):
		# default is Prescribed
		self.BCTreatments = {n:self.BCTreatment.Prescribed for n in range(len(self.BCType))}
		self.BCTreatments[self.BCType.FullState] = self.BCTreatment.Riemann

	class ConvFluxType(IntEnum):
	    Upwind = 0
	    LaxFriedrichs = 1

	class AdvectionOperatorType(IntEnum):
		ConstVel=0
		Burgers=1

	def SetSource(self, **kwargs):
		#append src data to Sources list 
		Source = SourceData()
		self.Sources.append(Source)
		Source.Set(**kwargs)

	def QuadOrder(self, Order):
		return 2*Order+1

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
		c = self.c
		#a = self.Params["Velocity"]
		if F is None:
			F = np.zeros(u.shape + (self.Dim,))
		# for d in range(self.Dim):
		# 	F[:,:,d] = c*u
		F[:] = np.expand_dims(c*u, axis=1)
		# F = a*u
		# F.shape = u.shape + (self.Dim,) 
		return F

	def ConvFluxUpwind(self, uL, uR, c, n):
		Vn = c*n 

		# upwind
		if Vn >= 0.:
			sL = 1.
			sR = 0.
		else:
			sL = 0. 
			sR = 1.

		F = Vn*(sL*uL + sR*uR)

		return F

	# def ConvFluxLaxFriedrichs(self, uL, uR, c, n):

	# 	# Left state
	# 	ul1 = self.getAdvOperator(uL) 
	# 	fL      = (ul1*uL)

	# 	# Right state
	# 	ur1 = self.getAdvOperator(uR)
	# 	fR 		= (ur1*uR)

	# 	du = uR - uL

	# 	# Check flow direction
	# 	if np.sign(c)<0:
	# 		c = -c

	# 	# flux assembly 
	# 	F = (0.5*n*(fL+fR) - 0.5*c*du)

	# 	return F

	# def ConvFluxLaxFriedrichs(self, UL, UR, n, F):
	# 	'''
	# 	Function: ConvFluxLaxFriedrichs
	# 	-------------------
	# 	This function computes the numerical flux (dotted with the normal)
	# 	using the Lax-Friedrichs flux function

	# 	INPUTS:
	# 	    gam: specific heat ratio
	# 	    UL: Left state
	# 	    UR: Right state
	# 	    n: Normal vector (assumed left to right)

	# 	OUTPUTS:
	# 	    F: Numerical flux dotted with the normal, i.e. F_hat dot n
	# 	'''

	# 	nq = F.shape[0]

	# 	# Extract intermediate arrays
	# 	# data = self.DataStorage
	# 	# try: 
	# 	# 	NN = data.NN
	# 	# except AttributeError: 
	# 	# 	data.NN = NN = np.zeros([nq,1])
	# 	# try: 
	# 	# 	n1 = data.n1
	# 	# except AttributeError: 
	# 	# 	data.n1 = n1 = np.zeros_like(n)
	# 	# try: 
	# 	# 	FL = data.FL
	# 	# except AttributeError: 
	# 	# 	data.FL = FL = np.zeros_like(F)
	# 	# try: 
	# 	# 	FL = data.FL
	# 	# except AttributeError: 
	# 	# 	data.FL = FL = np.zeros_like(F)

	# 	NN = np.linalg.norm(n, axis=1, keepdims=True)
	# 	n1 = n/NN

	# 	# Left State
	# 	FL = self.ConvFluxProjected(UL, n1)

	# 	# Right State
	# 	FR = self.ConvFluxProjected(UR, n1)

	# 	du = UR-UL

	# 	# max characteristic speed
	# 	lam = self.ComputeScalars("MaxWaveSpeed", UL, None, FlagNonPhysical=True).reshape(-1)
	# 	lamr = self.ComputeScalars("MaxWaveSpeed", UR, None, FlagNonPhysical=True).reshape(-1)
	# 	idx = lamr > lam
	# 	lam[idx] = lamr[idx]

	# 	# flux assembly 
	# 	F = NN*(0.5*(FL+FR) - 0.5*lam.reshape(-1,1)*du)

	# 	return F

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
		F = self.ConvFluxFcn.ComputeFlux(self, uL, uR, normals)
		
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

	#Source state takes multiple source terms (if needed) and sums them together. 
	def SourceState(self, nq, xglob, Time, u, s=None):
		for Source in self.Sources:

			#loop through available source terms
			Source.x = xglob
			Source.nq = nq
			Source.Time = Time
			Source.U = u
			s += self.CallSourceFunction(Source)

		return s

	def ConvFluxProjected(self, u, nvec):

		F = self.ConvFluxInterior(u, None)
		return np.sum(F.transpose(1,0,2)*nvec, axis=2).transpose()

	def ConvFluxBoundary(self, BC, uI, uB, normals, nq, data):
		bctreatment = self.BCTreatments[BC.BCType]
		if bctreatment == self.BCTreatment.Riemann:
			F = self.ConvFluxNumerical(uI, uB, normals, nq, data)
		else:
			# Prescribe analytic flux
			try:
				Fa = data.Fa
			except AttributeError:
				data.Fa = Fa = np.zeros([nq, self.StateRank, self.Dim])
			# Fa = self.ConvFluxInterior(uB, Fa)
			# # Take dot product with n
			try: 
				F = data.F
			except AttributeError:
				data.F = F = np.zeros_like(uI)
			F[:] = self.ConvFluxProjected(uB, normals)

		return F

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
			scalar[:] = self.cspeed
		else:
			raise NotImplementedError

		return scalar

	def CallFunction(self, FcnData, **kwargs):
		for key in kwargs:
			if key is "x":
				FcnData.x = kwargs[key]
				FcnData.nq = FcnData.x.shape[0]
			elif key is "Time":
				FcnData.Time = kwargs[key]
			else:
				raise Exception("Input error")

		nq = FcnData.nq
		sr = self.StateRank
		if FcnData.U is None or FcnData.U.shape != (nq, sr):
			FcnData.U = np.zeros([nq, sr],dtype=self.U.dtype)

		FcnData.U[:] = FcnData.Function(FcnData)

		return FcnData.U

	def CallSourceFunction(self, FcnData, **kwargs):
		for key in kwargs:
			if key is "x":
				FcnData.x = kwargs[key]
				FcnData.nq = FcnData.x.shape[0]
			elif key is "Time":
				FcnData.Time = kwargs[key]
			else:
				raise Exception("Input error")

		nq = FcnData.nq
		sr = self.StateRank
		if FcnData.S is None or FcnData.S.shape != (nq, sr):
			FcnData.S = np.zeros([nq, sr],dtype=self.S.dtype)

		FcnData.S[:] = FcnData.Function(FcnData)

		return FcnData.S

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
		c = self.c
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
		c = self.c
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

		r = np.linalg.norm(x-x0-self.c*t, axis=1, keepdims=True)
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
	def SetParams(self,**kwargs):
		Params = self.Params
		# Default values
		if not Params:
			Params["ConstXVelocity"] = 1.
			Params["ConstYVelocity"] = 1.
			Params["ConvFlux"] = self.ConvFluxType["LaxFriedrichs"]
		# Overwrite
		for key in kwargs:
			if key not in Params.keys(): raise Exception("Input error")
			if key is "ConvFlux":
				Params[key] = self.ConvFluxType[kwargs[key]]
			elif key is "AdvectionOperator":
				Params[key] = self.AdvectionOperatorType[kwargs[key]]
			else:
				Params[key] = kwargs[key]

		if Params["ConvFlux"] == self.ConvFluxType["LaxFriedrichs"]:
			self.ConvFluxFcn = LaxFriedrichsFlux()
		self.c = np.array([Params["ConstXVelocity"],Params["ConstYVelocity"]])
		self.cspeed = np.linalg.norm(self.c)

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