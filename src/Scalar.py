import numpy as np
from Basis import *
from General import *
import code
from Data import ArrayList, ICData, BCData, ExactData


class Scalar(object):
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

		# Boundary conditions
		# self.BCs = []
		# for ibfgrp in range(mesh.nBFaceGroup):
		# 	self.BCs.append(BCData(Name=mesh.BFGNames[ibfgrp]))
		self.nBC = mesh.nBFaceGroup
		self.BCs = [BCData() for ibfgrp in range(mesh.nBFaceGroup)]
		for ibfgrp in range(mesh.nBFaceGroup):
			self.BCs[ibfgrp].Name = mesh.BFGNames[ibfgrp]
			# self.BCs[0].Set(Name=mesh.BFGNames[ibfgrp])

		# Basis, order data for each element group
		# For now, ssume uniform basis and order for each element group 
		if type(basis) is str:
			basis = BasisType[basis]
		self.Bases = [basis for egrp in range(mesh.nElemGroup)] 
		if type(Order) is int:
			self.Orders = [Order for egrp in range(mesh.nElemGroup)]
		elif type(Order) is list:
			self.Orders = Order
		else:
			raise Exception("Input error")

		# State 
		# self.U = ArrayList(nArray=mesh.nElemGroup,nEntriesPerArray=mesh.nElems,FullDim=[mesh.nElemTot,nn,self.StateRank])
		# self.U = ArrayList(nArray=mesh.nElemGroup,ArrayDims=[[mesh.nElemTot,nn,self.StateRank]])
		ArrayDims = [[mesh.nElems[egrp],Order2nNode(self.Bases[egrp], self.Orders[egrp]), self.StateRank] \
					for egrp in range(mesh.nElemGroup)]
		self.U = ArrayList(nArray=mesh.nElemGroup,ArrayDims=ArrayDims)

		# BC treatments
		self.SetBCTreatment()

		# State indices
		self.StateIndices = {}
		for key in self.StateVariables.__members__.keys():
			self.StateIndices[key] = self.StateVariables.__members__.keys().index(key)

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
			Params["Velocity"] = 1.
			Params["ConvFlux"] = self.ConvFluxType["Upwind"]
		# Overwrite
		for key in kwargs:
			if key not in Params.keys(): raise Exception("Input error")
			if key is "ConvFlux":
				Params[key] = self.ConvFluxType[kwargs[key]]
			else:
				Params[key] = kwargs[key]

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
	    pass

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

	def QuadOrder(self, Order):
		return 2*Order+1

	def getWaveSpeed(self):
		return self.Params["Velocity"]

	def ConvFluxInterior(self, u, F):
		a = self.Params["Velocity"]
		if F is None:
			F = np.zeros(u.shape + (self.Dim,))
		for d in range(self.Dim):
			F[:,:,d] = a*u
		# F = a*u
		# F.shape = u.shape + (self.Dim,) 
		return F

	def ConvFluxUpwind(self, uL, uR, a, n):
		Vn = a*n 

		# upwind
		if Vn >= 0.:
			sL = 1.
			sR = 0.
		else:
			sL = 0. 
			sR = 1.

		F = Vn*(sL*uL + sR*uR)

		return F

	def ConvFluxNumerical(self, uL, uR, NData, nq, data):
		# nq = NData.nq
		if nq != uL.shape[0] or nq != uR.shape[0]:
			raise Exception("Wrong nq")

		if self.Params["ConvFlux"] != self.ConvFluxType.Upwind:
			raise Exception("Invalid inviscid flux function")

		try: 
			F = data.F
		except AttributeError: 
			data.F = F = np.zeros_like(uL)

		a = self.Params["Velocity"]
		ConvFlux = self.Params["ConvFlux"]

		for iq in range(nq):
			nvec = NData.nvec[iq,:]
			n = nvec/np.linalg.norm(nvec)

			if ConvFlux == self.ConvFluxType.Upwind:
				F[iq,:] = self.ConvFluxUpwind(uL[iq,:], uR[iq,:], a, n)
			else:
				raise Exception("Invalid flux function")

		return F

	def BoundaryState(self, BC, nq, xglob, Time, NData, uI, uB=None):
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

	def ConvFluxBoundary(self, BC, uI, uB, NData, nq, data):
		bctreatment = self.BCTreatments[BC.BCType]
		if bctreatment == self.BCTreatment.Riemann:
			F = self.ConvFluxNumerical(uI, uB, NData, nq, data)
		else:
			# Prescribe analytic flux
			try:
				Fa = data.Fa
			except AttributeError:
				data.Fa = Fa = np.zeros([nq, self.StateRank, self.Dim])
			Fa = self.ConvFluxInterior(uB, Fa)
			# Take dot product with n
			try: 
				F = data.F
			except AttributeError:
				data.F = F = np.zeros_like(uI)
			for jr in range(self.StateRank):
				F[:,jr] = np.sum(Fa[:,jr,:]*NData.nvec, axis=1)

		return F

	def ComputeScalars(self, ScalarNames, U, nq, scalar=None):
		if type(ScalarNames) is list:
			nscalar = len(ScalarNames)
		elif type(ScalarNames) is str:
			nscalar = 1
			ScalarNames = [ScalarNames]
		else:
			raise TypeError

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
				scalar[:,iscalar:iscalar+1] = self.AdditionalScalars(sname, U, scalar[:,iscalar:iscalar+1])

		return scalar

	def AdditionalScalars(self, ScalarName, U, scalar):
		raise NotImplementedError

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
			FcnData.U = np.zeros([nq, sr],dtype=self.U.Arrays[0].dtype)

		FcnData.U[:] = FcnData.Function(FcnData)

		return FcnData.U

	def FcnUniform(self, FcnData):
		Data = FcnData.Data
		U = FcnData.U
		sr = self.StateRank

		for k in range(sr):
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

		a = self.Params["Velocity"]

		U[:] = np.sin(omega*(x-a*t))

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

	