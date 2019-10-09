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
	def __init__(self,Order,Basis,mesh,StateRank=1):
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

		# Boundary conditions
		# self.BCs = []
		# for ibfgrp in range(mesh.nBFaceGroup):
		# 	self.BCs.append(BCData(Title=mesh.BFGTitles[ibfgrp]))
		self.BCs = [BCData() for ibfgrp in range(mesh.nBFaceGroup)]
		for ibfgrp in range(mesh.nBFaceGroup):
			self.BCs[ibfgrp].Title = mesh.BFGTitles[ibfgrp]
			# self.BCs[0].Set(Title=mesh.BFGTitles[ibfgrp])

		# Basis, order data for each element group
		if type(Basis) is str:
			Basis = BasisType[Basis]
		self.Bases = [Basis for egrp in range(mesh.nElemGroup)] 
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

	class VariableType(IntEnum):
	    Scalar = 0
	    # Additional scalars go here

	class VarLabelType(Enum):
		# LaTeX format
	    Scalar = "u"
	    # Additional scalars go here

	class BCType(IntEnum):
	    FullState = 0
	    Extrapolation = 1

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

	def BoundaryState(self, BC, nq, xglob, Time, uI, uB=None):

		if uB is not None:
			BC.U = uB

		bctype = BC.BCType
		if bctype == self.BCType.FullState:
			BC.x = xglob
			BC.nq = nq
			BC.Time = Time
			uB = self.CallFunction(BC)
		elif bctype == self.BCType.Extrapolation:
			uB[:] = uI[:]
		else:
			raise Exception("BC type not supported")

		return uB

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
			FcnData.U = np.zeros([nq, sr])

		FcnData.U[:] = FcnData.Function(FcnData)

		return FcnData.U

	def FcnUniform(self, FcnData):
		Data = FcnData.Data
		U = FcnData.U
		sr = self.StateRank

		for k in range(sr):
			U[k] = Data.U[k]

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

	