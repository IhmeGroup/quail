from abc import ABC, abstractmethod
import code
import numpy as np
from scipy.optimize import root
import sys

from numerics.basis.basis import order_to_num_basis_coeff
from general import *
import errors
from physics.base.data import ICData, BCData, ExactData, SourceData
import physics.base.functions as base_fcns
from physics.base.functions import FcnType as base_fcn_type
from physics.base.functions import BCType as base_BC_type


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

	def compute_flux(self, EqnSet, UL, UR, n):
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


def process_map(fcn_type, fcn_map):
	if fcn_type != "":
		# Update kwargs with reference to desired function 
		for fcn_keys in fcn_map.keys():
			if fcn_keys.name == fcn_type:
				# kwargs.update(Function=fcn_map[fcn_keys])
				fcn_ref = fcn_map[fcn_keys]
				break
	return fcn_ref


class PhysicsBase(object):
	'''
	Class: IFace
	--------------------------------------------------------------------------
	This is a class defined to encapsulate the temperature table with the 
	relevant methods
	'''
	@property
	@abstractmethod
	def StateRank(self):
		pass

	def __init__(self, order, basis, mesh):
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
		self.Params = {}
		self.IC = None
		self.ExactSoln = None
		self.ConvFluxFcn = None
		self.BCTreatments = {}
		self.Sources = []
		# Boundary conditions
		# self.BCs = []
		# for ibfgrp in range(mesh.nBFaceGroup):
		# 	self.BCs.append(BCData(Name=mesh.BFGNames[ibfgrp]))
		self.nBC = mesh.nBFaceGroup
		# self.BCs = [BCData() for ibfgrp in range(mesh.nBFaceGroup)]
		# for ibfgrp in range(mesh.nBFaceGroup):
		# 	self.BCs[ibfgrp].Name = mesh.BFGNames[ibfgrp]
		# 	# self.BCs[0].Set(Name=mesh.BFGNames[ibfgrp])
		self.BCs = [None]*mesh.nBFaceGroup

		# Basis, Order data for each element group
		# For now, ssume uniform basis and Order for each element group 
		if type(basis) is str:
			basis = BasisType[basis]
		self.Basis = basis
		if type(order) is int:
			self.order = order
		elif type(order) is list:
			self.order = order[0]
		else:
			raise Exception("Input error")

		# State 
		# self.U = ArrayList(nArray=mesh.nElemGroup,nEntriesPerArray=mesh.nElems,FullDim=[mesh.nElemTot,nn,self.StateRank])
		# self.U = ArrayList(nArray=mesh.nElemGroup,ArrayDims=[[mesh.nElemTot,nn,self.StateRank]])
		# ArrayDims = [[mesh.nElems[egrp],order_to_num_basis_coeff(self.Bases[egrp], self.Orders[egrp]), self.StateRank] \
		# 			for egrp in range(mesh.nElemGroup)]
		# self.U = ArrayList(nArray=mesh.nElemGroup,ArrayDims=ArrayDims)
		# self.S = ArrayList(nArray=mesh.nElemGroup,ArrayDims=ArrayDims)
		self.U = np.zeros([mesh.nElem, order_to_num_basis_coeff(self.Basis, self.order), self.StateRank])
		self.S = np.zeros([mesh.nElem, order_to_num_basis_coeff(self.Basis, self.order), self.StateRank])

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

		self.fcn_map = {
			base_fcn_type.Uniform : base_fcns.Uniform,
		}

		self.BC_map = {
			base_BC_type.FullState : base_fcns.FullState,
			base_BC_type.Extrapolate : base_fcns.Extrapolate,
		}

		self.source_map = {
		}


	@abstractmethod
	def SetParams(self,**kwargs):
		Params = self.Params
		# Overwrite
		for key in kwargs:
			# Params[key] = kwargs[key]
			# if key not in Params.keys(): raise Exception("Input error")
			if key is "ConvFlux":
				Params[key] = self.ConvFluxType[kwargs[key]]
			else:
				Params[key] = kwargs[key]

	def set_IC(self, IC_type, **kwargs):
		fcn_ref = process_map(IC_type, self.fcn_map)
		self.IC = fcn_ref(**kwargs)

	def set_exact(self, exact_type, **kwargs):
		fcn_ref = process_map(exact_type, self.fcn_map)
		self.ExactSoln = fcn_ref(**kwargs)

	def set_BC(self, BC_type, fcn_type=None, **kwargs):
		for i in range(len(self.BCs)):
			BC = self.BCs[i]
			if BC is None:
				if fcn_type is not None:
					fcn_ref = process_map(fcn_type, self.fcn_map)
					kwargs.update(function=fcn_ref)
				BC_ref = process_map(BC_type, self.BC_map)
				BC = BC_ref(**kwargs)
				self.BCs[i] = BC
	def SetBC(self, BCName, **kwargs):
		found = False
		code.interact(local=locals())
		for BC in self.BCs:
			if BC.Name == BCName:
				BC.Set(**kwargs)
				found = True
				break

		if not found:
			raise NameError

	def set_source(self, source_type, **kwargs):
		source_ref = process_map(source_type, self.source_map)
		Source = source_ref(**kwargs)
		self.Sources.append(Source)
		
	@abstractmethod
	class StateVariables(Enum):
		pass

	@abstractmethod
	class AdditionalVariables(Enum):
		pass

	def GetStateIndex(self, VariableName):
		# idx = self.VariableType[VariableName]
		idx = self.StateIndices[VariableName]
		# idx = self.StateVariables.__members__.keys().index(VariableName)
		return idx

	@abstractmethod
	class BCType(IntEnum):
		pass

	@abstractmethod
	class BCTreatment(IntEnum):
		pass

	def SetBCTreatment(self):
		# default is Prescribed
		self.BCTreatments = {n:self.BCTreatment.Prescribed for n in range(len(self.BCType))}
		self.BCTreatments[self.BCType.FullState] = self.BCTreatment.Riemann

	@abstractmethod
	class ConvFluxType(IntEnum):
		pass

	# def SetSource(self, **kwargs):
	# 	#append src data to Sources list 
	# 	Source = SourceData()
	# 	self.Sources.append(Source)
	# 	Source.Set(**kwargs)

	def QuadOrder(self, Order):
		return 2*Order+1

	@abstractmethod
	def ConvFluxInterior(self, u):
		pass

	@abstractmethod
	def ConvFluxNumerical(self, uL, uR, normals):
		pass

	@abstractmethod
	def BoundaryState(self, BC, nq, xglob, Time, normals, uI):
		pass

	#Source state takes multiple source terms (if needed) and sums them together. 
	def SourceState(self, nq, xglob, Time, u, s=None):
		for Source in self.Sources:

			#loop through available source terms
			Source.x = xglob
			Source.nq = nq
			Source.Time = Time
			Source.U = u
			s += self.CallSourceFunction(Source,Source.x,Source.Time)

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

	@abstractmethod
	def AdditionalScalars(self, ScalarName, U, scalar, FlagNonPhysical):
		pass

	def CallFunction(self, FcnData, x, t):
		# for key in kwargs:
		# 	if key is "x":
		# 		FcnData.x = kwargs[key]
		# 		FcnData.nq = FcnData.x.shape[0]
		# 	elif key is "Time":
		# 		FcnData.Time = kwargs[key]
		# 	else:
		# 		raise Exception("Input error")

		# nq = FcnData.nq
		# sr = self.StateRank
		# if FcnData.U is None or FcnData.U.shape != (nq, sr):
		# 	FcnData.U = np.zeros([nq, sr], dtype=self.U.dtype)

		# FcnData.U[:] = FcnData.Function(self, FcnData)
		# FcnData.alloc_helpers([x.shape[0], self.StateRank])
		FcnData.Up = FcnData.get_state(self, x, t)

		return FcnData.Up

	def CallSourceFunction(self, FcnData, x, t):
		# for key in kwargs:
		# 	if key is "x":
		# 		FcnData.x = kwargs[key]
		# 		FcnData.nq = FcnData.x.shape[0]
		# 	elif key is "Time":
		# 		FcnData.Time = kwargs[key]
		# 	else:
		# 		raise Exception("Input error")

		# nq = FcnData.nq
		# sr = self.StateRank
		# if FcnData.S is None or FcnData.S.shape != (nq, sr):
		# 	FcnData.S = np.zeros([nq, sr], dtype=self.S.dtype)
		# code.interact(local=locals())
		FcnData.S = FcnData.get_source(self, FcnData, x, t)

		return FcnData.S

	def FcnUniform(self, FcnData):
		Data = FcnData.Data
		U = FcnData.U
		ns = self.StateRank

		for k in range(ns):
			U[:,k] = Data.State[k]

		return U