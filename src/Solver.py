import numpy as np 
from Basis import *
from Quadrature import *
from Mesh import *
import code
import copy
from Data import ArrayList, GenericData
from SolverTools import *
from stepper import *
import time
import MeshTools
import Post
import Errors
from scipy.optimize import root

import Limiter

global echeck
echeck = -1

class DG_Solver(object):
	'''
	Class: DG_Solver
	--------------------------------------------------------------------------
	Use the vanilla DG method to solve a given set of PDEs
	'''
	def __init__(self,Params,EqnSet,mesh):

		self.Params = Params
		self.EqnSet = EqnSet
		self.mesh = mesh
		self.DataSet = GenericData()

		self.Time = Params["StartTime"]
		self.nTimeStep = 0 # will be set later

		TimeScheme = Params["TimeScheme"]
		if TimeScheme is "FE":
			Stepper = FE()
		elif TimeScheme is "RK4":
			Stepper = RK4()
		elif TimeScheme is "LSRK4":
			Stepper = LSRK4()
		elif TimeScheme is "SSPRK3":
			Stepper = SSPRK3()
		elif TimeScheme is "ADER":
			Stepper = ADER()
		else:
			raise NotImplementedError("Time scheme not supported")
		# if Params["nTimeStep"] > 0:
		# 	Stepper.dt = Params["EndTime"]/Params["nTimeStep"]
		self.Stepper = Stepper

		# Limiter
		limiterType = Params["ApplyLimiter"]
		self.Limiter = Limiter.SetLimiter(limiterType)

		# Check validity of parameters
		self.CheckSolverParams()

		# Initialize state
		self.InitState()


	def CheckSolverParams(self):
		Params = self.Params
		mesh = self.mesh
		EqnSet = self.EqnSet
		### Check interp basis validity
		if BasisType[Params["InterpBasis"]] == BasisType.SegLagrange or BasisType[Params["InterpBasis"]] == BasisType.SegLegendre:
		    if mesh.Dim != 1:
		        raise Errors.IncompatibleError
		else:
		    if mesh.Dim != 2:
		        raise Errors.IncompatibleError

		### Check uniform mesh
		if Params["UniformMesh"] is True:
		    ''' 
		    Check that element volumes are the same
		    Note that this is a necessary but not sufficient requirement
		    '''
		    TotVol, ElemVols = MeshTools.ElementVolumes(mesh)
		    if (ElemVols.Max - ElemVols.Min)/TotVol > 1.e-8:
		        raise ValueError

		### Check linear geometric mapping
		if Params["LinearGeomMapping"] is True:
		    for EG in mesh.ElemGroups:
		        if EG.QOrder != 1:
		            raise Errors.IncompatibleError
		        if EG.QBasis == BasisType.QuadLagrange \
		            and Params["UniformMesh"] is False:
		            raise Errors.IncompatibleError

		### Check limiter ###
		if Params["ApplyLimiter"] is 'ScalarPositivityPreserving' \
			and EqnSet.StateRank > 1:
				raise IncompatibleError
		if Params["ApplyLimiter"] is 'PositivityPreserving' \
			and EqnSet.StateRank == 1:
				raise IncompatibleError

		### Check time integration scheme ###
		TimeScheme = Params["TimeScheme"]		
		if TimeScheme is "ADER":
			raise Errors.IncompatibleError


	def InitState(self):
		mesh = self.mesh
		EqnSet = self.EqnSet
		U = EqnSet.U.Arrays
		sr = EqnSet.StateRank
		Params = self.Params
		# f = np.zeros([nq,sr])

		# Get mass matrices
		try:
			MMinv_all = self.DataSet.MMinv_all
		except AttributeError:
			# not found; need to compute
			MMinv_all = ComputeInvMassMatrices(mesh, EqnSet, solver=self)

		InterpolateIC = Params["InterpolateIC"]
		quadData = None
		JData = JacobianData(mesh)
		GeomPhiData = None
		xq = None; xphys = None; QuadChanged = True
		for egrp in range(mesh.nElemGroup):
			U_ = U[egrp]

			basis = EqnSet.Bases[egrp]
			Order = EqnSet.Orders[egrp]
			rhs = np.zeros([Order2nNode(basis,Order),sr],dtype=U_.dtype)
			for elem in range(mesh.ElemGroups[egrp].nElem):

				if not InterpolateIC:
					QuadOrder,QuadChanged = GetQuadOrderElem(mesh, egrp, EqnSet.Bases[egrp],
						# 1, EqnSet, quadData)
						2*np.amax([Order,1]), EqnSet, quadData)
					if QuadChanged:
						quadData = QuadData(mesh, mesh.ElemGroups[egrp].QBasis, EntityType.Element, QuadOrder)

					nq = quadData.nquad
					xq = quadData.xquad
					wq = quadData.wquad

					if QuadChanged:
						# PhiData = BasisData(egrp,Order,EntityType.Element,nq,xq,mesh,True,False)
						PhiData = BasisData(basis,Order,nq,mesh)
						PhiData.EvalBasis(xq, Get_Phi=True)
						xphys = np.zeros([nq, mesh.Dim])

					JData.ElemJacobian(egrp,elem,nq,xq,mesh,get_djac=True)

					# MMinv = GetInvMassMatrix(mesh, egrp, 0, basis, EqnSet.Orders[egrp])
					MMinv = MMinv_all.Arrays[egrp][elem]

					nn = PhiData.nn
				else:
					xq, nq = EquidistantNodes(basis, Order, xq)
					nn = nq

				xphys, GeomPhiData = Ref2Phys(mesh, egrp, elem, GeomPhiData, nq, xq, xphys, QuadChanged)
				# if sr == 1: f = f_ic(xphys)
				# else : 
				# 	# f = SmoothIsentropic1D(x=xphys,t=0.,gam=EqnSet.Params["SpecificHeatRatio"])
				# 	f = EqnSet.CallFunction(EqnSet.IC, x=xphys, Time=0.)
				f = EqnSet.CallFunction(EqnSet.IC, x=xphys, Time=self.Time)
				f.shape = nq,sr

				if not InterpolateIC:
					rhs *= 0.
					for n in range(nn):
						for iq in range(nq):
							rhs[n,:] += f[iq,:]*PhiData.Phi[iq,n]*wq[iq]*JData.djac[iq*(JData.nq != 1)]

					U_[elem,:,:] = np.dot(MMinv,rhs)
				else:
					U_[elem] = f

	def ApplyLimiter(self, U):
		if self.Limiter is not None:
			self.Limiter.LimitSolution(self, U)


	def CalculateResidualElem(self, egrp, elem, U, ER, StaticData):
		# U = EqnSet.U[egrp][elem] # [nn,sr]
		mesh = self.mesh
		EqnSet = self.EqnSet

		basis = EqnSet.Bases[egrp]
		Order = EqnSet.Orders[egrp]
		entity = EntityType.Element
		sr = EqnSet.StateRank
		dim = EqnSet.Dim
		# ER = R[egrp][elem] # [nn,sr]

		if Order == 0:
			return ER, StaticData

		if StaticData is None:
			pnq = -1
			quadData = None
			PhiData = None
			JData = JData = JacobianData(mesh)
			xglob = None
			u = None
			F = None
			s = None
			NData = None
			GeomPhiData = None
			StaticData = GenericData()
		else:
			nq = StaticData.pnq
			quadData = StaticData.quadData
			PhiData = StaticData.PhiData
			JData = StaticData.JData
			xglob = StaticData.xglob
			u = StaticData.u
			F = StaticData.F
			s = StaticData.s
			NData = StaticData.NData
			GeomPhiData = StaticData.GeomPhiData


		QuadOrder,QuadChanged = GetQuadOrderElem(mesh, egrp, mesh.ElemGroups[egrp].QBasis, Order, EqnSet, quadData)
		if QuadChanged:
			quadData = QuadData(mesh, basis, entity, QuadOrder)

		nq = quadData.nquad
		xq = quadData.xquad
		wq = quadData.wquad

		# PhiData = BasisData(egrp,Order,entity,nq,xq,mesh,True,True)
		if QuadChanged:
			PhiData = BasisData(EqnSet.Bases[egrp],Order,nq,mesh)
			PhiData.EvalBasis(xq, Get_Phi=True, Get_GPhi=True)
			# PhiData.PhysicalGrad(JData)

			xglob = np.zeros([nq, dim])
			u = np.zeros([nq, sr])
			F = np.zeros([nq, sr, dim])
			s = np.zeros([nq, sr])



		JData.ElemJacobian(egrp,elem,nq,xq,mesh,get_djac=True,get_jac=False,get_ijac=True)
		PhiData.EvalBasis(xq, Get_gPhi=True, JData=JData)

		xglob, GeomPhiData = Ref2Phys(mesh, egrp, elem, GeomPhiData, nq, xq, xglob, QuadChanged)

		nn = PhiData.nn

		# interpolate state and gradient at quad points
		u[:] = np.matmul(PhiData.Phi, U)

		F = EqnSet.ConvFluxInterior(u, F) # [nq,sr,dim]

		for ir in range(sr):
			for jn in range(nn):
				for iq in range(nq):
					gPhi = PhiData.gPhi[iq,jn] # dim
					ER[jn,ir] += np.dot(gPhi, F[iq,ir,:])*wq[iq]*JData.djac[iq*(JData.nq!=1)]

		s = np.zeros([nq,sr])
		s = EqnSet.SourceState(nq, xglob, self.Time, NData, u, s) # [nq,sr,dim]
		# Calculate source term integral
		for ir in range(sr):
			for jn in range(nn):
				for iq in range(nq):
					Phi = PhiData.Phi[iq,jn]
					ER[jn,ir] += Phi*s[iq,ir]*wq[iq]*JData.djac[iq*(JData.nq!=1)]

		if elem == echeck:
			code.interact(local=locals())

		# Store in StaticData
		StaticData.pnq = nq
		StaticData.quadData = quadData
		StaticData.PhiData = PhiData
		StaticData.JData = JData
		StaticData.xglob = xglob
		StaticData.u = u
		StaticData.F = F
		StaticData.s = s
		StaticData.NData = NData
		StaticData.GeomPhiData = GeomPhiData

		return ER, StaticData


	def CalculateResidualElems(self, U, R):

		mesh = self.mesh
		EqnSet = self.EqnSet
		StaticData = None

		for egrp in range(mesh.nElemGroup):
			for elem in range(mesh.nElems[egrp]):
				R[egrp][elem], StaticData = self.CalculateResidualElem(egrp, elem, U[egrp][elem], R[egrp][elem], StaticData)


	def CalculateResidualIFace(self, iiface, UL, UR, RL, RR, StaticData):
		mesh = self.mesh
		dim = mesh.Dim
		EqnSet = self.EqnSet
		sr = EqnSet.StateRank

		IFace = mesh.IFaces[iiface]
		egrpL = IFace.ElemGroupL
		egrpR = IFace.ElemGroupR
		elemL = IFace.ElemL
		elemR = IFace.ElemR
		faceL = IFace.faceL
		faceR = IFace.faceR
		OrderL = EqnSet.Orders[egrpL]
		OrderR = EqnSet.Orders[egrpR]
		nFacePerElemL = mesh.ElemGroups[egrpL].nFacePerElem
		nFacePerElemR = mesh.ElemGroups[egrpR].nFacePerElem

		if StaticData is None:
			pnq = -1
			quadData = None
			PhiDataL = None
			PhiDataR = None
			xelemL = None
			xelemR = None
			uL = None
			uR = None
			StaticData = GenericData()
			NData = None
		else:
			nq = StaticData.pnq
			quadData = StaticData.quadData
			PhiDataL = StaticData.PhiDataL
			PhiDataR = StaticData.PhiDataR
			xelemL = StaticData.xelemL
			xelemR = StaticData.xelemR
			uL = StaticData.uL
			uR = StaticData.uR
			NData = StaticData.NData
			Faces2PhiDataL = StaticData.Faces2PhiDataL
			Faces2PhiDataR = StaticData.Faces2PhiDataR

		QuadOrder, QuadChanged = GetQuadOrderIFace(mesh, IFace, mesh.ElemGroups[egrpL].QBasis, np.amax([OrderL,OrderR]), EqnSet, quadData)

		if QuadChanged:
			quadData = QuadData(mesh, EqnSet.Bases[egrpL], EntityType.IFace, QuadOrder)

		nq = quadData.nquad
		xq = quadData.xquad
		wq = quadData.wquad

		if QuadChanged:
			Faces2PhiDataL = [None for i in range(nFacePerElemL)]
			Faces2PhiDataR = [None for i in range(nFacePerElemR)]
			# PhiDataL = BasisData(EqnSet.Bases[egrpL],OrderL,nq,mesh)
			# PhiDataR = BasisData(EqnSet.Bases[egrpR],OrderR,nq,mesh)

			xelemL = np.zeros([nq, dim])
			xelemR = np.zeros([nq, dim])
			uL = np.zeros([nq, sr])
			uR = np.zeros([nq, sr])

		PhiDataL = Faces2PhiDataL[faceL]
		PhiDataR = Faces2PhiDataR[faceR]

		if PhiDataL is None or QuadChanged:
			Faces2PhiDataL[faceL] = PhiDataL = BasisData(EqnSet.Bases[egrpL],OrderL,nq,mesh)
			xelemL = PhiDataL.EvalBasisOnFace(mesh, egrpL, faceL, xq, xelemL, Get_Phi=True)
		if PhiDataR is None or QuadChanged:
			Faces2PhiDataR[faceR] = PhiDataR = BasisData(EqnSet.Bases[egrpR],OrderR,nq,mesh)
			xelemR = PhiDataR.EvalBasisOnFace(mesh, egrpR, faceR, xq[::-1], xelemR, Get_Phi=True)

		# if faceL != PhiDataL.face:
		# 	xelemL = PhiDataL.EvalBasisOnFace(mesh, egrpL, faceL, xq, xelemL, Get_Phi=True)
		# if faceR != PhiDataR.face:
		# 	xelemR = PhiDataR.EvalBasisOnFace(mesh, egrpR, faceR, xq[::-1], xelemR, Get_Phi=True)

		# PhiDataL = BasisData(ShapeType.Segment,OrderL,nq,mesh)
		# PhiDataL.EvalBasisOnFace(mesh, egrpL, faceL, xq, True, False, False, None)
		# PhiDataR = BasisData(ShapeType.Segment,OrderR,nq,mesh)
		# PhiDataR.EvalBasisOnFace(mesh, egrpR, faceR, xq, True, False, False, None)

		NData = IFaceNormal(mesh, IFace, nq, xq, NData)
		# NData.nvec *= wq

		nL = PhiDataL.nn
		nR = PhiDataR.nn
		nn = np.amax([nL,nR])

		# interpolate state and gradient at quad points
		# uL = np.zeros([nq, sr])
		# uR = np.zeros([nq, sr])
		for ir in range(sr):
			uL[:,ir] = np.matmul(PhiDataL.Phi, UL[:,ir])
			uR[:,ir] = np.matmul(PhiDataR.Phi, UR[:,ir])

		F = EqnSet.ConvFluxNumerical(uL, uR, NData, nq, StaticData) # [nq,sr]

		RL -= np.matmul(PhiDataL.Phi.transpose(), F*wq) # [nn,sr]
		RR += np.matmul(PhiDataR.Phi.transpose(), F*wq) # [nn,sr]

		if elemL == echeck or elemR == echeck:
			if elemL == echeck: print("Left!")
			else: print("Right!")
			code.interact(local=locals())

		# Store in StaticData
		StaticData.pnq = nq
		StaticData.quadData = quadData
		StaticData.PhiDataL = PhiDataL
		StaticData.PhiDataR = PhiDataR
		StaticData.xelemL = xelemL
		StaticData.xelemR = xelemR
		StaticData.uL = uL
		StaticData.uR = uR
		StaticData.NData = NData
		StaticData.Faces2PhiDataL = Faces2PhiDataL
		StaticData.Faces2PhiDataR = Faces2PhiDataR

		return RL, RR, StaticData



	def CalculateResidualIFaces(self, U, R):
		mesh = self.mesh
		EqnSet = self.EqnSet
		StaticData = None

		for iiface in range(mesh.nIFace):
			IFace = mesh.IFaces[iiface]
			egrpL = IFace.ElemGroupL
			egrpR = IFace.ElemGroupR
			elemL = IFace.ElemL
			elemR = IFace.ElemR
			faceL = IFace.faceL
			faceR = IFace.faceR

			UL = U[egrpL][elemL]
			UR = U[egrpR][elemR]
			RL = R[egrpL][elemL]
			RR = R[egrpR][elemR]

			RL, RR, StaticData = self.CalculateResidualIFace(iiface, UL, UR, RL, RR, StaticData)

	def CalculateResidualBFace(self, ibfgrp, ibface, U, R, StaticData):
		mesh = self.mesh
		EqnSet = self.EqnSet
		sr = EqnSet.StateRank
		dim = mesh.Dim

		BFG = mesh.BFaceGroups[ibfgrp]
		BFace = BFG.BFaces[ibface]
		egrp = BFace.ElemGroup
		elem = BFace.Elem
		face = BFace.face
		Order = EqnSet.Orders[egrp]
		nFacePerElem = mesh.ElemGroups[egrp].nFacePerElem

		if StaticData is None:
			pnq = -1
			quadData = None
			PhiData = None
			xglob = None
			uI = None
			uB = None
			NData = None
			GeomPhiData = None
			StaticData = GenericData()
			Faces2xelem = None
		else:
			nq = StaticData.pnq
			quadData = StaticData.quadData
			PhiData = StaticData.PhiData
			xglob = StaticData.xglob
			uI = StaticData.uI
			uB = StaticData.uB
			NData = StaticData.NData
			GeomPhiData = StaticData.GeomPhiData
			Faces2PhiData = StaticData.Faces2PhiData
			Faces2xelem = StaticData.Faces2xelem

		QuadOrder, QuadChanged = GetQuadOrderBFace(mesh, BFace, mesh.ElemGroups[egrp].QBasis, Order, EqnSet, quadData)

		if QuadChanged:
			quadData = QuadData(mesh, EqnSet.Bases[egrp], EntityType.BFace, QuadOrder)

		nq = quadData.nquad
		xq = quadData.xquad
		wq = quadData.wquad

		if QuadChanged:
			Faces2PhiData = [None for i in range(nFacePerElem)]
			# PhiData = BasisData(EqnSet.Bases[egrp],Order,nq,mesh)
			xglob = np.zeros([nq, dim])
			uI = np.zeros([nq, sr])
			uB = np.zeros([nq, sr])
			Faces2xelem = np.zeros([nFacePerElem, nq, dim])

		PhiData = Faces2PhiData[face]
		xelem = Faces2xelem[face]
		if PhiData is None or QuadChanged:
			Faces2PhiData[face] = PhiData = BasisData(EqnSet.Bases[egrp],Order,nq,mesh)
			Faces2xelem[face] = xelem = PhiData.EvalBasisOnFace(mesh, egrp, face, xq, xelem, Get_Phi=True)

		# if face != PhiData.face:
		# 	xelem = PhiData.EvalBasisOnFace(mesh, egrp, face, xq, xelem, Get_Phi=True)

		# PhiData.EvalBasisOnFace(mesh, egrp, face, xq, True, False, False, None)

		NData = BFaceNormal(mesh, BFace, nq, xq, NData)

		PointsChanged = QuadChanged or face != GeomPhiData.face
		xglob, GeomPhiData = Ref2Phys(mesh, egrp, elem, GeomPhiData, nq, xelem, xglob, PointsChanged)

		nn = PhiData.nn

		# interpolate state and gradient at quad points
		for ir in range(sr):
			uI[:,ir] = np.matmul(PhiData.Phi, U[:,ir])

		# Get boundary state
		BC = EqnSet.BCs[ibfgrp]
		uB = EqnSet.BoundaryState(BC, nq, xglob, self.Time, NData, uI, uB)

		# NData.nvec *= wq
		F = EqnSet.ConvFluxBoundary(BC, uI, uB, NData, nq, StaticData) # [nq,sr]

		R -= np.matmul(PhiData.Phi.transpose(), F*wq) # [nn,sr]

		if elem == echeck:
			code.interact(local=locals())

		# Store in StaticData
		StaticData.pnq = nq
		StaticData.quadData = quadData
		StaticData.PhiData = PhiData
		StaticData.uI = uI
		StaticData.uB = uB
		StaticData.F = F
		StaticData.xglob = xglob
		StaticData.NData = NData
		StaticData.GeomPhiData = GeomPhiData
		StaticData.Faces2PhiData = Faces2PhiData
		StaticData.Faces2xelem = Faces2xelem

		return R, StaticData


	def CalculateResidualBFaces(self, U, R):
		mesh = self.mesh
		EqnSet = self.EqnSet
		StaticData = None

		for ibfgrp in range(mesh.nBFaceGroup):
			BFG = mesh.BFaceGroups[ibfgrp]

			for ibface in range(BFG.nBFace):
				BFace = BFG.BFaces[ibface]
				egrp = BFace.ElemGroup
				elem = BFace.Elem
				face = BFace.face

				R[egrp][elem], StaticData = self.CalculateResidualBFace(ibfgrp, ibface, U[egrp][elem], R[egrp][elem], StaticData)


	def CalculateResidual(self, U, R):
		mesh = self.mesh
		EqnSet = self.EqnSet

		if R is None:
			R = ArrayList(SimilarArray=U)
		# Initialize residual to zero
		# for egrp in range(mesh.nElemGroup): R[egrp][:] = 0.
		R.SetUniformValue(0.)

		self.CalculateResidualBFaces(U.Arrays, R.Arrays)
		self.CalculateResidualElems(U.Arrays, R.Arrays)
		self.CalculateResidualIFaces(U.Arrays, R.Arrays)

		return R


	def ApplyTimeScheme(self, fhistory=None):

		EqnSet = self.EqnSet
		mesh = self.mesh
		order = self.Params["InterpOrder"]
		Stepper = self.Stepper
		Time = self.Time

		# Parameters
		TrackOutput = self.Params["TrackOutput"]

		t0 = time.time()
		for iStep in range(self.nTimeStep):

			# Integrate in time
			# self.Time is used for local time
			R = Stepper.TakeTimeStep(self)

			# Increment time
			Time += Stepper.dt
			self.Time = Time

			# Info to print
			PrintInfo = (iStep+1, self.Time, R.VectorNorm(ord=1))
			PrintString = "%d: Time = %g, Residual norm = %g" % (PrintInfo)

			# Output
			if TrackOutput:
				output,_ = Post.L2_error(mesh,EqnSet,Time,"Entropy",False)
				OutputString = ", Output = %g" % (output)
				PrintString += OutputString

			# Print info
			print(PrintString)

			# Write to file if requested
			if fhistory is not None:
				fhistory.write("%d %g %g" % (PrintInfo))
				if TrackOutput:
					fhistory.write(" %g" % (output))
				fhistory.write("\n")


		t1 = time.time()
		print("Wall clock time = %g seconds" % (t1-t0))


	def solve(self):
		mesh = self.mesh
		EqnSet = self.EqnSet

		OrderSequencing = self.Params["OrderSequencing"]
		InterpOrder = self.Params["InterpOrder"]
		nTimeStep = self.Params["nTimeStep"]
		EndTime = self.Params["EndTime"]
		WriteTimeHistory = self.Params["WriteTimeHistory"]
		if WriteTimeHistory:
			fhistory = open("TimeHistory.txt", "w")
		else:
			fhistory = None


		''' Convert to lists '''
		# InterpOrder
		if np.issubdtype(type(InterpOrder), np.integer):
			InterpOrders = [InterpOrder]
		elif type(InterpOrder) is list:
			InterpOrders = InterpOrder 
		else:
			raise TypeError
		nOrder = len(InterpOrders)
		# nTimeStep
		if np.issubdtype(type(nTimeStep), np.integer):
			nTimeSteps = [nTimeStep]*nOrder
		elif type(nTimeStep) is list:
			nTimeSteps = nTimeStep 
		else:
			raise TypeError
		# EndTime
		if np.issubdtype(type(EndTime), np.floating):
			EndTimes = []
			for i in range(nOrder):
				EndTimes.append(EndTime*(i+1))
		elif type(EndTime) is list:
			EndTimes = EndTime 
		else:
			raise TypeError


		''' Check compatibility '''
		if nOrder != len(nTimeSteps) or nOrder != len(EndTimes):
			raise ValueError

		if np.any(np.diff(EndTimes) < 0.):
			raise ValueError

		if OrderSequencing:
			if mesh.nElemGroup != 1:
				# only compatible with 
				raise Errors.IncompatibleError
		else:
			if len(InterpOrders) != 1:
				raise ValueError


		''' Loop through orders '''
		Time = 0.
		for iOrder in range(nOrder):
			Order = InterpOrders[iOrder]
			''' Compute time step '''
			self.Stepper.dt = (EndTimes[iOrder]-Time)/nTimeSteps[iOrder]
			self.nTimeStep = nTimeSteps[iOrder]

			''' After first iteration, project solution to next order '''
			if iOrder > 0:
				# Clear DataSet
				delattr(self, "DataSet")
				self.DataSet = GenericData()
				# Increment order
				Order_old = EqnSet.Orders[0]
				EqnSet.Orders[0] = Order
				# Project
				ProjectStateToNewBasis(self, EqnSet, mesh, EqnSet.Bases[0], Order_old)

			''' Apply time scheme '''
			self.ApplyTimeScheme(fhistory)

			Time = EndTimes[iOrder]


		if WriteTimeHistory:
			fhistory.close()

class ADERDG_Solver(DG_Solver):
	'''
	Class: ADERDG_Solver
	--------------------------------------------------------------------------
	Use the ADER DG method to solve a given set of PDEs
	'''
	
	def CheckSolverParams(self):
		Params = self.Params
		mesh = self.mesh
		EqnSet = self.EqnSet
		### Check interp basis validity
		if BasisType[Params["InterpBasis"]] == BasisType.SegLagrange or BasisType[Params["InterpBasis"]] == BasisType.SegLegendre:
		    if mesh.Dim != 1:
		        raise Errors.IncompatibleError
		else:
		    if mesh.Dim != 2:
		        raise Errors.IncompatibleError

		### Check uniform mesh
		if Params["UniformMesh"] is True:
		    ''' 
		    Check that element volumes are the same
		    Note that this is a necessary but not sufficient requirement
		    '''
		    TotVol, ElemVols = MeshTools.ElementVolumes(mesh)
		    if (ElemVols.Max - ElemVols.Min)/TotVol > 1.e-8:
		        raise ValueError

		### Check linear geometric mapping
		if Params["LinearGeomMapping"] is True:
		    for EG in mesh.ElemGroups:
		        if EG.QOrder != 1:
		            raise Errors.IncompatibleError
		        if EG.QBasis == BasisType.QuadLagrange \
		            and Params["UniformMesh"] is False:
		            raise Errors.IncompatibleError

		### Check limiter ###
		if Params["ApplyLimiter"] is 'ScalarPositivityPreserving' \
			and EqnSet.StateRank > 1:
				raise IncompatibleError
		if Params["ApplyLimiter"] is 'PositivityPreserving' \
			and EqnSet.StateRank == 1:
				raise IncompatibleError

		### Check time integration scheme ###
		TimeScheme = Params["TimeScheme"]
		if TimeScheme is not "ADER":
			raise Errors.IncompatibleError

		### Check flux/source coefficient interpolation compatability with basis functions.
		if Params["InterpolateFlux"] is True and BasisType[Params["InterpBasis"]] == BasisType.SegLegendre:
			raise Errors.IncompatibleError

		### Current build only supports scalar equations
		if EqnSet.StateRank > 1:
			raise Errors.IncompatibleError
	def CalculatePredictorStep(self, dt, W, Up):
		mesh = self.mesh
		EqnSet = self.EqnSet

		# Initialize predictor step to zero
		Up.SetUniformValue(0.)

		self.CalculatePredictorStepElems(dt, W.Arrays, Up.Arrays)

		return Up
	

	def CalculatePredictorStepElems(self, dt, W, Up):

		mesh = self.mesh
		EqnSet = self.EqnSet
		StaticData = None

		for egrp in range(mesh.nElemGroup):
			for elem in range(mesh.nElems[egrp]):
				Up[egrp][elem], StaticData = self.CalculatePredictorElemADER(egrp, elem, dt, W[egrp][elem], Up[egrp][elem], StaticData)


	def CalculatePredictorElemADER(self, egrp, elem, dt, W, Up, StaticData):

	
		mesh = self.mesh
		EqnSet = self.EqnSet
		sr = EqnSet.StateRank
		dim = mesh.Dim
		basis2 = EqnSet.Bases[egrp]
		basis1 = EqnSet.BasesADER[egrp]
		Order = EqnSet.Orders[egrp]
		entity = EntityType.Element

		quadData = None
		quadDataST = None
		QuadChanged = True; QuadChangedST = True
		EGroup=mesh.ElemGroups[egrp]
		Elem2Nodes = EGroup.Elem2Nodes[elem]
		dx = np.abs(mesh.Coords[Elem2Nodes[1],0]-mesh.Coords[Elem2Nodes[0],0])

		#Flux matrices in time
		FTL,_= GetTemporalFluxADER(mesh, basis1, basis1, Order, PhysicalSpace=False, egrp=0, elem=0, StaticData=None)
		FTR,_= GetTemporalFluxADER(mesh, basis1, basis2, Order, PhysicalSpace=False, egrp=0, elem=0, StaticData=None)
		
		#Stiffness matrix in time
		gradDir = 1
		SMT,_= GetStiffnessMatrixADER(gradDir,mesh, Order, egrp=0, elem=0, basis=basis1)
		gradDir = 0
		SMS,_= GetStiffnessMatrixADER(gradDir,mesh, Order, egrp=0, elem=0, basis=basis1)
		SMS = np.transpose(SMS)
		#MMinv,_= GetElemInvMassMatrixADER(mesh, basis1, Order, PhysicalSpace=True, egrp=-1, elem=-1, StaticData=None)

		QuadOrder,QuadChanged = GetQuadOrderElem(mesh, egrp, basis2, Order, EqnSet, quadData)
		QuadOrderST, QuadChangedST = GetQuadOrderElem(mesh, egrp, basis1, Order, EqnSet, quadDataST)
		QuadOrderST-=1

		if QuadChanged:
			quadData = QuadData(mesh, basis2, entity, QuadOrder)

		if QuadChangedST:
			quadDataST = QuadData(mesh, basis1, entity, QuadOrderST)

		nq = quadData.nquad
		xq = quadData.xquad
		wq = quadData.wquad

		nqST = quadDataST.nquad
		xqST = quadDataST.xquad
		wqST = quadDataST.wquad

		PhiData = BasisData(basis1,Order,nqST,mesh)
		PsiData = BasisData(basis2,Order,nq,mesh)

		nnST = PhiData.nn
		nn   = PsiData.nn

		#Make initial guess for the predictor step
		Up = np.reshape(Up,(nn,nn,sr))

		for ir in range(sr):
			#for i in range(nn):
			for j in range(nn):
				Up[0,j,ir]=W[j,ir]
		Up = np.reshape(Up,(nnST,sr))

		def F(x):
			x = np.reshape(x,(nnST,1))
			fluxpoly = self.FluxState(egrp, elem, dt, Order, basis1, x)
			f = np.matmul(FTL,x)-np.matmul(FTR,W)-np.matmul(SMT,x)+np.matmul(SMS,fluxpoly)
			#code.interact(local=locals())
			#print fluxpoly
			f = np.reshape(f,(nnST))
			return f

		for ir in range(sr):
			sol = root(F, Up[:,ir], method='krylov',tol=1.0e-6)
			#print sol
			#code.interact(local=locals())
			Up[:,ir] = sol.x

		return Up, StaticData


	def CalculateResidualBFace(self, ibfgrp, ibface, U, R, StaticData):
		mesh = self.mesh
		EqnSet = self.EqnSet
		sr = EqnSet.StateRank
		dim = mesh.Dim

		BFG = mesh.BFaceGroups[ibfgrp]
		BFace = BFG.BFaces[ibface]
		egrp = BFace.ElemGroup
		elem = BFace.Elem
		face = BFace.face
		Order = EqnSet.Orders[egrp]
		nFacePerElem = mesh.ElemGroups[egrp].nFacePerElem

		if StaticData is None:
			pnq = -1
			quadData = None
			quadDataST = None
			PhiData = None
			PsiData = None
			xglob = None
			tglob = None
			uI = None
			uB = None
			NData = None
			GeomPhiData = None
			TimePhiData = None
			xelemPsi = None
			xelemPhi = None
			StaticData = GenericData()

		else:
			nq = StaticData.pnq
			quadData = StaticData.quadData
			quadDataST = StaticData.quadDataST
			PhiData = StaticData.PhiData
			PsiData = StaticData.PsiData
			xglob = StaticData.xglob
			tglob = StaticData.tglob
			uI = StaticData.uI
			uB = StaticData.uB
			NData = StaticData.NData
			GeomPhiData = StaticData.GeomPhiData
			TimePhiData = StaticData.TimePhiData
			xelemPsi = StaticData.xelemPsi
			xelemPhi = StaticData.xelemPhi
		
		#Hard code basisType to Quads (currently only designed for 1D)
		basis2 = EqnSet.Bases[egrp]
		basis1 = EqnSet.BasesADER[egrp]

		QuadOrderST, QuadChangedST = GetQuadOrderBFace(mesh, BFace, basis1, Order, EqnSet, quadDataST)
		QuadOrderST-=1
		QuadOrder, QuadChanged = GetQuadOrderBFace(mesh, BFace, mesh.ElemGroups[egrp].QBasis, Order, EqnSet, quadData)

		if QuadChanged:
			quadData = QuadData(mesh, EqnSet.Bases[egrp], EntityType.BFace, QuadOrder)
		if QuadChangedST:
			quadDataST = QuadDataADER(mesh,basis1,EntityType.BFace, QuadOrderST)
		
		nqST = quadDataST.nquad
		xqST = quadDataST.xquad
		wqST = quadDataST.wquad

		nq = quadData.nquad
		xq = quadData.xquad
		wq = quadData.wquad

		if face == 0:
			faceST = 3
		elif face == 1:
			faceST = 1
		else:
			return IncompatibleError
		if QuadChanged or QuadChangedST:
			xglob = np.zeros([nq, dim])			
			uI = np.zeros([nqST, sr])
			uB = np.zeros([nqST, sr])
			xelemPhi = np.zeros([nqST, mesh.Dim+1])
			xelemPsi = np.zeros([nq, mesh.Dim])
			PhiData = BasisData(basis1,Order,nqST,mesh)
			xelemPhi = PhiData.EvalBasisOnFaceADER(mesh, basis1, egrp, faceST, xqST, xelemPhi, Get_Phi=True)
			PsiData = BasisData(basis2,Order,nq,mesh)
			xelemPsi = PsiData.EvalBasisOnFace(mesh, egrp, face, xq, xelemPsi, Get_Phi=True)

		NData = BFaceNormal(mesh, BFace, nq, xq, NData)
		PointsChanged = QuadChanged or face != GeomPhiData.face
		xglob, GeomPhiData = Ref2Phys(mesh, egrp, elem, GeomPhiData, nq, xelemPsi, xglob, PointsChanged)

		tglob, TimePhiData = Ref2PhysTime(mesh, egrp, elem, self.Time, self.Stepper.dt, TimePhiData, nqST, xelemPhi, tglob, PointsChanged)
		nn = PsiData.nn

		# interpolate state and gradient at quad points
		for ir in range(sr):
			uI[:,ir] = np.matmul(PhiData.Phi, U[:,ir])

		# Get boundary state
		BC = EqnSet.BCs[ibfgrp]
		uB = EqnSet.BoundaryState(BC, nqST, xglob, tglob, NData, uI, uB)

		F = EqnSet.ConvFluxBoundary(BC, uI, uB, NData, nqST, StaticData) # [nq,sr]

		F = np.reshape(F,(nq,nqST,sr))

		for ir in range(sr):
			#for k in range(nn): # Loop over basis function in space
			for i in range(nqST): # Loop over time
				for j in range(nq): # Loop over space
					Psi = PsiData.Phi[j,:]
					R[:,ir] -= wqST[i]*wq[j]*F[j,i]*Psi
	
		F = np.reshape(F,(nqST,sr))

		if elem == echeck:
			code.interact(local=locals())

		# Store in StaticData
		StaticData.pnq = nq
		StaticData.quadData = quadData
		StaticData.quadDataST = quadDataST
		StaticData.PhiData = PhiData
		StaticData.PsiData = PsiData
		StaticData.uI = uI
		StaticData.uB = uB
		StaticData.F = F
		StaticData.xglob = xglob
		StaticData.tglob = tglob
		StaticData.NData = NData
		StaticData.xelemPsi = xelemPsi
		StaticData.xelemPhi = xelemPhi
		StaticData.GeomPhiData = GeomPhiData
		StaticData.TimePhiData = TimePhiData


		return R, StaticData

	def CalculateResidualElem(self, egrp, elem, U, ER, StaticData):
		# U = EqnSet.U[egrp][elem] # [nn,sr]
		mesh = self.mesh
		EqnSet = self.EqnSet

		basis2 = EqnSet.Bases[egrp]
		basis1 = EqnSet.BasesADER[egrp]
		Order = EqnSet.Orders[egrp]
		entity = EntityType.Element
		sr = EqnSet.StateRank
		dim = EqnSet.Dim
		# ER = R[egrp][elem] # [nn,sr]

		if Order == 0:
			return ER, StaticData

		if StaticData is None:
			pnq = -1
			quadData = None
			quadDataST = None
			PhiData = None
			PsiData = None
			JData = JData = JacobianData(mesh)
			xglob = None
			tglob = None
			u = None
			F = None
			s = None
			NData = None
			GeomPhiData = None
			TimePhiData = None
			StaticData = GenericData()
		else:
			nq = StaticData.pnq
			quadData = StaticData.quadData
			quadDataST = StaticData.quadDataST
			PhiData = StaticData.PhiData
			PsiData = StaticData.PsiData
			JData = StaticData.JData
			xglob = StaticData.xglob
			tglob = StaticData.tglob
			u = StaticData.u
			F = StaticData.F
			s = StaticData.s
			NData = StaticData.NData
			GeomPhiData = StaticData.GeomPhiData
			TimePhiData = StaticData.TimePhiData

		QuadOrder,QuadChanged = GetQuadOrderElem(mesh, egrp, mesh.ElemGroups[egrp].QBasis, Order, EqnSet, quadData)
		QuadOrderST, QuadChangedST = GetQuadOrderElem(mesh, egrp, basis1, Order, EqnSet, quadDataST)
		QuadOrderST-=1

		if QuadChanged:
			quadData = QuadData(mesh, basis2, entity, QuadOrder)
		if QuadChangedST:
			quadDataST = QuadData(mesh, basis1, entity, QuadOrderST)

		nq = quadData.nquad
		xq = quadData.xquad
		wq = quadData.wquad

		nqST = quadDataST.nquad
		xqST = quadDataST.xquad
		wqST = quadDataST.wquad

		if QuadChanged:
			PhiData = BasisData(basis1,Order,nqST,mesh)
			PhiData.EvalBasis(xqST, Get_Phi=True, Get_GPhi=False)
			#PhiData.EvalBasis(xqST, Get_Phi=True, Get_GPhi=False)
			PsiData = BasisData(basis2,Order,nq,mesh)
			PsiData.EvalBasis(xq, Get_Phi=True, Get_GPhi=True)

			xglob = np.zeros([nq, dim])
			u = np.zeros([nqST, sr])
			F = np.zeros([nqST, sr, dim])
			s = np.zeros([nqST, sr])

		JData.ElemJacobian(egrp,elem,nq,xq,mesh,get_djac=True,get_jac=False,get_ijac=True)
		PsiData.EvalBasis(xq, Get_gPhi=True, JData=JData)

		xglob, GeomPhiData = Ref2Phys(mesh, egrp, elem, GeomPhiData, nq, xq, xglob, QuadChanged)
		tglob, TimePhiData = Ref2PhysTime(mesh, egrp, elem, self.Time, self.Stepper.dt, TimePhiData, nqST, xqST, tglob, QuadChangedST)


		nn = PsiData.nn

		#Calculate dx and dt(this is for constant delta_x, but works on different mesh resolutions)
		EGroup=mesh.ElemGroups[egrp]
		Elem2Nodes = EGroup.Elem2Nodes[elem]
		dx = np.abs(mesh.Coords[Elem2Nodes[1],0]-mesh.Coords[Elem2Nodes[0],0])
		Stepper = self.Stepper
		dt = Stepper.dt

		# interpolate state and gradient at quad points
		# u = np.zeros([nq, sr])
		u[:] = np.matmul(PhiData.Phi, U)

		#u = np.reshape(u,(nq,nn))
		F = EqnSet.ConvFluxInterior(u, F) # [nq,sr,dim]

		F = np.reshape(F,(nq,nq,sr,dim))

		for ir in range(sr):
			for k in range(nn): # Loop over basis function in space
				for i in range(nq): # Loop over time
					for j in range(nq): # Loop over space
						gPsi = PsiData.gPhi[j,k]
						ER[k,ir] += wq[i]*wq[j]*JData.djac[j*(JData.nq!=1)]*F[i,j]*gPsi

		F = np.reshape(F,(nqST,sr,dim))

		s = np.zeros([nqST,sr])
		s = EqnSet.SourceState(nqST, xglob, tglob, NData, u, s) # [nq,sr,dim]

		s = np.reshape(s,(nq,nq,sr))
		# Calculate source term integral
		for ir in range(sr):
			for k in range(nn):
				for i in range(nq): # Loop over time
					for j in range(nq): # Loop over space
						Psi = PsiData.Phi[j,k]
						ER[k,ir] += wq[i]*wq[j]*s[i,j,ir]*JData.djac[iq*(JData.nq!=1)]*Psi

		s = np.reshape(s,(nqST,sr))


		# s = EqnSet.SourceState(nq, xglob, self.Time, NData, u, s) # [nq,sr,dim]
		# # Calculate source term integral
		# for ir in range(sr):
		# 	for jn in range(nn):
		# 		for iq in range(nq):
		# 			Phi = PhiData.Phi[iq,jn]
		# 			ER[jn,ir] += Phi*s[iq,ir]*wq[iq]*JData.djac[iq*(JData.nq!=1)]

		if elem == echeck:
			code.interact(local=locals())

		# Store in StaticData
		StaticData.pnq = nq
		StaticData.quadData = quadData
		StaticData.quadDataST = quadDataST
		StaticData.PhiData = PhiData
		StaticData.PsiData = PsiData
		StaticData.JData = JData
		StaticData.xglob = xglob
		StaticData.tglob = tglob
		StaticData.u = u
		StaticData.F = F
		StaticData.s = s
		StaticData.NData = NData
		StaticData.GeomPhiData = GeomPhiData
		StaticData.TimePhiData = TimePhiData

		return ER, StaticData

	def CalculateResidualIFace(self, iiface, UL, UR, RL, RR, StaticData):
		mesh = self.mesh
		dim = mesh.Dim
		EqnSet = self.EqnSet
		sr = EqnSet.StateRank

		IFace = mesh.IFaces[iiface]
		egrpL = IFace.ElemGroupL
		egrpR = IFace.ElemGroupR
		elemL = IFace.ElemL
		elemR = IFace.ElemR
		faceL = IFace.faceL
		faceR = IFace.faceR
		OrderL = EqnSet.Orders[egrpL]
		OrderR = EqnSet.Orders[egrpR]
		nFacePerElemL = mesh.ElemGroups[egrpL].nFacePerElem
		nFacePerElemR = mesh.ElemGroups[egrpR].nFacePerElem

		if StaticData is None:
			pnq = -1
			quadData = None
			quadDataST = None
			PhiDataL = None
			PhiDataR = None
			PsiDataL = None
			PsiDataR = None
			xelemLPhi = None
			xelemRPhi = None
			xelemLPsi = None
			xelemRPsi = None
			uL = None
			uR = None
			StaticData = GenericData()
			NData = None
		else:
			nq = StaticData.pnq
			quadDataST = StaticData.quadDataST
			quadData = StaticData.quadData
			PhiDataL = StaticData.PhiDataL
			PhiDataR = StaticData.PhiDataR
			PsiDataL = StaticData.PsiDataL
			PsiDataR = StaticData.PsiDataR
			xelemLPhi = StaticData.xelemLPhi
			xelemRPhi = StaticData.xelemRPhi
			xelemLPsi = StaticData.xelemLPsi
			xelemRPsi = StaticData.xelemRPsi
			uL = StaticData.uL
			uR = StaticData.uR
			NData = StaticData.NData


		#Hard code basisType to Quads (currently only designed for 1D)
		basis2 = EqnSet.Bases[egrpL]
		basis1 = EqnSet.BasesADER[egrpL]

		QuadOrderST, QuadChangedST = GetQuadOrderIFace(mesh, IFace, basis1, np.amax([OrderL,OrderR]), EqnSet, quadDataST)
		QuadOrderST-=1
		QuadOrder, QuadChanged = GetQuadOrderIFace(mesh, IFace, mesh.ElemGroups[egrpL].QBasis, np.amax([OrderL,OrderR]), EqnSet, quadData)

		if QuadChanged:
			quadData = QuadData(mesh, EqnSet.Bases[egrpL], EntityType.IFace, QuadOrder)
		if QuadChangedST:
			quadDataST = QuadDataADER(mesh,basis1,EntityType.IFace, QuadOrderST)
		
		nqST = quadDataST.nquad
		xqST = quadDataST.xquad
		wqST = quadDataST.wquad

		nq = quadData.nquad
		xq = quadData.xquad
		wq = quadData.wquad

		if faceL == 0:
			faceSTL = 3
		elif faceL == 1:
			faceSTL = 1
		else:
			return IncompatibleError
		if faceR == 0:
			faceSTR = 3
		elif faceR == 1:
			faceSTR = 1
		else:
			return IncompatibleError

		if QuadChanged or QuadChangedST:

			xelemLPhi = np.zeros([nqST, dim+1])
			xelemRPhi = np.zeros([nqST, dim+1])
			xelemLPsi = np.zeros([nq, dim])
			xelemRPsi = np.zeros([nq, dim])
			uL = np.zeros([nqST, sr])
			uR = np.zeros([nqST, sr])

			#Evaluate space-time basis functions
			PhiDataL = BasisData(basis1,OrderL,nqST,mesh)
			PhiDataR = BasisData(basis1,OrderR,nqST,mesh)
			xelemLPhi = PhiDataL.EvalBasisOnFaceADER(mesh, basis1, egrpL, faceSTL, xqST, xelemLPhi, Get_Phi=True)
			xelemRPhi = PhiDataR.EvalBasisOnFaceADER(mesh, basis1, egrpR, faceSTR, xqST[::-1], xelemRPhi, Get_Phi=True)

			#Evaluate spacial basis functions
			PsiDataL = BasisData(basis2,OrderL,nq,mesh)
			PsiDataR = BasisData(basis2,OrderR,nq,mesh)
			xelemLPsi = PsiDataL.EvalBasisOnFace(mesh, egrpL, faceL, xq, xelemLPsi, Get_Phi=True)
			xelemRPsi = PsiDataR.EvalBasisOnFace(mesh, egrpR, faceR, xq[::-1], xelemRPsi, Get_Phi=True)



		NData = IFaceNormal(mesh, IFace, nq, xq, NData)
		# NData.nvec *= wq

		nL = PsiDataL.nn
		nR = PsiDataR.nn
		nn = np.amax([nL,nR])

		Stepper = self.Stepper
		dt = Stepper.dt

		for ir in range(sr):
			uL[:,ir] = np.matmul(PhiDataL.Phi, UL[:,ir])
			uR[:,ir] = np.matmul(PhiDataR.Phi, UR[:,ir])

		F = EqnSet.ConvFluxNumerical(uL, uR, NData, nqST, StaticData) # [nq,sr]

		F = np.reshape(F,(nq,nqST,sr))

		for ir in range(sr):
			#for k in range(nn): # Loop over basis function in space
			for i in range(nqST): # Loop over time
				for j in range(nq): # Loop over space
					PsiL = PsiDataL.Phi[j,:]
					PsiR = PsiDataR.Phi[j,:]
					RL[:,ir] -= wqST[i]*wq[j]*F[j,i]*PsiL
					RR[:,ir] += wqST[i]*wq[j]*F[j,i]*PsiR

		F = np.reshape(F,(nqST,sr))

		if elemL == echeck or elemR == echeck:
			if elemL == echeck: print("Left!")
			else: print("Right!")
			code.interact(local=locals())

		# Store in StaticData
		StaticData.pnq = nq
		StaticData.quadData = quadData
		StaticData.quadDataST = quadDataST
		StaticData.PhiDataL = PhiDataL
		StaticData.PhiDataR = PhiDataR
		StaticData.PsiDataL = PsiDataL
		StaticData.PsiDataR = PsiDataR
		StaticData.xelemLPhi = xelemLPhi
		StaticData.xelemRPhi = xelemRPhi
		StaticData.xelemLPsi = xelemLPsi
		StaticData.xelemRPsi = xelemRPsi
		StaticData.uL = uL
		StaticData.uR = uR
		StaticData.NData = NData

		return RL, RR, StaticData


	def FluxState(self, egrp, elem, dt, Order, basis, U):

		mesh = self.mesh
		dim = mesh.Dim
		EqnSet = self.EqnSet
		sr = EqnSet.StateRank
		Params = self.Params
		entity = EntityType.Element
		InterpolateFlux = Params["InterpolateFlux"]

		quadData = None
		quadDataST = None
		JData = JacobianData(mesh)
		GeomPhiData = None
		xq = None; xphys = None; QuadChanged = True; QuadChangedST = True;

		rhs = np.zeros([Order2nNode(basis,Order),sr],dtype=U.dtype)

		if not InterpolateFlux:


			QuadOrder,QuadChanged = GetQuadOrderElem(mesh, egrp, mesh.ElemGroups[egrp].QBasis, 4*Order, EqnSet, quadData)
			QuadOrderST, QuadChangedST = GetQuadOrderElem(mesh, egrp, basis, 4*Order, EqnSet, quadDataST)
			QuadOrderST-=1

			if QuadChanged:
				quadData = QuadData(mesh, mesh.ElemGroups[egrp].QBasis, entity, QuadOrder)
			if QuadChangedST:
				quadDataST = QuadData(mesh, basis, entity, QuadOrderST)

			nq = quadData.nquad
			xq = quadData.xquad
			wq = quadData.wquad

			nqST = quadDataST.nquad
			xqST = quadDataST.xquad
			wqST = quadDataST.wquad

			if QuadChanged:

				PhiData = BasisData(basis,Order,nqST,mesh)
				PhiData.EvalBasis(xqST, Get_Phi=True)
				xphys = np.zeros([nqST, mesh.Dim])

			JData.ElemJacobian(egrp,elem,nqST,xqST,mesh,get_djac=True)
			MMinv,_= GetElemInvMassMatrixADER(mesh, basis, Order, PhysicalSpace=True, egrp=-1, elem=-1, StaticData=None)
			nn = PhiData.nn
		else:

			xq, nq = EquidistantNodes(basis, Order, xq)
			nn = nq
			JData.ElemJacobian(egrp,elem,nq,xq,mesh,get_djac=True)

		if not InterpolateFlux:

			u = np.zeros([nqST, sr])
			u[:] = np.matmul(PhiData.Phi, U)

			#u = np.reshape(u,(nq,nn))
			F = np.zeros([nqST,sr,dim])
			f = EqnSet.ConvFluxInterior(u, F) # [nq,sr]

			f = np.reshape(f,(nq,nq,sr,dim))
			Phi = PhiData.Phi
			Phi = np.reshape(Phi,(nq,nq,nn))
		
			rhs *=0.
			for ir in range(sr):
				for k in range(nn): # Loop over basis function in space
					for i in range(nq): # Loop over time
						for j in range(nq): # Loop over space
							#Phi = PhiData.Phi[j,k]
							rhs[k,ir] += wq[i]*wq[j]*JData.djac[j*(JData.nq!=1)]*f[i,j]*Phi[i,j,k]

			#F = np.reshape(F,(nqST,sr,dim))
			F = np.dot(MMinv,rhs)*(1.0/JData.djac)*dt/2.0

		else:
			F1 = np.zeros([nn,sr,dim])
			f = EqnSet.ConvFluxInterior(U,F1)
			F = f[:,:,0]*(1.0/JData.djac)*dt/2.0
		return F
















