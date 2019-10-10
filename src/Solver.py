import numpy as np 
from Basis import *
from Quadrature import *
from Mesh import *
import code
import copy
from Data import ArrayList, GenericData
from SolverTools import *
from TimeStepping import *
import time



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
		self.nTimeStep = Params["nTimeStep"]

		TimeScheme = Params["TimeScheme"]
		if TimeScheme is "FE":
			TimeStepper = FE()
		elif TimeScheme is "RK4":
			TimeStepper = RK4()
		elif TimeScheme is "LSRK4":
			TimeStepper = LSRK4()
		else:
			raise Exception("Time scheme not supported")
		if Params["nTimeStep"] > 0:
			TimeStepper.dt = Params["EndTime"]/Params["nTimeStep"]
		self.TimeStepper = TimeStepper

		# Initialize state
		self.InitState()


	def InitState(self):
		mesh = self.mesh
		EqnSet = self.EqnSet
		U = EqnSet.U.Arrays
		sr = EqnSet.StateRank
		# f = np.zeros([nq,sr])

		# Get mass matrices
		try:
			MMinv_all = self.DataSet.MMinv_all
		except AttributeError:
			# not found; need to compute
			MMinv_all = ComputeInvMassMatrices(mesh, EqnSet, solver=self)

		quadData = None
		JData = JacobianData(mesh)
		for egrp in range(mesh.nElemGroup):
			U_ = U[egrp]

			basis = EqnSet.Bases[egrp]
			Order = EqnSet.Orders[egrp]
			rhs = np.zeros([Order2nNode(basis,Order),sr])
			for elem in range(mesh.ElemGroups[egrp].nElem):

				QuadOrder,QuadChanged = GetQuadOrderElem(mesh, egrp, EqnSet.Bases[egrp], \
					2*np.amax([Order,1]), EqnSet, quadData)
				if QuadChanged:
					quadData = QuadData(mesh, egrp, EntityType.Element, QuadOrder)

				nq = quadData.nquad
				xq = quadData.xquad
				wq = quadData.wquad

				if QuadChanged:
					# PhiData = BasisData(egrp,Order,EntityType.Element,nq,xq,mesh,True,False)
					PhiData = BasisData(basis,Order,nq,mesh)
					PhiData.EvalBasis(xq, Get_Phi=True)
					xphys = np.zeros([nq, mesh.Dim])

				JData.ElemJacobian(egrp,elem,nq,xq,mesh,Get_detJ=True)

				# MMinv = GetInvMassMatrix(mesh, egrp, 0, basis, EqnSet.Orders[egrp])
				MMinv = MMinv_all.Arrays[egrp][elem]

				nn = PhiData.nn

				rhs *= 0.
				xphys = Ref2Phys(mesh, egrp, elem, PhiData, nq, xq, xphys)
				# if sr == 1: f = f_ic(xphys)
				# else : 
				# 	# f = SmoothIsentropic1D(x=xphys,t=0.,gam=EqnSet.Params["SpecificHeatRatio"])
				# 	f = EqnSet.CallFunction(EqnSet.IC, x=xphys, Time=0.)
				f = EqnSet.CallFunction(EqnSet.IC, x=xphys, Time=self.Time)
				f.shape = nq,sr
				for n in range(nn):
					for iq in range(nq):
						rhs[n,:] += f[iq,:]*PhiData.Phi[iq,n]*wq[iq]*JData.detJ[iq*(JData.nq != 1)]

				U_[elem,:,:] = np.dot(MMinv,rhs)



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

		if StaticData is None:
			pnq = -1
			quadData = None
			PhiData = None
			JData = JData = JacobianData(mesh)
			xglob = None
			u = None
			F = None
			StaticData = GenericData()
		else:
			nq = StaticData.pnq
			quadData = StaticData.quadData
			PhiData = StaticData.PhiData
			JData = StaticData.JData
			xglob = StaticData.xglob
			u = StaticData.u
			F = StaticData.F


		QuadOrder,QuadChanged = GetQuadOrderElem(mesh, egrp, EqnSet.Bases[egrp], Order, EqnSet, quadData)
		if QuadChanged:
			quadData = QuadData(mesh, egrp, entity, QuadOrder)

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


		JData.ElemJacobian(egrp,elem,nq,xq,mesh,Get_detJ=True,Get_J=False,Get_iJ=True)
		PhiData.EvalBasis(xq, Get_gPhi=True, JData=JData)

		xglob = Ref2Phys(mesh, egrp, elem, PhiData, nq, xq, xglob)

		nn = PhiData.nn

		# interpolate state and gradient at quad points
		# u = np.zeros([nq, sr])
		for ir in range(sr):
			u[:,ir] = np.matmul(PhiData.Phi, U[:,ir])
		# gu = np.zeros([nq,sr,dim])
		# for iq in range(nq):
		# 	gPhi = PhiData.gPhi[iq] # [nn,dim]
		# 	for ir in range(sr):
		# 		gu[iq,ir,:] = np.matmul(gPhi.transpose(), U[:,ir])[0]

		F = EqnSet.ConvFluxInterior(u, F) # [nq,sr,dim]

		for ir in range(sr):
			for jn in range(nn):
				for iq in range(nq):
					gPhi = PhiData.gPhi[iq,jn] # dim
					ER[jn,ir] += np.dot(gPhi, F[iq,ir,:])*wq[iq]*JData.detJ[iq*(JData.nq!=1)]

		# Store in StaticData
		StaticData.pnq = nq
		StaticData.quadData = quadData
		StaticData.PhiData = PhiData
		StaticData.JData = JData
		StaticData.xglob = xglob
		StaticData.u = u
		StaticData.F = F

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
		else:
			nq = StaticData.pnq
			quadData = StaticData.quadData
			PhiDataL = StaticData.PhiDataL
			PhiDataR = StaticData.PhiDataR
			xelemL = StaticData.xelemL
			xelemR = StaticData.xelemR
			uL = StaticData.uL
			uR = StaticData.uR


		QuadOrder, QuadChanged = GetQuadOrderIFace(mesh, IFace, EqnSet.Bases[egrpL], np.amax([OrderL,OrderR]), EqnSet, quadData)

		if QuadChanged:
			quadData = QuadData(mesh, egrpL, EntityType.IFace, QuadOrder)

		nq = quadData.nquad
		xq = quadData.xquad
		wq = quadData.wquad

		if QuadChanged:
			PhiDataL = BasisData(EqnSet.Bases[egrpL],OrderL,nq,mesh)
			PhiDataR = BasisData(EqnSet.Bases[egrpR],OrderR,nq,mesh)

			xelemL = np.zeros([nq, dim])
			xelemR = np.zeros([nq, dim])
			uL = np.zeros([nq, sr])
			uR = np.zeros([nq, sr])

		if faceL != PhiDataL.face:
			xelemL = PhiDataL.EvalBasisOnFace(mesh, egrpL, faceL, xq, xelemL, Get_Phi=True)
		if faceR != PhiDataR.face:
			xelemR = PhiDataR.EvalBasisOnFace(mesh, egrpR, faceR, xq, xelemR, Get_Phi=True)

		# PhiDataL = BasisData(ShapeType.Segment,OrderL,nq,mesh)
		# PhiDataL.EvalBasisOnFace(mesh, egrpL, faceL, xq, True, False, False, None)
		# PhiDataR = BasisData(ShapeType.Segment,OrderR,nq,mesh)
		# PhiDataR.EvalBasisOnFace(mesh, egrpR, faceR, xq, True, False, False, None)

		NData = IFaceNormal(mesh, IFace, nq, xq)

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

		RL -= np.matmul(PhiDataL.Phi.transpose(), F) # [nn,sr]
		RR += np.matmul(PhiDataR.Phi.transpose(), F) # [nn,sr]

		# if elemL == 24: code.interact(local=locals())

		# Store in StaticData
		StaticData.pnq = nq
		StaticData.quadData = quadData
		StaticData.PhiDataL = PhiDataL
		StaticData.PhiDataR = PhiDataR
		StaticData.xelemL = xelemL
		StaticData.xelemR = xelemR
		StaticData.uL = uL
		StaticData.uR = uR

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


		if StaticData is None:
			pnq = -1
			quadData = None
			PhiData = None
			xglob = None
			xelem = None
			uI = None
			uB = None
			StaticData = GenericData()
		else:
			nq = StaticData.pnq
			quadData = StaticData.quadData
			PhiData = StaticData.PhiData
			xglob = StaticData.xglob
			xelem = StaticData.xelem
			uI = StaticData.uI
			uB = StaticData.uB


		QuadOrder, QuadChanged = GetQuadOrderBFace(mesh, BFace, EqnSet.Bases[egrp], Order, EqnSet, quadData)

		if QuadChanged:
			quadData = QuadData(mesh, egrp, EntityType.BFace, QuadOrder)

		nq = quadData.nquad
		xq = quadData.xquad
		wq = quadData.wquad

		if QuadChanged:
			PhiData = BasisData(EqnSet.Bases[egrp],Order,nq,mesh)
			
			xelem = np.zeros([nq, dim])
			xglob = np.zeros([nq, dim])
			uI = np.zeros([nq, sr])
			uB = np.zeros([nq, sr])

		if face != PhiData.face:
			xelem = PhiData.EvalBasisOnFace(mesh, egrp, face, xq, xelem, Get_Phi=True)

		# PhiData.EvalBasisOnFace(mesh, egrp, face, xq, True, False, False, None)

		NData = BFaceNormal(mesh, BFace, nq, xq)

		xglob = Ref2Phys(mesh, egrp, elem, PhiData, nq, xelem, xglob)

		nn = PhiData.nn

		# interpolate state and gradient at quad points
		for ir in range(sr):
			uI[:,ir] = np.matmul(PhiData.Phi, U[:,ir])

		# Get boundary state
		BC = EqnSet.BCs[ibfgrp]
		uB = EqnSet.BoundaryState(BC, nq, xglob, self.Time, uI, uB)

		F = EqnSet.ConvFluxNumerical(uI, uB, NData, nq, StaticData) # [nq,sr]

		R -= np.matmul(PhiData.Phi.transpose(), F) # [nn,sr]

		# Store in StaticData
		StaticData.pnq = nq
		StaticData.quadData = quadData
		StaticData.PhiData = PhiData
		StaticData.uI = uI
		StaticData.uB = uB
		StaticData.F = F
		StaticData.xglob = xglob
		StaticData.xelem = xelem

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

		self.CalculateResidualElems(U.Arrays, R.Arrays)
		self.CalculateResidualIFaces(U.Arrays, R.Arrays)
		self.CalculateResidualBFaces(U.Arrays, R.Arrays)

		return R


	def ApplyTimeScheme(self):

		EqnSet = self.EqnSet
		mesh = self.mesh

		TimeStepper = self.TimeStepper
		Time = self.Time

		t0 = time.time()
		for iStep in range(self.nTimeStep):

			# Integrate in time
			# self.Time is used for local time
			R = TimeStepper.TakeTimeStep(self)

			# Increment time
			Time += TimeStepper.dt
			self.Time = Time

			# Print info
			print("%d: Time = %g, Max residual = %g" % (iStep+1, self.Time, R.Max()))

		t1 = time.time()
		print("Wall clock time = %g seconds" % (t1-t0))





