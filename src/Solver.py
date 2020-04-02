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
		if BasisType[Params["InterpBasis"]] == BasisType.LagrangeSeg or BasisType[Params["InterpBasis"]] == BasisType.LegendreSeg:
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
			if mesh.QOrder != 1:
			    raise Errors.IncompatibleError
			if mesh.QBasis == BasisType.LagrangeQuad \
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
		U = EqnSet.U
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
		quad_pts = None; xphys = None

		basis = EqnSet.Basis
		Order = EqnSet.Order
		rhs = np.zeros([Order2nNode(basis,Order),sr],dtype=U.dtype)

		# Precompute basis and quadrature
		if not InterpolateIC:
			QuadOrder,_ = get_gaussian_quadrature_elem(mesh, EqnSet.Basis,
				2*np.amax([Order,1]), EqnSet, quadData)

			quadData = QuadData(mesh, mesh.QBasis, EntityType.Element, QuadOrder)

			nq = quadData.nquad
			quad_pts = quadData.quad_pts
			quad_wts = quadData.quad_wts

			PhiData = BasisData(basis,Order,nq,mesh)
			PhiData.EvalBasis(quad_pts, Get_Phi=True)
			xphys = np.zeros([nq, mesh.Dim])
		else:
			quad_pts, nq = EquidistantNodes(basis, Order, quad_pts)
			nn = nq

		for elem in range(mesh.nElem):

			xphys, GeomPhiData = Ref2Phys(mesh, elem, GeomPhiData, nq, quad_pts, xphys)

			f = EqnSet.CallFunction(EqnSet.IC, x=xphys, Time=self.Time)
			f.shape = nq,sr

			if not InterpolateIC:

				JData.ElemJacobian(elem,nq,quad_pts,mesh,get_djac=True)

				MMinv = MMinv_all[elem]

				nn = PhiData.nn

				# rhs *= 0.
				# for n in range(nn):
				# 	for iq in range(nq):
				# 		rhs[n,:] += f[iq,:]*PhiData.Phi[iq,n]*quad_wts[iq]*JData.djac[iq*(JData.nq != 1)]
				rhs[:] = np.matmul(PhiData.Phi.transpose(), f*quad_wts*JData.djac) # [nn, sr]

				U[elem,:,:] = np.matmul(MMinv,rhs)
			else:
				U[elem] = f

	def ApplyLimiter(self, U):
		if self.Limiter is not None:
			self.Limiter.LimitSolution(self, U)


	def CalculateResidualElem(self, elem, U, ER, StaticData):
		mesh = self.mesh
		EqnSet = self.EqnSet

		basis = EqnSet.Basis
		Order = EqnSet.Order
		entity = EntityType.Element
		sr = EqnSet.StateRank
		dim = EqnSet.Dim

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


		QuadOrder,QuadChanged = get_gaussian_quadrature_elem(mesh, mesh.QBasis, Order, EqnSet, quadData)
		if QuadChanged:
			quadData = QuadData(mesh, basis, entity, QuadOrder)

		nq = quadData.nquad
		xq = quadData.quad_pts
		wq = quadData.quad_wts

		if QuadChanged:
			PhiData = BasisData(EqnSet.Basis,Order,nq,mesh)
			PhiData.EvalBasis(xq, Get_Phi=True, Get_GPhi=True)
			# PhiData.PhysicalGrad(JData)

			xglob = np.zeros([nq, dim])
			u = np.zeros([nq, sr])
			F = np.zeros([nq, sr, dim])
			s = np.zeros([nq, sr])



		JData.ElemJacobian(elem,nq,xq,mesh,get_djac=True,get_jac=False,get_ijac=True)
		PhiData.EvalBasis(xq, Get_gPhi=True, JData=JData) # gPhi is [nq,nn,dim]

		xglob, GeomPhiData = Ref2Phys(mesh, elem, GeomPhiData, nq, xq, xglob, QuadChanged)

		nn = PhiData.nn

		# interpolate state and gradient at quad points
		u[:] = np.matmul(PhiData.Phi, U)

		F = EqnSet.ConvFluxInterior(u, F) # [nq,sr,dim]

		# for ir in range(sr):
		# 	for jn in range(nn):
		# 		for iq in range(nq):
		# 			gPhi = PhiData.gPhi[iq,jn] # dim
		# 			ER[jn,ir] += np.dot(gPhi, F[iq,ir,:])*wq[iq]*JData.djac[iq*(JData.nq!=1)]
		ER[:] += np.tensordot(PhiData.gPhi, F*(wq*JData.djac).reshape(nq,1,1), axes=([0,2],[0,2])) # [nn, sr]

		s = np.zeros([nq,sr])
		s = EqnSet.SourceState(nq, xglob, self.Time, NData, u, s) # [nq,sr]
		# Calculate source term integral
		# for ir in range(sr):
		# 	for jn in range(nn):
		# 		for iq in range(nq):
		# 			Phi = PhiData.Phi[iq,jn]
		# 			ER[jn,ir] += Phi*s[iq,ir]*wq[iq]*JData.djac[iq*(JData.nq!=1)]
		ER[:] += np.matmul(PhiData.Phi.transpose(), s*wq*JData.djac) # [nn, sr]
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

		for elem in range(mesh.nElem):
			R[elem], StaticData = self.CalculateResidualElem(elem, U[elem], R[elem], StaticData)


	def CalculateResidualIFace(self, iiface, UL, UR, RL, RR, StaticData):
		mesh = self.mesh
		dim = mesh.Dim
		EqnSet = self.EqnSet
		sr = EqnSet.StateRank

		IFace = mesh.IFaces[iiface]
		elemL = IFace.ElemL
		elemR = IFace.ElemR
		faceL = IFace.faceL
		faceR = IFace.faceR
		Order = EqnSet.Order
		nFacePerElem = mesh.nFacePerElem

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

		QuadOrder, QuadChanged = get_gaussian_quadrature_face(mesh, IFace, mesh.QBasis, Order, EqnSet, quadData)

		if QuadChanged:
			quadData = QuadData(mesh, EqnSet.Basis, EntityType.IFace, QuadOrder)

		nq = quadData.nquad
		xq = quadData.quad_pts
		wq = quadData.quad_wts

		if QuadChanged:
			Faces2PhiDataL = [None for i in range(nFacePerElem)]
			Faces2PhiDataR = [None for i in range(nFacePerElem)]

			xelemL = np.zeros([nq, dim])
			xelemR = np.zeros([nq, dim])
			uL = np.zeros([nq, sr])
			uR = np.zeros([nq, sr])

		PhiDataL = Faces2PhiDataL[faceL]
		PhiDataR = Faces2PhiDataR[faceR]

		if PhiDataL is None or QuadChanged:
			Faces2PhiDataL[faceL] = PhiDataL = BasisData(EqnSet.Basis,Order,nq,mesh)
			xelemL = PhiDataL.EvalBasisOnFace(mesh, faceL, xq, xelemL, Get_Phi=True)
		if PhiDataR is None or QuadChanged:
			Faces2PhiDataR[faceR] = PhiDataR = BasisData(EqnSet.Basis,Order,nq,mesh)
			xelemR = PhiDataR.EvalBasisOnFace(mesh, faceR, xq[::-1], xelemR, Get_Phi=True)

		NData = IFaceNormal(mesh, IFace, nq, xq, NData)
		# NData.nvec *= wq

		nL = PhiDataL.nn
		nR = PhiDataR.nn
		nn = np.amax([nL,nR])

		# interpolate state and gradient at quad points
		# for ir in range(sr):
		# 	uL[:,ir] = np.matmul(PhiDataL.Phi, UL[:,ir])
		# 	uR[:,ir] = np.matmul(PhiDataR.Phi, UR[:,ir])
		uL[:] = np.matmul(PhiDataL.Phi, UL)
		uR[:] = np.matmul(PhiDataR.Phi, UR)

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
			elemL = IFace.ElemL
			elemR = IFace.ElemR
			faceL = IFace.faceL
			faceR = IFace.faceR

			UL = U[elemL]
			UR = U[elemR]
			RL = R[elemL]
			RR = R[elemR]

			RL, RR, StaticData = self.CalculateResidualIFace(iiface, UL, UR, RL, RR, StaticData)

	def CalculateResidualBFace(self, ibfgrp, ibface, U, R, StaticData):
		mesh = self.mesh
		EqnSet = self.EqnSet
		sr = EqnSet.StateRank
		dim = mesh.Dim

		BFG = mesh.BFaceGroups[ibfgrp]
		BFace = BFG.BFaces[ibface]
		elem = BFace.Elem
		face = BFace.face
		Order = EqnSet.Order
		nFacePerElem = mesh.nFacePerElem

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

		QuadOrder, QuadChanged = get_gaussian_quadrature_face(mesh, BFace, mesh.QBasis, Order, EqnSet, quadData)

		if QuadChanged:
			quadData = QuadData(mesh, EqnSet.Basis, EntityType.BFace, QuadOrder)

		nq = quadData.nquad
		xq = quadData.quad_pts
		wq = quadData.quad_wts

		if QuadChanged:
			Faces2PhiData = [None for i in range(nFacePerElem)]
			# PhiData = BasisData(EqnSet.Basis,Order,nq,mesh)
			xglob = np.zeros([nq, dim])
			uI = np.zeros([nq, sr])
			uB = np.zeros([nq, sr])
			Faces2xelem = np.zeros([nFacePerElem, nq, dim])

		PhiData = Faces2PhiData[face]
		xelem = Faces2xelem[face]
		if PhiData is None or QuadChanged:
			Faces2PhiData[face] = PhiData = BasisData(EqnSet.Basis,Order,nq,mesh)
			Faces2xelem[face] = xelem = PhiData.EvalBasisOnFace(mesh, face, xq, xelem, Get_Phi=True)

		NData = BFaceNormal(mesh, BFace, nq, xq, NData)

		PointsChanged = QuadChanged or face != GeomPhiData.face
		xglob, GeomPhiData = Ref2Phys(mesh, elem, GeomPhiData, nq, xelem, xglob, PointsChanged)

		nn = PhiData.nn

		# interpolate state and gradient at quad points
		# for ir in range(sr):
		# 	uI[:,ir] = np.matmul(PhiData.Phi, U[:,ir])
		uI[:] = np.matmul(PhiData.Phi, U)

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
				elem = BFace.Elem
				face = BFace.face

				R[elem], StaticData = self.CalculateResidualBFace(ibfgrp, ibface, U[elem], R[elem], StaticData)


	def CalculateResidual(self, U, R):
		mesh = self.mesh
		EqnSet = self.EqnSet

		if R is None:
			# R = ArrayList(SimilarArray=U)
			R = np.copy(U)
		# Initialize residual to zero
		# R.SetUniformValue(0.)
		R[:] = 0.

		self.CalculateResidualBFaces(U, R)
		self.CalculateResidualElems(U, R)
		self.CalculateResidualIFaces(U, R)

		return R


	def ApplyTimeScheme(self, fhistory=None):

		EqnSet = self.EqnSet
		mesh = self.mesh
		Order = self.Params["InterpOrder"]
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
			# PrintInfo = (iStep+1, self.Time, R.VectorNorm(ord=1))
			PrintInfo = (iStep+1, self.Time, np.linalg.norm(np.reshape(R,-1), ord=1))
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

		# if OrderSequencing:
		# 	if mesh.nElemGroup != 1:
		# 		# only compatible with 
		# 		raise Errors.IncompatibleError
		# else:
		# 	if len(InterpOrders) != 1:
		# 		raise ValueError

		if not OrderSequencing:
			if len(InterpOrders) != 1:
				raise ValueError



		''' Loop through Orders '''
		Time = 0.
		for iOrder in range(nOrder):
			Order = InterpOrders[iOrder]
			''' Compute time step '''
			self.Stepper.dt = (EndTimes[iOrder]-Time)/nTimeSteps[iOrder]
			self.nTimeStep = nTimeSteps[iOrder]

			''' After first iteration, project solution to next Order '''
			if iOrder > 0:
				# Clear DataSet
				delattr(self, "DataSet")
				self.DataSet = GenericData()
				# Increment Order
				Order_old = EqnSet.Order
				EqnSet.Order = Order
				# Project
				ProjectStateToNewBasis(self, EqnSet, mesh, EqnSet.Basis, Order_old)

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
		if BasisType[Params["InterpBasis"]] == BasisType.LagrangeSeg or BasisType[Params["InterpBasis"]] == BasisType.LegendreSeg:
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
			if mesh.QOrder != 1:
			    raise Errors.IncompatibleError
			if mesh.QBasis == BasisType.LagrangeQuad \
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
		if Params["InterpolateFlux"] is True and BasisType[Params["InterpBasis"]] == BasisType.LegendreSeg:
			raise Errors.IncompatibleError

		### Current build only supports scalar equations
		#if EqnSet.StateRank > 1:
		#	raise Errors.IncompatibleError
	def CalculatePredictorStep(self, dt, W, Up):
		mesh = self.mesh
		EqnSet = self.EqnSet

		# Initialize predictor step to zero

		self.CalculatePredictorStepElems(dt, W, Up)

		return Up
	

	def CalculatePredictorStepElems(self, dt, W, Up):

		mesh = self.mesh
		EqnSet = self.EqnSet
		StaticData = None

		for elem in range(mesh.nElem):
			Up[elem], StaticData = self.CalculatePredictorElemADER(elem, dt, W[elem], Up[elem], StaticData)


	def CalculatePredictorElemADER(self, elem, dt, W, Up, StaticData):

	
		mesh = self.mesh
		EqnSet = self.EqnSet
		sr = EqnSet.StateRank
		dim = mesh.Dim
		basis2 = EqnSet.Basis
		basis1 = EqnSet.BasisADER
		Order = EqnSet.Order
		entity = EntityType.Element

		quadData = None
		quadDataST = None
		QuadChanged = True; QuadChangedST = True
		Elem2Nodes = mesh.Elem2Nodes[elem]
		dx = np.abs(mesh.Coords[Elem2Nodes[1],0]-mesh.Coords[Elem2Nodes[0],0])

		#Flux matrices in time
		FTL,_= GetTemporalFluxADER(mesh, basis1, basis1, Order, PhysicalSpace=False, elem=0, StaticData=None)
		FTR,_= GetTemporalFluxADER(mesh, basis1, basis2, Order, PhysicalSpace=False, elem=0, StaticData=None)
		
		#Stiffness matrix in time
		gradDir = 1
		SMT,_= GetStiffnessMatrixADER(gradDir,mesh, Order, elem=0, basis=basis1)
		gradDir = 0
		SMS,_= GetStiffnessMatrixADER(gradDir,mesh, Order, elem=0, basis=basis1)
		SMS = np.transpose(SMS)
		MM,_=  GetElemMassMatrixADER(mesh, basis1, Order, PhysicalSpace=False, elem=-1, StaticData=None)

		#MMinv,_= GetElemInvMassMatrixADER(mesh, basis1, Order, PhysicalSpace=True, elem=-1, StaticData=None)

		QuadOrder,QuadChanged = get_gaussian_quadrature_elem(mesh, basis2, Order, EqnSet, quadData)
		QuadOrderST, QuadChangedST = get_gaussian_quadrature_elem(mesh, basis1, Order, EqnSet, quadDataST)
		QuadOrderST-=1

		if QuadChanged:
			quadData = QuadData(mesh, basis2, entity, QuadOrder)

		if QuadChangedST:
			quadDataST = QuadData(mesh, basis1, entity, QuadOrderST)

		nq = quadData.nquad
		xq = quadData.quad_pts
		wq = quadData.quad_wts

		nqST = quadDataST.nquad
		xqST = quadDataST.quad_pts
		wqST = quadDataST.quad_wts

		PhiData = BasisData(basis1,Order,nqST,mesh)
		PsiData = BasisData(basis2,Order,nq,mesh)

		nnST = PhiData.nn
		nn   = PsiData.nn

		#Make initial guess for the predictor step
		Up = Up.reshape(nn,nn,sr)

		for ir in range(sr):
			#for i in range(nn):
			for j in range(nn):
				Up[0,j,ir]=W[j,ir]
		#Up = np.pad(W,pad_width=((0,nnST-nn),(0,0)))
		Up = Up.reshape(nnST,sr)

		def F(u):
			u = u.reshape(nnST,sr)
			fluxpoly = self.FluxCoefficients(elem, dt, Order, basis1, u)
			srcpoly = self.SourceCoefficients(elem, dt, Order, basis1, u)
			f = np.matmul(FTL,u)-np.matmul(FTR,W)-np.matmul(SMT,u)+np.matmul(SMS,fluxpoly)-np.matmul(MM,srcpoly)
			f = f.reshape(nnST*sr)
			return f

		Up = Up.reshape(nnST*sr)
		sol = root(F, Up, method='krylov',tol=1.0e-6)
		Up = sol.x
		Up = Up.reshape(nnST,sr)

		return Up, StaticData


	def CalculateResidualBFace(self, ibfgrp, ibface, U, R, StaticData):
		mesh = self.mesh
		EqnSet = self.EqnSet
		sr = EqnSet.StateRank
		dim = mesh.Dim

		BFG = mesh.BFaceGroups[ibfgrp]
		BFace = BFG.BFaces[ibface]
		elem = BFace.Elem
		face = BFace.face
		Order = EqnSet.Order
		nFacePerElem = mesh.nFacePerElem
		nFacePerElemADER = nFacePerElem + 2 # Hard-coded for 1D ADER method

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
			#xelemPsi = StaticData.xelemPsi
			#xelemPhi = StaticData.xelemPhi
			Faces2PhiData = StaticData.Faces2PhiData
			Faces2PsiData = StaticData.Faces2PsiData
			Faces2xelemADER = StaticData.Faces2xelemADER
			Faces2xelem = StaticData.Faces2xelem
		
		basis2 = EqnSet.Basis
		basis1 = EqnSet.BasisADER

		QuadOrderST, QuadChangedST = get_gaussian_quadrature_face(mesh, BFace, basis1, Order, EqnSet, quadDataST)
		QuadOrderST-=1
		QuadOrder, QuadChanged = get_gaussian_quadrature_face(mesh, BFace, mesh.QBasis, Order, EqnSet, quadData)

		if QuadChanged:
			quadData = QuadData(mesh, EqnSet.Basis, EntityType.BFace, QuadOrder)
		if QuadChangedST:
			quadDataST = QuadDataADER(mesh,basis1,EntityType.BFace, QuadOrderST)
		
		nqST = quadDataST.nquad
		xqST = quadDataST.quad_pts
		wqST = quadDataST.quad_wts

		nq = quadData.nquad
		xq = quadData.quad_pts
		wq = quadData.quad_wts

		if face == 0:
			faceST = 3
		elif face == 1:
			faceST = 1
		else:
			return IncompatibleError

		if QuadChanged or QuadChangedST:
			Faces2PsiData = [None for i in range(nFacePerElem)]
			Faces2PhiData = [None for i in range(nFacePerElemADER)]
			PsiData = BasisData(basis2, Order, nq, mesh)
			PhiData = BasisData(basis1, Order, nqST, mesh)
			xglob = np.zeros([nq, sr])
			uI = np.zeros([nqST, sr])
			uB = np.zeros([nqST, sr])
			Faces2xelem = np.zeros([nFacePerElem, nq, dim])
			Faces2xelemADER = np.zeros([nFacePerElem, nqST, dim+1])

		PsiData = Faces2PhiData[face]
		PhiData = Faces2PsiData[face]
		xelemPsi = Faces2xelem[face]
		xelemPhi = Faces2xelemADER[face]

		if PsiData is None or QuadChanged:
			Faces2PsiData[face] = PsiData = BasisData(basis2,Order,nq,mesh)
			Faces2xelem[face] = xelemPsi = PsiData.EvalBasisOnFace(mesh, face, xq, xelemPsi, Get_Phi=True)
		if PhiData is None or QuadChangedST:
			Faces2PhiData[face] = PhiData = BasisData(basis1,Order,nqST,mesh)
			Faces2xelemADER[face] = xelemPhi = PhiData.EvalBasisOnFaceADER(mesh, basis1, faceST, xqST, xelemPhi, Get_Phi=True)


		NData = BFaceNormal(mesh, BFace, nq, xq, NData)
		PointsChanged = QuadChanged or face != GeomPhiData.face
		xglob, GeomPhiData = Ref2Phys(mesh, elem, GeomPhiData, nq, xelemPsi, xglob, PointsChanged)

		tglob, TimePhiData = Ref2PhysTime(mesh, elem, self.Time, self.Stepper.dt, TimePhiData, nqST, xelemPhi, tglob, PointsChanged)
		nn = PsiData.nn

		# interpolate state and gradient at quad points
		uI[:] = np.matmul(PhiData.Phi, U)

		# Get boundary state
		BC = EqnSet.BCs[ibfgrp]
		uB = EqnSet.BoundaryState(BC, nqST, xglob, tglob, NData, uI, uB)

		F = EqnSet.ConvFluxBoundary(BC, uI, uB, NData, nqST, StaticData) # [nq,sr]

		# F = np.reshape(F,(nq,nqST,sr))

		# for ir in range(sr):
		# 	for i in range(nqST): # Loop over time
		# 		for j in range(nq): # Loop over space
		# 			Psi = PsiData.Phi[j,:]
		# 			R[:,ir] -= wqST[i]*wq[j]*F[j,i,ir]*Psi
	
		# F = np.reshape(F,(nqST,sr))

		R[:] -= np.matmul(np.tile(PsiData.Phi,(nn,1)).transpose(),F*wqST*wq)

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
		StaticData.Faces2PhiData = Faces2PhiData
		StaticData.Faces2PsiData = Faces2PsiData
		StaticData.Faces2xelemADER = Faces2xelemADER
		StaticData.Faces2xelem = Faces2xelem

		return R, StaticData

	def CalculateResidualElem(self, elem, U, ER, StaticData):
		mesh = self.mesh
		EqnSet = self.EqnSet

		basis2 = EqnSet.Basis
		basis1 = EqnSet.BasisADER
		Order = EqnSet.Order
		entity = EntityType.Element
		sr = EqnSet.StateRank
		dim = EqnSet.Dim

		if Order == 0:
			return ER, StaticData

		if StaticData is None:
			pnq = -1
			quadData = None
			quadDataST = None
			PhiData = None
			PsiData = None
			JData = JacobianData(mesh)
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

		QuadOrder,QuadChanged = get_gaussian_quadrature_elem(mesh, mesh.QBasis, Order, EqnSet, quadData)
		QuadOrderST, QuadChangedST = get_gaussian_quadrature_elem(mesh, basis1, Order, EqnSet, quadDataST)
		QuadOrderST-=1

		if QuadChanged:
			quadData = QuadData(mesh, basis2, entity, QuadOrder)
		if QuadChangedST:
			quadDataST = QuadData(mesh, basis1, entity, QuadOrderST)

		nq = quadData.nquad
		xq = quadData.quad_pts
		wq = quadData.quad_wts

		nqST = quadDataST.nquad
		xqST = quadDataST.quad_pts
		wqST = quadDataST.quad_wts

		if QuadChanged:
			PhiData = BasisData(basis1,Order,nqST,mesh)
			PhiData.EvalBasis(xqST, Get_Phi=True, Get_GPhi=False)
			PsiData = BasisData(basis2,Order,nq,mesh)
			PsiData.EvalBasis(xq, Get_Phi=True, Get_GPhi=True)

			xglob = np.zeros([nq, dim])
			u = np.zeros([nqST, sr])
			F = np.zeros([nqST, sr, dim])
			s = np.zeros([nqST, sr])

		JData.ElemJacobian(elem,nq,xq,mesh,get_djac=True,get_jac=False,get_ijac=True)
		PsiData.EvalBasis(xq, Get_gPhi=True, JData=JData)

		xglob, GeomPhiData = Ref2Phys(mesh, elem, GeomPhiData, nq, xq, xglob, QuadChanged)
		tglob, TimePhiData = Ref2PhysTime(mesh, elem, self.Time, self.Stepper.dt, TimePhiData, nqST, xqST, tglob, QuadChangedST)


		nn = PsiData.nn

		# interpolate state and gradient at quad points
		u[:] = np.matmul(PhiData.Phi, U)

		F = EqnSet.ConvFluxInterior(u, F) # [nq,sr,dim]

		# F = np.reshape(F,(nq,nq,sr,dim))

		# for ir in range(sr):
		# 	for k in range(nn): # Loop over basis function in space
		# 		for i in range(nq): # Loop over time
		# 			for j in range(nq): # Loop over space
		# 				gPsi = PsiData.gPhi[j,k]
		# 				ER[k,ir] += wq[i]*wq[j]*JData.djac[j*(JData.nq!=1)]*F[i,j,ir]*gPsi

		# F = np.reshape(F,(nqST,sr,dim))
		
		ER[:] += np.tensordot(np.tile(PsiData.gPhi,(nn,1,1)), F*(wqST.reshape(nq,nq)*JData.djac).reshape(nqST,1,1), axes=([0,2],[0,2])) # [nn, sr]


		s = np.zeros([nqST,sr])
		s = EqnSet.SourceState(nqST, xglob, tglob, NData, u, s) # [nq,sr,dim]

		# s = np.reshape(s,(nq,nq,sr))
		# #Calculate source term integral
		# for ir in range(sr):
		# 	for k in range(nn):
		# 		for i in range(nq): # Loop over time
		# 			for j in range(nq): # Loop over space
		# 				Psi = PsiData.Phi[j,k]
		# 				ER[k,ir] += wq[i]*wq[j]*s[i,j,ir]*JData.djac[j*(JData.nq!=1)]*Psi
		# s = np.reshape(s,(nqST,sr))

		ER[:] += np.matmul(np.tile(PsiData.Phi,(nn,1)).transpose(),s*(wqST.reshape(nq,nq)*JData.djac).reshape(nqST,1))

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
		elemL = IFace.ElemL
		elemR = IFace.ElemR
		faceL = IFace.faceL
		faceR = IFace.faceR
		Order = EqnSet.Order
		nFacePerElem = mesh.nFacePerElem
		nFacePerElemADER = nFacePerElem + 2 # Hard-coded for the 1D ADER method

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
			Faces2PhiDataL = StaticData.Faces2PhiDataL
			Faces2PhiDataR = StaticData.Faces2PhiDataR
			Faces2PsiDataL = StaticData.Faces2PsiDataL
			Faces2PsiDataR = StaticData.Faces2PsiDataR
			uL = StaticData.uL
			uR = StaticData.uR
			NData = StaticData.NData


		basis2 = EqnSet.Basis
		basis1 = EqnSet.BasisADER

		QuadOrderST, QuadChangedST = get_gaussian_quadrature_face(mesh, IFace, basis1, Order, EqnSet, quadDataST)
		QuadOrderST-=1
		QuadOrder, QuadChanged = get_gaussian_quadrature_face(mesh, IFace, mesh.QBasis, Order, EqnSet, quadData)

		if QuadChanged:
			quadData = QuadData(mesh, EqnSet.Basis, EntityType.IFace, QuadOrder)
		if QuadChangedST:
			quadDataST = QuadDataADER(mesh,basis1,EntityType.IFace, QuadOrderST)
		
		nqST = quadDataST.nquad
		xqST = quadDataST.quad_pts
		wqST = quadDataST.quad_wts

		nq = quadData.nquad
		xq = quadData.quad_pts
		wq = quadData.quad_wts

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

			Faces2PsiDataL = [None for i in range(nFacePerElem)]
			Faces2PsiDataR = [None for i in range(nFacePerElem)]
			Faces2PhiDataL = [None for i in range(nFacePerElemADER)]
			Faces2PhiDataR = [None for i in range(nFacePerElemADER)]
			PsiDataL = BasisData(basis2, Order, nq, mesh)
			PsiDataR = BasisData(basis2, Order, nq, mesh)
			PhiDataL = BasisData(basis1, Order, nqST, mesh)
			PhiDataR = BasisData(basis1, Order, nqST, mesh)

			xglob = np.zeros([nq, sr])
			uL = np.zeros([nqST, sr])
			uR = np.zeros([nqST, sr])

			xelemLPhi = np.zeros([nqST, dim+1])
			xelemRPhi = np.zeros([nqST, dim+1])
			xelemLPsi = np.zeros([nq, dim])
			xelemRPsi = np.zeros([nq, dim])
			
		PsiDataL = Faces2PhiDataL[faceL]
		PsiDataR = Faces2PsiDataR[faceR]
		PhiDataL = Faces2PsiDataL[faceL]
		PhiDataR = Faces2PhiDataR[faceR]

		#Evaluate space-time basis functions
		if PhiDataL is None or QuadChangedST:
			Faces2PhiDataL[faceL] = PhiDataL = BasisData(basis1,Order,nqST,mesh)
			xelemLPhi = PhiDataL.EvalBasisOnFaceADER(mesh, basis1, faceSTL, xqST, xelemLPhi, Get_Phi=True)
		if PhiDataR is None or QuadChangedST:
			Faces2PhiDataR[faceR] = PhiDataR = BasisData(basis1,Order,nqST,mesh)
			xelemRPhi = PhiDataR.EvalBasisOnFaceADER(mesh, basis1, faceSTR, xqST[::-1], xelemRPhi, Get_Phi=True)

		#Evaluate spacial basis functions
		if PsiDataL is None or QuadChanged:
			Faces2PsiDataL[faceL] = PsiDataL = BasisData(basis2,Order,nq,mesh)
			xelemLPsi = PsiDataL.EvalBasisOnFace(mesh, faceL, xq, xelemLPsi, Get_Phi=True)
		if PsiDataR is None or QuadChanged:
			Faces2PsiDataR[faceR] = PsiDataR = BasisData(basis2,Order,nq,mesh)
			xelemRPsi = PsiDataR.EvalBasisOnFace(mesh, faceR, xq[::-1], xelemRPsi, Get_Phi=True)

		NData = IFaceNormal(mesh, IFace, nq, xq, NData)

		nL = PsiDataL.nn
		nR = PsiDataR.nn
		nn = np.amax([nL,nR])

		Stepper = self.Stepper
		dt = Stepper.dt

		uL[:] = np.matmul(PhiDataL.Phi, UL)
		uR[:] = np.matmul(PhiDataR.Phi, UR)

		F = EqnSet.ConvFluxNumerical(uL, uR, NData, nqST, StaticData) # [nq,sr]

		# F = np.reshape(F,(nq,nqST,sr))
		
		# for ir in range(sr):
		# 	#for k in range(nn): # Loop over basis function in space
		# 	for i in range(nqST): # Loop over time
		# 		for j in range(nq): # Loop over space
		# 			PsiL = PsiDataL.Phi[j,:]
		# 			PsiR = PsiDataR.Phi[j,:]
		# 			RL[:,ir] -= wqST[i]*wq[j]*F[j,i,ir]*PsiL
		# 			RR[:,ir] += wqST[i]*wq[j]*F[j,i,ir]*PsiR

		# F = np.reshape(F,(nqST,sr))


		RL -= np.matmul(np.tile(PsiDataL.Phi,(nn,1)).transpose(), F*wqST*wq)
		RR += np.matmul(np.tile(PsiDataR.Phi,(nn,1)).transpose(), F*wqST*wq)

		if elemL == echeck or elemR == echeck:
			if elemL == echeck: print("Left!")
			else: print("Right!")
			code.interact(local=locals())

		# Store in StaticData
		StaticData.pnq = nq
		StaticData.quadData = quadData
		StaticData.quadDataST = quadDataST
		StaticData.Faces2PhiDataL = Faces2PhiDataL
		StaticData.Faces2PhiDataR = Faces2PhiDataR
		StaticData.Faces2PsiDataL = Faces2PsiDataL
		StaticData.Faces2PsiDataR = Faces2PsiDataR
		StaticData.uL = uL
		StaticData.uR = uR
		StaticData.NData = NData

		return RL, RR, StaticData


	def FluxCoefficients(self, elem, dt, Order, basis, U):

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


			QuadOrder,QuadChanged = get_gaussian_quadrature_elem(mesh, mesh.QBasis, Order, EqnSet, quadData)
			QuadOrderST, QuadChangedST = get_gaussian_quadrature_elem(mesh, basis, Order, EqnSet, quadDataST)
			QuadOrderST-=1

			if QuadChanged:
				quadData = QuadData(mesh, mesh.QBasis, entity, QuadOrder)
			if QuadChangedST:
				quadDataST = QuadData(mesh, basis, entity, QuadOrderST)

			nq = quadData.nquad
			xq = quadData.quad_pts
			wq = quadData.quad_wts

			nqST = quadDataST.nquad
			xqST = quadDataST.quad_pts
			wqST = quadDataST.quad_wts

			if QuadChanged:

				PhiData = BasisData(basis,Order,nqST,mesh)
				PhiData.EvalBasis(xqST, Get_Phi=True)
				xphys = np.zeros([nqST, mesh.Dim])

			JData.ElemJacobian(elem,nqST,xqST,mesh,get_djac=True)
			MMinv,_= GetElemInvMassMatrixADER(mesh, basis, Order, PhysicalSpace=True, elem=-1, StaticData=None)
			nn = PhiData.nn
		else:

			xq, nq = EquidistantNodes(basis, Order, xq)
			nn = nq
			JData.ElemJacobian(elem,nq,xq,mesh,get_djac=True)

		if not InterpolateFlux:

			u = np.zeros([nqST, sr])
			u[:] = np.matmul(PhiData.Phi, U)

			#u = np.reshape(u,(nq,nn))
			F = np.zeros([nqST,sr,dim])
			f = EqnSet.ConvFluxInterior(u, F) # [nq,sr]
			f = f.reshape(nqST,sr)
			#f = np.reshape(f,(nq,nq,sr,dim))
			#Phi = PhiData.Phi
			#Phi = np.reshape(Phi,(nq,nq,nn))
		
			rhs *=0.
			# for ir in range(sr):
			# 	for k in range(nn): # Loop over basis function in space
			# 		for i in range(nq): # Loop over time
			# 			for j in range(nq): # Loop over space
			# 				#Phi = PhiData.Phi[j,k]
			# 				rhs[k,ir] += wq[i]*wq[j]*JData.djac[j*(JData.nq!=1)]*f[i,j,ir]*Phi[i,j,k]

			rhs[:] = np.matmul(PhiData.Phi.transpose(), f*(wqST*JData.djac))


			#F = np.reshape(F,(nqST,sr,dim))
			F = np.dot(MMinv,rhs)*(1.0/JData.djac)*dt/2.0

		else:
			F1 = np.zeros([nn,sr,dim])
			#code.interact(local=locals())
			f = EqnSet.ConvFluxInterior(U,F1)
			F = f[:,:,0]*(1.0/JData.djac)*dt/2.0
			#code.interact(local=locals())
		return F

	def SourceCoefficients(self, elem, dt, Order, basis, U):

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
		TimePhiData = None
		xq = None; xphys = None; QuadChanged = True; QuadChangedST = True;
		xglob = None; tglob = None; NData = None;

		rhs = np.zeros([Order2nNode(basis,Order),sr],dtype=U.dtype)

		if not InterpolateFlux:

			QuadOrder,QuadChanged = get_gaussian_quadrature_elem(mesh, mesh.QBasis, Order, EqnSet, quadData)
			QuadOrderST, QuadChangedST = get_gaussian_quadrature_elem(mesh, basis, Order, EqnSet, quadDataST)
			QuadOrderST-=1

			if QuadChanged:
				quadData = QuadData(mesh, mesh.QBasis, entity, QuadOrder)
			if QuadChangedST:
				quadDataST = QuadData(mesh, basis, entity, QuadOrderST)

			nq = quadData.nquad
			xq = quadData.quad_pts
			wq = quadData.quad_wts

			nqST = quadDataST.nquad
			xqST = quadDataST.quad_pts
			wqST = quadDataST.quad_wts

			xglob, GeomPhiData = Ref2Phys(mesh, elem, GeomPhiData, nq, xq, xglob, QuadChanged)
			tglob, TimePhiData = Ref2PhysTime(mesh, elem, self.Time, self.Stepper.dt, TimePhiData, nqST, xqST, tglob, QuadChangedST)
			
			if QuadChanged:

				PhiData = BasisData(basis,Order,nqST,mesh)
				PhiData.EvalBasis(xqST, Get_Phi=True)
				xphys = np.zeros([nqST, mesh.Dim])

			JData.ElemJacobian(elem,nqST,xqST,mesh,get_djac=True)
			MMinv,_= GetElemInvMassMatrixADER(mesh, basis, Order, PhysicalSpace=True, elem=-1, StaticData=None)
			nn = PhiData.nn
		else:

			xq, nq = EquidistantNodes(mesh.QBasis, Order, xq)
			nn = nq
			JData.ElemJacobian(elem,nq,xq,mesh,get_djac=True)
			
			QuadOrderST, QuadChangedST = get_gaussian_quadrature_elem(mesh, basis, Order, EqnSet, quadDataST)
			QuadOrderST-=1

			if QuadChangedST:
				quadDataST = QuadData(mesh, basis, entity, QuadOrderST)

			nqST = quadDataST.nquad
			xqST = quadDataST.quad_pts

			xglob, GeomPhiData = Ref2Phys(mesh, elem, GeomPhiData, nq, xq, xglob, QuadChanged)
			tglob, TimePhiData = Ref2PhysTime(mesh, elem, self.Time, self.Stepper.dt, TimePhiData, nqST, xqST, tglob, QuadChangedST)

		if not InterpolateFlux:

			u = np.zeros([nqST, sr])
			u[:] = np.matmul(PhiData.Phi, U)

			#u = np.reshape(u,(nq,nn))
			#S = np.zeros([nqST,sr,dim])
			s = np.zeros([nqST,sr])
			s = EqnSet.SourceState(nqST, xglob, tglob, NData, u, s) # [nq,sr]
			
			#s = np.reshape(s,(nq,nq,sr,dim))
			#Phi = PhiData.Phi
			#Phi = np.reshape(Phi,(nq,nq,nn))
		
			rhs *=0.
			# for ir in range(sr):
			# 	for k in range(nn): # Loop over basis function in space
			# 		for i in range(nq): # Loop over time
			# 			for j in range(nq): # Loop over space
			# 				#Phi = PhiData.Phi[j,k]
			# 				rhs[k,ir] += wq[i]*wq[j]*JData.djac[j*(JData.nq!=1)]*s[i,j,ir]*Phi[i,j,k]
			rhs[:] = np.matmul(PhiData.Phi.transpose(),s*wqST*JData.djac)
			S = np.dot(MMinv,rhs)*dt/2.0

		else:
			S1 = np.zeros([nqST,sr])
			s = EqnSet.SourceState(nqST, xglob, tglob, NData, U, S1)
			S = s*dt/2.0
		return S
