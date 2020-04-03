import numpy as np 
import General
import Data
import Quadrature
import Mesh
import Basis
import MeshTools
import code

PosTol = 1.e-10

def SetLimiter(limiterType):
	if limiterType is None:
		return None
	elif limiterType is General.LimiterType.PositivityPreserving.name:
		return PPLimiter()
	elif limiterType is General.LimiterType.ScalarPositivityPreserving.name:
		return PPScalarLimiter()
	else:
		raise NotImplementedError


class PPLimiter(object):
	def __init__(self):
		# self.StaticData = Data.GenericData()
		pass

	def LimitSolution(self, solver, U):
		EqnSet = solver.EqnSet
		mesh = solver.mesh
		# U = EqnSet.U.Arrays
		StaticData = None

		for elem in range(mesh.nElem):
			U[elem] = self.LimitElement(solver, elem, U[elem], StaticData)

	def LimitElement(self, solver, elem, U, StaticData):
		EqnSet = solver.EqnSet
		mesh = solver.mesh

		basis = EqnSet.Basis
		Order = EqnSet.Order
		entity = General.EntityType.Element
		sr = EqnSet.StateRank
		dim = EqnSet.Dim
		Faces = mesh.Faces[elem]
		nFacePerElem = mesh.nFacePerElem
		_, ElemVols = MeshTools.ElementVolumes(mesh, solver)

		scalar1 = "Density"
		scalar2 = "Pressure"

		if StaticData is None:
			nq_prev = 0
			quadElem = None
			quadFace = None
			PhiElem = None
			xelem = None
			JData = JData = Basis.JacobianData(mesh)
			u = None
			u_bar = None
			rho_bar = None
			p_bar = None
			u_D = None
			rho_D = None
			p_D = None
			theta = None
			StaticData = Data.GenericData()
		else:
			nq_prev = StaticData.nq_prev
			quadElem = StaticData.quadElem
			quadFace = StaticData.quadFace
			PhiElem = StaticData.PhiElem
			xelem = StaticData.xelem
			JData = StaticData.JData
			u = StaticData.u
			u_bar = StaticData.u_bar
			rho_bar = StaticData.rho_bar
			p_bar = StaticData.p_bar
			u_D = StaticData.u_D
			rho_D = StaticData.rho_D
			p_D = StaticData.p_D
			theta = StaticData.theta
			Faces2PhiData = StaticData.Faces2PhiData

		QuadOrder,QuadChanged = Quadrature.get_gaussian_quadrature_elem(mesh, basis, Order, EqnSet, quadElem)
		if QuadChanged:
			quadElem = Quadrature.QuadData(mesh, basis, entity, QuadOrder)

		nq = quadElem.nquad
		xq = quadElem.quad_pts
		wq = quadElem.quad_wts

		if QuadChanged:
			PhiElem = Basis.BasisData(EqnSet.Basis,Order,nq,mesh)
			PhiElem.eval_basis(xq, Get_Phi=True, Get_GPhi=True) # [nq, nn]

			u = np.zeros([nq, sr])
			u_bar = np.zeros([1, sr])
			F = np.zeros([nq, sr, dim])


		JData.element_jacobian(mesh,elem,nq,xq,get_djac=True,get_jac=False,get_ijac=True)

		#nn = PhiElem.nn

		# interpolate state and gradient at quad points
		# u = np.zeros([nq, sr])
		u[:] = np.matmul(PhiElem.Phi, U)

		# Multiply quadrature weights by Jacobian determinant
		# wq *= JData.djac

		# Average value of state
		# vol = np.sum(wq*JData.djac)
		vol = ElemVols[elem]
		u_bar[:] = np.matmul(u.transpose(), wq*JData.djac).T/vol
		# u_bar.shape = 1, -1

		# Density and pressure
		rho_bar = EqnSet.ComputeScalars(scalar1, u_bar, 1, rho_bar)
		p_bar = EqnSet.ComputeScalars(scalar2, u_bar, 1, p_bar)

		if np.any(rho_bar < 0.) or np.any(p_bar < 0.):
			raise Errors.NotPhysicalError

		''' Get relevant quadrature points '''

		nq_eval = nq
		nq_elem = nq
		# Loop through faces
		for face in range(nFacePerElem):
			Face = Faces[face]
			eN, faceN = MeshTools.NeighborAcrossFace(mesh, elem, face)
			if Face.Type == Mesh.FaceType.Boundary:
				# boundary face
				eN = elem; faceN = face
				BFG = mesh.BFaceGroups[Face.Group]
				BF = BFG.BFaces[Face.Number]
				entity = General.EntityType.IFace
				QuadOrder, QuadChanged = Quadrature.get_gaussian_quadrature_face(mesh, BF, mesh.QBasis, Order, EqnSet, quadFace)
			else:
				IF = mesh.IFaces[Face.Number]
				OrderN = EqnSet.Order
				entity = General.EntityType.BFace
				QuadOrder, QuadChanged = Quadrature.get_gaussian_quadrature_face(mesh, IF, mesh.QBasis, np.amax([Order,OrderN]), EqnSet, quadFace)

			if QuadChanged:
				quadFace = Quadrature.QuadData(mesh, basis, entity, QuadOrder)

			nq = quadFace.nquad
			xq = quadFace.quad_pts
			wq = quadFace.quad_wts

			if QuadChanged:
				Faces2PhiData = [None for i in range(nFacePerElem)]
				uf = np.zeros([nq, sr])

			PhiData = Faces2PhiData[face]
			if PhiData is None or QuadChanged:
				Faces2PhiData[face] = PhiData = Basis.BasisData(EqnSet.Basis,Order,nq,mesh)
				xelem = PhiData.eval_basis_on_face(mesh, face, xq, xelem, Get_Phi=True)

			if face == 0:
				# first face
				if nq_prev == 0: 
					# Best guess for size
					nq_prev = nq_elem + nFacePerElem*nq
					u_D = np.zeros([nq_prev, sr])

				# Fill in element interior values
				u_D[:nq_eval,:] = u

			''' Interpolate state to face quadrature points '''
			uf[:] = np.matmul(PhiData.Phi, U)
			# Increment nq_eval
			nq_eval += nq
			if nq_eval > nq_prev:
				raise ValueError
				# resize
				u_D = np.concatenate((u_D, np.zeros([nq_eval-nq_prev,sr])))
				nq_prev = nq_eval
			# Add to u_D
			u_D[nq_eval-nq:nq_eval,:] = uf

		# Final resize
		if nq_prev != nq_eval:
			nq_prev = nq_eval
			u_D = u_D[:nq_eval,:]

		if theta is None or theta.shape[0] != nq_eval:
			theta = np.zeros([nq_eval,1])

		np.seterr(divide='ignore')

		''' Limit density '''
		# Compute density
		rho_D = EqnSet.ComputeScalars(scalar1, u_D, nq_eval, rho_D)
		theta[:] = np.abs((rho_bar - PosTol)/(rho_bar - rho_D))
		theta1 = np.amin([1.,np.amin(theta)])

		# Rescale
		if theta1 < 1.:
			irho = EqnSet.GetStateIndex(scalar1)
			U[:,irho] = theta1*U[:,irho] + (1. - theta1)*rho_bar

			# Intermediate limited solution
			u_D[:nq_elem] = np.matmul(PhiElem.Phi, U)
			qcount = nq_elem
			for face in range(nFacePerElem):
				PhiData = Faces2PhiData[face]
				nq = PhiData.nq
				u_D[qcount:qcount+nq] = np.matmul(PhiData.Phi, U)
				qcount += nq

			if qcount != nq_eval:
				raise ValueError

		''' Limit pressure '''
		p_D = EqnSet.ComputeScalars(scalar2, u_D, nq_eval, p_D)
		# theta = np.abs((p_bar - PosTol)/(p_bar - p_D))
		theta[:] = 1.
		iposP = (p_D < 0.).reshape(-1) # indices where pressure is negative
		theta[iposP] = p_bar/(p_bar - p_D[iposP])
		theta2 = np.amin(theta)
		if theta2 < 1.:
			U[:] = theta2*U + (1. - theta2)*u_bar

		np.seterr(divide='warn')

		# Store in StaticData
		StaticData.nq_prev = nq_prev
		StaticData.quadElem = quadElem
		StaticData.quadFace = quadFace
		StaticData.PhiElem = PhiElem
		StaticData.xelem = xelem
		StaticData.JData = JData
		StaticData.u = u
		StaticData.u_bar = u_bar
		StaticData.rho_bar = rho_bar
		StaticData.p_bar = p_bar
		StaticData.u_D = u_D
		StaticData.rho_D = rho_D
		StaticData.p_D = p_D
		StaticData.theta = theta
		StaticData.Faces2PhiData = Faces2PhiData

		return U

class PPScalarLimiter(object):
	def __init__(self):
		# self.StaticData = Data.GenericData()
		pass

	def LimitSolution(self, solver, U):
		EqnSet = solver.EqnSet
		mesh = solver.mesh
		# U = EqnSet.U.Arrays
		StaticData = None

		for elem in range(mesh.nElem):
			U[elem] = self.LimitElement(solver, elem, U[elem], StaticData)

	def LimitElement(self, solver, elem, U, StaticData):
		EqnSet = solver.EqnSet
		mesh = solver.mesh

		basis = EqnSet.Basis
		Order = EqnSet.Order
		entity = General.EntityType.Element
		sr = EqnSet.StateRank
		dim = EqnSet.Dim
		Faces = mesh.Faces[elem]
		nFacePerElem = mesh.nFacePerElem
		_, ElemVols = MeshTools.ElementVolumes(mesh, solver)

		scalar1 = "u"

		if StaticData is None:
			nq_prev = 0
			quadElem = None
			quadFace = None
			PhiElem = None
			xelem = None
			JData = JData = Basis.JacobianData(mesh)
			u = None
			u_bar = None
			u_D = None
			theta = None
			StaticData = Data.GenericData()
		else:
			nq_prev = StaticData.nq_prev
			quadElem = StaticData.quadElem
			quadFace = StaticData.quadFace
			PhiElem = StaticData.PhiElem
			xelem = StaticData.xelem
			JData = StaticData.JData
			u = StaticData.u
			u_bar = StaticData.u_bar
			u_D = StaticData.u_D
			theta = StaticData.theta
			Faces2PhiData = StaticData.Faces2PhiData

		QuadOrder,QuadChanged = Quadrature.get_gaussian_quadrature_elem(mesh, basis, Order, EqnSet, quadElem)
		if QuadChanged:
			quadElem = Quadrature.QuadData(mesh, basis, entity, QuadOrder)

		nq = quadElem.nquad
		xq = quadElem.quad_pts
		wq = quadElem.quad_wts

		if QuadChanged:
			PhiElem = Basis.BasisData(EqnSet.Basis,Order,nq,mesh)
			PhiElem.eval_basis(xq, Get_Phi=True, Get_GPhi=True) # [nq, nn]

			u = np.zeros([nq, sr])
			u_bar = np.zeros([1, sr])
			F = np.zeros([nq, sr, dim])


		JData.element_jacobian(mesh,elem,nq,xq,get_djac=True,get_jac=False,get_ijac=True)

		#nn = PhiElem.nn

		# interpolate state and gradient at quad points
		# u = np.zeros([nq, sr])
		u[:] = np.matmul(PhiElem.Phi, U)

		# Multiply quadrature weights by Jacobian determinant
		# wq *= JData.djac

		# Average value of state
		# vol = np.sum(wq*JData.djac)
		vol = ElemVols[elem]
		u_bar[:] = np.matmul(u.transpose(), wq*JData.djac).T/vol

		''' Get relevant quadrature points '''

		nq_eval = nq
		nq_elem = nq
		# Loop through faces
		for face in range(nFacePerElem):
			Face = Faces[face]
			eN, faceN = MeshTools.NeighborAcrossFace(mesh, elem, face)
			if Face.Type == Mesh.FaceType.Boundary:
				# boundary face
				eN = elem; faceN = face
				BFG = mesh.BFaceGroups[Face.Group]
				BF = BFG.BFaces[Face.Number]
				entity = General.EntityType.IFace
				QuadOrder, QuadChanged = Quadrature.get_gaussian_quadrature_face(mesh, BF, mesh.QBasis, Order, EqnSet, quadFace)
			else:
				IF = mesh.IFaces[Face.Number]
				OrderN = EqnSet.Order
				entity = General.EntityType.BFace
				QuadOrder, QuadChanged = Quadrature.get_gaussian_quadrature_face(mesh, IF, mesh.QBasis, np.amax([Order,OrderN]), EqnSet, quadFace)

			if QuadChanged:
				quadFace = Quadrature.QuadData(mesh, basis, entity, QuadOrder)

			nq = quadFace.nquad
			xq = quadFace.quad_pts
			wq = quadFace.quad_wts

			if QuadChanged:
				Faces2PhiData = [None for i in range(nFacePerElem)]
				uf = np.zeros([nq, sr])

			PhiData = Faces2PhiData[face]
			if PhiData is None or QuadChanged:
				Faces2PhiData[face] = PhiData = Basis.BasisData(EqnSet.Basis,Order,nq,mesh)
				xelem = PhiData.eval_basis_on_face(mesh, face, xq, xelem, Get_Phi=True)

			if face == 0:
				# first face
				if nq_prev == 0: 
					# Best guess for size
					nq_prev = nq_elem + nFacePerElem*nq
					u_D = np.zeros([nq_prev, sr])

				# Fill in element interior values
				u_D[:nq_eval,:] = u

			''' Interpolate state to face quadrature points '''
			uf[:] = np.matmul(PhiData.Phi, U)
			# Increment nq_eval
			nq_eval += nq
			if nq_eval > nq_prev:
				raise ValueError
				# resize
				u_D = np.concatenate((u_D, np.zeros([nq_eval-nq_prev,sr])))
				nq_prev = nq_eval
			# Add to u_D
			u_D[nq_eval-nq:nq_eval,:] = uf

		# Final resize
		if nq_prev != nq_eval:
			nq_prev = nq_eval
			u_D = u_D[:nq_eval,:]

		if theta is None or theta.shape[0] != nq_eval:
			theta = np.zeros([nq_eval,1])

		np.seterr(divide='ignore')

		''' Limit density '''
		# Compute density
		theta[:] = np.abs((u_bar - PosTol)/(u_bar - u_D))
		theta1 = np.amin([1.,np.amin(theta)])

		# Rescale
		if theta1 < 1.:
			#		iU = EqnSet.GetStateIndex(scalar1)
			iU = 1
			#code.interact(local=locals())
			#U[:,iU] = theta1*U[:,iU] + (1. - theta1)*u_bar
			U[:] = theta1*U[:] + (1. - theta1)*u_bar
		np.seterr(divide='warn')

		# Store in StaticData
		StaticData.nq_prev = nq_prev
		StaticData.quadElem = quadElem
		StaticData.quadFace = quadFace
		StaticData.PhiElem = PhiElem
		StaticData.xelem = xelem
		StaticData.JData = JData
		StaticData.u = u
		StaticData.u_bar = u_bar
		StaticData.u_D = u_D
		StaticData.theta = theta
		StaticData.Faces2PhiData = Faces2PhiData

		return U












