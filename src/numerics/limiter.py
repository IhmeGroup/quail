import code
import numpy as np 

import general
import data as Data

import numerics.quadrature.quadrature as quadrature
import numerics.basis.basis as Basis

import meshing.meshbase as Mesh
import meshing.tools as MeshTools

PosTol = 1.e-10

def set_limiter(limiterType):
	'''
    Method: set_limiter
    ----------------------------
	selects limiter bases on input deck

    INPUTS:
		limiterType: type of limiter selected (Default: None)
	'''
	if limiterType is None:
		return None
	elif limiterType is general.LimiterType.PositivityPreserving.name:
		return PPLimiter()
	elif limiterType is general.LimiterType.ScalarPositivityPreserving.name:
		return PPScalarLimiter()
	else:
		raise NotImplementedError


class PPLimiter(object):
	'''
    Class: PPLimiter
    ------------------
    This class contains information about the positivity preserving limiter
    '''
	def __init__(self):
		'''
		Method: __init__
		-------------------
		Initializes PPLimiter object
		'''
		pass

	def limit_solution(self, solver, U):
		'''
		Method: limit_solution
		------------------------
		Calls the limiter function for each element
		INPUTS:
			solver: type of solver (i.e. DG, ADER-DG, etc...)

		OUTPUTS:
			U: solution array
		'''
		EqnSet = solver.EqnSet
		mesh = solver.mesh
		# U = EqnSet.U.Arrays
		StaticData = None

		for elem in range(mesh.nElem):
			U[elem] = self.limit_element(solver, elem, U[elem], StaticData)

	def limit_element(self, solver, elem, U, StaticData):
		'''
		Method: limit_element
		------------------------
		Limits the solution on each element

		INPUTS:
			solver: type of solver (i.e. DG, ADER-DG, etc...)
			elem: element index

		OUTPUTS:
			U: solution array
		'''
		EqnSet = solver.EqnSet
		mesh = solver.mesh
		basis = solver.basis

		order = EqnSet.order
		entity = general.EntityType.Element
		ns = EqnSet.StateRank
		dim = EqnSet.Dim
		Faces = mesh.Faces[elem]
		nFacePerElem = mesh.nFacePerElem
		_, ElemVols = MeshTools.element_volumes(mesh, solver)

		scalar1 = "Density"
		scalar2 = "Pressure"

		if StaticData is None:
			nq_prev = 0
			quadElem = None
			quadFace = None
			# PhiElem = None
			xelem = None
			# JData = JData = Basis.JacobianData(mesh)
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
			# PhiElem = StaticData.PhiElem
			xelem = StaticData.xelem
			# JData = StaticData.JData
			u = StaticData.u
			u_bar = StaticData.u_bar
			rho_bar = StaticData.rho_bar
			p_bar = StaticData.p_bar
			u_D = StaticData.u_D
			rho_D = StaticData.rho_D
			p_D = StaticData.p_D
			theta = StaticData.theta
			Faces2PhiData = StaticData.Faces2PhiData

		QuadOrder,QuadChanged = quadrature.get_gaussian_quadrature_elem(mesh, basis, order, EqnSet, quadElem)
		if QuadChanged:
			quadElem = quadrature.QuadData(mesh, basis, entity, QuadOrder)

		quad_pts = quadElem.quad_pts
		quad_wts = quadElem.quad_wts
		nq = quad_pts.shape[0]

		if QuadChanged:
			# PhiElem = Basis.BasisData(EqnSet.Basis,order,mesh)
			basis.eval_basis(quad_pts, Get_Phi=True, Get_GPhi=True) # [nq, nn]

			u = np.zeros([nq, ns])
			u_bar = np.zeros([1, ns])
			F = np.zeros([nq, ns, dim])


		djac,_,ijac = Basis.element_jacobian(mesh,elem,quad_pts,get_djac=True,get_jac=False,get_ijac=True)

		#nn = PhiElem.nn

		# interpolate state and gradient at quad points
		# u = np.zeros([nq, ns])
		u[:] = np.matmul(basis.basis_val, U)

		# Multiply quadrature weights by Jacobian determinant
		# wq *= JData.djac

		# Average value of state
		# vol = np.sum(wq*JData.djac)
		vol = ElemVols[elem]
		u_bar[:] = np.matmul(u.transpose(), quad_wts*djac).T/vol
		# u_bar.shape = 1, -1

		# Density and pressure
		rho_bar = EqnSet.ComputeScalars(scalar1, u_bar, rho_bar)
		p_bar = EqnSet.ComputeScalars(scalar2, u_bar, p_bar)

		if np.any(rho_bar < 0.) or np.any(p_bar < 0.):
			raise Errors.NotPhysicalError

		''' Get relevant quadrature points '''

		nq_eval = nq
		nq_elem = nq

		# Loop through faces
		for face in range(nFacePerElem):
			Face = Faces[face]
			eN, faceN = MeshTools.neighbor_across_face(mesh, elem, face)
			if Face.Type == Mesh.FaceType.Boundary:
				# boundary face
				eN = elem; faceN = face
				BFG = mesh.BFaceGroups[Face.Group]
				BF = BFG.BFaces[Face.Number]
				entity = general.EntityType.IFace
				QuadOrder, QuadChanged = quadrature.get_gaussian_quadrature_face(mesh, BF, mesh.gbasis, order, EqnSet, quadFace)
			else:
				IF = mesh.IFaces[Face.Number]
				order_n = EqnSet.order
				entity = general.EntityType.BFace
				QuadOrder, QuadChanged = quadrature.get_gaussian_quadrature_face(mesh, IF, mesh.gbasis, np.amax([order,order_n]), EqnSet, quadFace)


			if QuadChanged:
				quadFace = quadrature.QuadData(mesh, basis, entity, QuadOrder)

			quad_pts = quadFace.quad_pts
			quad_wts = quadFace.quad_wts
			nq = quad_pts.shape[0]

			if QuadChanged:
				Faces2PhiData = [None for i in range(nFacePerElem)]
				uf = np.zeros([nq,ns])

			#PhiData = Faces2PhiData[face]
			#if QuadChanged:
			Faces2PhiData[face] = basis # PhiData = Basis.BasisData(EqnSet.Basis,order,mesh)
			xelem = basis.eval_basis_on_face(mesh, face, quad_pts, xelem, Get_Phi=True)

			if face == 0:
				# first face
				if nq_prev == 0: 
					# Best guess for size
					nq_prev = nq_elem + nFacePerElem*nq
					u_D = np.zeros([nq_prev, ns])

				# Fill in element interior values
				u_D[:nq_eval,:] = u

			''' Interpolate state to face quadrature points '''
			uf[:] = np.matmul(basis.basis_val, U)
			# Increment nq_eval
			nq_eval += nq
			if nq_eval > nq_prev:
				raise ValueError
				# resize
				u_D = np.concatenate((u_D, np.zeros([nq_eval-nq_prev,ns])))
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
		rho_D = EqnSet.ComputeScalars(scalar1, u_D, rho_D)
		theta[:] = np.abs((rho_bar - PosTol)/(rho_bar - rho_D))
		theta1 = np.amin([1.,np.amin(theta)])

		# Rescale
		if theta1 < 1.:
			irho = EqnSet.GetStateIndex(scalar1)
			U[:,irho] = theta1*U[:,irho] + (1. - theta1)*rho_bar

			# Intermediate limited solution
			u_D[:nq_elem] = np.matmul(basis.basis_val, U)
			qcount = nq_elem
			for face in range(nFacePerElem):
				basis = Faces2PhiData[face]
				nq = basis.basis_val.shape[0]
				#nq = PhiData.nq
				u_D[qcount:qcount+nq] = np.matmul(basis.basis_val, U)
				qcount += nq

			if qcount != nq_eval:
				raise ValueError

		''' Limit pressure '''
		p_D = EqnSet.ComputeScalars(scalar2, u_D, p_D)
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
		# StaticData.PhiElem = PhiElem
		StaticData.xelem = xelem
		# StaticData.JData = JData
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
	'''
	Class: PPScalarLimiter
	-------------------
	This class contains information about the scalar positivity preserving limiter
	'''
	def __init__(self):
		'''
		Method: __init__
		-------------------
		Initializes PPLimiter object
		'''
		pass

	def limit_solution(self, solver, U):
		'''
		Method: limit_solution
		------------------------
		Calls the limiter function for each element
		INPUTS:
			solver: type of solver (i.e. DG, ADER-DG, etc...)

		OUTPUTS:
			U: solution array
		'''
		EqnSet = solver.EqnSet
		mesh = solver.mesh
		# U = EqnSet.U.Arrays
		StaticData = None

		for elem in range(mesh.nElem):
			U[elem] = self.limit_element(solver, elem, U[elem], StaticData)

	def limit_element(self, solver, elem, U, StaticData):
		'''
		Method: limit_element
		------------------------
		Limits the solution on each element

		INPUTS:
			solver: type of solver (i.e. DG, ADER-DG, etc...)
			elem: element index

		OUTPUTS:
			U: solution array
		'''
		EqnSet = solver.EqnSet
		mesh = solver.mesh

		basis = EqnSet.Basis
		order = EqnSet.order
		entity = general.EntityType.Element
		ns = EqnSet.StateRank
		dim = EqnSet.Dim
		Faces = mesh.Faces[elem]
		nFacePerElem = mesh.nFacePerElem
		_, ElemVols = MeshTools.element_volumes(mesh, solver)

		scalar1 = "u"

		if StaticData is None:
			nq_prev = 0
			quadElem = None
			quadFace = None
			PhiElem = None
			xelem = None
			JData = Basis.JacobianData(mesh)
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

		QuadOrder,QuadChanged = quadrature.get_gaussian_quadrature_elem(mesh, basis, order, EqnSet, quadElem)
		if QuadChanged:
			quadElem = quadrature.QuadData(mesh, basis, entity, QuadOrder)

		quad_pts = quadElem.quad_pts
		quad_wts = quadElem.quad_wts
		nq = quad_pts.shape[0]

		if QuadChanged:
			PhiElem = Basis.BasisData(EqnSet.Basis,order,mesh)
			PhiElem.eval_basis(quad_pts, Get_Phi=True, Get_GPhi=True) # [nq, nn]

			u = np.zeros([nq, ns])
			u_bar = np.zeros([1, ns])
			F = np.zeros([nq, ns, dim])


		JData.element_jacobian(mesh,elem,quad_pts,get_djac=True,get_jac=False,get_ijac=True)

		#nn = PhiElem.nn

		# interpolate state and gradient at quad points
		# u = np.zeros([nq, ns])
		u[:] = np.matmul(PhiElem.Phi, U)

		# Multiply quadrature weights by Jacobian determinant
		# wq *= JData.djac

		# Average value of state
		# vol = np.sum(wq*JData.djac)
		vol = ElemVols[elem]
		u_bar[:] = np.matmul(u.transpose(), quad_wts*JData.djac).T/vol

		''' Get relevant quadrature points '''

		nq_eval = nq
		nq_elem = nq
		# Loop through faces
		for face in range(nFacePerElem):
			Face = Faces[face]
			eN, faceN = MeshTools.neighbor_across_face(mesh, elem, face)
			if Face.Type == Mesh.FaceType.Boundary:
				# boundary face
				eN = elem; faceN = face
				BFG = mesh.BFaceGroups[Face.Group]
				BF = BFG.BFaces[Face.Number]
				entity = general.EntityType.IFace
				QuadOrder, QuadChanged = quadrature.get_gaussian_quadrature_face(mesh, BF, mesh.QBasis, order, EqnSet, quadFace)
			else:
				IF = mesh.IFaces[Face.Number]
				order_n = EqnSet.order
				entity = general.EntityType.BFace
				QuadOrder, QuadChanged = quadrature.get_gaussian_quadrature_face(mesh, IF, mesh.QBasis, np.amax([order,order_n]), EqnSet, quadFace)

			if QuadChanged:
				quadFace = quadrature.QuadData(mesh, basis, entity, QuadOrder)

			quad_pts = quadFace.quad_pts
			quad_wts = quadFace.quad_wts
			nq = quad_pts.shape[0]

			if QuadChanged:
				Faces2PhiData = [None for i in range(nFacePerElem)]
				uf = np.zeros([nq, ns])

			PhiData = Faces2PhiData[face]
			if PhiData is None or QuadChanged:
				Faces2PhiData[face] = PhiData = Basis.BasisData(EqnSet.Basis,order,mesh)
				xelem = PhiData.eval_basis_on_face(mesh, face, quad_pts, xelem, Get_Phi=True)

			if face == 0:
				# first face
				if nq_prev == 0: 
					# Best guess for size
					nq_prev = nq_elem + nFacePerElem*nq
					u_D = np.zeros([nq_prev, ns])

				# Fill in element interior values
				u_D[:nq_eval,:] = u

			''' Interpolate state to face quadrature points '''
			uf[:] = np.matmul(PhiData.Phi, U)
			# Increment nq_eval
			nq_eval += nq
			if nq_eval > nq_prev:
				raise ValueError
				# resize
				u_D = np.concatenate((u_D, np.zeros([nq_eval-nq_prev,ns])))
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












