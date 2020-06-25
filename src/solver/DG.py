from abc import ABC, abstractmethod
import code
import copy
import numpy as np 
import time

import errors
from data import ArrayList, GenericData

import general
from general import SetSolverParams, BasisType, ShapeType, EntityType
import meshing.meshbase as mesh_defs

import numerics.basis.tools as basis_tools

import numerics.limiting.positivitypreserving as pp_limiter

# from numerics.quadrature.quadrature import QuadData
import numerics.timestepping.stepper as stepper

import processing.post as post_defs
import processing.readwritedatafiles as ReadWriteDataFiles

import solver.tools as solver_tools
global echeck
echeck = -1


def set_limiter(limiter_type, physics_type):
	'''
    Method: set_limiter
    ----------------------------
	selects limiter bases on input deck

    INPUTS:
		limiterType: type of limiter selected (Default: None)
	'''
	if limiter_type is None:
		return None
	elif general.LimiterType[limiter_type] is general.LimiterType.PositivityPreserving:
		limiter_ref = pp_limiter.PositivityPreserving
	elif general.LimiterType[limiter_type] is general.LimiterType.ScalarPositivityPreserving:
		limiter_ref = pp_limiter.ScalarPositivityPreserving
	else:
		raise NotImplementedError

	limiter = limiter_ref(physics_type)

	return limiter


class SolverBase(ABC):
	def __init__(self,Params,EqnSet,mesh):
		'''
		Method: __init__
		--------------------------------------------------------------------------
		Initializes the DG object, verifies parameters, and initializes the state

		INPUTS:
			Params: list of parameters for the solver
			EqnSet: solver object (current implementation supports Scalar and Euler equations)
			mesh: mesh object
		'''

		self.Params = Params
		self.EqnSet = EqnSet
		self.mesh = mesh
		self.DataSet = GenericData()

		self.Time = Params["StartTime"]
		self.nTimeStep = 0 # will be set later

		TimeScheme = Params["TimeScheme"]
		self.Stepper = stepper.set_stepper(TimeScheme)

		# Set the basis functions for the solver
		basis_name  = Params["InterpBasis"]
		self.basis = basis_tools.set_basis(mesh, EqnSet.order, basis_name)

		# check for compatibility
		if mesh.gbasis.shape_type != self.basis.shape_type:
			raise errors.IncompatibleError

		# Limiter
		limiterType = Params["ApplyLimiter"]
		self.Limiter = set_limiter(limiterType, EqnSet.PHYSICS_TYPE)

		# Check validity of parameters
		self.check_solver_params()

		# Precompute operators
		self.precompute_matrix_operators()
		if self.Limiter is not None:
			self.Limiter.precompute_operators(self)

		# Initialize state
		if Params["RestartFile"] is None:
			self.init_state_from_fcn()

	@abstractmethod
	def check_solver_params(self):
		pass

	@abstractmethod
	def precompute_matrix_operators(self):
		pass

	def init_state_from_fcn(self):
		mesh = self.mesh
		EqnSet = self.EqnSet
		basis = self.basis
		Params = self.Params
		iMM_elems = self.elem_operators.iMM_elems

		U = EqnSet.U
		ns = EqnSet.StateRank
		order = EqnSet.order

		if Params["InterpolateIC"]:
			eval_pts, npts = basis.equidistant_nodes(order)
		else:
			order = 2*np.amax([EqnSet.order, 1])
			order = EqnSet.QuadOrder(order)
			# quad_order, _ = get_gaussian_quadrature_elem(mesh, basis, order)
			quad_order = basis.get_quadrature(mesh, order)
			basis.get_quad_data(quad_order, 0)
			# quad_data = QuadData(mesh, mesh.gbasis, general.EntityType.Element, quad_order)

			eval_pts = basis.quad_pts
			# eval_pts = quad_data.quad_pts
			npts = eval_pts.shape[0]

		for elem in range(mesh.nElem):
			xphys, _ = mesh_defs.ref_to_phys(mesh, elem, None, eval_pts)
			f = EqnSet.CallFunction(EqnSet.IC, x=xphys, t=self.Time)
			# f.shape = npts,ns

			if Params["InterpolateIC"]:
				solver_tools.interpolate_to_nodes(f, U[elem,:,:])
			else:
				solver_tools.L2_projection(mesh, iMM_elems[elem], basis, elem, f, U[elem,:,:])


	def project_state_to_new_basis(self, U_old, basis_old, order_old):
		mesh = self.mesh
		EqnSet = self.EqnSet
		basis = self.basis
		Params = self.Params
		iMM_elems = self.elem_operators.iMM_elems

		U = EqnSet.U
		ns = EqnSet.StateRank

		# basis_old = basis_tools.set_basis(mesh, order_old, basis_name_old)
		if basis_old.shape_type != basis.shape_type:
			raise errors.IncompatibleError

		if Params["InterpolateIC"]:
			eval_pts, npts = basis.equidistant_nodes(EqnSet.order)
		else:
			order = 2*np.amax([EqnSet.order, order_old])
			quad_order = basis.get_quadrature(mesh, order)
			# quad_order, _ = get_gaussian_quadrature_elem(mesh, basis, order)
			# quad_data = QuadData(mesh, mesh.gbasis, general.EntityType.Element, quad_order)
			basis.get_quad_data(quad_order,0)
			eval_pts = basis.quad_pts
			npts = eval_pts.shape[0]

		basis_old.eval_basis(eval_pts, Get_Phi=True)

		for elem in range(mesh.nElem):
			Up_old = np.matmul(basis_old.basis_val, U_old[elem,:,:])

			if Params["InterpolateIC"]:
				solver_tools.interpolate_to_nodes(Up_old, U[elem,:,:])
			else:
				solver_tools.L2_projection(mesh, iMM_elems[elem], basis, elem, Up_old, U[elem,:,:])


	# def init_state_from_fcn(self):
	# 	'''
	# 	Method: init_state
	# 	-------------------------
	# 	Initializes the state based on prescribed initial conditions
		
	# 	'''
	# 	mesh = self.mesh
	# 	EqnSet = self.EqnSet
	# 	basis = self.basis

	# 	nb = basis.nb

	# 	U = EqnSet.U
	# 	ns = EqnSet.StateRank
	# 	Params = self.Params

	# 	# Get mass matrices
	# 	try:
	# 		iMM = self.DataSet.iMM_all
	# 	except AttributeError:
	# 		# not found; need to compute
	# 		iMM_all = basis_tools.get_inv_mass_matrices(mesh, EqnSet, solver=self)

	# 	InterpolateIC = Params["InterpolateIC"]
	# 	quadData = None
	# 	# JData = JacobianData(mesh)
	# 	GeomPhiData = None
	# 	quad_pts = None
	# 	xphys = None

	# 	#basis = EqnSet.Basis
	# 	order = EqnSet.order
	# 	rhs = np.zeros([nb,ns],dtype=U.dtype)

	# 	# Precompute basis and quadrature
	# 	if not InterpolateIC:
	# 		QuadOrder,_ = get_gaussian_quadrature_elem(mesh, basis,
	# 			2*np.amax([order,1]), EqnSet, quadData)

	# 		quadData = QuadData(mesh, mesh.gbasis, EntityType.Element, QuadOrder)

	# 		quad_pts = quadData.quad_pts
	# 		quad_wts = quadData.quad_wts
	# 		nq = quad_pts.shape[0]

	# 		#PhiData = BasisData(basis,order,mesh)
	# 		basis.eval_basis(quad_pts, Get_Phi=True)
	# 		xphys = np.zeros([nq, mesh.Dim])
		
	# 	else:
	# 		quad_pts, nq = basis.equidistant_nodes(order, quad_pts)
	# 		#nb = nq

	# 	for elem in range(mesh.nElem):

	# 		xphys, GeomPhiData = mesh_defs.ref_to_phys(mesh, elem, GeomPhiData, quad_pts, xphys)

	# 		f = EqnSet.CallFunction(EqnSet.IC, x=xphys, t=self.Time)
	# 		f.shape = nq,ns

	# 		if not InterpolateIC:

	# 			djac,_,_ = basis_tools.element_jacobian(mesh,elem,quad_pts,get_djac=True)

	# 			iMM = iMM_all[elem]

	# 			# rhs *= 0.
	# 			# for n in range(nn):
	# 			# 	for iq in range(nq):
	# 			# 		rhs[n,:] += f[iq,:]*PhiData.Phi[iq,n]*quad_wts[iq]*JData.djac[iq*(JData.nq != 1)]

	# 			rhs[:] = np.matmul(basis.basis_val.transpose(), f*quad_wts*djac) # [nb, ns]

	# 			U[elem,:,:] = np.matmul(iMM,rhs)
	# 		else:
	# 			U[elem] = f

	@abstractmethod
	def calculate_residual_elem(self, elem, Up, ER):
		pass

	@abstractmethod
	def calculate_residual_iface(self, iiface, UpL, UpR, RL, RR):
		pass

	@abstractmethod
	def calculate_residual_bface(self, ibfgrp, ibface, U, R):
		pass


class ElemOperators(object):
	def __init__(self):
		self.quad_pts = None
		self.quad_wts = None
		self.basis_val = None 
		self.basis_grad = None 
		self.basis_pgrad_elems = None 
		self.gbasis_val = None 
		self.jac_elems = None 
		self.ijac_elems = None 
		self.djac_elems = None 
		self.x_elems = None
		self.Uq = None 
		self.Fq = None 
		self.Sq = None 
		self.iMM_elems = np.zeros(0)

	def get_gaussian_quadrature(self, mesh, EqnSet, basis, order):

		# QuadOrder, _ = get_gaussian_quadrature_elem(mesh, mesh.gbasis, order, EqnSet, None)
		gbasis = mesh.gbasis
		quad_order = gbasis.get_quadrature(mesh, order, physics = EqnSet)
		# quadData = QuadData(mesh, basis, EntityType.Element, QuadOrder)
		basis.get_quad_data(quad_order, 0)
		self.quad_pts = basis.quad_pts
		self.quad_wts = basis.quad_wts

	def get_basis_and_geom_data(self, mesh, basis, order):
		# separate these later

		# Unpack
		dim = mesh.Dim 
		nElem = mesh.nElem 
		quad_pts = self.quad_pts 
		nq = quad_pts.shape[0]
		nb = basis.nb

		# Allocate
		self.jac_elems = np.zeros([nElem,nq,dim,dim])
		self.ijac_elems = np.zeros([nElem,nq,dim,dim])
		self.djac_elems = np.zeros([nElem,nq,1])
		self.x_elems = np.zeros([nElem,nq,dim])
		self.basis_pgrad_elems = np.zeros([nElem,nq,nb,dim])

		GeomPhiData = None

		# basis data
		basis.eval_basis(self.quad_pts, Get_Phi=True, Get_GPhi=True)

		self.basis_val = basis.basis_val 
		self.basis_grad = basis.basis_grad 

		for elem in range(mesh.nElem):
			# Jacobian
			djac, jac, ijac = basis_tools.element_jacobian(mesh, elem, quad_pts, get_djac=True, get_jac=True, get_ijac=True)
			# Store
			self.jac_elems[elem] = jac
			self.ijac_elems[elem] = ijac
			self.djac_elems[elem] = djac

			# Physical coordinates of quadrature points
			x, GeomPhiData = mesh_defs.ref_to_phys(mesh, elem, GeomPhiData, quad_pts)
			# Store
			self.x_elems[elem] = x
			# Physical gradient
			basis.eval_basis(quad_pts, Get_gPhi=True, ijac=ijac) # gPhi is [nq,nb,dim]
			self.basis_pgrad_elems[elem] = basis.basis_pgrad

	def alloc_other_arrays(self, EqnSet, basis, order):
		quad_pts = self.quad_pts 
		nq = quad_pts.shape[0]
		nb = basis.nb
		ns = EqnSet.StateRank
		dim = EqnSet.dim

		self.Uq = np.zeros([nq, ns]) 
		self.Fq = np.zeros([nq, ns, dim])
		self.Sq = np.zeros([nq, ns])  

	def compute_operators(self, mesh, EqnSet, basis, order):
		self.get_gaussian_quadrature(mesh, EqnSet, basis, order)
		self.get_basis_and_geom_data(mesh, basis, order)
		self.alloc_other_arrays(EqnSet, basis, order)
		self.iMM_elems = basis_tools.get_inv_mass_matrices(mesh, EqnSet, basis)


class IFaceOperators(ElemOperators):
	def __init__(self):
		self.quad_pts = None
		self.quad_wts = None
		self.faces_to_basisL = None
		self.faces_to_basisR = None
		self.normals_ifaces = None
		self.UqL = None 
		self.UqR = None 
		self.Fq = None 

	def get_gaussian_quadrature(self, mesh, EqnSet, basis, order):

		gbasis = mesh.gbasis
		quad_order = gbasis.get_face_quadrature(mesh,order,physics = EqnSet)
		basis.get_face_quad_data(quad_order, 0)
		# quadData = QuadData(mesh, basis, EntityType.IFace, QuadOrder)
		self.quad_pts = basis.quad_pts
		self.quad_wts = basis.quad_wts

	def get_basis_and_geom_data(self, mesh, basis, order):
		# separate these later

		# Unpack
		dim = mesh.Dim
		quad_pts = self.quad_pts 
		nq = quad_pts.shape[0]
		nb = basis.nb
		nFacePerElem = mesh.nFacePerElem

		# Allocate
		self.faces_to_basisL = np.zeros([nFacePerElem,nq,nb])
		self.faces_to_basisR = np.zeros([nFacePerElem,nq,nb])
		self.normals_ifaces = np.zeros([mesh.nIFace,nq,dim])

		# basis data
		#PhiData = BasisData(basis, order, mesh)

		for f in range(nFacePerElem):
			# Left
			_ = basis.eval_basis_on_face(mesh, f, quad_pts, None, Get_Phi=True)
			self.faces_to_basisL[f] = basis.basis_val
			# Right
			_ = basis.eval_basis_on_face(mesh, f, quad_pts[::-1], None, Get_Phi=True)
			self.faces_to_basisR[f] = basis.basis_val

		i = 0
		for IFace in mesh.IFaces:
			# Normals
			nvec = mesh_defs.iface_normal(mesh, IFace, quad_pts)
			self.normals_ifaces[i] = nvec
			i += 1

	def alloc_other_arrays(self, EqnSet, basis, order):
		quad_pts = self.quad_pts 
		nq = quad_pts.shape[0]
		ns = EqnSet.StateRank

		self.UqL = np.zeros([nq, ns])
		self.UqR = np.zeros([nq, ns])
		self.Fq = np.zeros([nq, ns])

	def compute_operators(self, mesh, EqnSet, basis, order):
		self.get_gaussian_quadrature(mesh, EqnSet, basis, order)
		self.get_basis_and_geom_data(mesh, basis, order)
		self.alloc_other_arrays(EqnSet, basis, order)


class BFaceOperators(IFaceOperators):
	def __init__(self):
		self.quad_pts = None
		self.quad_wts = None
		self.faces_to_basis = None
		self.faces_to_xref = None
		self.normals_bfgroups = None
		self.x = None
		self.UqI = None 
		self.UqB = None 
		self.Fq = None 

	def get_basis_and_geom_data(self, mesh, basis, order):
		# separate these later

		# Unpack
		dim = mesh.Dim
		quad_pts = self.quad_pts 
		nq = quad_pts.shape[0]
		nb = basis.nb
		nFacePerElem = mesh.nFacePerElem

		# Allocate
		self.faces_to_basis = np.zeros([nFacePerElem,nq,nb])
		self.faces_to_xref = np.zeros([nFacePerElem,nq,dim])
		self.normals_bfgroups = []
		self.x_bfgroups = []

		# basis data
		# PhiData = BasisData(basis, order, mesh)
		GeomPhiData = None

		for f in range(nFacePerElem):
			# Left
			self.faces_to_xref[f] = xref = basis.eval_basis_on_face(mesh, f, quad_pts, None, Get_Phi=True)
			self.faces_to_basis[f] = basis.basis_val

		i = 0
		for BFG in mesh.BFaceGroups.values():
			self.normals_bfgroups.append(np.zeros([BFG.nBFace,nq,dim]))
			self.x_bfgroups.append(np.zeros([BFG.nBFace,nq,dim]))
			normal_bfgroup = self.normals_bfgroups[i]
			x_bfgroup = self.x_bfgroups[i]
			j = 0
			for BFace in BFG.BFaces:
				# Normals
				nvec = mesh_defs.bface_normal(mesh, BFace, quad_pts)
				normal_bfgroup[j] = nvec

				# Physical coordinates of quadrature points
				x, GeomPhiData = mesh_defs.ref_to_phys(mesh, BFace.Elem, GeomPhiData, self.faces_to_xref[BFace.face], None)
				# Store
				x_bfgroup[j] = x

				# Increment
				j += 1
			i += 1

	def alloc_other_arrays(self, EqnSet, basis, order):
		quad_pts = self.quad_pts 
		nq = quad_pts.shape[0]
		ns = EqnSet.StateRank

		self.UqI = np.zeros([nq, ns])
		self.UqB = np.zeros([nq, ns])
		self.Fq = np.zeros([nq, ns])

	def compute_operators(self, mesh, EqnSet, basis, order):
		self.get_gaussian_quadrature(mesh, EqnSet, basis, order)
		self.get_basis_and_geom_data(mesh, basis, order)
		self.alloc_other_arrays(EqnSet, basis, order)


class DG(SolverBase):
	'''
	Class: DG
	--------------------------------------------------------------------------
	Discontinuous Galerkin method designed to solve a given set of PDEs

	ATTRIBUTES:
		Params: list of parameters for the solver
		EqnSet: solver object (current implementation supports Scalar and Euler equations)
		mesh: mesh object
		DataSet: location to store generic data
		Time: current time in the simulation
		nTimeStep: number of time steps
	'''
	def check_solver_params(self):
		'''
		Method: check_solver_params
		--------------------------------------------------------------------------
		Checks the validity of the solver parameters

		'''
		Params = self.Params
		mesh = self.mesh
		EqnSet = self.EqnSet
		### Check interp basis validity
		if BasisType[Params["InterpBasis"]] == BasisType.LagrangeEqSeg or BasisType[Params["InterpBasis"]] == BasisType.LegendreSeg:
		    if mesh.Dim != 1:
		        raise errors.IncompatibleError
		else:
		    if mesh.Dim != 2:
		        raise errors.IncompatibleError

		### Check uniform mesh
		# if Params["UniformMesh"] is True:
		#     ''' 
		#     Check that element volumes are the same
		#     Note that this is a necessary but not sufficient requirement
		#     '''
		#     TotVol, ElemVols = MeshTools.element_volumes(mesh)
		#     if (ElemVols.Max - ElemVols.Min)/TotVol > 1.e-8:
		#         raise ValueError

		# ### Check linear geometric mapping
		# if Params["LinearGeomMapping"] is True:
		# 	if mesh.QOrder != 1:
		# 	    raise errors.IncompatibleError
		# 	if mesh.QBasis == BasisType.LagrangeEqQuad \
		# 	    and Params["UniformMesh"] is False:
		# 	    raise errors.IncompatibleError

		### Check limiter ###
		if Params["ApplyLimiter"] is 'ScalarPositivityPreserving' \
			and EqnSet.StateRank > 1:
				raise IncompatibleError
		if Params["ApplyLimiter"] is 'PositivityPreserving' \
			and EqnSet.StateRank == 1:
				raise IncompatibleError


	def precompute_matrix_operators(self):
		mesh = self.mesh 
		EqnSet = self.EqnSet
		basis = self.basis

		self.elem_operators = ElemOperators()
		self.elem_operators.compute_operators(mesh, EqnSet, basis, EqnSet.order)
		self.iface_operators = IFaceOperators()
		self.iface_operators.compute_operators(mesh, EqnSet, basis, EqnSet.order)
		self.bface_operators = BFaceOperators()
		self.bface_operators.compute_operators(mesh, EqnSet, basis, EqnSet.order)

	def apply_limiter(self, U):
		'''
		Method: apply_limiter
		-------------------------
		Applies the limiter to the solution array, U.
		
		INPUTS:
			U: solution array

		OUTPUTS:
			U: limited solution array

		Notes: See Limiter.py for details
		'''
		if self.Limiter is not None:
			self.Limiter.limit_solution(self, U)


	def calculate_residual_elem(self, elem, Up, ER):
		'''
		Method: calculate_residual_elem
		---------------------------------
		Calculates the volume integral for a specified element
		
		INPUTS:
			elem: element index
			U: solution array
			
		OUTPUTS:
			ER: calculated residiual array (for volume integral of specified element)
		'''
		EqnSet = self.EqnSet
		ns = EqnSet.StateRank
		dim = EqnSet.dim
		elem_ops = self.elem_operators
		basis_val = elem_ops.basis_val
		quad_wts = elem_ops.quad_wts

		x_elems = elem_ops.x_elems
		nq = quad_wts.shape[0]
		x = x_elems[elem]

		# interpolate state and gradient at quad points
		Uq = np.matmul(basis_val, Up)

		'''
		Evaluate the inviscid flux integral.
		'''
		Fq = EqnSet.ConvFluxInterior(Uq, F=None) # [nq,ns,dim]
		ER += solver_tools.calculate_inviscid_flux_volume_integral(self, elem_ops, elem, Fq)

		'''
		Evaluate the source term integral
		'''
		Sq = elem_ops.Sq
		Sq[:] = 0. # SourceState is an additive function so source needs to be initialized to zero for each time step
		Sq = EqnSet.SourceState(nq, x, self.Time, Uq, Sq) # [nq,ns]

		ER += solver_tools.calculate_source_term_integral(elem_ops, elem, Sq)

		if elem == echeck:
			code.interact(local=locals())

		return ER

	def calculate_residual_elems(self, U, R):
		'''
		Method: calculate_residual_elems
		---------------------------------
		Calculates the volume integral across the entire domain
		
		INPUTS:
			U: solution array
			
		OUTPUTS:
			R: calculated residiual array
		'''
		mesh = self.mesh
		EqnSet = self.EqnSet

		for elem in range(mesh.nElem):
			R[elem] = self.calculate_residual_elem(elem, U[elem], R[elem])


	def calculate_residual_iface(self, iiface, UpL, UpR, RL, RR):
		'''
		Method: calculate_residual_iface
		---------------------------------
		Calculates the boundary integral for the internal faces
		
		INPUTS:
			iiface: internal face index
			UL: solution array from left neighboring element
			UR: solution array from right neighboring element
			
		OUTPUTS:
			RL: calculated residual array (left neighboring element contribution)
			RR: calculated residual array (right neighboring element contribution)
		'''
		mesh = self.mesh
		EqnSet = self.EqnSet
		IFace = mesh.IFaces[iiface]
		elemL = IFace.ElemL
		elemR = IFace.ElemR
		faceL = IFace.faceL
		faceR = IFace.faceR

		iface_ops = self.iface_operators
		quad_pts = iface_ops.quad_pts
		quad_wts = iface_ops.quad_wts
		faces_to_basisL = iface_ops.faces_to_basisL
		faces_to_basisR = iface_ops.faces_to_basisR
		normals_ifaces = iface_ops.normals_ifaces
		UqL = iface_ops.UqL
		UqR = iface_ops.UqR
		Fq = iface_ops.Fq

		nq = quad_wts.shape[0]
		basis_valL = faces_to_basisL[faceL]
		basis_valR = faces_to_basisR[faceR]

		# interpolate state and gradient at quad points
		UqL = np.matmul(basis_valL, UpL)
		UqR = np.matmul(basis_valR, UpR)

		normals = normals_ifaces[iiface]

		Fq = EqnSet.ConvFluxNumerical(UqL, UqR, normals) # [nq,ns]

		RL -= solver_tools.calculate_inviscid_flux_boundary_integral(basis_valL, quad_wts, Fq)
		RR += solver_tools.calculate_inviscid_flux_boundary_integral(basis_valR, quad_wts, Fq)

		if elemL == echeck or elemR == echeck:
			if elemL == echeck: print("Left!")
			else: print("Right!")
			code.interact(local=locals())

		return RL, RR



	def calculate_residual_ifaces(self, U, R):
		'''
		Method: calculate_residual_ifaces
		-----------------------------------
		Calculates the boundary integral for all internal faces
		
		INPUTS:
			U: solution array
			
		OUTPUTS:
			R: calculated residual array (includes all face contributions)
		'''
		mesh = self.mesh
		EqnSet = self.EqnSet

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

			RL, RR = self.calculate_residual_iface(iiface, UL, UR, RL, RR)

	def calculate_residual_bface(self, BFG, ibface, U, R):
		'''
		Method: calculate_residual_bface
		---------------------------------
		Calculates the boundary integral for the boundary faces
		
		INPUTS:
			ibfgrp: index of boundary face groups (groups indicate different boundary conditions)
			ibface: boundary face index
			U: solution array from internal cell
			
		OUTPUTS:
			R: calculated residual array from boundary face
		'''
		mesh = self.mesh
		EqnSet = self.EqnSet
		# BFG = mesh.BFaceGroups[ibfgrp]
		ibfgrp = BFG.number
		BFace = BFG.BFaces[ibface]
		elem = BFace.Elem
		face = BFace.face

		bface_ops = self.bface_operators
		quad_pts = bface_ops.quad_pts
		quad_wts = bface_ops.quad_wts
		faces_to_basis = bface_ops.faces_to_basis
		normals_bfgroups = bface_ops.normals_bfgroups
		x_bfgroups = bface_ops.x_bfgroups
		UqI = bface_ops.UqI
		UqB = bface_ops.UqB
		Fq = bface_ops.Fq

		nq = quad_wts.shape[0]
		basis_val = faces_to_basis[face]

		# interpolate state and gradient at quad points
		UqI = np.matmul(basis_val, U)

		normals = normals_bfgroups[ibfgrp][ibface]
		x = x_bfgroups[ibfgrp][ibface]

		# Get boundary state
		BC = EqnSet.BCs[BFG.name]

		Fq = BC.get_boundary_flux(EqnSet, x, self.Time, normals, UqI)

		R -= solver_tools.calculate_inviscid_flux_boundary_integral(basis_val, quad_wts, Fq)

		if elem == echeck:
			code.interact(local=locals())

		return R


	def calculate_residual_bfaces(self, U, R):
		'''
		Method: calculate_residual_bfaces
		-----------------------------------
		Calculates the boundary integral for all the boundary faces
		
		INPUTS:
			U: solution array from internal cell
			
		OUTPUTS:
			R: calculated residual array from boundary face
		'''
		mesh = self.mesh
		EqnSet = self.EqnSet

		# for ibfgrp in range(mesh.nBFaceGroup):
		# 	BFG = mesh.BFaceGroups[ibfgrp]

		for BFG in mesh.BFaceGroups.values():

			for ibface in range(BFG.nBFace):
				BFace = BFG.BFaces[ibface]
				elem = BFace.Elem
				face = BFace.face

				R[elem] = self.calculate_residual_bface(BFG, ibface, U[elem], R[elem])


	def calculate_residual(self, U, R):
		'''
		Method: calculate_residual
		-----------------------------------
		Calculates the boundary + volume integral for the DG formulation
		
		INPUTS:
			U: solution array
			
		OUTPUTS:
			R: residual array
		'''

		mesh = self.mesh
		EqnSet = self.EqnSet

		if R is None:
			# R = ArrayList(SimilarArray=U)
			R = np.copy(U)
		# Initialize residual to zero
		# R.SetUniformValue(0.)
		R[:] = 0.

		self.calculate_residual_bfaces(U, R)
		self.calculate_residual_elems(U, R)
		self.calculate_residual_ifaces(U, R)

		return R


	def apply_time_scheme(self, fhistory=None):
		'''
		Method: apply_time_scheme
		-----------------------------
		Applies the specified time scheme to update the solution

		'''
		EqnSet = self.EqnSet
		mesh = self.mesh
		Order = self.Params["InterpOrder"]
		Stepper = self.Stepper
		Time = self.Time

		# Parameters
		TrackOutput = self.Params["TrackOutput"]
		WriteInterval = self.Params["WriteInterval"]
		if WriteInterval == -1:
			WriteInterval = np.NAN
		WriteFinalSolution = self.Params["WriteFinalSolution"]
		WriteInitialSolution = self.Params["WriteInitialSolution"]

		if WriteInitialSolution:
			ReadWriteDataFiles.write_data_file(self, 0)

		t0 = time.time()
		iwrite = 1
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
				output,_ = post_defs.L2_error(mesh,EqnSet,Time,"Entropy",False)
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

			# Write data file
			if (iStep + 1) % WriteInterval == 0:
				ReadWriteDataFiles.write_data_file(self, iwrite)
				iwrite += 1


		t1 = time.time()
		print("Wall clock time = %g seconds" % (t1-t0))

		if WriteFinalSolution:
			ReadWriteDataFiles.write_data_file(self, -1)

	def solve(self):
		'''
		Method: solve
		-----------------------------
		Performs the main solve of the DG method. Initializes the temporal loop. 

		'''
		mesh = self.mesh
		EqnSet = self.EqnSet
		basis = self.basis

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

		if not OrderSequencing:
			if len(InterpOrders) != 1:
				raise ValueError

		''' Loop through Orders '''
		Time = self.Time
		for iOrder in range(nOrder):
			Order = InterpOrders[iOrder]
			''' Compute time step '''
			if nTimeSteps[iOrder] != 0:
				self.Stepper.dt = (EndTimes[iOrder]-Time)/nTimeSteps[iOrder]
			self.nTimeStep = nTimeSteps[iOrder]

			''' After first iteration, project solution to next Order '''
			if iOrder > 0:
				# Clear DataSet
				delattr(self, "DataSet")
				self.DataSet = GenericData()
				# Increment Order
				Order_old = EqnSet.order
				EqnSet.order = Order
				# Project
				solver_tools.project_state_to_new_basis(self, mesh, EqnSet, basis, Order_old)

				basis.order = Order
				basis.nb = basis.get_num_basis_coeff(Order)				

				self.precompute_matrix_operators()

			''' Apply time scheme '''
			self.apply_time_scheme(fhistory)

			Time = EndTimes[iOrder]


		if WriteTimeHistory:
			fhistory.close()
