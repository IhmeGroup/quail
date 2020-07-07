import code
import numpy as np
from scipy.linalg import solve_sylvester

from data import GenericData
import errors

from general import ModalOrNodal, StepperType

import meshing.meshbase as mesh_defs
import meshing.tools as mesh_tools

import numerics.basis.tools as basis_tools
import numerics.basis.ader_tools as basis_st_tools

import numerics.limiting.tools as limiter_tools

import numerics.timestepping.stepper as stepper

global echeck
echeck = -1

import solver.ader_tools as solver_tools
import solver.tools as dg_tools
import solver.base as base
import solver.DG as DG


class ElemOperatorsADER(DG.ElemOperators):

	def get_basis_and_geom_data(self, mesh, basis, order):
		# separate these later

		dim = mesh.Dim 
		quad_pts = self.quad_pts 
		nElem = mesh.nElem 
		nq = quad_pts.shape[0]
		nb = basis.nb

		self.jac_elems = np.zeros([nElem,nq,dim,dim])
		self.ijac_elems = np.zeros([nElem,nq,dim,dim])
		self.djac_elems = np.zeros([nElem,nq,1])
		self.x_elems = np.zeros([nElem,nq,dim])
		self.basis_pgrad_elems = np.zeros([nElem,nq,nb,dim])

		GeomPhiData = None

		basis.eval_basis(quad_pts, Get_Phi=True, Get_GPhi=True)

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
			# basis.eval_basis(quad_pts, Get_gPhi=True, ijac=ijac) # gPhi is [nq,nb,dim]
			# self.basis_pgrad_elems[elem] = basis.basis_pgrad

class IFaceOperatorsADER(DG.IFaceOperators):

	def get_gaussian_quadrature(self, mesh, EqnSet, basis, order):
		
		gbasis = mesh.gbasis
		quad_order = gbasis.get_face_quadrature(mesh, order, physics=EqnSet)
		self.quad_pts, self.quad_wts = basis.get_face_quad_data(quad_order)

		# self.quad_pts = basis.quad_pts
		# self.quad_wts = basis.quad_wts

	def get_basis_and_geom_data(self, mesh, basis, order):
		# separate these later

		# Unpack
		dim = mesh.Dim
		quad_pts = self.quad_pts 
		nq = quad_pts.shape[0]
		nb = basis.get_num_basis_coeff(order)
		nFacePerElem = mesh.nFacePerElem + 2

		# Allocate
		self.faces_to_basisL = np.zeros([nFacePerElem,nq,nb])
		self.faces_to_basisR = np.zeros([nFacePerElem,nq,nb])
		self.normals_ifaces = np.zeros([mesh.nIFace,nq,dim])

		for f in range(nFacePerElem):
			# Left
			#eval_basis_on_face_ader(mesh, basis_st, face_stL, quad_pts_st, xelemLPhi, Get_Phi=True)
			_ = basis.eval_basis_on_face(mesh, f, quad_pts, None, basis, Get_Phi=True)
			self.faces_to_basisL[f] = basis.basis_val
			# Right
			_ = basis.eval_basis_on_face(mesh, f, quad_pts[::-1], None, basis, Get_Phi=True)
			self.faces_to_basisR[f] = basis.basis_val

		i = 0
		for IFace in mesh.IFaces:
			# Normals
			nvec = mesh_defs.iface_normal(mesh, IFace, quad_pts)
			self.normals_ifaces[i] = nvec
			i += 1

class BFaceOperatorsADER(IFaceOperatorsADER):

	def get_basis_and_geom_data(self, mesh, basis, order):
		# separate these later

		# Unpack
		dim = mesh.Dim
		quad_pts = self.quad_pts 
		nq = quad_pts.shape[0]
		nb = basis.get_num_basis_coeff(order)
		nFacePerElem = mesh.nFacePerElem + 2

		# Allocate
		self.faces_to_basis = np.zeros([nFacePerElem,nq,nb])
		self.faces_to_xref = np.zeros([nFacePerElem,nq,dim+1])
		self.normals_bfgroups = []
		self.x_bfgroups = []

		GeomPhiData = None

		for f in range(nFacePerElem):
			# Left
			self.faces_to_xref[f] = xref = basis.eval_basis_on_face(mesh, f, quad_pts, None, basis, Get_Phi=True)
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
		ns = EqnSet.NUM_STATE_VARS

		self.UqI = np.zeros([nq, ns])
		self.UqB = np.zeros([nq, ns])
		self.Fq = np.zeros([nq, ns])


class ADEROperators(object):
	def __init__(self):
		self.MM = None
		self.iMM = None
		self.iMM_elems = None
		self.K = None
		self.iK = None 
		self.FTL = None 
		self.FTR = None
		self.SMT = None
		self.SMS_elems = None	
		self.jac_elems = None 
		self.ijac_elems = None 
		self.djac_elems = None 
		self.x_elems = None
		self.vol_elems = None


	def calc_ader_matrices(self, mesh, basis, basis_st, dt, order):

		dim = mesh.Dim
		nb = basis_st.nb
		SMS_elems = np.zeros([mesh.nElem,nb,nb,dim])
		iMM_elems = np.zeros([mesh.nElem,nb,nb])
		# Get flux matrices in time
		FTL = basis_st_tools.get_temporal_flux_ader(mesh, basis_st, basis_st, order, elem=0, PhysicalSpace=False)
		FTR = basis_st_tools.get_temporal_flux_ader(mesh, basis_st, basis, order, elem=0, PhysicalSpace=False)

		# Get stiffness matrix in time
		SMT = basis_st_tools.get_stiffness_matrix_ader(mesh, basis, basis_st, order, dt, elem=0, gradDir=1,PhysicalSpace = False)

		# Get stiffness matrix in space
		for elem in range(mesh.nElem):
			SMS = basis_st_tools.get_stiffness_matrix_ader(mesh, basis, basis_st, order, dt, elem, gradDir=0,PhysicalSpace = True)
			SMS_elems[elem,:,:,0] = SMS.transpose()

			iMM =  basis_st_tools.get_elem_inv_mass_matrix_ader(mesh, basis_st, order, elem, PhysicalSpace=True)
			iMM_elems[elem] = iMM

		iMM =  basis_st_tools.get_elem_inv_mass_matrix_ader(mesh, basis_st, order, elem, PhysicalSpace=False)

		# Get mass matrix in space-time
		MM =  basis_st_tools.get_elem_mass_matrix_ader(mesh, basis_st, order, elem=-1, PhysicalSpace=False)
		# iMM =  get_elem_inv_mass_matrix_ader(mesh, basis_st, order, elem=-1, PhysicalSpace=False)

		_, ElemVols = mesh_tools.element_volumes(mesh)

		self.vol_elems = ElemVols
		self.FTL = FTL
		self.FTR = FTR
		self.SMT = SMT
		self.SMS_elems = SMS_elems
		self.MM = MM
		self.iMM = iMM
		self.iMM_elems = iMM_elems
		self.K = FTL - SMT
		self.iK = np.linalg.inv(self.K) 


	def get_geom_data(self, mesh, basis, order):

		# Unpack
		shape_name = basis.__class__.__bases__[1].__name__

		dim = mesh.Dim 
		nElem = mesh.nElem 
		nb = basis.nb
		xnode = None
		gbasis=mesh.gbasis
		xnode, nnode = gbasis.equidistant_nodes(order,xnode)

		# Allocate
		self.jac_elems = np.zeros([nElem,nb,dim,dim])
		self.ijac_elems = np.zeros([nElem,nb,dim,dim])
		self.djac_elems = np.zeros([nElem,nb,1])
		self.x_elems = np.zeros([nElem,nb,dim])

		GeomPhiData = None

		for elem in range(mesh.nElem):
			# Jacobian
			djac, jac, ijac = basis_tools.element_jacobian(mesh, elem, xnode, get_djac=True, get_jac=True, get_ijac=True)

			if shape_name is 'HexShape':
				self.jac_elems[elem] = np.tile(jac,(int(np.sqrt(nnode)),1,1))
				self.ijac_elems[elem] = np.tile(ijac,(int(np.sqrt(nnode)),1,1))
				self.djac_elems[elem] = np.tile(djac,(int(np.sqrt(nnode)),1))

				# Physical coordinates of nodal points
				x, GeomPhiData = mesh_defs.ref_to_phys(mesh, elem, GeomPhiData, xnode)
				# Store
				self.x_elems[elem] = np.tile(x,(int(np.sqrt(nnode)),1))

			elif shape_name is 'QuadShape':
				self.jac_elems[elem] = np.tile(jac,(nnode,1,1))
				self.ijac_elems[elem] = np.tile(ijac,(nnode,1,1))
				self.djac_elems[elem] = np.tile(djac,(nnode,1))

				# Physical coordinates of nodal points
				x, GeomPhiData = mesh_defs.ref_to_phys(mesh, elem, GeomPhiData, xnode)
				# Store
				self.x_elems[elem] = np.tile(x,(nnode,1))

	def compute_operators(self, mesh, EqnSet, basis, basis_st, dt, order):
		self.calc_ader_matrices(mesh, basis, basis_st, dt, order)
		self.get_geom_data(mesh, basis_st, order)


class ADERDG(base.SolverBase):
	'''
	Class: ADERDG
	--------------------------------------------------------------------------
	Use the ADER-DG method to solve a given set of PDEs
	'''
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

		ns = EqnSet.NUM_STATE_VARS

		TimeScheme = Params["TimeScheme"]
		if StepperType[TimeScheme] != StepperType.ADER:
			raise errors.IncompatibleError
		self.Stepper = stepper.ADER()

		# Set the basis functions for the solver
		BasisFunction  = Params["InterpBasis"]
		self.basis = basis_tools.set_basis(EqnSet.order, BasisFunction)
		self.basis_st = basis_st_tools.set_basis_spacetime(mesh, EqnSet.order, BasisFunction)

		# Set quadrature
		self.basis.set_elem_quadrature_type(Params["ElementQuadrature"])
		self.basis.set_face_quadrature_type(Params["FaceQuadrature"])
		self.basis_st.set_elem_quadrature_type(Params["ElementQuadrature"])
		self.basis_st.set_face_quadrature_type(Params["FaceQuadrature"])
		mesh.gbasis.set_elem_quadrature_type(Params["ElementQuadrature"])
		mesh.gbasis.set_face_quadrature_type(Params["FaceQuadrature"])

		# Allocate array for predictor step in ADER-Scheme
		EqnSet.Up = np.zeros([self.mesh.nElem, self.basis_st.get_num_basis_coeff(EqnSet.order), EqnSet.NUM_STATE_VARS])

		# Set predictor function
		SourceTreatment = Params["SourceTreatment"]
		self.calculate_predictor_elem = solver_tools.set_source_treatment(ns, SourceTreatment)

		# Limiter
		limiterType = Params["ApplyLimiter"]
		self.Limiter = limiter_tools.set_limiter(limiterType, EqnSet.PHYSICS_TYPE)

		# Check validity of parameters
		self.check_compatibility()

		# Precompute operators
		self.precompute_matrix_operators()
		if self.Limiter is not None:
			self.Limiter.precompute_operators(self)

		# Initialize state
		if Params["RestartFile"] is None:
			self.init_state_from_fcn()
	def __repr__(self):
		return '{self.__class__.__name__}(Physics: {self.EqnSet},\n   Basis: {self.basis},\n   Basis_st: {self.basis_st},\n   Stepper: {self.Stepper})'.format(self=self)
	def check_compatibility(self):
		'''
		Method: check_compatibility
		--------------------------------------------------------------------------
		Checks the validity of the solver parameters
		'''
		super().check_compatibility()

		basis = self.basis
		Params = self.Params

		if Params["InterpolateFlux"] and basis.MODAL_OR_NODAL != ModalOrNodal.Nodal:
			raise errors.IncompatibleError

		# Params = self.Params
		# mesh = self.mesh
		# EqnSet = self.EqnSet
		# ### Check interp basis validity
		# if BasisType[Params["InterpBasis"]] == BasisType.LagrangeEqSeg or BasisType[Params["InterpBasis"]] == BasisType.LegendreSeg:
		#     if mesh.Dim != 1:
		#         raise errors.IncompatibleError
		# else:
		#     if mesh.Dim != 2:
		#         raise errors.IncompatibleError

		# ### Check limiter ###
		# if Params["ApplyLimiter"] is 'PositivityPreserving' \
		# 	and EqnSet.NUM_STATE_VARS == 1:
		# 		raise IncompatibleError
  

	def precompute_matrix_operators(self):
		mesh = self.mesh 
		EqnSet = self.EqnSet

		basis = self.basis
		basis_st = self.basis_st
		
		dt = self.Params['EndTime']/self.Params['nTimeStep']


		self.elem_operators = DG.ElemOperators()
		self.elem_operators.compute_operators(mesh, EqnSet, basis, EqnSet.order)
		self.iface_operators = DG.IFaceOperators()
		self.iface_operators.compute_operators(mesh, EqnSet, basis, EqnSet.order)
		self.bface_operators = DG.BFaceOperators()
		self.bface_operators.compute_operators(mesh, EqnSet, basis, EqnSet.order)

		# Calculate ADER specific space-time operators
		self.elem_operators_st = ElemOperatorsADER()
		self.elem_operators_st.compute_operators(mesh, EqnSet, basis_st, EqnSet.order)
		self.iface_operators_st = IFaceOperatorsADER()
		self.iface_operators_st.compute_operators(mesh, EqnSet, basis_st, EqnSet.order)
		self.bface_operators_st = BFaceOperatorsADER()
		self.bface_operators_st.compute_operators(mesh, EqnSet, basis_st, EqnSet.order)

		self.ader_operators = ADEROperators()
		self.ader_operators.compute_operators(mesh, EqnSet, basis, basis_st, dt, EqnSet.order)

	def calculate_predictor_step(self, dt, W, Up):
		'''
		Method: calculate_predictor_step
		-------------------------------------------
		Calls the predictor step for each element

		INPUTS:
			dt: time step 
			W: previous time step solution in space only

		OUTPUTS:
			Up: predicted solution in space-time
		'''
		mesh = self.mesh
		EqnSet = self.EqnSet

		for elem in range(mesh.nElem):
			Up[elem] = self.calculate_predictor_elem(self, elem, dt, W[elem], Up[elem])

		return Up

	# def calculate_predictor_elem(self, elem, dt, Wp, Up):
	# 	'''
	# 	Method: calculate_predictor_elem
	# 	-------------------------------------------
	# 	Calculates the predicted solution state for the ADER-DG method using a nonlinear solve of the
	# 	weak form of the DG discretization in time.

	# 	INPUTS:
	# 		elem: element index
	# 		dt: time step 
	# 		Wp: previous time step solution in space only

	# 	OUTPUTS:
	# 		Up: predicted solution in space-time
	# 	'''
	# 	EqnSet = self.EqnSet
	# 	ns = EqnSet.NUM_STATE_VARS
	# 	mesh = self.mesh

	# 	basis = self.basis
	# 	basis_st = self.basis_st

	# 	order = EqnSet.order
		
	# 	elem_ops = self.elem_operators
	# 	ader_ops = self.ader_operators
		
	# 	quad_wts = elem_ops.quad_wts
	# 	basis_val = elem_ops.basis_val 
	# 	djac_elems = elem_ops.djac_elems 
		
	# 	djac = djac_elems[elem]
	# 	# _, ElemVols = MeshTools.element_volumes(mesh, self)

	# 	FTR = ader_ops.FTR
	# 	MM = ader_ops.MM
	# 	SMS = ader_ops.SMS_elems[elem]
	# 	# K = ader_ops.K
	# 	iK = ader_ops.iK
	# 	# W_bar = np.zeros([1,ns])
	# 	# Wq = np.matmul(basis_val, Wp)
	# 	# vol = ElemVols[elem]

	# 	# W_bar[:] = np.matmul(Wq.transpose(),quad_wts*djac).T/vol

	# 	# Wh = np.average(W)

	# 	# def F(u):
	# 	# 	S = 0.
	# 	# 	S = EqnSet.SourceState(1, 0., 0., u, S)
	# 	# 	F = u - S[0,0] - W_bar[0,0]
	# 	# 	return F

	# 	#U_bar = fsolve(F, W_bar)
	# 	# nu= -100000.
	# 	# dsdu = nu
	# 	# Up[:] = U_bar
	# 	#code.interact(local=locals())
	# 	#dsdu = (1./nu)*(2.*U_bar-3.*U_bar**2 - 0.5 +2.*U_bar*0.5)
	# 	#dsdu = (1./nu)*(2.*Up-3.*Up**2 - 0.5 +2.*Up*0.5)
	# 	#code.interact(local=locals())

	# 	### Hacky implementation of implicit source term
	# 	# Kp = K-MM*dt*dsdu

	# 	# iK = np.linalg.inv(Kp)

	# 	srcpoly = self.source_coefficients(elem, dt, order, basis_st, Up)
	# 	flux = self.flux_coefficients(elem, dt, order, basis_st, Up)
	# 	ntest = 10
	# 	for i in range(ntest):

	# 		# Up_new = np.matmul(iK,(np.matmul(MM,srcpoly)-np.matmul(SMS,fluxpoly)+np.matmul(FTR,Wp)-np.matmul(MM,dt*dsdu*Up)))
	# 		Up_new = np.matmul(iK,(np.matmul(MM,srcpoly)-np.einsum('ijk,jlk->il',SMS,flux)+np.matmul(FTR,Wp)))
	# 		err = Up_new - Up

	# 		if np.amax(np.abs(err))<1e-10:
	# 			Up = Up_new
	# 			break

	# 		Up = Up_new
			
	# 		srcpoly = self.source_coefficients(elem, dt, order, basis_st, Up)
	# 		flux = self.flux_coefficients(elem, dt, order, basis_st, Up)

	# 	return Up

	def calculate_residual_elem(self, elem, Up, ER):
		'''
		Method: calculate_residual_elem
		-------------------------------------------
		Calculates the residual from the volume integral for each element

		INPUTS:
			elem: element index
			U: solution state

		OUTPUTS:
		ER: calculated residiual array (for volume integral of specified element)
		'''
		EqnSet = self.EqnSet
		mesh = self.mesh
		ns = EqnSet.NUM_STATE_VARS
		dim = EqnSet.dim

		elem_ops = self.elem_operators
		elem_ops_st = self.elem_operators_st

		quad_wts = elem_ops.quad_wts
		quad_wts_st = elem_ops_st.quad_wts
		quad_pts_st = elem_ops_st.quad_pts

		basis_val_st = elem_ops_st.basis_val
		x_elems = elem_ops.x_elems
		x = x_elems[elem]

		nq = quad_wts.shape[0]
		nq_st = quad_wts_st.shape[0]

		# interpolate state and gradient at quad points
		Uq = np.matmul(basis_val_st, Up)

		if self.Params["ConvFluxSwitch"] == True:
			'''
			Evaluate the inviscid flux integral.
			'''
			Fq = EqnSet.ConvFluxInterior(Uq) # [nq,sr,dim]
			ER += solver_tools.calculate_inviscid_flux_volume_integral(self, elem_ops, elem_ops_st, elem, Fq)
		
		if self.Params["SourceSwitch"] == True:
			'''
			Evaluate the source term integral
			'''
			t = np.zeros([nq_st,dim])
			TimePhiData = None
			t, TimePhiData = solver_tools.ref_to_phys_time(mesh, elem, self.Time, self.Stepper.dt, TimePhiData, quad_pts_st, t, None)
			
			Sq = elem_ops_st.Sq
			Sq[:] = 0.
			Sq = EqnSet.SourceState(nq_st, x, t, Uq, Sq) # [nq,sr,dim]

			ER += solver_tools.calculate_source_term_integral(elem_ops, elem_ops_st, elem, Sq)

		if elem == echeck:
			code.interact(local=locals())

		return ER

	def calculate_residual_iface(self, iiface, UpL, UpR, RL, RR):
		'''
		Method: calculate_residual_iface
		-------------------------------------------
		Calculates the residual from the boundary integral for each internal face

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

		if faceL == 0:
			faceL_st = 3
		elif faceL == 1:
			faceL_st = 1
		else:
			return IncompatibleError
		if faceR == 0:
			faceR_st = 3
		elif faceR == 1:
			faceR_st = 1
		else:
			return IncompatibleError

		iface_ops = self.iface_operators
		iface_ops_st = self.iface_operators_st
		quad_wts_st = iface_ops_st.quad_wts

		faces_to_basisL = iface_ops.faces_to_basisL
		faces_to_basisR = iface_ops.faces_to_basisR

		faces_to_basisL_st = iface_ops_st.faces_to_basisL
		faces_to_basisR_st = iface_ops_st.faces_to_basisR

		UqL = iface_ops.UqL
		UqR = iface_ops.UqR
		Fq = iface_ops.Fq

		basis_valL = faces_to_basisL[faceL]
		basis_valR = faces_to_basisR[faceR]

		basis_valL_st = faces_to_basisL_st[faceL_st]
		basis_valR_st = faces_to_basisR_st[faceR_st]

		UqL = np.matmul(basis_valL_st, UpL)
		UqR = np.matmul(basis_valR_st, UpR)

		normals_ifaces = iface_ops.normals_ifaces
		normals = normals_ifaces[iiface]

		if self.Params["ConvFluxSwitch"] == True:

			Fq = EqnSet.ConvFluxNumerical(UqL, UqR, normals) # [nq_st,ns]

			RL -= solver_tools.calculate_inviscid_flux_boundary_integral(basis_valL, quad_wts_st, Fq)
			RR += solver_tools.calculate_inviscid_flux_boundary_integral(basis_valR, quad_wts_st, Fq)

		if elemL == echeck or elemR == echeck:
			if elemL == echeck: print("Left!")
			else: print("Right!")
			code.interact(local=locals())

		return RL, RR

	def calculate_residual_bface(self, BFG, ibface, U, R):
		'''
		Method: calculate_residual_bface
		-------------------------------------------
		Calculates the residual from the boundary integral for each boundary face

		INPUTS:
			ibfgrp: index of BC group
			ibface: index of boundary face
			U: solution array from internal element
			
		OUTPUTS:
			R: calculated residual array (from boundary face)
		'''
		mesh = self.mesh
		dim = mesh.Dim
		EqnSet = self.EqnSet
		ns = EqnSet.NUM_STATE_VARS
		# BFG = mesh.BFaceGroups[ibfgrp]
		ibfgrp = BFG.number
		BFace = BFG.BFaces[ibface]
		elem = BFace.Elem
		face = BFace.face

		bface_ops = self.bface_operators
		bface_ops_st = self.bface_operators_st
		quad_wts_st = bface_ops_st.quad_wts
		faces_to_xref_st = bface_ops_st.faces_to_xref

		faces_to_basis = bface_ops.faces_to_basis
		faces_to_basis_st = bface_ops_st.faces_to_basis

		normals_bfgroups = bface_ops.normals_bfgroups
		x_bfgroups = bface_ops.x_bfgroups

		UqI = bface_ops_st.UqI
		UqB = bface_ops_st.UqB
		Fq = bface_ops_st.Fq

		if face == 0:
			face_st = 3
		elif face == 1:
			face_st = 1
		else:
			return IncompatibleError
	
		basis_val = faces_to_basis[face]
		basis_val_st = faces_to_basis_st[face_st]
		xref_st = faces_to_xref_st[face_st]

		nq_st = quad_wts_st.shape[0]

		TimePhiData = None

		t = np.zeros([nq_st,dim])
		t, TimePhiData = solver_tools.ref_to_phys_time(mesh, elem, self.Time, self.Stepper.dt, TimePhiData, xref_st, t, None)

		# interpolate state and gradient at quad points
		UqI = np.matmul(basis_val_st, U)

		normals = normals_bfgroups[ibfgrp][ibface]
		x = x_bfgroups[ibfgrp][ibface]

		# Get boundary state
		BC = EqnSet.BCs[BFG.name]
		# Fq = BC.get_boundary_flux(EqnSet, x, t, normals, UqI)

		if self.Params["ConvFluxSwitch"] == True:
			# Loop over time to apply BC at each temporal quadrature point
			for i in range(len(t)):	
				Fq[i,:] = BC.get_boundary_flux(EqnSet, x, t[i], normals, UqI[i,:].reshape([1,ns]))

			R -= solver_tools.calculate_inviscid_flux_boundary_integral(basis_val, quad_wts_st, Fq)

		if elem == echeck:
			code.interact(local=locals())

		return R

	def flux_coefficients(self, elem, dt, order, basis, Up):
		'''
		Method: flux_coefficients
		-------------------------------------------
		Calculates the polynomial coefficients for the flux functions in ADER-DG

		INPUTS:
			elem: element index
			dt: time step size
			order: solution order
			U: solution array
			
		OUTPUTS:
			F: polynomical coefficients of the flux function
		'''

		EqnSet = self.EqnSet
		mesh = self.mesh
		ns = EqnSet.NUM_STATE_VARS
		dim = EqnSet.dim
		Params = self.Params

		InterpolateFlux = Params["InterpolateFlux"]

		elem_ops = self.elem_operators
		elem_ops_st = self.elem_operators_st
		djac_elems = elem_ops.djac_elems 
		djac = djac_elems[elem]

		rhs = np.zeros([basis.get_num_basis_coeff(order),ns,dim],dtype=Up.dtype)
		F = np.zeros_like(rhs)

		if Params["InterpolateFlux"]:

			Fq = EqnSet.ConvFluxInterior(Up)
			dg_tools.interpolate_to_nodes(Fq, F)
		else:
			ader_ops = self.ader_operators
			basis_val_st = elem_ops_st.basis_val
			quad_wts_st = elem_ops_st.quad_wts
			quad_wts = elem_ops.quad_wts
			quad_pts_st = elem_ops_st.quad_pts
			quad_pts = elem_ops.quad_pts
			nq_st = quad_wts_st.shape[0]
			nq = quad_wts.shape[0]
			iMM = ader_ops.iMM_elems[elem]

			Uq = np.matmul(basis_val_st,Up)

			Fq = EqnSet.ConvFluxInterior(Uq)
			solver_tools.L2_projection(mesh, iMM, basis, quad_pts_st, quad_wts_st, np.tile(djac,(nq,1)), Fq[:,:,0], F[:,:,0])

		return F*dt/2.0

	def source_coefficients(self, elem, dt, order, basis, Up):
		'''
		Method: source_coefficients
		-------------------------------------------
		Calculates the polynomial coefficients for the source functions in ADER-DG

		INPUTS:
			elem: element index
			dt: time step size
			order: solution order
			U: solution array
			
		OUTPUTS:
			S: polynomical coefficients of the flux function
		'''
		mesh = self.mesh
		dim = mesh.Dim
		EqnSet = self.EqnSet
		ns = EqnSet.NUM_STATE_VARS
		Params = self.Params

		InterpolateFlux = Params["InterpolateFlux"]

		elem_ops = self.elem_operators
		elem_ops_st = self.elem_operators_st
		djac_elems = elem_ops.djac_elems 
		djac = djac_elems[elem]
		S = elem_ops_st.Sq

		x_elems = elem_ops.x_elems
		x = x_elems[elem]

		ader_ops = self.ader_operators
		x_elems_ader = ader_ops.x_elems
		x_ader = x_elems_ader[elem]

		TimePhiData = None

		if Params["InterpolateFlux"]:
			xnode, nb = basis.equidistant_nodes(order)

			t = np.zeros([nb,dim])
			t, TimePhiData = solver_tools.ref_to_phys_time(mesh, elem, self.Time, self.Stepper.dt, TimePhiData, xnode, t, None)
			Sq = np.zeros([t.shape[0],ns])
			Sq = EqnSet.SourceState(nb, x_ader, t, Up, Sq)

			dg_tools.interpolate_to_nodes(Sq, S)
		else:

			ader_ops = self.ader_operators
			basis_val_st = elem_ops_st.basis_val
			quad_wts_st = elem_ops_st.quad_wts
			quad_wts = elem_ops.quad_wts
			quad_pts_st = elem_ops_st.quad_pts
			nq_st = quad_wts_st.shape[0]
			nq = quad_wts.shape[0]
			iMM = ader_ops.iMM_elems[elem]

			Uq = np.matmul(basis_val_st,Up)

			t = np.zeros([nq_st,dim])
			t, TimePhiData = solver_tools.ref_to_phys_time(mesh, elem, self.Time, self.Stepper.dt, TimePhiData, quad_pts_st, t, None)
		
			Sq = np.zeros([t.shape[0],ns])
			Sq = EqnSet.SourceState(nq_st, x, t, Uq, Sq) # [nq,sr,dim]		
			solver_tools.L2_projection(mesh, iMM, basis, quad_pts_st, quad_wts_st, np.tile(djac,(nq,1)), Sq, S)

		return S*dt/2.0