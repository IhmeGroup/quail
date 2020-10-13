# ------------------------------------------------------------------------ #
#
#       File : src/numerics/solver/ADERDG.py
#
#       Contains class definitions for the ADER-DG solver.
#      
# ------------------------------------------------------------------------ #
import numpy as np
from scipy.linalg import solve_sylvester

import errors

from general import ModalOrNodal, StepperType

import meshing.meshbase as mesh_defs
import meshing.tools as mesh_tools

import numerics.basis.tools as basis_tools
import numerics.basis.ader_tools as basis_st_tools

import numerics.helpers.helpers as helpers
import numerics.limiting.tools as limiter_tools

import numerics.timestepping.tools as stepper_tools
import numerics.timestepping.stepper as stepper_defs

import solver.ader_tools as solver_tools
import solver.tools as dg_tools
import solver.base as base
import solver.DG as DG


class ElemHelpersADER(DG.ElemHelpers):
	'''
    ElemHelpersADER inherits attributes and methods from the DG.ElemHelpers 
    class. See DG.ElemHelpers for detailed comments of attributes and 
    methods.

    Additional methods and attributes are commented below.
	'''
	def __init__(self):
		super().__init__()
		self.basis_time = None 
			# basis object for converting time from ref to phys space

	def get_basis_and_geom_data(self, mesh, basis, order):
		dim = mesh.dim 
		quad_pts = self.quad_pts 
		num_elems = mesh.num_elems 
		nq = quad_pts.shape[0]
		nb = basis.nb

		self.jac_elems = np.zeros([num_elems,nq,dim,dim])
		self.ijac_elems = np.zeros([num_elems,nq,dim,dim])
		self.djac_elems = np.zeros([num_elems,nq,1])
		self.x_elems = np.zeros([num_elems,nq,dim])
		self.basis_phys_grad_elems = np.zeros([num_elems,nq,nb,dim])

		basis.get_basis_val_grads(quad_pts, get_val=True, get_ref_grad=True)

		self.basis_val = basis.basis_val
		self.basis_ref_grad = basis.basis_ref_grad

		for elem in range(mesh.num_elems):
			# get Jacobian data
			djac, jac, ijac = basis_tools.element_jacobian(mesh, elem,
					quad_pts, get_djac=True, get_jac=True, get_ijac=True)
			# store the Jacobian data
			self.jac_elems[elem] = jac
			self.ijac_elems[elem] = ijac
			self.djac_elems[elem] = djac

			# physical coordinates of quadrature points
			x = mesh_tools.ref_to_phys(mesh, elem, quad_pts)
			# store
			self.x_elems[elem] = x


class InteriorFaceHelpersADER(DG.InteriorFaceHelpers):
	'''
    InteriorHelpersADER inherits attributes and methods from the
    DG.InteriorFaceHelpers class. See DG.InteriorFaceHelpers for detailed
    comments of attributes and methods.

    Additional methods and attributes are commented below.
	'''
	def get_gaussian_quadrature(self, mesh, physics, basis, order):
		
		gbasis = mesh.gbasis
		quad_order = gbasis.FACE_SHAPE.get_quadrature_order(mesh, 
				order, physics=physics)
		self.quad_pts, self.quad_wts = \
				basis.FACE_SHAPE.get_quadrature_data(quad_order)

	def get_basis_and_geom_data(self, mesh, basis, order):

		dim = mesh.dim
		quad_pts = self.quad_pts 
		nq = quad_pts.shape[0]
		nb = basis.get_num_basis_coeff(order)

		# primary difference is that their are two additional faces 
		# per element for the ADER-DG approach
		nfaces_per_elem = mesh.gbasis.NFACES + 2

		# allocate
		self.faces_to_basisL = np.zeros([nfaces_per_elem,nq,nb])
		self.faces_to_basisR = np.zeros([nfaces_per_elem,nq,nb])
		self.normals_ifaces = np.zeros([mesh.num_interior_faces,nq,dim])

		for f in range(nfaces_per_elem):
			# left
			basis.get_basis_face_val_grads(mesh, f, quad_pts, 
					basis, get_val=True)
			self.faces_to_basisL[f] = basis.basis_val
			# right
			basis.get_basis_face_val_grads(mesh, f, quad_pts[::-1], 
					basis, get_val=True)
			self.faces_to_basisR[f] = basis.basis_val

		# normals
		i = 0
		for interior_face in mesh.interior_faces:
			normals = mesh.gbasis.calculate_normals(mesh, interior_face.elemL_ID, 
				interior_face.faceL_ID, quad_pts)
			self.normals_ifaces[i] = normals
			i += 1

class BoundaryFaceHelpersADER(InteriorFaceHelpersADER):
	'''
    BoundaryHelpersADER inherits attributes and methods from the
    InteriorFaceHelpersADER class. See InteriorFaceHelpersADER for detailed
    comments of attributes and methods.

    Additional methods and attributes are commented below.
	'''
	def get_basis_and_geom_data(self, mesh, basis, order):

		dim = mesh.dim
		quad_pts = self.quad_pts 
		nq = quad_pts.shape[0]
		nb = basis.get_num_basis_coeff(order)

		# primary difference is that their are two additional faces 
		# per element for the ADER-DG approach
		nfaces_per_elem = mesh.gbasis.NFACES + 2

		# Allocate
		self.faces_to_basis = np.zeros([nfaces_per_elem, nq, nb])
		self.faces_to_xref = np.zeros([nfaces_per_elem, nq, dim+1])
		self.normals_bgroups = []
		self.x_bfgroups = []


		for f in range(nfaces_per_elem):
			# Left
			self.faces_to_xref[f] = basis.get_elem_ref_from_face_ref(f, 
					quad_pts)
			basis.get_basis_face_val_grads(mesh, f, quad_pts, basis, 
					get_val=True)
			self.faces_to_basis[f] = basis.basis_val

		i = 0
		for bgroup in mesh.boundary_groups.values():
			self.normals_bgroups.append(np.zeros([bgroup.num_boundary_faces,nq,
					dim]))
			self.x_bfgroups.append(np.zeros([bgroup.num_boundary_faces,nq,dim]))
			normal_bgroup = self.normals_bgroups[i]
			x_bfgroup = self.x_bfgroups[i]

			# normals
			j = 0
			for boundary_face in bgroup.boundary_faces:
				nvec = mesh.gbasis.calculate_normals(mesh, 
						boundary_face.elem_ID, boundary_face.face_ID, 
						quad_pts)
				normal_bgroup[j] = nvec

				# physical coordinates of quadrature points
				x = mesh_tools.ref_to_phys(mesh, boundary_face.elem_ID, 
						self.faces_to_xref[boundary_face.face_ID])
				# store
				x_bfgroup[j] = x

				# increment
				j += 1
			i += 1

	def alloc_other_arrays(self, physics, basis, order):
		quad_pts = self.quad_pts 
		nq = quad_pts.shape[0]
		ns = physics.NUM_STATE_VARS

		self.UqI = np.zeros([nq, ns])
		self.UqB = np.zeros([nq, ns])
		self.Fq = np.zeros([nq, ns])


class ADERHelpers(object):
	'''
	The ADERHelpers class contains the methods and attributes that are 
	required prior to the main solver temporal loop for the ADER scheme. 
	They are used to precompute the attributes pertaining to the space-time
	element in both physical and reference space

	Attributes:
	-----------
	MM: numpy array
		space-time mass matrix evaluated on the reference element
	iMM: numpy array
		space-time inverse mass matrix evaluated on the reference element
	iMM_elems: numpy array
		space-time inverse mass matrix evaluated on the physical element
	K: numpy array
		space-time matrix defined as st-flux matrix minus the stiffness 
		matrix whose gradient is taken in the temporal direction
	iK: numpy array
		inverse of space-time matrix K
	FTL: numpy array
		flux matrix in space-time reference space (evaluated at tau=1)
	FTR: numpy array
		flux matrix in space-time reference space (evaulated at tau=-1)
	SMT: numpy array
		stiffness matrix in time (gradient taken in temporal "direction")
	SMS_elems: numpy array
		stiffness matrix in spatial direction evaluated for each element in 
		physical space
	jac_elems: numpy array
		Jacobian evaluated at the element nodes
	ijac_elems: numpy array
		inverse Jacobian evaluated at the element nodes
	djac_elems: numpy array
		determinant of the Jacobian evaluated at the element nodes
	x_elems: numpy array
		physical coordinates of nodal points
	'''
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

	def calc_ader_matrices(self, mesh, basis, basis_st, dt, order):
		'''
		Precomputes the matries for the ADER-DG scheme

		Inputs:
		-------
			mesh: mesh object
			basis: basis object
			basis_st: space-time basis object
			dt: time step
			order: solution order

		Outputs:
		--------
			self.FTL: flux matrix in space-time reference space 
				[nb_st, nb_st]
			self.FTR: flux matrix in reference space [nb_st, nb]
			self.SMT: stiffness matrix in time [nb_st, nb_st]
			self.SMS_elems: stiffness matrix in space for each 
				element [num_elems, nb_st, nb_st]
			self.MM: space-time mass matrix in reference space
				[nb_st, nb_st]
			self.iMM: space-time inverse mass matrix in ref space
				[nb_st, nb_st]
			self.iMM_elems: space-time inverse mass matrix in physical
				space [num_elems, nb_st, nb_st]
			self.K: space-time matrix FTL - SMT [nb_st, nb_st]
			self.iK: inverse of space-time matrix K [nb_st, nb_st]
		'''		
		dim = mesh.dim
		nb = basis_st.nb
		SMS_elems = np.zeros([mesh.num_elems,nb,nb,dim])
		iMM_elems = np.zeros([mesh.num_elems,nb,nb])
		# Get flux matrices in time
		FTL = basis_st_tools.get_temporal_flux_ader(mesh, basis_st, basis_st,
				order, physical_space=False)
		FTR = basis_st_tools.get_temporal_flux_ader(mesh, basis_st, basis,
				order, physical_space=False)

		# Get stiffness matrix in time
		SMT = basis_st_tools.get_stiffness_matrix_ader(mesh, basis, basis_st,
				order, dt, elem=0, grad_dir=1, physical_space=False)

		# Get stiffness matrix in space
		for elem in range(mesh.num_elems):
			SMS = basis_st_tools.get_stiffness_matrix_ader(mesh, basis,
					basis_st, order, dt, elem, grad_dir=0, physical_space=True)
			SMS_elems[elem,:,:,0] = SMS.transpose()

			iMM = basis_st_tools.get_elem_inv_mass_matrix_ader(mesh,
					basis_st, order, elem, physical_space=True)
			iMM_elems[elem] = iMM

		iMM = basis_st_tools.get_elem_inv_mass_matrix_ader(mesh, basis_st,
				order, elem=-1, physical_space=False)
		MM = basis_st_tools.get_elem_mass_matrix_ader(mesh, basis_st, order,
				elem=-1, physical_space=False)

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
		'''
		Precomputes the geometric data for the ADER-DG scheme

		Inputs:
		-------
			mesh: mesh object
			basis: basis object
			order: solution order

		Outputs:
		--------
			self.jac_elems: precomputed Jacobian for each element 
				[num_elems, nb, dim, dim]
			self.ijac_elems: precomputed inverse Jacobian for each element
				[num_elems, nb, dim, dim]
			self.djac_elems: precomputed determinant of the Jacobian for each
				element [num_elems, nb, 1]
			self.x_elems: precomputed coordinates of the nodal points
				in physical space [num_elems, nb, dim]
		'''
		shape_name = basis.__class__.__bases__[1].__name__

		dim = mesh.dim 
		num_elems = mesh.num_elems 
		nb = basis.nb
		gbasis = mesh.gbasis
		xnode = gbasis.get_nodes(order)
		nnode = xnode.shape[0]

		# allocate
		self.jac_elems = np.zeros([num_elems,nb,dim,dim])
		self.ijac_elems = np.zeros([num_elems,nb,dim,dim])
		self.djac_elems = np.zeros([num_elems,nb,1])
		self.x_elems = np.zeros([num_elems,nb,dim])


		for elem in range(mesh.num_elems):
			# Jacobian
			djac, jac, ijac = basis_tools.element_jacobian(mesh, elem, xnode,
				 	get_djac=True, get_jac=True, get_ijac=True)

			if shape_name == 'HexShape':
				self.jac_elems[elem] = np.tile(jac,(int(np.sqrt(nnode)),1,1))
				self.ijac_elems[elem] = np.tile(ijac,
						(int(np.sqrt(nnode)),1,1))
				self.djac_elems[elem] = np.tile(djac,(int(np.sqrt(nnode)),1))

				# Physical coordinates of nodal points
				x = mesh_tools.ref_to_phys(mesh, elem, xnode)
				# Store
				self.x_elems[elem] = np.tile(x,(int(np.sqrt(nnode)),1))

			elif shape_name == 'QuadShape':
				self.jac_elems[elem] = np.tile(jac,(nnode,1,1))
				self.ijac_elems[elem] = np.tile(ijac,(nnode,1,1))
				self.djac_elems[elem] = np.tile(djac,(nnode,1))

				# Physical coordinates of nodal points
				x = mesh_tools.ref_to_phys(mesh, elem, xnode)
				# Store
				self.x_elems[elem] = np.tile(x,(nnode,1))

	def compute_helpers(self, mesh, physics, basis, basis_st, dt, order):
		self.calc_ader_matrices(mesh, basis, basis_st, dt, order)
		self.get_geom_data(mesh, basis_st, order)


class ADERDG(base.SolverBase):
	'''
    ADERDG inherits attributes and methods from the SolverBase class.
    See SolverBase for detailed comments of attributes and methods.

    Additional methods and attributes are commented below.
	'''
	def __init__(self, params, physics, mesh):
		super().__init__(params, physics, mesh)

		ns = physics.NUM_STATE_VARS

		TimeStepper = params["TimeStepper"]
		if StepperType[TimeStepper] != StepperType.ADER:
			raise errors.IncompatibleError

		self.stepper = stepper_defs.ADER(physics.U)
		stepper_tools.set_time_stepping_approach(self.stepper, params)

		# Set the space-time basis functions for the solver
		basis_name  = params["SolutionBasis"]
		self.basis_st = basis_st_tools.set_basis_spacetime(mesh, 
				physics.order, basis_name)

		# Set quadrature for space-time basis
		self.basis_st.set_elem_quadrature_type(params["ElementQuadrature"])
		self.basis_st.set_face_quadrature_type(params["FaceQuadrature"])

		self.basis.force_colocated_nodes_quad_pts(params["ColocatedPoints"])
		self.basis_st.force_colocated_nodes_quad_pts(params["ColocatedPoints"])

		# Allocate array for predictor step in ADER-Scheme
		physics.U_pred = np.zeros([self.mesh.num_elems, 
				self.basis_st.get_num_basis_coeff(physics.order), 
				physics.NUM_STATE_VARS])

		# Set predictor function
		source_treatment = params["SourceTreatmentADER"]
		self.calculate_predictor_elem = solver_tools.set_source_treatment(ns,
				source_treatment)

		# Precompute helpers
		self.precompute_matrix_helpers()

		if self.limiter is not None:
			self.limiter.precompute_helpers(self)

		physics.conv_flux_fcn.alloc_helpers(np.zeros([
				self.int_face_helpers_st.quad_wts.shape[0], 
				physics.NUM_STATE_VARS]))

		# Initialize state
		if params["RestartFile"] is None:
			self.init_state_from_fcn()
	
	def __repr__(self):
		return '{self.__class__.__name__}(Physics: {self.physics},\n   \
				Basis: {self.basis}, \n   Basis_st: {self.basis_st},\n   \
				Stepper: {self.stepper})'.format(self=self)
	
	def check_compatibility(self):
		super().check_compatibility()

		basis = self.basis
		params = self.params

		if params["InterpolateFluxADER"] and \
				basis.MODAL_OR_NODAL != ModalOrNodal.Nodal:
			raise errors.IncompatibleError

		if params["CFL"] != None:
			print("Error Message")
			print("-------------------------------------------------------")
			print("CFL-based time-stepping not currently supported in " + \
					"ADERDG")
			print("")
			raise errors.IncompatibleError

	def precompute_matrix_helpers(self):
		mesh = self.mesh 
		physics = self.physics

		basis = self.basis
		basis_st = self.basis_st
		stepper = self.stepper

		self.elem_helpers = DG.ElemHelpers()
		self.elem_helpers.compute_helpers(mesh, physics, basis, 
				physics.order)
		self.int_face_helpers = DG.InteriorFaceHelpers()
		self.int_face_helpers.compute_helpers(mesh, physics, basis, 
				physics.order)
		self.bface_helpers = DG.BoundaryFaceHelpers()
		self.bface_helpers.compute_helpers(mesh, physics, basis,
				physics.order)

		# Calculate ADER specific space-time helpers
		self.elem_helpers_st = ElemHelpersADER()
		self.elem_helpers_st.compute_helpers(mesh, physics, basis_st,
				physics.order)
		self.int_face_helpers_st = InteriorFaceHelpersADER()
		self.int_face_helpers_st.compute_helpers(mesh, physics, basis_st,
				physics.order)
		self.bface_helpers_st = BoundaryFaceHelpersADER()
		self.bface_helpers_st.compute_helpers(mesh, physics, basis_st,
				physics.order)

		stepper.dt = stepper.get_time_step(stepper, self)
		dt = stepper.dt
		self.ader_helpers = ADERHelpers()
		self.ader_helpers.compute_helpers(mesh, physics, basis, 
				basis_st, dt, physics.order)

	def calculate_predictor_step(self, dt, W, Up):
		'''
		Calls the predictor step for each element

		Inputs:
		-------
			dt: time step 
			W: previous time step solution in space only [num_elems, nb, ns]

		Outputs:
		--------
			Up: predicted solution in space-time [num_elems, nb_st, ns]
		'''
		mesh = self.mesh
		physics = self.physics

		for elem in range(mesh.num_elems):
			Up[elem] = self.calculate_predictor_elem(self, elem, dt, W[elem],
					Up[elem])

		return Up

	def get_element_residual(self, elem, Uc, R_elem):
		physics = self.physics
		mesh = self.mesh
		ns = physics.NUM_STATE_VARS
		dim = physics.DIM

		elem_helpers = self.elem_helpers
		elem_helpers_st = self.elem_helpers_st

		quad_wts = elem_helpers.quad_wts
		quad_wts_st = elem_helpers_st.quad_wts
		quad_pts_st = elem_helpers_st.quad_pts

		basis_val_st = elem_helpers_st.basis_val
		x_elems = elem_helpers.x_elems
		x = x_elems[elem]

		nq = quad_wts.shape[0]
		nq_st = quad_wts_st.shape[0]

		# interpolate state and gradient at quadrature points
		Uq = helpers.evaluate_state(Uc, basis_val_st)

		if self.params["ConvFluxSwitch"] == True:
			# Evaluate the inviscid flux integral.
			Fq = physics.get_conv_flux_interior(Uq) # [nq,sr,dim]
			R_elem += solver_tools.calculate_inviscid_flux_volume_integral(self, 
					elem_helpers, elem_helpers_st, elem, Fq)
		
		if self.params["SourceSwitch"] == True:
			# Evaluate the source term integral
			t, elem_helpers_st.basis_time = solver_tools.ref_to_phys_time(
					mesh, elem, self.time, self.stepper.dt, 
					quad_pts_st[:, -1:], elem_helpers_st.basis_time)
			
			Sq = elem_helpers_st.Sq
			Sq[:] = 0.
			Sq = physics.eval_source_terms(Uq, x, t, Sq) # [nq,sr,dim]

			R_elem += solver_tools.calculate_source_term_integral(elem_helpers, 
					elem_helpers_st, elem, Sq)

		return R_elem

	def get_interior_face_residual(self, int_face_ID, Uc_L, Uc_R, R_L, R_R):
		mesh = self.mesh
		physics = self.physics
		interior_face = mesh.interior_faces[int_face_ID]
		elemL = interior_face.elemL_ID
		elemR = interior_face.elemR_ID
		faceL_ID = interior_face.faceL_ID
		faceR_ID = interior_face.faceR_ID

		# Convert 1D face numbering to "2D" face numbering
		if faceL_ID == 0:
			faceL_ID_st = 3
		elif faceL_ID == 1:
			faceL_ID_st = 1
		else:
			return ValueError
		if faceR_ID == 0:
			faceR_ID_st = 3
		elif faceR_ID == 1:
			faceR_ID_st = 1
		else:
			return ValueError

		int_face_helpers = self.int_face_helpers
		int_face_helpers_st = self.int_face_helpers_st
		quad_wts_st = int_face_helpers_st.quad_wts

		faces_to_basisL = int_face_helpers.faces_to_basisL
		faces_to_basisR = int_face_helpers.faces_to_basisR

		faces_to_basisL_st = int_face_helpers_st.faces_to_basisL
		faces_to_basisR_st = int_face_helpers_st.faces_to_basisR

		UqL = int_face_helpers.UqL
		UqR = int_face_helpers.UqR
		Fq = int_face_helpers.Fq

		basis_valL = faces_to_basisL[faceL_ID]
		basis_valR = faces_to_basisR[faceR_ID]

		basis_valL_st = faces_to_basisL_st[faceL_ID_st]
		basis_valR_st = faces_to_basisR_st[faceR_ID_st]

		UqL = helpers.evaluate_state(Uc_L, basis_valL_st)
		UqR = helpers.evaluate_state(Uc_R, basis_valR_st)

		normals_ifaces = int_face_helpers.normals_ifaces
		normals = normals_ifaces[int_face_ID]

		if self.params["ConvFluxSwitch"] == True:

			Fq = physics.get_conv_flux_numerical(UqL, UqR, normals) # [nq_st,ns]
			R_L -= solver_tools.calculate_inviscid_flux_boundary_integral(
					basis_valL, quad_wts_st, Fq)
			R_R += solver_tools.calculate_inviscid_flux_boundary_integral(
					basis_valR, quad_wts_st, Fq)

		return R_L, R_R

	def get_boundary_face_residual(self, bgroup, bface_ID, Uc, R_B):

		mesh = self.mesh
		dim = mesh.dim
		physics = self.physics
		ns = physics.NUM_STATE_VARS
		bgroup_num = bgroup.number
		boundary_face = bgroup.boundary_faces[bface_ID]
		elem = boundary_face.elem_ID
		face = boundary_face.face_ID

		bface_helpers = self.bface_helpers
		bface_helpers_st = self.bface_helpers_st
		quad_wts_st = bface_helpers_st.quad_wts
		faces_to_xref_st = bface_helpers_st.faces_to_xref

		faces_to_basis = bface_helpers.faces_to_basis
		faces_to_basis_st = bface_helpers_st.faces_to_basis

		normals_bgroups = bface_helpers.normals_bgroups
		x_bfgroups = bface_helpers.x_bfgroups

		UqI = bface_helpers_st.UqI
		UqB = bface_helpers_st.UqB
		Fq = bface_helpers_st.Fq

		# Convert 1D face ID to "2D" face ID
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

		t, self.elem_helpers_st.basis_time = solver_tools.ref_to_phys_time(mesh, elem, self.time,
				self.stepper.dt, xref_st[:, -1:], self.elem_helpers_st.basis_time)

		# interpolate state and gradient at quadrature points
		UqI = helpers.evaluate_state(Uc, basis_val_st)


		normals = normals_bgroups[bgroup_num][bface_ID]
		x = x_bfgroups[bgroup_num][bface_ID]

		# Get boundary state
		BC = physics.BCs[bgroup.name]

		if self.params["ConvFluxSwitch"] == True:
			# loop over time to apply BC at each temporal quadrature point
			for i in range(t.shape[0]):	
				# Fq[i,:] = BC.get_boundary_flux(physics, x, t[i], normals, 
				# 		UqI[i,:].reshape([1,ns]))
				Fq[i,:] = BC.get_boundary_flux(physics, 
						UqI[i, :].reshape([1, ns]), normals, x, t[i])

			R_B -= solver_tools.calculate_inviscid_flux_boundary_integral(
					basis_val, quad_wts_st, Fq)

		return R_B

	def flux_coefficients(self, elem, dt, order, basis, Up):
		'''
		Calculates the polynomial coefficients for the flux functions in 
		ADER-DG

		Inputs:
		-------
			elem: element index
			dt: time step size
			order: solution order
			basis: basis object
			Up: solution array [nb_st, ns]
			
		Outputs:
		--------
			F: polynomical coefficients of the flux function [nb_st, ns, dim]
		'''
		physics = self.physics
		mesh = self.mesh
		ns = physics.NUM_STATE_VARS
		dim = physics.DIM
		params = self.params

		InterpolateFluxADER = params["InterpolateFluxADER"]

		elem_helpers = self.elem_helpers
		elem_helpers_st = self.elem_helpers_st
		djac_elems = elem_helpers.djac_elems 
		djac = djac_elems[elem]

		rhs = np.zeros([basis.get_num_basis_coeff(order),ns,dim],
				dtype=Up.dtype)
		F = np.zeros_like(rhs)

		if params["InterpolateFluxADER"]:

			Fq = physics.get_conv_flux_interior(Up)
			dg_tools.interpolate_to_nodes(Fq, F)
		else:
			ader_helpers = self.ader_helpers
			basis_val_st = elem_helpers_st.basis_val
			quad_wts_st = elem_helpers_st.quad_wts
			quad_wts = elem_helpers.quad_wts
			quad_pts_st = elem_helpers_st.quad_pts
			quad_pts = elem_helpers.quad_pts
			nq_st = quad_wts_st.shape[0]
			nq = quad_wts.shape[0]
			iMM = ader_helpers.iMM_elems[elem]

			Uq = helpers.evaluate_state(Up, basis_val_st)

			Fq = physics.get_conv_flux_interior(Uq)
			solver_tools.L2_projection(mesh, iMM, basis, quad_pts_st, 
					quad_wts_st, np.tile(djac,(nq,1)), Fq[:,:,0], F[:,:,0])

		return F*dt/2.0 # [nb_st, ns, dim]

	def source_coefficients(self, elem, dt, order, basis, Up):
		'''
		Calculates the polynomial coefficients for the source functions in 
		ADER-DG

		Inputs:
		-------
			elem: element index
			dt: time step size
			order: solution order
			basis: basis object
			Up: solution array [nb_st, ns]
			
		Outputs:
		--------
			S: polynomical coefficients of the flux function [nb_st, ns]
		'''
		mesh = self.mesh
		dim = mesh.dim
		physics = self.physics
		ns = physics.NUM_STATE_VARS
		params = self.params

		elem_helpers = self.elem_helpers
		elem_helpers_st = self.elem_helpers_st
		djac_elems = elem_helpers.djac_elems 
		djac = djac_elems[elem]

		x_elems = elem_helpers.x_elems
		x = x_elems[elem]

		ader_helpers = self.ader_helpers
		x_elems_ader = ader_helpers.x_elems
		x_ader = x_elems_ader[elem]

		if params["InterpolateFluxADER"]:
			xnode = basis.get_nodes(order)
			nb = xnode.shape[0]
			t = np.zeros([nb,dim])
			t, elem_helpers_st.basis_time = solver_tools.ref_to_phys_time(mesh, elem, 
					self.time, self.stepper.dt, xnode[:, -1:], elem_helpers_st.basis_time)
			Sq = np.zeros([t.shape[0],ns])
			S = np.zeros_like(Sq)
			Sq = physics.eval_source_terms(Up, x_ader, t, Sq)
			dg_tools.interpolate_to_nodes(Sq, S)

		else:
			ader_helpers = self.ader_helpers
			basis_val_st = elem_helpers_st.basis_val
			nb_st = basis_val_st.shape[1]
			quad_wts_st = elem_helpers_st.quad_wts
			quad_wts = elem_helpers.quad_wts
			quad_pts_st = elem_helpers_st.quad_pts
			nq_st = quad_wts_st.shape[0]
			nq = quad_wts.shape[0]
			iMM = ader_helpers.iMM_elems[elem]

			Uq = helpers.evaluate_state(Up, basis_val_st)

			t = np.zeros([nq_st,dim])
			t, elem_helpers_st.basis_time = solver_tools.ref_to_phys_time(mesh, elem, 
					self.time, self.stepper.dt, quad_pts_st[:, -1:], elem_helpers_st.basis_time)
		
			Sq = np.zeros([t.shape[0],ns])
			S = np.zeros([nb_st, ns])
			Sq = physics.eval_source_terms(Uq, x, t, Sq) # [nq,sr,dim]		
			solver_tools.L2_projection(mesh, iMM, basis, quad_pts_st, 
					quad_wts_st, np.tile(djac,(nq,1)), Sq, S)

		return S*dt/2.0 # [nb_st, ns]