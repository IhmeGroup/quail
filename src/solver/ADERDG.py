# ------------------------------------------------------------------------ #
#
#       File : src/solver/ADERDG.py
#
#       Contains class definitions for the ADER-DG solver
#
# ------------------------------------------------------------------------ #
import numpy as np
from scipy.linalg import solve_sylvester

import errors

from general import ModalOrNodal, StepperType

import meshing.meshbase as mesh_defs
import meshing.tools as mesh_tools

import numerics.basis.basis as basis_defs
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
		self.need_phys_grad = False
		self.basis_time = None
		self.nq_tile_constant = None
		self.nn_tile_constant = None
		self.nb_tile_constant = None


class InteriorFaceHelpersADER(DG.InteriorFaceHelpers):
	'''
    InteriorFaceHelpersADER inherits attributes and methods from the 
    DG.InteriorFaceHelpers class. See DG.InteriorFaceHelpers for detailed 
    comments of attributes and methods.

    Additional methods and attributes are commented below.
	'''
	def __init__(self):
		super().__init__()
		self.faceL_IDs_st = np.empty(0, dtype=int)
		self.faceR_IDs_st = np.empty(0, dtype=int)

	def store_neighbor_info(self, mesh):
		'''
		Store the element and face IDs on the left and right of each face
		for the space-time faces.

		Inputs:
		-------
			mesh: mesh object

		Outputs:
		--------
			self.elemL_IDs: Element IDs to the left of each interior face
			[num_interior_faces]
			self.elemR_IDs: Element IDs to the right of each interior face
			[num_interior_faces]
			self.faceL_IDs: Face IDs to the left of each interior face
			[num_interior_faces]
			self.faceR_IDs: Face IDs to the right of each interior face
			[num_interior_faces]
		'''
		super().store_neighbor_info(mesh)

		# Convert spacial face numbering to space-time
		if mesh.ndims == 1:
			self.faceL_IDs_st = np.ones(mesh.num_interior_faces, dtype=int)
			self.faceR_IDs_st = np.ones(mesh.num_interior_faces, dtype=int)

			indL = np.where(self.faceL_IDs == 0)[0]
			indR = np.where(self.faceR_IDs == 0)[0]
			self.faceL_IDs_st[indL] = 3
			self.faceR_IDs_st[indR] = 3

		elif mesh.ndims == 2:
			self.faceL_IDs_st = self.faceL_IDs
			self.faceR_IDs_st = self.faceR_IDs
		else:
			raise NotImplementedError


class BoundaryFaceHelpersADER(DG.BoundaryFaceHelpers):
	'''
    BoundaryFaceHelpersADER inherits attributes and methods from the 
    DG.BoundaryFaceHelpers class. See DG.BoundaryFaceHelpers for detailed 
    comments of attributes and methods.

    Additional methods and attributes are commented below.
	'''
	def __init__(self):
		super().__init__()
		self.face_IDs_st = []

	def store_neighbor_info(self, mesh):
		'''
		Store the element and face IDs on the left and right of each face
		for the space-time boundary faces.

		Inputs:
		-------
			mesh: mesh object

		Outputs:
		--------
			self.elemL_IDs: Element IDs to the left of each interior face
			[num_interior_faces]
			self.elemR_IDs: Element IDs to the right of each interior face
			[num_interior_faces]
			self.faceL_IDs: Face IDs to the left of each interior face
			[num_interior_faces]
			self.faceR_IDs: Face IDs to the right of each interior face
			[num_interior_faces]
		'''
		super().store_neighbor_info(mesh)
		# Convert spacial boundary face numbering to space-time
		for bgroup in mesh.boundary_groups.values():	
			bgroup_face_IDs_st = np.ones(bgroup.num_boundary_faces, 
					dtype=int)
			if mesh.ndims == 1:
				ind = np.where(self.face_IDs[bgroup.number] == 0)[0]
				bgroup_face_IDs_st[ind] = 3
			elif mesh.ndims == 2:
				bgroup_face_IDs_st = self.face_IDs[bgroup.number]
			else:
				raise NotImplementedError
			self.face_IDs_st.append(bgroup_face_IDs_st)


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
		self.MM = np.zeros(0)
		self.iMM = np.zeros(0)
		self.iMM_elems = np.zeros(0)
		self.K = np.zeros(0)
		self.iK = np.zeros(0)
		self.FTL = np.zeros(0)
		self.FTR = np.zeros(0)
		self.SMT = np.zeros(0)
		self.SMS_elems = np.zeros(0)
		self.jac_elems = np.zeros(0)
		self.ijac_elems = np.zeros(0)
		self.djac_elems = np.zeros(0)
		self.x_elems = np.zeros(0)

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
		ndims = mesh.ndims
		nb = basis_st.nb
		SMS_elems = np.zeros([mesh.num_elems, nb, nb, ndims])
		iMM_elems = np.zeros([mesh.num_elems, nb, nb])

		# Get flux matrices in time
		FTL = basis_st_tools.get_temporal_flux_ader(mesh, basis_st, basis_st,
				order, physical_space=False)
		FTR = basis_st_tools.get_temporal_flux_ader(mesh, basis_st, basis,
				order, physical_space=False)

		# Get stiffness matrix in time
		SMT = basis_st_tools.get_stiffness_matrix_ader(mesh, basis, basis_st,
				order, dt, elem_ID=0, grad_dir=-1, physical_space=False)

		# Get stiffness matrices in space and inverse mass matrices
		# (physical space)
		for elem_ID in range(mesh.num_elems):
			for nd in range(ndims):
				SMS = basis_st_tools.get_stiffness_matrix_ader(mesh, basis,
						basis_st, order, dt, elem_ID, grad_dir=nd,
						physical_space=True)
				SMS_elems[elem_ID, :, :, nd] = SMS.transpose()

			iMM = basis_st_tools.get_elem_inv_mass_matrix_ader(mesh,
					basis_st, order, elem_ID, physical_space=True)
			iMM_elems[elem_ID] = iMM

		# Get mass matrix (and inverse) in reference space
		iMM = basis_st_tools.get_elem_inv_mass_matrix_ader(mesh, basis_st,
				order, elem_ID=-1, physical_space=False)
		MM = basis_st_tools.get_elem_mass_matrix_ader(mesh, basis_st, order,
				elem_ID=-1, physical_space=False)

		# Store
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
				[num_elems, nb, ndims, ndims]
			self.ijac_elems: precomputed inverse Jacobian for each element
				[num_elems, nb, ndims, ndims]
			self.djac_elems: precomputed determinant of the Jacobian for each
				element [num_elems, nb, 1]
			self.x_elems: precomputed coordinates of the nodal points
				in physical space [num_elems, nb, ndims]
		'''
		ndims = mesh.ndims
		num_elems = mesh.num_elems
		nb = basis.nb
		gbasis = mesh.gbasis
		tile_basis = basis_defs.LagrangeSeg(order)

		# Define geometric basis for tiling jac, ijac, and djac
		xnodes = gbasis.get_nodes(order)
		nnodes = xnodes.shape[0]

		tile_xnodes = tile_basis.get_nodes(order)
		tile_nnodes = tile_xnodes.shape[0]

		# Allocate
		self.jac_elems = np.zeros([num_elems, nb, ndims, ndims])
		self.ijac_elems = np.zeros([num_elems, nb, ndims, ndims])
		self.djac_elems = np.zeros([num_elems, nb, 1])
		self.x_elems = np.zeros([num_elems, nb, ndims])

		for elem_ID in range(mesh.num_elems):
			# Jacobian
			djac, jac, ijac = basis_tools.element_jacobian(mesh, elem_ID,
					xnodes, get_djac=True, get_jac=True, get_ijac=True)

			self.jac_elems[elem_ID] = np.tile(jac, (tile_nnodes, 1, 1))
			self.ijac_elems[elem_ID] = np.tile(ijac, (tile_nnodes, 1, 1))
			self.djac_elems[elem_ID] = np.tile(djac, (tile_nnodes, 1))

			# Physical coordinates of nodal points
			x = mesh_tools.ref_to_phys(mesh, elem_ID, xnodes)
			# Store
			self.x_elems[elem_ID] = np.tile(x, (tile_nnodes, 1))

	def set_tiling_constants(self, basis, elem_helpers_st, bface_helpers_st):
		'''
		Sets the tiling constants for the ADER-DG scheme. Tiling
		constants are used with the numpy 'tile' function to build arrays
		to maintain consistency for the tensor multiplications throughout
		the solver.

		Inputs:
		-------
			basis: basis object
			elem_helpers_st: element helper object in space-time
			bface_helpers_st: boundary face helper object in space-time

		Outputs:
		--------
			Sets the tiling constants in elem_helpers_st
		'''
		bface_quad_pts_st = bface_helpers_st.quad_pts
		nq_t, nb_t, time_skip, time_tile = basis.get_tiling_constants(
				bface_quad_pts_st)

		elem_helpers_st.nq_tile_constant = nq_t
		elem_helpers_st.nb_tile_constant = nb_t
		elem_helpers_st.time_skip = time_skip
		elem_helpers_st.time_tile = time_tile

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

		# Time stepping
		time_stepper = params["TimeStepper"]
		if StepperType[time_stepper] != StepperType.ADER:
			raise errors.IncompatibleError

		self.stepper = stepper_defs.ADER(self.state_coeffs)
		stepper_tools.set_time_stepping_approach(self.stepper, params)

		# Set the space-time basis functions for the solver
		basis_name = params["SolutionBasis"]
		self.basis_st = basis_st_tools.set_basis_spacetime(mesh,
				self.order, basis_name)

		# Set quadrature for space-time basis
		self.basis_st.set_elem_quadrature_type(params["ElementQuadrature"])
		self.basis_st.set_face_quadrature_type(params["FaceQuadrature"])

		self.basis_st.force_colocated_nodes_quad_pts(
				params["ColocatedPoints"])

		# Allocate array for predictor step in ADER-Scheme
		self.state_coeffs_pred = np.zeros([self.mesh.num_elems,
				self.basis_st.get_num_basis_coeff(self.order),
				physics.NUM_STATE_VARS])

		# Set predictor function
		source_treatment = params["SourceTreatmentADER"]
		self.calculate_predictor_step = solver_tools.set_source_treatment(ns,
				source_treatment)

		# Precompute helpers
		self.precompute_matrix_helpers()

		if self.limiters:
			for limiter in self.limiters:
				limiter.precompute_helpers(self)

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

		order = self.order
		basis = self.basis
		basis_st = self.basis_st
		stepper = self.stepper

		self.elem_helpers = DG.ElemHelpers()
		self.elem_helpers.compute_helpers(mesh, physics, basis,
				order)
		self.int_face_helpers = DG.InteriorFaceHelpers()
		self.int_face_helpers.compute_helpers(mesh, physics, basis,
				order)
		self.bface_helpers = DG.BoundaryFaceHelpers()
		self.bface_helpers.compute_helpers(mesh, physics, basis,
				order)

		# Calculate ADER specific space-time helpers
		self.elem_helpers_st = ElemHelpersADER()
		self.elem_helpers_st.compute_helpers(mesh, physics, basis_st,
				order)
		self.int_face_helpers_st = InteriorFaceHelpersADER()
		self.int_face_helpers_st.compute_helpers(mesh, physics, basis_st,
				order)
		self.bface_helpers_st = BoundaryFaceHelpersADER()
		self.bface_helpers_st.compute_helpers(mesh, physics, basis_st,
				order)

		stepper.dt = stepper.get_time_step(stepper, self)
		dt = stepper.dt
		self.ader_helpers = ADERHelpers()
		self.ader_helpers.compute_helpers(mesh, physics, basis,
				basis_st, dt, order)

		self.ader_helpers.set_tiling_constants(basis,
				self.elem_helpers_st, self.bface_helpers_st)


	def get_element_residual(self, Uc, res_elem):
		physics = self.physics
		mesh = self.mesh
		ns = physics.NUM_STATE_VARS
		ndims = physics.NDIMS

		elem_helpers = self.elem_helpers
		elem_helpers_st = self.elem_helpers_st

		quad_wts = elem_helpers.quad_wts
		quad_wts_st = elem_helpers_st.quad_wts
		quad_pts_st = elem_helpers_st.quad_pts

		basis_val_st = elem_helpers_st.basis_val
		x_elems = elem_helpers.x_elems

		nq = quad_wts.shape[0]
		nq_st = quad_wts_st.shape[0]

		# Interpolate state at quad points
		Uq = helpers.evaluate_state(Uc, basis_val_st) # [ne, nq_st, ns]

		if self.verbose:
			# Get min and max of state variables for reporting
			self.get_min_max_state(Uq)

		if self.params["ConvFluxSwitch"] == True:
			# Evaluate the inviscid flux integral.
			Fq = physics.get_conv_flux_interior(Uq)[0] # [ne, nq, ns, ndims]
			res_elem += solver_tools.calculate_inviscid_flux_volume_integral(
					self, elem_helpers, elem_helpers_st, Fq) # [ne, nb, ns]

		if self.params["SourceSwitch"] == True:
			# Evaluate the source term integral

			# Get array in physical time from ref time
			t, elem_helpers_st.basis_time = solver_tools.ref_to_phys_time(
					mesh, self.time, self.stepper.dt,
					quad_pts_st[:, -1:], elem_helpers_st.basis_time)

			# Evaluate the source term at the quadrature points
			Sq = elem_helpers_st.Sq
			Sq[:] = 0. # [ne, nq, sr, ndims]
			Sq = physics.eval_source_terms(Uq, x_elems, t, Sq)

			res_elem += solver_tools.calculate_source_term_integral(
					elem_helpers, elem_helpers_st, Sq) # [ne, nb, ns]

		return res_elem # [ne, nb, ns]

	def get_interior_face_residual(self, faceL_id, faceR_id, Uc_L, Uc_R):
		# Unpack
		mesh = self.mesh
		physics = self.physics
		ns = physics.NUM_STATE_VARS

		nq_t = self.elem_helpers_st.nq_tile_constant

		int_face_helpers = self.int_face_helpers
		int_face_helpers_st = self.int_face_helpers_st

		faceL_id_st = int_face_helpers_st.faceL_IDs_st
		faceR_id_st = int_face_helpers_st.faceR_IDs_st

		quad_wts_st = int_face_helpers_st.quad_wts

		faces_to_basisL = int_face_helpers.faces_to_basisL
		faces_to_basisR = int_face_helpers.faces_to_basisR

		faces_to_basisL_st = int_face_helpers_st.faces_to_basisL
		faces_to_basisR_st = int_face_helpers_st.faces_to_basisR

		basis_valL = faces_to_basisL[faceL_id]
		basis_valR = faces_to_basisR[faceR_id]

		basis_valL_st = faces_to_basisL_st[faceL_id_st]
		basis_valR_st = faces_to_basisR_st[faceR_id_st]

		# Interpolate state at quad points
		UqL = helpers.evaluate_state(Uc_L, basis_valL_st) # [nf, nq_st, ns]
		UqR = helpers.evaluate_state(Uc_R, basis_valR_st) # [nf, nq_st, ns]

		normals_int_faces = int_face_helpers.normals_int_faces
		normals_int_faces = np.tile(normals_int_faces, 
				(normals_int_faces.shape[1], 1))

		# Allocate resL and resR (needed for operator splitting)
		resL = np.zeros_like(self.stepper.res)
		resR = np.zeros_like(self.stepper.res)

		if self.params["ConvFluxSwitch"] == True:
			# Compute numerical flux
			Fq = physics.get_conv_flux_numerical(UqL, UqR, normals_int_faces)
					# [nf, nq_st, ns]
			resL = solver_tools.calculate_inviscid_flux_boundary_integral(
					nq_t, basis_valL, quad_wts_st, Fq)
			resR = solver_tools.calculate_inviscid_flux_boundary_integral(
					nq_t, basis_valR, quad_wts_st, Fq)

		return resL, resR # [nif, nb, ns]

	def get_boundary_face_residual(self, bgroup, face_ID, Uc, resB):
		# Unpack
		mesh = self.mesh
		ndims = mesh.ndims
		physics = self.physics
		ns = physics.NUM_STATE_VARS
		bgroup_num = bgroup.number
		nq_t = self.elem_helpers_st.nq_tile_constant 
		nb_t = self.elem_helpers_st.nb_tile_constant 
		time_skip = self.elem_helpers_st.time_skip
		time_tile = self.elem_helpers_st.time_tile

		bface_helpers = self.bface_helpers
		bface_helpers_st = self.bface_helpers_st
		quad_wts_st = bface_helpers_st.quad_wts
		faces_to_xref_st = bface_helpers_st.faces_to_xref

		faces_to_basis = bface_helpers.faces_to_basis
		faces_to_basis_st = bface_helpers_st.faces_to_basis

		normals_bgroups = bface_helpers.normals_bgroups
		x_bgroups = bface_helpers.x_bgroups
		face_ID_st = bface_helpers_st.face_IDs_st[bgroup_num] 

		basis_val = faces_to_basis[face_ID]
		basis_val_st = faces_to_basis_st[face_ID_st]
		xref_st = faces_to_xref_st[face_ID_st]

		nq_st = quad_wts_st.shape[0]

		# Get array in physical time from ref time
		t, self.elem_helpers_st.basis_time = solver_tools.ref_to_phys_time(
				mesh, self.time, self.stepper.dt, xref_st[:, :, -1:],
				self.elem_helpers_st.basis_time)

		''' 
		Define time slice to make time arrays one dimensional cases even
		in the 2D case.
		'''

		# Build tiled time array. Removes unnecessary extra data.
		time_slice = slice(0, t.shape[1], time_skip)
		time = t[:, time_slice]
		time_t = np.tile(time,(1, time_tile, 1))

		# Interpolate state at quadrature points
		UqI = helpers.evaluate_state(Uc, basis_val_st) # [nbf, nq, ns]

		# Unpack normals and x on boundary faces
		normals = normals_bgroups[bgroup_num]
		x = x_bgroups[bgroup_num]

		# Tile normals and x to prepare for looping over elements in time
		normals = np.tile(normals, (time_tile, 1))
		x = np.tile(x, (time_tile, 1))

		# Get boundary state
		BC = physics.BCs[bgroup.name]
		nbf = UqI.shape[0]
		Fq = np.zeros([nbf, nq_st, ns])
		if self.params["ConvFluxSwitch"] == True:
			# Loop over time to apply BC at each temporal quadrature point
			for i in range(t.shape[1]):
				# Need time to be constant not an array to work with 
				# get_boundary_flux appropriately
				t_ = time_t[:, i][0, 0]
				x_ = x[:, i].reshape([nbf, 1, ndims])
				normals_ = normals[:, i].reshape([nbf, 1, ndims])

				Fq[:, i, :] = BC.get_boundary_flux(physics,
						UqI[:, i, :].reshape([nbf, 1, ns]),
						normals_, x_, t_).reshape([nbf, ns])

			resB = solver_tools.calculate_inviscid_flux_boundary_integral(
					nq_t, basis_val, quad_wts_st, Fq) # [nbf, nb, ns]

		return resB # [nbf, nb, ns]

	def flux_coefficients(self, dt, order, basis, Up):
		'''
		Calculates the polynomial coefficients for the flux functions in
		ADER-DG

		Inputs:
		-------
			dt: time step size
			order: solution order
			basis: basis object
			Up: coefficients of predicted solution [ne, nb_st, ns]

		Outputs:
		--------
			F: polynomial coefficients of the flux function
				[ne, nb_st, ns, ndims]
		'''
		# Unpack
		physics = self.physics
		mesh = self.mesh
		ns = physics.NUM_STATE_VARS
		ndims = physics.NDIMS
		params = self.params

		InterpolateFluxADER = params["InterpolateFluxADER"]

		elem_helpers = self.elem_helpers
		elem_helpers_st = self.elem_helpers_st
		djac_elems = elem_helpers.djac_elems

		nq_t = self.elem_helpers_st.nq_tile_constant 
		nb_t = self.elem_helpers_st.nb_tile_constant 

		# Allocate flux coefficients
		F = np.zeros([Up.shape[0], basis.get_num_basis_coeff(order),
				ns, ndims], dtype=Up.dtype)

		# Flux coefficient calc from interpolation or L2-projection
		if InterpolateFluxADER is True:
			# Calculate flux
			Fq = physics.get_conv_flux_interior(Up)[0]
			# Interpolate flux coefficient to nodes
			dg_tools.interpolate_to_nodes(Fq, F)

		else:
			# Unpack for L2-projection
			ader_helpers = self.ader_helpers
			basis_val_st = elem_helpers_st.basis_val
			quad_wts_st = elem_helpers_st.quad_wts
			quad_wts = elem_helpers.quad_wts
			quad_pts_st = elem_helpers_st.quad_pts
			quad_pts = elem_helpers.quad_pts
			nq_st = quad_wts_st.shape[0]
			nq = quad_wts.shape[0]
			iMM_elems = ader_helpers.iMM_elems

			# Interpolate state at quadrature points
			Uq = helpers.evaluate_state(Up, basis_val_st)

			# Evaluate the inviscid flux
			Fq = physics.get_conv_flux_interior(Uq)[0]

			# Project Fq to the space-time basis coefficients
			for d in range(ndims):
				solver_tools.L2_projection(mesh, iMM_elems, basis, quad_pts_st,
						quad_wts_st, np.tile(djac_elems, (nq_t, 1)),
						Fq[:, :, :, d], F[:, :, :, d])

		return F*dt/2.0 # [ne, nb_st, ns, ndims]

	def source_coefficients(self, dt, order, basis, Up):
		'''
		Calculates the polynomial coefficients for the source functions in
		ADER-DG

		Inputs:
		-------
			elem_ID: element index
			dt: time step size
			order: solution order
			basis: basis object
			Up: coefficients of predicted solution [ne, nb_st, ns]

		Outputs:
		--------
			S: polynomical coefficients of the flux function [ne, nb_st, ns]
		'''
		# Unpack
		mesh = self.mesh
		ndims = mesh.ndims
		physics = self.physics
		ns = physics.NUM_STATE_VARS
		params = self.params

		elem_helpers = self.elem_helpers
		elem_helpers_st = self.elem_helpers_st
		djac_elems = elem_helpers.djac_elems
		x_elems = elem_helpers.x_elems

		ader_helpers = self.ader_helpers
		x_elems_ader = ader_helpers.x_elems

		if params["InterpolateFluxADER"]:
			xnodes = basis.get_nodes(order)
			nb = xnodes.shape[0]

			# Get array in physical time from ref time
			t, elem_helpers_st.basis_time = solver_tools.ref_to_phys_time(
					mesh, self.time, self.stepper.dt,
					xnodes[:, -1:], elem_helpers_st.basis_time)

			# Evaluate the source term at the quadrature points
			Sq = np.zeros([Up.shape[0], t.shape[0], ns])
			S = np.zeros_like(Sq)
			Sq = physics.eval_source_terms(Up, x_elems_ader, t, Sq)

			# Interpolate source coefficient to nodes
			dg_tools.interpolate_to_nodes(Sq, S)
		else:
			# Unpack for L2-projection
			ader_helpers = self.ader_helpers
			basis_val_st = elem_helpers_st.basis_val
			nb_st = basis_val_st.shape[1]
			quad_wts_st = elem_helpers_st.quad_wts
			quad_wts = elem_helpers.quad_wts
			quad_pts_st = elem_helpers_st.quad_pts
			nq_st = quad_wts_st.shape[0]
			nq = quad_wts.shape[0]
			iMM_elems = ader_helpers.iMM_elems

			# Interpolate state at quadrature points
			Uq = helpers.evaluate_state(Up, basis_val_st)

			# Get array in physical time from ref time
			t = np.zeros([nq_st, ndims])
			t, elem_helpers_st.basis_time = solver_tools.ref_to_phys_time(
					mesh, self.time, self.stepper.dt,
					quad_pts_st[:, -1:], elem_helpers_st.basis_time)

			# Evaluate the source term at the quadrature points
			Sq = np.zeros_like(Uq)
			S = np.zeros([Uq.shape[0], nb_st, ns])
			Sq = physics.eval_source_terms(Uq, x_elems, t, Sq)
				# [ne, nq, ns, ndims]

			# Project Sq to the space-time basis coefficients
			solver_tools.L2_projection(mesh, iMM_elems, basis, quad_pts_st,
					quad_wts_st, np.tile(djac_elems, (nq, 1)), Sq, S)

		return S*dt/2.0 # [ne, nb_st, ns]
