# ------------------------------------------------------------------------ #
#
#       File : src/solver/DG.py
#
#       Contains class definitions for the DG solver.
#
# ------------------------------------------------------------------------ #
from abc import ABC, abstractmethod
import numpy as np
import time

import errors

import meshing.meshbase as mesh_defs
import meshing.tools as mesh_tools

import numerics.adaptation.adapter as adapter_defs
import numerics.basis.tools as basis_tools
import numerics.helpers.helpers as helpers

import numerics.timestepping.tools as stepper_tools
import numerics.timestepping.stepper as stepper_defs

import solver.base as base
import solver.tools as solver_tools


class ElemHelpers(object):
	'''
	The ElemHelpers class contains the methods and attributes that are
	accessed prior to the main solver temporal loop. They are used to
	precompute attributes for the element volumes.

	Attributes:
	-----------
	quad_pts: numpy array
		coordinates for the quadrature point evaluations
	quad_wts: numpy array
		values for the weights of each quadrature point
	basis_val: numpy array
		stores the evaluated basis function
	basis_ref_grad: numpy array
		stores the evaluated basis function's gradient for the reference
		element
	basis_phys_grad_elems: numpy array
		stores the evaluated basis function's gradient for each individual
		physical element
	jac_elems: numpy array
		stores the evaluated geometric Jacobian for each element
	ijac_elems: numpy array
		stores the evaluated inverse of the geometric Jacobian for each
		element
	djac_elems: numpy array
		stores the evaluated determinant of the geometric Jacobian for
		each element
	x_elems: numpy array
		physical coordinates of quadrature points for each element
	Uq: numpy array
		solution state vector evaluated at the quadrature points
	Fq: numpy array
		flux vector evaluated at the quadrature points
	Sq: numpy array
		source vector evaluated at the quadrature points
	iMM_elems: numpy array
		stores the inverse mass matrix for each element
	vol_elems: numpy array
		stores the volume of each element
	domain_vol: float
		stores the total volume of the domain

	Methods:
	--------
	get_gaussian_quadrature
		precomputes the quadrature points and weights for the given
		quadrature type
	get_basis_and_geom_data
		precomputes the element's basis function, its gradients,
		geometric Jacobian info, and volume
	alloc_other_arrays
		allocate the solution, flux, and source vectors that are evaluated
		at the quadrature points
	compute_helpers
		call the functions to precompute the necessary helper data
	'''
	def __init__(self):
		self.quad_pts = np.zeros(0)
		self.quad_wts = np.zeros(0)
		self.basis_val = np.zeros(0)
		self.basis_ref_grad = np.zeros(0)
		self.basis_phys_grad_elems = np.zeros(0)
		self.jac_elems = np.zeros(0)
		self.ijac_elems = np.zeros(0)
		self.djac_elems = np.zeros(0)
		self.x_elems = np.zeros(0)
		self.Uq = np.zeros(0)
		self.Fq = np.zeros(0)
		self.Sq = np.zeros(0)
		self.iMM_elems = np.zeros(0)
		self.vol_elems = np.zeros(0)
		self.domain_vol = 0.
		self.need_phys_grad = True

	def get_gaussian_quadrature(self, mesh, physics, basis, order):
		'''
		Precomputes the quadrature points and weights given the computed
		quadrature order

		Inputs:
		-------
			mesh: mesh object
			physics: physics object
			basis: basis object
			order: solution order

		Outputs:
		--------
			self.quad_pts: precomputed quadrature points [nq, ndims]
			self.quad_wts: precomputed quadrature weights [nq, 1]
		'''
		gbasis = mesh.gbasis
		quad_order = gbasis.get_quadrature_order(mesh, order,
				physics=physics)
		self.quad_pts, self.quad_wts = basis.get_quadrature_data(quad_order)

	def get_basis_and_geom_data(self, mesh, basis, order):
		'''
		Precomputes the basis and geometric data for each element

		Inputs:
		-------
			mesh: mesh object
			basis: basis object
			order: solution order

		Outputs:
		--------
			self.basis_val: precomputed basis value [nq, nb]
			self.basis_ref_grad: precomputed basis gradient for the
				reference element [nq, nb, ndims]
			self.basis_phys_grad_elems: precomputed basis gradient for each
				physical element [num_elems, nq, nb, ndims]
			self.jac_elems: precomputed Jacobian for each element
				[num_elems, nq, ndims, ndims]
			self.ijac_elems: precomputed inverse Jacobian for each element
				[num_elems, nq, ndims, ndims]
			self.djac_elems: precomputed determinant of the Jacobian for each
				element [num_elems, nq, 1]
			self.x_elems: precomputed coordinates of the quadrature points
				in physical space [num_elems, nq, ndims]
		'''
		ndims = mesh.ndims
		num_elems = mesh.num_elems
		quad_pts = self.quad_pts
		nq = quad_pts.shape[0]
		nb = basis.nb

		# Allocate
		self.jac_elems = np.zeros([num_elems, nq, ndims, ndims])
		self.ijac_elems = np.zeros([num_elems, nq, ndims, ndims])
		self.djac_elems = np.zeros([num_elems, nq, 1])
		self.x_elems = np.zeros([num_elems, nq, ndims])
		self.basis_phys_grad_elems = np.zeros([num_elems, nq, nb, ndims])

		# Basis data
		basis.get_basis_val_grads(self.quad_pts, get_val=True,
				get_ref_grad=True)

		self.basis_val = basis.basis_val
		self.basis_ref_grad = basis.basis_ref_grad

		for elem_ID in range(mesh.num_elems):
			# Jacobian
			djac, jac, ijac = basis_tools.element_jacobian(mesh, elem_ID,
					quad_pts, get_djac=True, get_jac=True, get_ijac=True)
			# Store
			self.jac_elems[elem_ID] = jac
			self.ijac_elems[elem_ID] = ijac
			self.djac_elems[elem_ID] = djac

			# Physical coordinates of quadrature points
			x = mesh_tools.ref_to_phys(mesh, elem_ID, quad_pts)
			# Store
			self.x_elems[elem_ID] = x

			if self.need_phys_grad:
				# Physical gradient
				basis.get_basis_val_grads(quad_pts, get_phys_grad=True,
						ijac=ijac)
				self.basis_phys_grad_elems[elem_ID] = basis.basis_phys_grad
					# [nq, nb, ndims]

		# Volumes
		self.vol_elems, self.domain_vol = mesh_tools.element_volumes(mesh)

	def alloc_other_arrays(self, physics, basis, order):
		'''
		Allocates the solution, flux, and source vectors that are evaluated
		at the quadrature points

		Inputs:
		-------
			physics: physics object
			basis: basis object
			order: solution order
		'''
		quad_pts = self.quad_pts
		nq = quad_pts.shape[0]
		nb = basis.nb
		ns = physics.NUM_STATE_VARS
		ndims = physics.NDIMS
		nelem = self.vol_elems.shape[0]

		self.Uq = np.zeros([nelem, nq, ns])
		self.Fq = np.zeros([nelem, nq, ns, ndims])
		self.Sq = np.zeros([nelem, nq, ns])

	def compute_helpers(self, mesh, physics, basis, order):
		'''
		Calls the functions to precompute the necessary helper data

		Inputs:
		-------
			mesh: mesh object
			physics: physics object
			basis: basis object
			order: solution order

		Outputs:
		--------
			self.iMM_elems: precomputed inverse mass matrix for each element
				[mesh.num_elems, nb, nb]
		'''
		self.get_gaussian_quadrature(mesh, physics, basis, order)
		self.get_basis_and_geom_data(mesh, basis, order)
		self.alloc_other_arrays(physics, basis, order)
		self.iMM_elems = basis_tools.get_inv_mass_matrices(mesh,
				basis, order)


class InteriorFaceHelpers(ElemHelpers):
	'''
	The InteriorFaceHelpers class contains the methods and attributes that
	are accessed prior to the main solver temporal loop. They are used to
	precompute attributes for the interior faces in the domain.

	InteriorFaceHelpers inherits attributes from the ElemHelpers parent
	class. See ElemHelpers class for additional comments of methods.

	Attributes:
	-----------
	quad_pts: numpy array
		coordinates for the quadrature point evaluations
	quad_wts: numpy array
		values for the weights of each quadrature point
	faces_to_basisL: numpy array
		basis values evaluated at quadrature points of each face for
		left element
	faces_to_basisR: numpy array
		basis values evaluated at quadrature points of each face for
		right element
	normals_int_faces: numpy array
		normal vector array for each interior face
	UqL: numpy array
		solution vector evaluated at the face quadrature points for left
		element
	UqR: numpy array
		solution vector evaluated at the face quadrature points for right
		element
	Fq: numpy array
		flux vector evaluated at the face quadrature points

	Methods:
	--------
	get_gaussian_quadrature
		precomputes the quadrature points and weights for the given
		quadrature type
	get_basis_and_geom_data
		precomputes the face's basis function, its gradients,
		and normals
	alloc_other_arrays
		allocate the solution and flux vectors that are evaluated
		at the quadrature points
	compute_helpers
		call the functions to precompute the necessary helper data
	'''
	def __init__(self):
		self.quad_pts = np.zeros(0)
		self.quad_wts = np.zeros(0)
		self.faces_to_basisL = np.zeros(0)
		self.faces_to_basisR = np.zeros(0)
		self.normals_int_faces = np.zeros(0)
		self.UqL = np.zeros(0)
		self.UqR = np.zeros(0)
		self.Fq = np.zeros(0)
		self.elemL_IDs = np.empty(0, dtype=int)
		self.elemR_IDs = np.empty(0, dtype=int)
		self.faceL_IDs = np.empty(0, dtype=int)
		self.faceR_IDs = np.empty(0, dtype=int)

	def get_gaussian_quadrature(self, mesh, physics, basis, order):
		'''
		Precomputes the quadrature points and weights given the computed
		quadrature order

		Inputs:
		-------
			mesh: mesh object
			physics: physics object
			basis: basis object
			order: solution order

		Outputs:
		--------
			self.quad_pts: precomputed quadrature points [nq, ndims]
			self.quad_wts: precomputed quadrature weights [nq, 1]
		'''
		gbasis = mesh.gbasis
		quad_order = gbasis.FACE_SHAPE.get_quadrature_order(mesh,
				order, physics=physics)
		self.quad_pts, self.quad_wts = \
				basis.FACE_SHAPE.get_quadrature_data(quad_order)

	def get_basis_and_geom_data(self, mesh, basis, order):
		'''
		Precomputes the basis and geometric data for each interior face

		Inputs:
		-------
			mesh: mesh object
			basis: basis object
			order: solution order

		Outputs:
		--------
			self.faces_to_basisL: basis values evaluated at quadrature
				points of each face for left element
				[nfaces_per_elem, nq, nb]
			self.faces_to_basisR: basis values evaluated at quadrature
				points of each face for right element
				[nfaces_per_elem, nq, nb]
			self.normals_int_faces: precomputed normal vectors at each
				interior face [num_interior_faces, nq, ndims]
		'''
		ndims = mesh.ndims
		quad_pts = self.quad_pts
		nq = quad_pts.shape[0]
		nb = basis.nb
		nfaces_per_elem = basis.NFACES

		# Allocate
		self.faces_to_basisL = np.zeros([nfaces_per_elem, nq, nb])
		self.faces_to_basisR = np.zeros([nfaces_per_elem, nq, nb])
		self.normals_int_faces = np.zeros([mesh.num_interior_faces, nq,
				ndims])

		# Get values on each face (from both left and right perspectives)
		for face_ID in range(nfaces_per_elem):
			# Left
			basis.get_basis_face_val_grads(mesh, face_ID, quad_pts,
					get_val=True)
			self.faces_to_basisL[face_ID] = basis.basis_val
			# Right
			basis.get_basis_face_val_grads(mesh, face_ID, quad_pts[::-1],
					get_val=True)
			self.faces_to_basisR[face_ID] = basis.basis_val

		# Normals
		i = 0
		for interior_face in mesh.interior_faces:
			normals = mesh.gbasis.calculate_normals(mesh,
					interior_face.elemL_ID, interior_face.faceL_ID, quad_pts)
			self.normals_int_faces[i] = normals
			i += 1

	def alloc_other_arrays(self, physics, basis, order):
		quad_pts = self.quad_pts
		nq = quad_pts.shape[0]
		ns = physics.NUM_STATE_VARS

		self.UqL = np.zeros([nq, ns])
		self.UqR = np.zeros([nq, ns])
		self.Fq = np.zeros([nq, ns])

	def store_neighbor_info(self, mesh):
		'''
		Store the element and face IDs on the left and right of each face.

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
		self.elemL_IDs = np.empty(mesh.num_interior_faces, dtype=int)
		self.elemR_IDs = np.empty(mesh.num_interior_faces, dtype=int)
		self.faceL_IDs = np.empty(mesh.num_interior_faces, dtype=int)
		self.faceR_IDs = np.empty(mesh.num_interior_faces, dtype=int)
		for face_ID in range(mesh.num_interior_faces):
			int_face = mesh.interior_faces[face_ID]
			self.elemL_IDs[face_ID] = int_face.elemL_ID
			self.elemR_IDs[face_ID] = int_face.elemR_ID
			self.faceL_IDs[face_ID] = int_face.faceL_ID
			self.faceR_IDs[face_ID] = int_face.faceR_ID

	def compute_helpers(self, mesh, physics, basis, order):
		self.get_gaussian_quadrature(mesh, physics, basis, order)
		self.get_basis_and_geom_data(mesh, basis, order)
		self.alloc_other_arrays(physics, basis, order)
		self.store_neighbor_info(mesh)


class BoundaryFaceHelpers(InteriorFaceHelpers):
	'''
	The BoundaryFaceHelpers class contains the methods and attributes that
	are accessed prior to the main solver temporal loop. They are used to
	precompute attributes for the boundary faces in the domain.

	BoundaryFaceHelpers inherits attributes from the InteriorFaceHelpers
	parent class. See InteriorFaceHelpers class for additional comments of
	methods.

	Attributes:
	-----------
	quad_pts: numpy array
		coordinates for the quadrature point evaluations
	quad_wts: numpy array
		values for the weights of each quadrature point
	faces_to_basis: numpy array
		basis values evaluated at quadrature points of each face
	faces_to_xref: numpy array
		coordinates of quadrature points of each face converted to
		element reference space
	normals_bgroups: numpy array
		normal vector array for each boundary face
	x: numpy array
		coordinates of the quadrature points in physical space at the
		boundary face
	UqI: numpy array
		values of interior state at the quadrature points
	UqB: numpy array
		values of boundary (exterior) state at the quadrature points
	Fq: numpy array
		flux vector evaluated at the face quadrature points

	Methods:
	--------
	get_basis_and_geom_data
		precomputes the boundary face's basis function, its gradients,
		and normals
	alloc_other_arrays
		allocate the solution and flux vectors that are evaluated
		at the quadrature points
	compute_helpers
		call the functions to precompute the necessary helper data
	'''
	def __init__(self):
		self.quad_pts = np.zeros(0)
		self.quad_wts = np.zeros(0)
		self.faces_to_basis = np.zeros(0)
		self.faces_to_xref = np.zeros(0)
		self.normals_bgroups = []
		self.x_bgroups = []
		self.UqI = np.zeros(0)
		self.UqB = np.zeros(0)
		self.Fq = np.zeros(0)
		self.elem_IDs = []
		self.face_IDs = []

	def get_basis_and_geom_data(self, mesh, basis, order):
		'''
		Precomputes the basis and geometric data for each boundary face

		Inputs:
		-------
			mesh: mesh object
			basis: basis object
			order: solution order

		Outputs:
		--------
			self.faces_to_basis: basis values evaluated at quadrature points
				of each face [nfaces_per_elem, nq, nb]
			self.faces_to_xref: coordinates of quadrature points of each
				face converted to element reference space
				[nfaces_per_elem, nq, ndims]
			self.normals_bgroups: precomputed normal vectors at each
				boundary face [num_boundary_faces, nq, ndims]
			self.x_bgroups: precomputed physical coordinates of the
				quadrature points [num_boundary_faces, nq, ndims]
		'''
		ndims = mesh.ndims
		quad_pts = self.quad_pts
		nq = quad_pts.shape[0]
		nb = basis.nb
		nfaces_per_elem = basis.NFACES

		# Allocate
		self.faces_to_basis = np.zeros([nfaces_per_elem, nq, nb])
		self.faces_to_xref = np.zeros([nfaces_per_elem, nq, basis.NDIMS])

		# Get values on each face (from interior perspective)
		for face_ID in range(nfaces_per_elem):
			self.faces_to_xref[face_ID] = basis.get_elem_ref_from_face_ref(
					face_ID, quad_pts)
			basis.get_basis_face_val_grads(mesh, face_ID, quad_pts,
					get_val=True)
			self.faces_to_basis[face_ID] = basis.basis_val

		# Get boundary information
		i = 0
		for bgroup in mesh.boundary_groups.values():
			self.normals_bgroups.append(np.zeros([bgroup.num_boundary_faces,
					nq, ndims]))
			self.x_bgroups.append(np.zeros([bgroup.num_boundary_faces,
					nq, ndims]))
			normal_bgroup = self.normals_bgroups[i]
			x_bgroup = self.x_bgroups[i]

			j = 0
			for boundary_face in bgroup.boundary_faces:
				# Normals
				normals = mesh.gbasis.calculate_normals(mesh,
						boundary_face.elem_ID,
						boundary_face.face_ID, quad_pts)
				normal_bgroup[j] = normals

				# Physical coordinates of quadrature points
				x = mesh_tools.ref_to_phys(mesh, boundary_face.elem_ID,
						self.faces_to_xref[boundary_face.face_ID])
				# Store
				x_bgroup[j] = x

				# Increment
				j += 1
			i += 1

	def alloc_other_arrays(self, physics, basis, order):
		quad_pts = self.quad_pts
		nq = quad_pts.shape[0]
		ns = physics.NUM_STATE_VARS

		self.UqI = np.zeros([nq, ns])
		self.UqB = np.zeros([nq, ns])
		self.Fq = np.zeros([nq, ns])

	def store_neighbor_info(self, mesh):
		'''
		Store the element and face IDs of the neighbors of each boundary
		face.

		Inputs:
		-------
			mesh: mesh object

		Outputs:
		--------
			self.elem_IDs: List containing arrays of element IDs of boundary
			face neighbors for each boundary group
			[num_boundary_groups][num_interior_faces]
			self.face_IDs: List containing arrays of face IDs of boundary
			face neighbors for each boundary group
			[num_boundary_groups][num_interior_faces]
		'''
		# Loop through boundary groups
		for bgroup in mesh.boundary_groups.values():
			bgroup_elem_IDs = np.empty(bgroup.num_boundary_faces, dtype=int)
			bgroup_face_IDs = np.empty(bgroup.num_boundary_faces, dtype=int)
			for bface_ID in range(bgroup.num_boundary_faces):
				boundary_face = bgroup.boundary_faces[bface_ID]
				bgroup_elem_IDs[bface_ID] = boundary_face.elem_ID
				bgroup_face_IDs[bface_ID] = boundary_face.face_ID
			self.elem_IDs.append(bgroup_elem_IDs)
			self.face_IDs.append(bgroup_face_IDs)

	def compute_helpers(self, mesh, physics, basis, order):
		self.get_gaussian_quadrature(mesh, physics, basis, order)
		self.get_basis_and_geom_data(mesh, basis, order)
		self.alloc_other_arrays(physics, basis, order)
		self.store_neighbor_info(mesh)


class DG(base.SolverBase):
	'''
	DG inherits attributes and methods from the SolverBase class.
	See SolverBase for detailed comments of attributes and methods.

	Additional methods and attributes are commented below.
	'''
	def __init__(self, params, physics, mesh):
		super().__init__(params, physics, mesh)

		self.stepper = stepper_tools.set_stepper(params, self.state_coeffs)
		stepper_tools.set_time_stepping_approach(self.stepper, params)

		# Precompute helpers
		self.precompute_matrix_helpers()

		if self.limiters:
			for limiter in self.limiters:
				limiter.precompute_helpers(self)

		physics.conv_flux_fcn.alloc_helpers(
				np.zeros([mesh.num_interior_faces,
				self.int_face_helpers.quad_wts.shape[0],
				physics.NUM_STATE_VARS]))

		# Initial condition
		if params["RestartFile"] is None:
			self.init_state_from_fcn()

		# Initialize adapter
		self.adapter = adapter_defs.Adapter(self)

	def precompute_matrix_helpers(self):
		mesh = self.mesh
		physics = self.physics
		basis = self.basis

		self.elem_helpers = ElemHelpers()
		self.elem_helpers.compute_helpers(mesh, physics, basis,
				self.order)
		self.int_face_helpers = InteriorFaceHelpers()
		self.int_face_helpers.compute_helpers(mesh, physics, basis,
				self.order)
		self.bface_helpers = BoundaryFaceHelpers()
		self.bface_helpers.compute_helpers(mesh, physics, basis,
				self.order)

	def get_element_residual(self, Uc, res_elem):
		# Unpack
		physics = self.physics
		ns = physics.NUM_STATE_VARS
		ndims = physics.NDIMS
		elem_helpers = self.elem_helpers
		basis_val = elem_helpers.basis_val
		quad_wts = elem_helpers.quad_wts

		x_elems = elem_helpers.x_elems
		nq = quad_wts.shape[0]

		# Interpolate state at quad points
		Uq = helpers.evaluate_state(Uc, basis_val,
				skip_interp=self.basis.skip_interp) # [ne, nq, ns]
		if self.verbose:
			# Get min and max of state variables for reporting
			self.get_min_max_state(Uq)

		if self.params["ConvFluxSwitch"] == True:
			# Evaluate the inviscid flux integral
			Fq = physics.get_conv_flux_interior(Uq)[0] # [ne, nq, ns, ndims]

			res_elem += solver_tools.calculate_inviscid_flux_volume_integral(
					self, elem_helpers, Fq) # [ne, nb, ns]

		if self.params["SourceSwitch"] == True:
			# Evaluate the source term integral
			# eval_source_terms is an additive function so source needs to be
			# initialized to zero for each time step
			Sq = np.zeros_like(Uq) # [ne, nq, ns]
			Sq = physics.eval_source_terms(Uq, x_elems, self.time, Sq)
					# [ne, nq, ns]

			res_elem += solver_tools.calculate_source_term_integral(
					elem_helpers, Sq) # [ne, nb, ns]

		return res_elem # [ne, nb, ns]

	def get_interior_face_residual(self, faceL_IDs, faceR_IDs, UcL, UcR):
		# Unpack
		mesh = self.mesh
		physics = self.physics

		int_face_helpers = self.int_face_helpers
		quad_wts = int_face_helpers.quad_wts
		faces_to_basisL = int_face_helpers.faces_to_basisL
		faces_to_basisR = int_face_helpers.faces_to_basisR
		normals_int_faces = int_face_helpers.normals_int_faces
				# [nf, nq, ndims]

		ns = physics.NUM_STATE_VARS
		nq = quad_wts.shape[0]

		# Interpolate state at quad points
		UqL = helpers.evaluate_state(UcL, faces_to_basisL[faceL_IDs])
				# [nf, nq, ns]
		UqR = helpers.evaluate_state(UcR, faces_to_basisR[faceR_IDs])
				# [nf, nq, ns]

		# Allocate resL and resR (needed for operator splitting)
		nifL = self.int_face_helpers.elemL_IDs.shape[0]
		nifR = self.int_face_helpers.elemR_IDs.shape[0]
		resL = np.zeros([nifL, nq, ns])
		resR = np.zeros([nifR, nq, ns])

		if self.params["ConvFluxSwitch"] == True:
			# Compute numerical flux
			Fq = physics.get_conv_flux_numerical(UqL, UqR, normals_int_faces)
					# [nf, nq, ns]

			# Compute contribution to left and right element residuals
			resL = solver_tools.calculate_inviscid_flux_boundary_integral(
					faces_to_basisL[faceL_IDs], quad_wts, Fq)
			resR = solver_tools.calculate_inviscid_flux_boundary_integral(
					faces_to_basisR[faceR_IDs], quad_wts, Fq)

		return resL, resR # [nif, nb, ns]

	def get_boundary_face_residual(self, bgroup, face_IDs, Uc, resB):
		# unpack
		mesh = self.mesh
		physics = self.physics
		bgroup_num = bgroup.number

		bface_helpers = self.bface_helpers
		quad_wts = bface_helpers.quad_wts
		normals_bgroups = bface_helpers.normals_bgroups
		x_bgroups = bface_helpers.x_bgroups

		basis_val = bface_helpers.faces_to_basis[face_IDs] # [nbf, nq, nb]

		# Interpolate state at quad points
		UqI = helpers.evaluate_state(Uc, basis_val) # [nbf, nq, ns]

		normals = normals_bgroups[bgroup_num] # [nbf, nq, ndims]
		x = x_bgroups[bgroup_num] # [nbf, nq, ndims]
		BC = physics.BCs[bgroup.name]

		# Interpolate state at quadrature points
		UqI = helpers.evaluate_state(Uc, basis_val)

		if self.params["ConvFluxSwitch"] == True:
			# Compute boundary flux
			Fq = BC.get_boundary_flux(physics, UqI, normals, x, self.time)
			# Compute contribution to adjacent element residual
			resB = solver_tools.calculate_inviscid_flux_boundary_integral(
					basis_val, quad_wts, Fq)

		return resB
