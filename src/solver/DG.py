# ------------------------------------------------------------------------ #
#
#       File : src/numerics/solver/DG.py
#
#       Contains class definitions for the DG solver.
#      
# ------------------------------------------------------------------------ #
from abc import ABC, abstractmethod
import code
import copy
import numpy as np 
import time

import errors

import meshing.meshbase as mesh_defs
import meshing.tools as mesh_tools

import numerics.basis.tools as basis_tools
import numerics.helpers.helpers as helpers

import numerics.timestepping.tools as stepper_tools
import numerics.timestepping.stepper as stepper_defs

import solver.base as base
import solver.tools as solver_tools


global echeck
echeck = -1


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
	gbasis_val: numpy array
		stores the evaluated geometric basis function
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
		self.quad_pts = None
		self.quad_wts = None
		self.basis_val = None 
		self.basis_ref_grad = None 
		self.basis_phys_grad_elems = None 
		self.gbasis_val = None 
		self.jac_elems = None 
		self.ijac_elems = None 
		self.djac_elems = None 
		self.x_elems = None
		self.Uq = None 
		self.Fq = None 
		self.Sq = None 
		self.iMM_elems = np.zeros(0)
		self.vol_elems = None
		self.domain_vol = 0.

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
			self.quad_pts: precomputed quadrature points [nq, dim]
			self.quad_wts: precomputed quadrature weights [nq, 1]
		'''
		gbasis = mesh.gbasis
		quad_order = gbasis.get_quadrature_order(mesh, order, 
				physics = physics)
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
				reference element [nq, nb, dim]
			self.basis_phys_grad_elems: precomputed basis gradient for each 
				physical element [num_elems, nq, nb, dim]
			self.jac_elems: precomputed Jacobian for each element 
				[num_elems, nq, dim, dim]
			self.ijac_elems: precomputed inverse Jacobian for each element
				[num_elems, nq, dim, dim]
			self.djac_elems: precomputed determinant of the Jacobian for each
				element [num_elems, nq, 1]
			self.x_elems: precomputed coordinates of the quadrature points
				in physical space [num_elems, nq, dim]
		'''
		dim = mesh.dim 
		num_elems = mesh.num_elems 
		quad_pts = self.quad_pts 
		nq = quad_pts.shape[0]
		nb = basis.nb

		# allocate
		self.jac_elems = np.zeros([num_elems, nq, dim, dim])
		self.ijac_elems = np.zeros([num_elems, nq, dim, dim])
		self.djac_elems = np.zeros([num_elems, nq, 1])
		self.x_elems = np.zeros([num_elems, nq, dim])
		self.basis_phys_grad_elems = np.zeros([num_elems, nq, nb, dim])

		# basis data
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
			# Physical gradient
			basis.get_basis_val_grads(quad_pts, get_phys_grad=True, 
					ijac=ijac)
			self.basis_phys_grad_elems[elem_ID] = basis.basis_phys_grad  
				# [nq,nb,dim]

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
		dim = physics.DIM

		self.Uq = np.zeros([nq, ns]) 
		self.Fq = np.zeros([nq, ns, dim])
		self.Sq = np.zeros([nq, ns])  

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
				physics, basis)


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
		stores the evaluated basis function of left element neighbor
	faces_to_basisR: numpy array
		stores the evaluated basis function of right element neighbor
	normals_ifaces: numpy array
		normal vector array for each interior face
	UqL: numpy array
		solution vector evaluated at the quadrature points for left element 
		neighbor
	UqR: numpy array
		solution vector evaluated at the quadrature points for right element 
		neighbor
	Fq: numpy array
		flux vector evaluated at the face quadrature points

	Methods:
	--------
	get_gaussian_quadrature
		precomputes the quadrature points and weights for the given 
		quadrature type
	get_basis_and_geom_data
		precomputes the face's basis function, it's gradients,
		and normals
	alloc_other_arrays
		allocate the solution and flux vectors that are evaluated
		at the quadrature points
	compute_helpers
		call the functions to precompute the necessary helper data
	'''
	def __init__(self):
		self.quad_pts = None
		self.quad_wts = None
		self.faces_to_basisL = None
		self.faces_to_basisR = None
		self.normals_ifaces = None
		self.UqL = None 
		self.UqR = None 
		self.Fq = None 

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
			self.quad_pts: precomputed quadrature points [nq, dim]
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
			self.faces_to_basisL: precomputed basis value of left
				neighboring element [nfaces_per_elem, nq, nb]
			self.faces_to_basisR: precomputed basis value of right
				neighboring element [nfaces_per_elem, nq, nb]
			self.normals_ifaces: precomputed normal vectors at each 
				interior face [num_interior_faces, nq, dim]
		'''
		dim = mesh.dim
		quad_pts = self.quad_pts 
		nq = quad_pts.shape[0]
		nb = basis.nb
		nfaces_per_elem = mesh.gbasis.NFACES

		# allocate
		self.faces_to_basisL = np.zeros([nfaces_per_elem, nq, nb])
		self.faces_to_basisR = np.zeros([nfaces_per_elem, nq, nb])
		self.normals_ifaces = np.zeros([mesh.num_interior_faces, nq, dim])

		for f in range(nfaces_per_elem):
			# left
			basis.get_basis_face_val_grads(mesh, f, quad_pts, get_val=True)
			self.faces_to_basisL[f] = basis.basis_val
			# right
			basis.get_basis_face_val_grads(mesh, f, quad_pts[::-1], 
					get_val=True)
			self.faces_to_basisR[f] = basis.basis_val
		
		# normals
		i = 0
		for IFace in mesh.interior_faces:
			normals = mesh.gbasis.calculate_normals(mesh, IFace.elemL_ID, 
					IFace.faceL_ID, quad_pts)
			self.normals_ifaces[i] = normals
			i += 1

	def alloc_other_arrays(self, physics, basis, order):
		quad_pts = self.quad_pts 
		nq = quad_pts.shape[0]
		ns = physics.NUM_STATE_VARS

		self.UqL = np.zeros([nq, ns])
		self.UqR = np.zeros([nq, ns])
		self.Fq = np.zeros([nq, ns])

	def compute_helpers(self, mesh, physics, basis, order):
		self.get_gaussian_quadrature(mesh, physics, basis, order)
		self.get_basis_and_geom_data(mesh, basis, order)
		self.alloc_other_arrays(physics, basis, order)


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
		stores the evaluated basis function of interior element
	faces_to_xref: numpy array
		stores the element reference nodes at boundary face
	normals_bfgroups: numpy array
		normal vector array for each boundary face
	x: numpy array
		coordinates of the quadrature points in physical space at the 
		boundary face
	UqI: numpy array
		solution vector evaluated at the quadrature points for the interior 
		element 
	UqB: numpy array
		solution vector evaluated at the quadrature points for the boundary 
		element 
	Fq: numpy array
		flux vector evaluated at the face quadrature points

	Methods:
	--------
	get_basis_and_geom_data
		precomputes the boundary face's basis function, it's gradients,
		and normals
	alloc_other_arrays
		allocate the solution and flux vectors that are evaluated
		at the quadrature points
	compute_helpers
		call the functions to precompute the necessary helper data
	'''
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
		'''
		Precomputes the basis and geometric data for each boundary face

		Inputs:
		-------
			mesh: mesh object
			basis: basis object
			order: solution order

		Outputs:
		--------
			self.faces_to_basis: precomputed basis value of interior
				neighboring element [nfaces_per_elem, nq, nb]
			self.faces_to_xref: precomputed element reference nodes at 
				the boundary face
			self.normals_bfgroups: precomputed normal vectors at each 
				boundary face [num_boundary_faces, nq, dim]
			self.x_bfgroups: precomputed physical coordinates of the 
				quadrature points [num_boundary_faces, nq, dim]
		'''
		dim = mesh.dim
		quad_pts = self.quad_pts 
		nq = quad_pts.shape[0]
		nb = basis.nb
		nfaces_per_elem = mesh.gbasis.NFACES

		# allocate
		self.faces_to_basis = np.zeros([nfaces_per_elem, nq, nb])
		self.faces_to_xref = np.zeros([nfaces_per_elem, nq, dim])
		self.normals_bfgroups = []
		self.x_bfgroups = []

		for f in range(nfaces_per_elem):
			# interior
			self.faces_to_xref[f] = basis.get_elem_ref_from_face_ref(f, 
					quad_pts)
			basis.get_basis_face_val_grads(mesh, f, quad_pts, get_val=True)
			self.faces_to_basis[f] = basis.basis_val

		i = 0
		for BFG in mesh.boundary_groups.values():
			self.normals_bfgroups.append(np.zeros([BFG.num_boundary_faces, 
					nq, dim]))
			self.x_bfgroups.append(np.zeros([BFG.num_boundary_faces, 
					nq, dim]))
			normal_bfgroup = self.normals_bfgroups[i]
			x_bfgroup = self.x_bfgroups[i]
			
			# normals
			j = 0
			for boundary_face in BFG.boundary_faces:

				nvec = mesh.gbasis.calculate_normals(mesh, 
						boundary_face.elem_ID, 
						boundary_face.face_ID, quad_pts)
				normal_bfgroup[j] = nvec

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

	def compute_helpers(self, mesh, physics, basis, order):
		self.get_gaussian_quadrature(mesh, physics, basis, order)
		self.get_basis_and_geom_data(mesh, basis, order)
		self.alloc_other_arrays(physics, basis, order)


class DG(base.SolverBase):
	'''
    DG inherits attributes and methods from the SolverBase class.
    See SolverBase for detailed comments of attributes and methods.

    Additional methods and attributes are commented below.
	'''
	def __init__(self, params, physics, mesh):
		super().__init__(params, physics, mesh)

		self.Stepper = stepper_tools.set_stepper(params, physics.U)
		stepper_tools.set_time_stepping_approach(self.Stepper, params)
	
		# Check validity of parameters
		self.check_compatibility()
		
		# Precompute helpers
		self.precompute_matrix_helpers()
		if self.limiter is not None:
			self.limiter.precompute_helpers(self)

		physics.conv_flux_fcn.alloc_helpers(
				np.zeros([self.iface_helpers.quad_wts.shape[0], 
				physics.NUM_STATE_VARS]))

		if params["RestartFile"] is None:
			self.init_state_from_fcn()
		
	def precompute_matrix_helpers(self):

		mesh = self.mesh 
		physics = self.physics
		basis = self.basis

		self.elem_helpers = ElemHelpers()
		self.elem_helpers.compute_helpers(mesh, physics, basis, 
				physics.order)
		self.iface_helpers = InteriorFaceHelpers()
		self.iface_helpers.compute_helpers(mesh, physics, basis, 
				physics.order)
		self.bface_helpers = BoundaryFaceHelpers()
		self.bface_helpers.compute_helpers(mesh, physics, basis, 
				physics.order)

	def get_element_residual(self, elem_ID, Up, R_elem):
		physics = self.physics
		ns = physics.NUM_STATE_VARS
		dim = physics.DIM
		elem_helpers = self.elem_helpers
		basis_val = elem_helpers.basis_val
		quad_wts = elem_helpers.quad_wts

		x_elems = elem_helpers.x_elems
		nq = quad_wts.shape[0]
		x = x_elems[elem_ID]

		# interpolate state and gradient at quad points
		Uq = helpers.evaluate_state(Up, basis_val, 
				skip_interp=self.basis.skip_interp)

		if self.params["ConvFluxSwitch"] == True:
			# evaluate the inviscid flux integral
			Fq = physics.get_conv_flux_interior(Uq) # [nq, ns, dim]
			R_elem += solver_tools.calculate_inviscid_flux_volume_integral(
					self, elem_helpers, elem_ID, Fq)

		if self.params["SourceSwitch"] == True:
			# evaluate the source term integral
			Sq = elem_helpers.Sq
			# eval_source_terms is an additive function so source needs to be 
			# initialized to zero for each time step
			Sq[:] = 0.
			Sq = physics.eval_source_terms(Uq, x, self.time, Sq) # [nq, ns]

			R_elem += solver_tools.calculate_source_term_integral(elem_helpers, 
					elem_ID, Sq)

		if elem_ID == echeck:
			code.interact(local=locals())

		return R_elem

	def get_interior_face_residual(self, iiface, UpL, UpR, R_L, R_R):
		mesh = self.mesh
		physics = self.physics
		IFace = mesh.interior_faces[iiface]
		elemL = IFace.elemL_ID
		elemR = IFace.elemR_ID
		faceL_ID = IFace.faceL_ID
		faceR_ID = IFace.faceR_ID

		iface_helpers = self.iface_helpers
		quad_pts = iface_helpers.quad_pts
		quad_wts = iface_helpers.quad_wts
		faces_to_basisL = iface_helpers.faces_to_basisL
		faces_to_basisR = iface_helpers.faces_to_basisR
		normals_ifaces = iface_helpers.normals_ifaces
		UqL = iface_helpers.UqL
		UqR = iface_helpers.UqR
		Fq = iface_helpers.Fq

		nq = quad_wts.shape[0]
		basis_valL = faces_to_basisL[faceL_ID]
		basis_valR = faces_to_basisR[faceR_ID]

		# interpolate state and gradient at quad points
		UqL = helpers.evaluate_state(UpL, basis_valL)
		UqR = helpers.evaluate_state(UpR, basis_valR)

		normals = normals_ifaces[iiface]
		
		if self.params["ConvFluxSwitch"] == True:
			Fq = physics.get_conv_flux_numerical(UqL, UqR, normals) # [nq,ns]

			R_L -= solver_tools.calculate_inviscid_flux_boundary_integral(
					basis_valL, quad_wts, Fq)
			R_R += solver_tools.calculate_inviscid_flux_boundary_integral(
					basis_valR, quad_wts, Fq)

		if elemL == echeck or elemR == echeck:
			if elemL == echeck: print("Left!")
			else: print("Right!")
			code.interact(local=locals())

		return R_L, R_R

	def get_boundary_face_residual(self, BFG, bface_ID, U, R_B):
		mesh = self.mesh
		physics = self.physics
		bgroup_num = BFG.number
		boundary_face = BFG.boundary_faces[bface_ID]
		elem_ID = boundary_face.elem_ID
		face_ID = boundary_face.face_ID

		bface_helpers = self.bface_helpers
		quad_pts = bface_helpers.quad_pts
		quad_wts = bface_helpers.quad_wts
		faces_to_basis = bface_helpers.faces_to_basis
		normals_bfgroups = bface_helpers.normals_bfgroups
		x_bfgroups = bface_helpers.x_bfgroups
		UqI = bface_helpers.UqI
		UqB = bface_helpers.UqB
		Fq = bface_helpers.Fq

		nq = quad_wts.shape[0]
		basis_val = faces_to_basis[face_ID]

		# interpolate state and gradient at quad points
		UqI = helpers.evaluate_state(U, basis_val)

		normals = normals_bfgroups[bgroup_num][bface_ID]
		x = x_bfgroups[bgroup_num][bface_ID]

		# get boundary state
		BC = physics.BCs[BFG.name]

		if self.params["ConvFluxSwitch"] == True:

			Fq = BC.get_boundary_flux(physics, UqI, normals, x, self.time)

			R_B -= solver_tools.calculate_inviscid_flux_boundary_integral(
					basis_val, quad_wts, Fq)

		if elem_ID == echeck:
			code.interact(local=locals())

		return R_B