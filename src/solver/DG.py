# ------------------------------------------------------------------------ #
#
#       quail: A lightweight discontinuous Galerkin code for
#              teaching and prototyping
#		<https://github.com/IhmeGroup/quail>
#       
#		Copyright (C) 2020-2021
#
#       This program is distributed under the terms of the GNU
#		General Public License v3.0. You should have received a copy
#       of the GNU General Public License along with this program.  
#		If not, see <https://www.gnu.org/licenses/>.
#
# ------------------------------------------------------------------------ #

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
	normals_elems: numpy array
		stores the normals of each face of each element
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
		self.normals_elems = np.zeros(0)
		self.domain_vol = 0.
		self.need_phys_grad = True

	def get_gaussian_quadrature(self, mesh, physics, basis, order):
		'''
		Precomputes the quadrature points and weights given the computed
		quadrature order. Also computes them for the faces of each element.

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
			self.face_quad_pts: precomputed quadrature points at faces [nqf, ndims]
		'''
		gbasis = mesh.gbasis
		quad_order = gbasis.get_quadrature_order(mesh, order,
				physics=physics)
		self.quad_pts, self.quad_wts = basis.get_quadrature_data(quad_order)
		face_quad_order = gbasis.FACE_SHAPE.get_quadrature_order(mesh,
			order, physics=physics)
		self.face_quad_pts, self.face_quad_wts = basis.FACE_SHAPE.get_quadrature_data(quad_order)

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
		self.basis_phys_grad_elems = np.zeros([num_elems, nq, nb, basis.NDIMS])
		self.normals_elems = np.empty([num_elems, mesh.gbasis.NFACES,
			self.face_quad_pts.shape[0], ndims])

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

			# Face normals
			for i in range(mesh.gbasis.NFACES):
				self.normals_elems[elem_ID, i] = mesh.gbasis.calculate_normals(
						mesh, elem_ID, i, self.face_quad_pts)

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
	faces_to_basis_ref_gradL: numpy array
		gradient of basis values evaluated at quadrature points of each face 
		for left element
	faces_to_basis_ref_gradR: numpy array
		gradient of basis values evaluated at quadrature points of each face 
		for right element
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
	elemL_IDs: numpy array
		element IDs to the left of each interior face
	elemR_IDs: numpy array
		element IDs to the right of each interior face
	faceL_IDs: numpy array
		face IDs to the left of each interior face
	faceR_IDs: numpy array
		face IDs to the right of each interior face
	ijacL_elems: numpy array
		stores the evaluated inverse of the geometric Jacobian for each
		left element
	ijacR_elems: numpy array
		stores the evaluated inverse of the geometric Jacobian for each
		right element
	self.x_faces: numpy array
		stores the coordinates of the quadrature points for each
		interior face


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
		self.faces_to_basis_ref_gradL = np.zeros(0)
		self.faces_to_basis_ref_gradR = np.zeros(0)
		self.normals_int_faces = np.zeros(0)
		self.UqL = np.zeros(0)
		self.UqR = np.zeros(0)
		self.Fq = np.zeros(0)
		self.elemL_IDs = np.empty(0, dtype=int)
		self.elemR_IDs = np.empty(0, dtype=int)
		self.faceL_IDs = np.empty(0, dtype=int)
		self.faceR_IDs = np.empty(0, dtype=int)
		self.ijacL_elems = np.zeros(0)
		self.ijacR_elems = np.zeros(0)
		self.faces_to_xref = np.zeros(0)

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
			self.faces_to_basis_ref_gradL: gradient of basis values 
				evaluated at quadrature points of each face for left element
				[nfaces_per_elem, nq, nb, ndims]
			self.faces_to_basis_ref_gradR: gradient of basis values 
				evaluated at quadrature points of each face for right element
				[nfaces_per_elem, nq, nb, ndims]
			self.normals_int_faces: precomputed normal vectors at each
				interior face [num_interior_faces, nq, ndims]
			self.ijacL_elems: stores the evaluated inverse of the geometric 
				Jacobian for each left element 
				[num_interior_faces, nq, ndims, ndims]
			self.ijacR_elems: stores the evaluated inverse of the geometric 
				Jacobian for each right element 
				[num_interior_faces, nq, ndims, ndims]
			self.face_lengths: stores the precomputed length of each face
				[num_interior_faces, 1]
			self.x_faces: stores the coordinates of the quadrature points
				for each interior face
				[num_interior_faces, nq, ndims]
		
		Note(s):
		--------
			We separate ndims_basis and ndims to allow for basis
			and mesh to have different number of dimensions
			(ex: when using a space-time basis function and 
			only a spatial mesh)
		'''
		ndims_basis = basis.NDIMS 
		ndims = mesh.ndims

		# unpack
		quad_pts = self.quad_pts
		quad_wts = self.quad_wts
		nq = quad_pts.shape[0]
		nb = basis.nb
		nfaces_per_elem = basis.NFACES
		nfaces = mesh.num_interior_faces

		# Allocate
		self.faces_to_basisL = np.zeros([nfaces_per_elem, nq, nb])
		self.faces_to_basisR = np.zeros([nfaces_per_elem, nq, nb])
		self.faces_to_basis_ref_gradL = np.zeros([nfaces_per_elem,
				nq, nb, ndims_basis])
		self.faces_to_basis_ref_gradR = np.zeros([nfaces_per_elem,
				nq, nb, ndims_basis])
		self.ijacL_elems = np.zeros([nfaces, nq, ndims, ndims])
		self.ijacR_elems = np.zeros([nfaces, nq, ndims, ndims])
		self.normals_int_faces = np.zeros([mesh.num_interior_faces, nq,
				ndims])
		djac_faces = np.zeros([mesh.num_interior_faces, nq])
		
		self.x_faces = np.zeros([nfaces, nq, ndims])
		
		# Get values on each face (from both left and right perspectives) 
		# for both the basis and the reference gradient of the basis
		for face_ID in range(nfaces_per_elem):
			# Left
			basis.get_basis_face_val_grads(mesh, face_ID, quad_pts,
					get_val=True, get_ref_grad=True)
			self.faces_to_basisL[face_ID] = basis.basis_val
			self.faces_to_basis_ref_gradL[face_ID] = basis.basis_ref_grad

			# Right
			basis.get_basis_face_val_grads(mesh, face_ID, quad_pts[::-1],
					get_val=True, get_ref_grad=True)
			self.faces_to_basisR[face_ID] = basis.basis_val
			self.faces_to_basis_ref_gradR[face_ID] = basis.basis_ref_grad

		# Normals
		i = 0
		for interior_face in mesh.interior_faces:
			normals = mesh.gbasis.calculate_normals(mesh,
					interior_face.elemL_ID, interior_face.faceL_ID, quad_pts)
			self.normals_int_faces[i] = normals

			# Left state
			# Convert from face ref space to element ref space
			elem_pts = basis.get_elem_ref_from_face_ref(
					interior_face.faceL_ID, quad_pts)

			_, _, ijacL = basis_tools.element_jacobian(mesh, 
					interior_face.elemL_ID, elem_pts, get_djac=False,
					get_jac=False, get_ijac=True)
					
			x = mesh_tools.ref_to_phys(mesh, interior_face.elemL_ID, elem_pts)
			# Store
			self.x_faces[i] = x
			
			# Right state
			# Convert from face ref space to element ref space
			elem_pts = basis.get_elem_ref_from_face_ref(
					interior_face.faceR_ID, quad_pts[::-1])
			_, _, ijacR = basis_tools.element_jacobian(mesh,
					interior_face.elemR_ID, elem_pts, get_djac=False,
					get_jac=False, get_ijac=True)

			# Store
			self.ijacL_elems[i] = ijacL
			self.ijacR_elems[i] = ijacR

			# Used for face_length calculations
			djac_faces[i] = np.linalg.norm(normals, axis=1)
			i += 1

		self.face_lengths = mesh_tools.get_face_lengths(djac_faces, quad_wts)
		
	def alloc_other_arrays(self, physics, basis, order):
		'''
		Allocates the solution and flux vectors that are evaluated
		at the quadrature points

		Inputs:
		-------
			physics: physics object
			basis: basis object
			order: solution order
		'''
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
	normals_bgroups: list of numpy arrays
		normal vector array for each boundary face
	x_bgroups: list of numpy arrays
		coordinates of the quadrature points in physical space at the
		boundary face	
	ijac_bgroups: list of numpy arrays
		stores the evaluated inverse of the geometric Jacobian for each
		interior boundary element
	face_lengths_bgroups: list of numpy arrays
		stores the precomputed face lengths for each boundary face
	UqI: numpy array
		values of interior state at the quadrature points
	UqB: numpy array
		values of boundary (exterior) state at the quadrature points
	Fq: numpy array
		flux vector evaluated at the face quadrature points
	elem_IDs: list of numpy arrays 
		list containing arrays of element IDs of boundary
		face neighbors for each boundary group
	face_IDs: list of numpy arrays 
		list containing arrays of face IDs of boundary
		face neighbors for each boundary group

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
		self.ijac_bgroups = []
		self.face_lengths_bgroups = []
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
			self.faces_to_basis_ref_grad: gradient of basis values 
				evaluated at quadrature points of each face for interior element
				[nfaces_per_elem, nq, nb, ndims]
			self.faces_to_xref: coordinates of quadrature points of each
				face converted to element reference space
				[nfaces_per_elem, nq, ndims]		
			self.normals_bgroups: precomputed normal vectors at each
				boundary face [num_boundary_faces, nq, ndims]
			self.x_bgroups: precomputed physical coordinates of the
				quadrature points [num_boundary_faces, nq, ndims]
			self.ijac_bgroups: stores the evaluated inverse of the geometric 
				Jacobian for each interior element 	
			self.face_lengths_bgroups: stores the precomputed length of each face
				[num_interior_faces, 1]
		'''
		ndims = mesh.ndims
		quad_pts = self.quad_pts
		quad_wts = self.quad_wts
		nq = quad_pts.shape[0]
		nb = basis.nb
		nfaces_per_elem = basis.NFACES

		# Allocate
		self.faces_to_basis = np.zeros([nfaces_per_elem, nq, nb])
		self.faces_to_xref = np.zeros([nfaces_per_elem, nq, basis.NDIMS])
		self.faces_to_basis_ref_grad = np.zeros([nfaces_per_elem,
				nq, nb, basis.NDIMS])

		# Get values on each face (from interior perspective)
		for face_ID in range(nfaces_per_elem):
			self.faces_to_xref[face_ID] = basis.get_elem_ref_from_face_ref(
					face_ID, quad_pts)
			basis.get_basis_face_val_grads(mesh, face_ID, quad_pts,
					get_val=True, get_ref_grad=True)
			self.faces_to_basis[face_ID] = basis.basis_val
			self.faces_to_basis_ref_grad[face_ID] = basis.basis_ref_grad

		# Get boundary information
		i = 0
		for bgroup in mesh.boundary_groups.values():
			self.normals_bgroups.append(np.zeros([bgroup.num_boundary_faces,
					nq, ndims]))
			self.x_bgroups.append(np.zeros([bgroup.num_boundary_faces,
					nq, ndims]))
			self.ijac_bgroups.append(np.zeros([bgroup.num_boundary_faces,
					nq, ndims, ndims]))
			self.face_lengths_bgroups.append(np.zeros([
					bgroup.num_boundary_faces, 1]))

			normal_bgroup = self.normals_bgroups[i]
			x_bgroup = self.x_bgroups[i]
			ijac_bgroup = self.ijac_bgroups[i]
			face_lengths_bgroup = self.face_lengths_bgroups[i]

			j = 0
			for boundary_face in bgroup.boundary_faces:
				# Normals
				normals = mesh.gbasis.calculate_normals(mesh,
						boundary_face.elem_ID,
						boundary_face.face_ID, quad_pts)
				elem_pts = basis.get_elem_ref_from_face_ref(
						boundary_face.face_ID, quad_pts)
				djac, jac, ijac = basis_tools.element_jacobian(mesh, 
					boundary_face.elem_ID, elem_pts, get_djac=True,
					get_jac=True, get_ijac=True)

				normal_bgroup[j] = normals
				ijac_bgroup[j] = ijac
				djac_faces = np.linalg.norm(normals, axis=1)
				face_lengths_bgroup[j] = mesh_tools.get_face_lengths(
						djac_faces.reshape([1, nq]), quad_wts)

				# Physical coordinates of quadrature points
				x = mesh_tools.ref_to_phys(mesh, boundary_face.elem_ID,
						self.faces_to_xref[boundary_face.face_ID])
				# Store
				x_bgroup[j] = x

				# Increment
				j += 1
			i += 1


	def alloc_other_arrays(self, physics, basis, order):
		'''
		Allocates the solution and flux vectors that are evaluated
		at the quadrature points

		Inputs:
		-------
			physics: physics object
			basis: basis object
			order: solution order
		'''
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
		stepper_tools.set_source_treatment(physics)
		# Precompute helpers
		self.precompute_matrix_helpers()

		if self.limiters:
			for limiter in self.limiters:
				limiter.precompute_helpers(self)

		physics.conv_flux_fcn.alloc_helpers(
				np.zeros([mesh.num_interior_faces,
				self.int_face_helpers.quad_wts.shape[0],
				physics.NUM_STATE_VARS]))

		if physics.diff_flux_fcn:
			physics.diff_flux_fcn.alloc_helpers(
					np.zeros([mesh.num_interior_faces,
					self.int_face_helpers.quad_wts.shape[0],
					physics.NUM_STATE_VARS]))

		# Construct the necessary functions dependent upon required physics
		solver_tools.set_function_definitions(self, params)

		# Initial condition
		if params["RestartFile"] is None:
			self.init_state_from_fcn()

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
		basis_phys_grad_elems = elem_helpers.basis_phys_grad_elems
		quad_wts = elem_helpers.quad_wts
		djac_elems=elem_helpers.djac_elems
		ijac_elems = elem_helpers.ijac_elems
		x_elems = elem_helpers.x_elems
		nq = quad_wts.shape[0]
		fluxes = self.params["ConvFluxSwitch"]
		sources = self.params["SourceSwitch"]

		# Interpolate state at quad points
		Uq = helpers.evaluate_state(Uc, basis_val,
				skip_interp=self.basis.skip_interp) # [ne, nq, ns]
		
		# Interpolate gradient of state at quad points
		gUq = self.evaluate_gradient(Uc, basis_phys_grad_elems)

		if self.verbose:
			# Get min and max of state variables for reporting
			self.get_min_max_state(Uq)

		if fluxes:
			# Evaluate the inviscid flux integral
			Fq = physics.get_conv_flux_interior(Uq, x_elems)[0] # [ne, nq, ns, ndims]

			if physics.diff_flux_fcn:
				# Evaluate the diffusion flux
				Fq -= physics.get_diff_flux_interior(Uq, gUq) 
					# [ne, nq, ns, ndims]
			
			res_elem += solver_tools.calculate_volume_flux_integral(
					self, elem_helpers, Fq) # [ne, nb, ns]

		if sources:
			# Evaluate the source term integral
			# eval_source_terms is an additive function so source needs to be
			# initialized to zero for each time step
			Sq = np.zeros_like(Uq) # [ne, nq, ns]
			Sq = physics.eval_source_terms(Uq, x_elems, self.time, Sq)
					# [ne, nq, ns]

			res_elem += solver_tools.calculate_source_term_integral(
					elem_helpers, Sq) # [ne, nb, ns]

		# Add artificial viscosity term
		if self.params["ArtificialViscosity"]:
			av_param = self.params["AVParameter"]
			res_elem -= solver_tools.calculate_artificial_viscosity_integral(
					physics, elem_helpers, Uc, av_param, self.order)

		return res_elem # [ne, nb, ns]

	def get_interior_face_residual(self, faceL_IDs, faceR_IDs, UcL, UcR):
		# Unpack
		mesh = self.mesh
		physics = self.physics
		fluxes = self.params["ConvFluxSwitch"]

		int_face_helpers = self.int_face_helpers
		elem_helpers = self.elem_helpers
		vol_elems = elem_helpers.vol_elems
		face_lengths = int_face_helpers.face_lengths
		x_faces = int_face_helpers.x_faces

		quad_wts = int_face_helpers.quad_wts
		faces_to_basisL = int_face_helpers.faces_to_basisL
		faces_to_basisR = int_face_helpers.faces_to_basisR
		faces_to_basis_ref_gradL = int_face_helpers.faces_to_basis_ref_gradL
		faces_to_basis_ref_gradR = int_face_helpers.faces_to_basis_ref_gradR
		ijacL_elems = int_face_helpers.ijacL_elems
		ijacR_elems = int_face_helpers.ijacR_elems

		normals_int_faces = int_face_helpers.normals_int_faces
				# [nf, nq, ndims]

		ns = physics.NUM_STATE_VARS
		nq = quad_wts.shape[0]

		# Interpolate state at quad points
		UqL = helpers.evaluate_state(UcL, faces_to_basisL[faceL_IDs])
				# [nf, nq, ns]
		UqR = helpers.evaluate_state(UcR, faces_to_basisR[faceR_IDs])
				# [nf, nq, ns]

		# Interpolate gradient of state at quad points
		gUqL_ref = self.evaluate_gradient(UcL, 
				faces_to_basis_ref_gradL[faceL_IDs])
		gUqR_ref = self.evaluate_gradient(UcR, 
				faces_to_basis_ref_gradR[faceR_IDs])

		# Make gradient the physical gradient at L/R states
		gUqL = self.ref_to_phys_grad(ijacL_elems, gUqL_ref)
		gUqR = self.ref_to_phys_grad(ijacR_elems, gUqR_ref)

		# Allocate resL and resR (needed for operator splitting)
		nifL = self.int_face_helpers.elemL_IDs.shape[0]
		nifR = self.int_face_helpers.elemR_IDs.shape[0]
		resL = np.zeros([nifL, nq, ns])
		resR = np.zeros([nifR, nq, ns])
		resL_diff = np.zeros([nifL, nq, ns])
		resR_diff = np.zeros([nifR, nq, ns])

		if physics.diff_flux_fcn:
			# Calculate diffusion flux helpers
			physics.diff_flux_fcn.compute_iface_helpers(self)

		if fluxes:
			# Compute numerical flux
			Fq = physics.get_conv_flux_numerical(UqL, UqR, normals_int_faces, x_faces)
					# [nf, nq, ns]

			# Compute diffusion flux
			Fq_diff, FL, FR = physics.get_diff_flux_numerical(UqL, UqR,
					gUqL, gUqR, normals_int_faces) # [nf, nq, ns], 
					# [nf, nq, ns, ndims], [nf, nq, ns, ndims]
			Fq -= Fq_diff

			FL_phys = self.ref_to_phys_grad(ijacL_elems, FL)
			FR_phys = self.ref_to_phys_grad(ijacR_elems, FR)

			# Compute contribution to left and right element residuals
			resL = solver_tools.calculate_boundary_flux_integral(
					faces_to_basisL[faceL_IDs], quad_wts, Fq)
			resR = solver_tools.calculate_boundary_flux_integral(
					faces_to_basisR[faceR_IDs], quad_wts, Fq)

			# Compute additional boundary flux integrals for diffusion terms
			resL_diff = self.calculate_boundary_flux_integral_sum(
					faces_to_basis_ref_gradL[faceL_IDs], quad_wts, FL_phys)

			resR_diff = self.calculate_boundary_flux_integral_sum(
					faces_to_basis_ref_gradR[faceR_IDs], quad_wts, 
					FR_phys)
			
		return resL, resR, resL_diff, resR_diff # [nif, nb, ns]

	def get_boundary_face_residual(self, bgroup, face_IDs, Uc, resB):
		# unpack
		mesh = self.mesh
		physics = self.physics
		bgroup_num = bgroup.number

		bface_helpers = self.bface_helpers
		elem_helpers = self.elem_helpers
		fluxes = self.params["ConvFluxSwitch"]

		quad_wts = bface_helpers.quad_wts
		normals_bgroups = bface_helpers.normals_bgroups
		x_bgroups = bface_helpers.x_bgroups
		ijac_bgroups = bface_helpers.ijac_bgroups

		basis_val = bface_helpers.faces_to_basis[face_IDs] # [nbf, nq, nb]
		basis_ref_grad = bface_helpers.faces_to_basis_ref_grad[face_IDs] 

		# Interpolate state at quad points
		UqI = helpers.evaluate_state(Uc, basis_val) # [nbf, nq, ns]

		normals = normals_bgroups[bgroup_num] # [nbf, nq, ndims]
		x = x_bgroups[bgroup_num] # [nbf, nq, ndims]
		ijac = ijac_bgroups[bgroup_num]

		BC = physics.BCs[bgroup.name]

		# Interpolate state at quadrature points
		UqI = helpers.evaluate_state(Uc, basis_val)

		# Interpolate gradient of state at quad points
		gUq_ref = self.evaluate_gradient(Uc, 
				basis_ref_grad)

		# Make ref gradient of state the physical gradient
		gUq = self.ref_to_phys_grad(ijac, gUq_ref)

		# Compute any additional helpers for diffusive flux fcn
		if physics.diff_flux_fcn:
			physics.diff_flux_fcn.compute_bface_helpers(self, bgroup_num)
		
		if fluxes:
			# Compute boundary flux
			Fq, FqB = BC.get_boundary_flux(physics, UqI, normals, x, self.time, gUq=gUq)
			FqB_phys = self.ref_to_phys_grad(ijac, FqB)

			# Compute contribution to adjacent element residual
			resB = solver_tools.calculate_boundary_flux_integral(
					basis_val, quad_wts, Fq)

			resB -= self.calculate_boundary_flux_integral_sum(
				basis_ref_grad, quad_wts, FqB_phys)

		return resB
