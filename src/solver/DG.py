# ------------------------------------------------------------------------ #
#
#       File : src/numerics/solver/DG.py
#
#       Contains class definitions for the DG solver available in the 
#		DG Python framework.
#
#       Authors: Eric Ching and Brett Bornhoft
#
#       Created: January 2020
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
		stores the evaluated geometric jacobian for each element
	ijac_elems: numpy array
		stores the evaluated inverse of the geometric jacobian for each 
		element
	djac_elems: numpy array
		stores the evaluated determinant of the geometric jacobian for 
		each element
	x_elems: numpy array
		physical coordinates of quadrature points
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
		precomputes the element's basis function, it's gradients,
		geometric jacobian info, and volume
	alloc_other_arrays
		allocate the solution, flux, and source vectors that are evaluated
		at the quadrature points
	compute_operators
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
			self.jac_elems: precomputed jacobian for each element 
				[num_elems, nq, dim, dim]
			self.ijac_elems: precomputed inverse jacobian for each element
				[num_elems, nq, dim, dim]
			self.djac_elems: precomputed determinant of the jacobian for each
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

		for elem in range(mesh.num_elems):
			# jacobian
			djac, jac, ijac = basis_tools.element_jacobian(mesh, elem,
					quad_pts, get_djac=True, get_jac=True, get_ijac=True)
			# store
			self.jac_elems[elem] = jac
			self.ijac_elems[elem] = ijac
			self.djac_elems[elem] = djac

			# physical coordinates of quadrature points
			x = mesh_tools.ref_to_phys(mesh, elem, quad_pts)
			# store
			self.x_elems[elem] = x
			# physical gradient
			basis.get_basis_val_grads(quad_pts, get_phys_grad=True, 
					ijac=ijac)
			self.basis_phys_grad_elems[elem] = basis.basis_phys_grad  
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
		dim = physics.dim

		self.Uq = np.zeros([nq, ns]) 
		self.Fq = np.zeros([nq, ns, dim])
		self.Sq = np.zeros([nq, ns])  

	def compute_operators(self, mesh, physics, basis, order):
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
	compute_operators
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
			normals = mesh.gbasis.calculate_normals(mesh, IFace.elemL_id, 
					IFace.faceL_id, quad_pts)
			self.normals_ifaces[i] = normals
			i += 1

	def alloc_other_arrays(self, physics, basis, order):
		quad_pts = self.quad_pts 
		nq = quad_pts.shape[0]
		ns = physics.NUM_STATE_VARS

		self.UqL = np.zeros([nq, ns])
		self.UqR = np.zeros([nq, ns])
		self.Fq = np.zeros([nq, ns])

	def compute_operators(self, mesh, physics, basis, order):
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
	compute_operators
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
						boundary_face.elem_id, 
						boundary_face.face_id, quad_pts)
				normal_bfgroup[j] = nvec

				# physical coordinates of quadrature points
				x = mesh_tools.ref_to_phys(mesh, boundary_face.elem_id, 
						self.faces_to_xref[boundary_face.face_id])
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

	def compute_operators(self, mesh, physics, basis, order):
		self.get_gaussian_quadrature(mesh, physics, basis, order)
		self.get_basis_and_geom_data(mesh, basis, order)
		self.alloc_other_arrays(physics, basis, order)


class DG(base.SolverBase):
	'''
    DG inherits attributes and methods from the SolverBase class.
    See SolverBase for detailed comments of attributes and methods.

    Additional methods and attributes are commented below.
	'''
	def __init__(self, Params, physics, mesh):
		super().__init__(Params, physics, mesh)

		self.Stepper = stepper_tools.set_stepper(Params, physics.U)
		stepper_tools.set_time_stepping_approach(self.Stepper, Params)
	
		# check validity of parameters
		self.check_compatibility()
		
		# precompute operators
		self.precompute_matrix_operators()
		if self.limiter is not None:
			self.limiter.precompute_operators(self)

		physics.ConvFluxFcn.alloc_helpers(
				np.zeros([self.iface_operators.quad_wts.shape[0], 
				physics.NUM_STATE_VARS]))

		if Params["RestartFile"] is None:
			self.init_state_from_fcn()
		
	def precompute_matrix_operators(self):

		mesh = self.mesh 
		physics = self.physics
		basis = self.basis

		self.elem_operators = ElemHelpers()
		self.elem_operators.compute_operators(mesh, physics, basis, 
				physics.order)
		self.iface_operators = InteriorFaceHelpers()
		self.iface_operators.compute_operators(mesh, physics, basis, 
				physics.order)
		self.bface_operators = BoundaryFaceHelpers()
		self.bface_operators.compute_operators(mesh, physics, basis, 
				physics.order)

	def get_element_residual(self, elem, Up, ER):

		physics = self.physics
		ns = physics.NUM_STATE_VARS
		dim = physics.dim
		elem_ops = self.elem_operators
		basis_val = elem_ops.basis_val
		quad_wts = elem_ops.quad_wts

		x_elems = elem_ops.x_elems
		nq = quad_wts.shape[0]
		x = x_elems[elem]

		# interpolate state and gradient at quad points
		Uq = helpers.evaluate_state(Up, basis_val, 
				skip_interp=self.basis.skip_interp)

		if self.Params["ConvFluxSwitch"] == True:
			# evaluate the inviscid flux integral
			Fq = physics.ConvFluxInterior(Uq) # [nq, ns, dim]
			ER += solver_tools.calculate_inviscid_flux_volume_integral(
					self, elem_ops, elem, Fq)

		if self.Params["SourceSwitch"] == True:
			# evaluate the source term integral
			Sq = elem_ops.Sq
			# SourceState is an additive function so source needs to be 
			# initialized to zero for each time step
			Sq[:] = 0.
			Sq = physics.SourceState(nq, x, self.time, Uq, Sq) # [nq, ns]

			ER += solver_tools.calculate_source_term_integral(elem_ops, 
					elem, Sq)

		if elem == echeck:
			code.interact(local=locals())

		return ER

	def get_interior_face_residual(self, iiface, UpL, UpR, RL, RR):

		mesh = self.mesh
		physics = self.physics
		IFace = mesh.interior_faces[iiface]
		elemL = IFace.elemL_id
		elemR = IFace.elemR_id
		faceL_id = IFace.faceL_id
		faceR_id = IFace.faceR_id

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
		basis_valL = faces_to_basisL[faceL_id]
		basis_valR = faces_to_basisR[faceR_id]

		# interpolate state and gradient at quad points
		UqL = helpers.evaluate_state(UpL, basis_valL)
		UqR = helpers.evaluate_state(UpR, basis_valR)

		normals = normals_ifaces[iiface]
		
		if self.Params["ConvFluxSwitch"] == True:

			Fq = physics.ConvFluxNumerical(UqL, UqR, normals) # [nq,ns]

			RL -= solver_tools.calculate_inviscid_flux_boundary_integral(
					basis_valL, quad_wts, Fq)
			RR += solver_tools.calculate_inviscid_flux_boundary_integral(
					basis_valR, quad_wts, Fq)

		if elemL == echeck or elemR == echeck:
			if elemL == echeck: print("Left!")
			else: print("Right!")
			code.interact(local=locals())

		return RL, RR

	def get_boundary_face_residual(self, BFG, ibface, U, R):

		mesh = self.mesh
		physics = self.physics
		ibfgrp = BFG.number
		boundary_face = BFG.boundary_faces[ibface]
		elem = boundary_face.elem_id
		face = boundary_face.face_id

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
		UqI = helpers.evaluate_state(U, basis_val)

		normals = normals_bfgroups[ibfgrp][ibface]
		x = x_bfgroups[ibfgrp][ibface]

		# get boundary state
		BC = physics.BCs[BFG.name]

		if self.Params["ConvFluxSwitch"] == True:

			Fq = BC.get_boundary_flux(physics, x, self.time, normals, UqI)

			R -= solver_tools.calculate_inviscid_flux_boundary_integral(
					basis_val, quad_wts, Fq)

		if elem == echeck:
			code.interact(local=locals())

		return R