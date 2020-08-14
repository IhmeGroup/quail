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


class ElemOperators(object):
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

		# QuadOrder, _ = get_gaussian_quadrature_elem(mesh, mesh.gbasis, order, physics, None)
		gbasis = mesh.gbasis
		quad_order = gbasis.get_quadrature_order(mesh, order, physics = physics)
		self.quad_pts, self.quad_wts = basis.get_quadrature_data(quad_order)

	def get_basis_and_geom_data(self, mesh, basis, order):
		# separate these later

		# Unpack
		dim = mesh.dim 
		num_elems = mesh.num_elems 
		quad_pts = self.quad_pts 
		nq = quad_pts.shape[0]
		nb = basis.nb

		# Allocate
		self.jac_elems = np.zeros([num_elems, nq, dim, dim])
		self.ijac_elems = np.zeros([num_elems, nq, dim, dim])
		self.djac_elems = np.zeros([num_elems, nq, 1])
		self.x_elems = np.zeros([num_elems, nq, dim])
		self.basis_phys_grad_elems = np.zeros([num_elems, nq, nb, dim])

		# basis data
		basis.get_basis_val_grads(self.quad_pts, get_val=True, get_ref_grad=True)

		self.basis_val = basis.basis_val 
		self.basis_ref_grad = basis.basis_ref_grad 

		for elem in range(mesh.num_elems):
			# Jacobian
			djac, jac, ijac = basis_tools.element_jacobian(mesh, elem, quad_pts, get_djac=True, get_jac=True, get_ijac=True)
			# Store
			self.jac_elems[elem] = jac
			self.ijac_elems[elem] = ijac
			self.djac_elems[elem] = djac

			# Physical coordinates of quadrature points
			x = mesh_tools.ref_to_phys(mesh, elem, quad_pts)
			# Store
			self.x_elems[elem] = x
			# Physical gradient
			basis.get_basis_val_grads(quad_pts, get_phys_grad=True, ijac=ijac) # gPhi is [nq,nb,dim]
			self.basis_phys_grad_elems[elem] = basis.basis_phys_grad

		# _, ElemVols = mesh_tools.element_volumes(mesh)
		self.vol_elems, self.domain_vol = mesh_tools.element_volumes(mesh)

	def alloc_other_arrays(self, physics, basis, order):
		quad_pts = self.quad_pts 
		nq = quad_pts.shape[0]
		nb = basis.nb
		ns = physics.NUM_STATE_VARS
		dim = physics.dim

		self.Uq = np.zeros([nq, ns]) 
		self.Fq = np.zeros([nq, ns, dim])
		self.Sq = np.zeros([nq, ns])  

	def compute_operators(self, mesh, physics, basis, order):
		self.get_gaussian_quadrature(mesh, physics, basis, order)
		self.get_basis_and_geom_data(mesh, basis, order)
		self.alloc_other_arrays(physics, basis, order)
		self.iMM_elems = basis_tools.get_inv_mass_matrices(mesh, physics, basis)


class InteriorFaceOperators(ElemOperators):
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

		gbasis = mesh.gbasis
		quad_order = gbasis.FACE_SHAPE.get_quadrature_order(mesh,order,physics=physics)
		self.quad_pts, self.quad_wts = basis.FACE_SHAPE.get_quadrature_data(quad_order)

	def get_basis_and_geom_data(self, mesh, basis, order):
		# separate these later

		# Unpack
		dim = mesh.dim
		quad_pts = self.quad_pts 
		nq = quad_pts.shape[0]
		nb = basis.nb
		# nFacePerElem = mesh.nFacePerElem
		nfaces_per_elem = mesh.gbasis.NFACES

		# Allocate
		self.faces_to_basisL = np.zeros([nfaces_per_elem, nq, nb])
		self.faces_to_basisR = np.zeros([nfaces_per_elem, nq, nb])
		self.normals_ifaces = np.zeros([mesh.num_interior_faces, nq, dim])

		# basis data
		#PhiData = BasisData(basis, order, mesh)

		for f in range(nfaces_per_elem):
			# Left
			basis.get_basis_face_val_grads(mesh, f, quad_pts, get_val=True)
			self.faces_to_basisL[f] = basis.basis_val
			# Right
			basis.get_basis_face_val_grads(mesh, f, quad_pts[::-1], get_val=True)
			self.faces_to_basisR[f] = basis.basis_val

		i = 0
		for IFace in mesh.interior_faces:
			# Normals
			# normals = mesh_defs.iface_normal(mesh, IFace, quad_pts)
			normals = mesh.gbasis.calculate_normals(mesh, IFace.elemL_id, IFace.faceL_id, quad_pts)
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


class BoundaryFaceOperators(InteriorFaceOperators):
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
		dim = mesh.dim
		quad_pts = self.quad_pts 
		nq = quad_pts.shape[0]
		nb = basis.nb
		# nFacePerElem = mesh.nFacePerElem
		nfaces_per_elem = mesh.gbasis.NFACES

		# Allocate
		self.faces_to_basis = np.zeros([nfaces_per_elem, nq, nb])
		self.faces_to_xref = np.zeros([nfaces_per_elem, nq, dim])
		self.normals_bfgroups = []
		self.x_bfgroups = []

		# basis data
		# PhiData = BasisData(basis, order, mesh)

		for f in range(nfaces_per_elem):
			# Left
			self.faces_to_xref[f] = basis.get_elem_ref_from_face_ref(f, quad_pts)
			basis.get_basis_face_val_grads(mesh, f, quad_pts, get_val=True)
			self.faces_to_basis[f] = basis.basis_val

		i = 0
		for BFG in mesh.boundary_groups.values():
			self.normals_bfgroups.append(np.zeros([BFG.num_boundary_faces,nq,dim]))
			self.x_bfgroups.append(np.zeros([BFG.num_boundary_faces,nq,dim]))
			normal_bfgroup = self.normals_bfgroups[i]
			x_bfgroup = self.x_bfgroups[i]
			j = 0
			for boundary_face in BFG.boundary_faces:
				# Normals
				# nvec = mesh_defs.bface_normal(mesh, boundary_face, quad_pts)
				nvec = mesh.gbasis.calculate_normals(mesh, boundary_face.elem_id, boundary_face.face_id, quad_pts)
				normal_bfgroup[j] = nvec

				# Physical coordinates of quadrature points
				x = mesh_tools.ref_to_phys(mesh, boundary_face.elem_id, self.faces_to_xref[boundary_face.face_id])
				# Store
				x_bfgroup[j] = x

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

	def compute_operators(self, mesh, physics, basis, order):
		self.get_gaussian_quadrature(mesh, physics, basis, order)
		self.get_basis_and_geom_data(mesh, basis, order)
		self.alloc_other_arrays(physics, basis, order)


class DG(base.SolverBase):
	'''
	Class: DG
	--------------------------------------------------------------------------
	Discontinuous Galerkin method designed to solve a given set of PDEs

	ATTRIBUTES:
		Params: list of parameters for the solver
		physics: solver object (current implementation supports Scalar and Euler equations)
		mesh: mesh object
		DataSet: location to store generic data
		Time: current time in the simulation
		nTimeStep: number of time steps
	'''
	def __init__(self, Params, physics, mesh):
		super().__init__(Params, physics, mesh)

		self.Stepper = stepper_tools.set_stepper(Params, physics.U)
		stepper_tools.set_time_stepping_approach(self.Stepper, Params)
	
		# Check validity of parameters
		self.check_compatibility()
		
		# Precompute operators
		self.precompute_matrix_operators()
		if self.limiter is not None:
			self.limiter.precompute_operators(self)

		physics.ConvFluxFcn.alloc_helpers(np.zeros([self.iface_operators.quad_wts.shape[0], physics.NUM_STATE_VARS]))

		if Params["RestartFile"] is None:
			self.init_state_from_fcn()
		
	def precompute_matrix_operators(self):
		mesh = self.mesh 
		physics = self.physics
		basis = self.basis

		self.elem_operators = ElemOperators()
		self.elem_operators.compute_operators(mesh, physics, basis, physics.order)
		self.iface_operators = InteriorFaceOperators()
		self.iface_operators.compute_operators(mesh, physics, basis, physics.order)
		self.bface_operators = BoundaryFaceOperators()
		self.bface_operators.compute_operators(mesh, physics, basis, physics.order)

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
		# Uq = np.matmul(basis_val, Up)
		Uq = helpers.evaluate_state(Up, basis_val, skip_interp=self.basis.skip_interp)

		
		if self.Params["ConvFluxSwitch"] == True:
			'''
			Evaluate the inviscid flux integral
			'''
			Fq = physics.ConvFluxInterior(Uq) # [nq, ns, dim]
			ER += solver_tools.calculate_inviscid_flux_volume_integral(self, elem_ops, elem, Fq)

		if self.Params["SourceSwitch"] == True:
			'''
			Evaluate the source term integral
			'''
			Sq = elem_ops.Sq
			Sq[:] = 0. # SourceState is an additive function so source needs to be initialized to zero for each time step
			Sq = physics.SourceState(nq, x, self.time, Uq, Sq) # [nq, ns]

			ER += solver_tools.calculate_source_term_integral(elem_ops, elem, Sq)

		if elem == echeck:
			code.interact(local=locals())

		return ER

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
		# UqL = np.matmul(basis_valL, UpL)
		# UqR = np.matmul(basis_valR, UpR)
		UqL = helpers.evaluate_state(UpL, basis_valL)
		UqR = helpers.evaluate_state(UpR, basis_valR)

		normals = normals_ifaces[iiface]
		
		if self.Params["ConvFluxSwitch"] == True:

			Fq = physics.ConvFluxNumerical(UqL, UqR, normals) # [nq,ns]

			RL -= solver_tools.calculate_inviscid_flux_boundary_integral(basis_valL, quad_wts, Fq)
			RR += solver_tools.calculate_inviscid_flux_boundary_integral(basis_valR, quad_wts, Fq)

		if elemL == echeck or elemR == echeck:
			if elemL == echeck: print("Left!")
			else: print("Right!")
			code.interact(local=locals())

		return RL, RR

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
		physics = self.physics
		# BFG = mesh.boundary_groups[ibfgrp]
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
		# UqI = np.matmul(basis_val, U)
		UqI = helpers.evaluate_state(U, basis_val)

		normals = normals_bfgroups[ibfgrp][ibface]
		x = x_bfgroups[ibfgrp][ibface]

		# Get boundary state
		BC = physics.BCs[BFG.name]

		if self.Params["ConvFluxSwitch"] == True:

			Fq = BC.get_boundary_flux(physics, x, self.time, normals, UqI)

			R -= solver_tools.calculate_inviscid_flux_boundary_integral(basis_val, quad_wts, Fq)

		if elem == echeck:
			code.interact(local=locals())

		return R
