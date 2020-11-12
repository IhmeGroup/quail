# ------------------------------------------------------------------------ #
#
#       File : src/numerics/limiting/wenolimiter.py
#
#       Contains class definitions for WENO limiters.
#      
# ------------------------------------------------------------------------ #
from abc import ABC, abstractmethod
import numpy as np 

import errors
import general

import meshing.tools as mesh_tools

import numerics.helpers.helpers as helpers
import numerics.limiting.base as base
import numerics.limiting.tools as limiter_tools

# REMOVE FOR MASTER
class ScalarWENO(base.LimiterBase):
	'''
	This class corresponds to the scalar WENO limiter. It inherits from 
	the LimiterBase class. See LimiterBase for detailed comments of 
	attributes and methods. See the following references:
	
		[1] X. Zhong, C. Shu, "A simple weighted essentially nonoscillatory
			limiter for Runge-Kutta discontinuous Galerkin methods," 
			Journal of Computational Physics. Vol. 232 pg. 397-415. 2013.

	Attributes:
	-----------
	elem_vols: numpy array
		element volumes
	basis_val_elem_faces: numpy array
		stores basis values for element and faces
	quad_wts_elem: numpy array
		quadrature points for element
	djac_elems: numpy array
		stores Jacobian determinants for each element
	elemP_IDs: numpy array
		stores the i+1/2 neighbor index for element i
	elemM_IDs: numpy array
		stores the i-1/2 neighbor index for element i
	'''
	COMPATIBLE_PHYSICS_TYPES = general.PhysicsType.Burgers

	def __init__(self, physics_type):
		super().__init__(physics_type)
		self.elem_vols = np.zeros(0)
		self.basis_val_elem_faces = np.zeros(0)
		self.quad_wts_elem = np.zeros(0)
		self.djac_elems = np.zeros(0)
		self.elemP_IDs = np.zeros(0)
		self.elemM_IDs = np.zeros(0)

	def precompute_helpers(self, solver):
		# Unpack
		mesh = solver.mesh
		basis = solver.basis
		elem_helpers = solver.elem_helpers
		int_face_helpers = solver.int_face_helpers

		ndims = mesh.ndims
		num_elems = mesh.num_elems
		nb = basis.nb
		nq = elem_helpers.quad_pts.shape[0]

		self.elem_vols, _ = mesh_tools.element_volumes(solver.mesh, solver)
		self.basis_phys_hessian_elems = np.zeros([num_elems, nq, nb, ndims])

		# Basis values in element interior and on faces
		if not solver.basis.skip_interp:
			basis_val_faces = int_face_helpers.faces_to_basisL.copy()
			bshape = basis_val_faces.shape
			basis_val_faces.shape = (bshape[0]*bshape[1], bshape[2])

			self.basis_val_elem_faces = np.vstack((elem_helpers.basis_val,
					basis_val_faces))
		else:
			self.basis_val_elem_faces = elem_helpers.basis_val

		# Jacobian determinant
		self.djac_elems = elem_helpers.djac_elems

		# Element quadrature weights
		self.quad_wts_elem = elem_helpers.quad_wts

		# identify neighboring elements and store
		elemP_IDs = np.empty([num_elems], dtype=int)
		elemM_IDs = np.empty([num_elems], dtype=int)

		for elem_ID in range(num_elems):
			elemP_IDs[elem_ID] = mesh.elements[elem_ID].face_to_neighbors[1]
			elemM_IDs[elem_ID] = mesh.elements[elem_ID].face_to_neighbors[0]
	
		self.elemP_IDs = elemP_IDs
		self.elemM_IDs = elemM_IDs

		# Calculate hessian for higher-order weno calculation
		self.basis_ref_hessian = limiter_tools.get_hessian(self, basis, 
				elem_helpers.quad_pts)
		ijac_elems = elem_helpers.ijac_elems

		for elem_ID in range(num_elems):
			self.basis_phys_hessian_elems[elem_ID] = limiter_tools.get_phys_hessian(
					self, basis, ijac_elems[elem_ID])
		

	def get_nonlinearwts(self, order, p, gamma, basis_phys_grad, quad_wts, vol):
		'''
		This method calculates the smoothness indicator. (See Eq. 3.10 in [1])

		Inputs:
		-------
			order: solution order
			p: polynomial coeffs of of element being smoothed [ne, nb, ns]
			gamma: weighting constants in weno scheme (See Eq. 3.11 in [1])
			basis_phys_grad: evaluated gradient of the basis function in 
            		physical space [nq, nb, ndims]
            quad_wts: quadrature weights [nq, 1]  
            vol: element volumes [ne, 1]

		Outputs:
		--------
			weno_wts: returns the linear weight for the weno reconstruction [ne] 
		'''

		# s = 1 corresponds to the first derivative of the basis function
		s = 1
		basis_p = (np.matmul(basis_phys_grad[:, :, :, 0], p)**2).transpose(0,2,1)
		basis_p_qwts = np.einsum('ikj,jl -> ik',basis_p, quad_wts)

		beta = vol**(2*s-1) * basis_p_qwts.reshape(p.shape[0])

		eps = 1.0e-6

		# s = 2 corresponds to the second der. and order 2
		if order == 2:
			s = 2
			#Unpack 
			basis_phys_hessian = self.basis_phys_hessian_elems
			hess_p = (np.matmul(basis_phys_hessian[:, :, :, 0], p)**2).transpose(0,2,1)
			hess_p_qwts = np.einsum('ikj, jk -> ik', hess_p, quad_wts)

			beta += vol**(2*s-1) * hess_p_qwts.reshape(p.shape[0])

		return gamma / (eps + beta)**2 # weno_wts [ne]


	def limit_element(self, solver, Uc):
		# Unpack		
		elem_helpers = solver.elem_helpers
		vols = self.elem_vols
		basis_phys_grads = elem_helpers.basis_phys_grad_elems
		quad_wts = elem_helpers.quad_wts
		
		weno_wts = np.zeros([Uc.shape[0], 3])

		# Determine if the elements requires limiting
		shock_indicated = self.shock_indicator(self, solver, Uc)

		# Unpack limiter info from shock indicator
		if self.U_elem is not None:
			p0 = self.Um_elem
			p1 = self.U_elem
			p2 = self.Up_elem

			p0_bar = self.Um_bar
			p1_bar = self.U_bar
			p2_bar = self.Up_bar
		else:
			raise NotImplementedError

		p0_tilde = p0 - p0_bar + p1_bar
		p2_tilde = p2 - p2_bar + p1_bar
	
		# Currently only implemented up to  P2
		if solver.order > 2:
			raise NotImplementedError

		# Calculate non-linear weights
		weno_wts[:, 0] = self.get_nonlinearwts(solver.order, p0, 0.001, 
				basis_phys_grads, quad_wts, vols)
		weno_wts[:, 1] = self.get_nonlinearwts(solver.order, p1, 0.998, 
				basis_phys_grads, quad_wts, vols)
		weno_wts[:, 2] = self.get_nonlinearwts(solver.order, p2, 0.001, 
				basis_phys_grads, quad_wts, vols)

		normal_wts = weno_wts / np.sum(weno_wts, axis=1).reshape(weno_wts.shape[0], 1)

		# Update state_coeffs for indicated elements
		Uc[shock_indicated] = np.einsum('i, ijk -> ijk', normal_wts[shock_indicated, 0], 
				p0_tilde[shock_indicated]) + \
				np.einsum('i, ijk -> ijk', normal_wts[shock_indicated, 1],
				p1[shock_indicated]) + \
				np.einsum('i, ijk -> ijk', normal_wts[shock_indicated, 2],
				p2_tilde[shock_indicated])

		return Uc # [ne, nq, 1]

class EulerWENO(base.LimiterBase):
	'''
	This class corresponds to the Euler WENO limiter. It inherits from 
	the LimiterBase class. See LimiterBase for detailed comments of 
	attributes and methods. See the following references:
	
		[1] X. Zhong, C. Shu, "A simple weighted essentially nonoscillatory
			limiter for Runge-Kutta discontinuous Galerkin methods," 
			Journal of Computational Physics. Vol. 232 pg. 397-415. 2013.

	Attributes:
	-----------
	elem_vols: numpy array
		element volumes
	basis_val_elem_faces: numpy array
		stores basis values for element and faces
	quad_wts_elem: numpy array
		quadrature points for element
	djac_elems: numpy array
		stores Jacobian determinants for each element
	elemP_IDs: numpy array
		stores the i+1/2 neighbor index for element i
	elemM_IDs: numpy array
		stores the i-1/2 neighbor index for element i
	'''
	COMPATIBLE_PHYSICS_TYPES = general.PhysicsType.Euler

	def __init__(self, physics_type):
		super().__init__(physics_type)
		self.elem_vols = np.zeros(0)
		self.basis_val_elem_faces = np.zeros(0)
		self.quad_wts_elem = np.zeros(0)
		self.djac_elems = np.zeros(0)
		self.elemP_IDs = np.zeros(0)
		self.elemM_IDs = np.zeros(0)

	def precompute_helpers(self, solver):
		# Unpack
		mesh = solver.mesh
		basis = solver.basis
		elem_helpers = solver.elem_helpers
		int_face_helpers = solver.int_face_helpers

		ndims = mesh.ndims
		num_elems = mesh.num_elems
		nb = basis.nb
		nq = elem_helpers.quad_pts.shape[0]

		self.elem_vols, _ = mesh_tools.element_volumes(solver.mesh, solver)
		self.basis_phys_hessian_elems = np.zeros([num_elems, nq, nb, ndims])

		# Basis values in element interior and on faces
		if not solver.basis.skip_interp:
			basis_val_faces = int_face_helpers.faces_to_basisL.copy()
			bshape = basis_val_faces.shape
			basis_val_faces.shape = (bshape[0]*bshape[1], bshape[2])

			self.basis_val_elem_faces = np.vstack((elem_helpers.basis_val,
					basis_val_faces))
		else:
			self.basis_val_elem_faces = elem_helpers.basis_val

		# Jacobian determinant
		self.djac_elems = elem_helpers.djac_elems

		# Element quadrature weights
		self.quad_wts_elem = elem_helpers.quad_wts

		# identify neighboring elements and store
		elemP_IDs = np.empty([num_elems], dtype=int)
		elemM_IDs = np.empty([num_elems], dtype=int)

		for elem_ID in range(num_elems):
			elemP_IDs[elem_ID] = mesh.elements[elem_ID].face_to_neighbors[1]
			elemM_IDs[elem_ID] = mesh.elements[elem_ID].face_to_neighbors[0]
	
		self.elemP_IDs = elemP_IDs
		self.elemM_IDs = elemM_IDs

		# Calculate hessian for higher-order weno calculation
		self.basis_ref_hessian = limiter_tools.get_hessian(self, basis, 
				elem_helpers.quad_pts)
		ijac_elems = elem_helpers.ijac_elems

		for elem_ID in range(num_elems):
			self.basis_phys_hessian_elems[elem_ID] = limiter_tools.get_phys_hessian(
					self, basis, ijac_elems[elem_ID])
		

	def get_nonlinearwts(self, order, p, gamma, basis_phys_grad, quad_wts, vol, djac):
		'''
		This method calculates the smoothness indicator. (See Eq. 3.10 in [1])

		Inputs:
		-------
			order: solution order
			p: polynomial coeffs of of element being smoothed [ne, nb, ns]
			gamma: weighting constants in weno scheme (See Eq. 3.11 in [1])
			basis_phys_grad: evaluated gradient of the basis function in 
            		physical space [nq, nb, ndims]
            quad_wts: quadrature weights [nq, 1]  
            vol: element volumes [ne, 1]

		Outputs:
		--------
			weno_wts: returns the linear weight for the weno reconstruction [ne] 
		'''

		# s = 1 corresponds to the first derivative of the basis function
		s = 1
		# basis_p = (np.matmul(basis_phys_grad[:, :, :, 0], p)**2).transpose(0,2,1)
		# basis_p_qwts = np.einsum('ikj,jl -> ik',basis_p, quad_wts)

		# should this be ijn?
		# basis_p= np.einsum('ikjl, ijn, kb -> ijn', basis_phys_grad, p, quad_wts)
		# beta = np.einsum('i, ikl -> il', vol**(2*s-1), basis_p)

		basis_p = np.einsum('ikjl, ijn -> ikn', basis_phys_grad, p)**2
		beta = np.einsum('i, ikn, ikb -> in', vol**(2*s-1), basis_p, quad_wts*djac)
		eps = 1.0e-6
		# s = 2 corresponds to the second der. and order 2
		# if order == 2:
		# 	s = 2
		# 	#Unpack 
		# 	basis_phys_hessian = self.basis_phys_hessian_elems
		# 	hess_p = (np.matmul(basis_phys_hessian[:, :, :, 0], p)**2).transpose(0,2,1)
		# 	hess_p_qwts = np.einsum('ikj, jk -> ik', hess_p, quad_wts)

		# 	beta += vol**(2*s-1) * hess_p_qwts

		return gamma / (eps + beta)**2 # weno_wts [ne]


	def limit_element(self, solver, Uc):
		# Unpack		
		physics = solver.physics
		mesh = solver.mesh
		elem_helpers = solver.elem_helpers
		int_face_helpers = solver.int_face_helpers

		vols = self.elem_vols
		basis_phys_grads = elem_helpers.basis_phys_grad_elems
		quad_wts = elem_helpers.quad_wts
		elemP_IDs = self.elemP_IDs
		elemM_IDs = self.elemM_IDs
		djacs = self.djac_elems
		djacP = self.djac_elems[elemP_IDs]
		djacM = self.djac_elems[elemM_IDs]
		
		weno_wts = np.zeros([Uc.shape[0], 3, 3])
		djacs = self.djac_elems
		# Interpolate state at quadrature points over element and on faces
		U_elem_faces = helpers.evaluate_state(Uc, self.basis_val_elem_faces, 
				skip_interp=solver.basis.skip_interp)

		nq_elem = self.quad_wts_elem.shape[0]
		U_elem = U_elem_faces[:, :nq_elem, :]
		U_face = U_elem_faces[:, nq_elem:, :]
		# Average value of states
		U_bar = helpers.get_element_mean(U_elem, self.quad_wts_elem, djacs, 
				vols)
		# Calculate the eigenvectors
		self.right_eigen, self.left_eigen = physics.get_conv_eigenvectors(U_bar)
		Vc = np.einsum('elij, elj -> eli', self.left_eigen, Uc)

		# Determine if the elements requires limiting
		shock_indicated = self.shock_indicator(self, solver, Uc)

		# Unpack limiter info from shock indicator
		if self.U_elem is not None:
			p0 = np.einsum('ebij, elj -> eli', self.left_eigen, self.Um_elem)
			p1 = np.einsum('ebij, elj -> eli', self.left_eigen, self.U_elem)
			p2 = np.einsum('ebij, elj -> eli', self.left_eigen, self.Up_elem)

			p0_bar = np.einsum('ebij, elj -> eli', self.left_eigen, self.Um_bar)
			p1_bar = np.einsum('ebij, elj -> eli', self.left_eigen, self.U_bar)
			p2_bar = np.einsum('ebij, elj -> eli', self.left_eigen, self.Up_bar)
		else:
			raise NotImplementedError

		p0_tilde = p0 - p0_bar + p1_bar
		p2_tilde = p2 - p2_bar + p1_bar
	
		# Currently only implemented up to  P2
		if solver.order > 2:
			raise NotImplementedError

		# Calculate non-linear weights
		weno_wts[:, 0, :] = self.get_nonlinearwts(solver.order, p0, 0.001, 
				basis_phys_grads, quad_wts, vols, djacM)
		weno_wts[:, 1, :] = self.get_nonlinearwts(solver.order, p1, 0.998, 
				basis_phys_grads, quad_wts, vols, djacs)
		weno_wts[:, 2, :] = self.get_nonlinearwts(solver.order, p2, 0.001, 
				basis_phys_grads, quad_wts, vols, djacP)

		normal_wts = weno_wts / np.sum(weno_wts, axis=1).reshape([Uc.shape[0], 1, 3])

		# Update state_coeffs for indicated elements
		Vc[shock_indicated] = np.einsum('ik, ijk -> ijk', normal_wts[shock_indicated, 0], 
				p0_tilde[shock_indicated]) + \
				np.einsum('ik, ijk -> ijk', normal_wts[shock_indicated, 1],
				p1[shock_indicated]) + \
				np.einsum('ik, ijk -> ijk', normal_wts[shock_indicated, 2],
				p2_tilde[shock_indicated])

		Uc[shock_indicated] = np.einsum('elij, elj -> eli', self.right_eigen[shock_indicated], Vc[shock_indicated])
		# import code; code.interact(local=locals())
		return Uc # [ne, nq, 1]