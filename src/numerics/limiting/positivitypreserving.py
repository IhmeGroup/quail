from abc import ABC, abstractmethod
import code
import numpy as np 

import errors
import general

import meshing.tools as mesh_tools

import numerics.limiting.base as base


POS_TOL = 1.e-10


class PositivityPreserving(base.LimiterBase):
	'''
    Class: PPLimiter
    ------------------
    This class contains information about the positivity preserving limiter
    '''

	COMPATIBLE_PHYSICS_TYPES = general.PhysicsType.Euler

	def __init__(self, physics_type):
		'''
		Method: __init__
		-------------------
		Initializes PPLimiter object
		'''
		super().__init__(physics_type)
		self.scalar1 = "Density"
		self.scalar2 = "Pressure"
		self.elem_vols = np.zeros(0)
		self.basis_val_elem_faces = None
		self.quad_wts_elem = np.zeros(0)
		self.djac_elems = np.zeros(0)

	def precompute_operators(self, solver):
		elem_ops = solver.elem_operators
		iface_ops = solver.iface_operators
		_, self.elem_vols = mesh_tools.element_volumes(solver.mesh, solver)

		# basis values in element interior and on faces
		basis_val_faces = iface_ops.faces_to_basisL.copy()
		bshape = basis_val_faces.shape
		basis_val_faces.shape = (bshape[0]*bshape[1], bshape[2])
		self.basis_val_elem_faces = np.vstack((elem_ops.basis_val, basis_val_faces))

		# Jacobian determinant
		self.djac_elems = elem_ops.djac_elems

		# Element quadrature weights
		self.quad_wts_elem = elem_ops.quad_wts

	def limit_element(self, solver, elem, U):
		'''
		Method: limit_element
		------------------------
		Limits the solution on each element

		INPUTS:
			solver: type of solver (i.e. DG, ADER-DG, etc...)
			elem: element index

		OUTPUTS:
			U: solution array
		'''
		physics = solver.EqnSet
		elem_ops = solver.elem_operators
		iface_ops = solver.iface_operators

		djac = self.djac_elems[elem]

		# interpolate state and gradient at quad points over element and on faces
		u_elem_faces = np.matmul(self.basis_val_elem_faces, U)
		nq_elem = self.quad_wts_elem.shape[0]
		u_elem = u_elem_faces[:nq_elem,:]

		# Average value of state
		vol = self.elem_vols[elem]
		u_bar = np.matmul(u_elem.transpose(), self.quad_wts_elem*djac).T/vol

		# Density and pressure
		rho_bar = physics.ComputeScalars(self.scalar1, u_bar)
		p_bar = physics.ComputeScalars(self.scalar2, u_bar)

		if np.any(rho_bar < 0.) or np.any(p_bar < 0.):
			raise Errors.NotPhysicalError

		np.seterr(divide='ignore')

		''' Limit density '''
		# Compute density
		rho_elem_faces = physics.ComputeScalars(self.scalar1, u_elem_faces)
		theta = np.abs((rho_bar - POS_TOL)/(rho_bar - rho_elem_faces))
		theta1 = np.amin([1., np.amin(theta)])

		# Rescale
		if theta1 < 1.:
			irho = physics.GetStateIndex(self.scalar1)
			U[:,irho] = theta1*U[:,irho] + (1. - theta1)*rho_bar

			# Intermediate limited solution
			u_elem_faces = np.matmul(self.basis_val_elem_faces, U)

		''' Limit pressure '''
		p_elem_faces = physics.ComputeScalars(self.scalar2, u_elem_faces)
		# theta = np.abs((p_bar - POS_TOL)/(p_bar - p_elem_faces))
		theta[:] = 1.0
		i_pos_p = (p_elem_faces < 0.).reshape(-1) # indices where pressure is negative
		theta[i_pos_p] = p_bar/(p_bar - p_elem_faces[i_pos_p])
		theta2 = np.amin(theta)
		if theta2 < 1.:
			U = theta2*U + (1. - theta2)*u_bar

		np.seterr(divide='warn')

		return U


class ScalarPositivityPreserving(base.LimiterBase):
	'''
	Class: PPScalarLimiter
	-------------------
	This class contains information about the scalar positivity preserving limiter
	'''

	COMPATIBLE_PHYSICS_TYPES = [general.PhysicsType.ConstAdvScalar, general.PhysicsType.Burgers]

	def __init__(self, physics_type):
		'''
		Method: __init__
		-------------------
		Initializes limiter object
		'''
		super().__init__(physics_type)
		self.elem_vols = np.zeros(0)
		self.basis_val_elem_faces = None
		self.quad_wts_elem = np.zeros(0)
		self.djac_elems = np.zeros(0)

	def precompute_operators(self, solver):
		elem_ops = solver.elem_operators
		iface_ops = solver.iface_operators
		_, self.elem_vols = mesh_tools.element_volumes(solver.mesh, solver)

		# basis values in element interior and on faces
		basis_val_faces = iface_ops.faces_to_basisL.copy()
		bshape = basis_val_faces.shape
		basis_val_faces.shape = (bshape[0]*bshape[1], bshape[2])
		self.basis_val_elem_faces = np.vstack((elem_ops.basis_val, basis_val_faces))

		# Jacobian determinant
		self.djac_elems = elem_ops.djac_elems

		# Element quadrature weights
		self.quad_wts_elem = elem_ops.quad_wts

	def limit_element(self, solver, elem, U):
		'''
		Method: limit_element
		------------------------
		Limits the solution on each element

		INPUTS:
			solver: type of solver (i.e. DG, ADER-DG, etc...)
			elem: element index

		OUTPUTS:
			U: solution array
		'''
		physics = solver.EqnSet
		elem_ops = solver.elem_operators
		iface_ops = solver.iface_operators

		djac = self.djac_elems[elem]

		# interpolate state and gradient at quad points over element and on faces
		u_elem_faces = np.matmul(self.basis_val_elem_faces, U)
		nq_elem = self.quad_wts_elem.shape[0]
		u_elem = u_elem_faces[:nq_elem,:]

		# Average value of state
		vol = self.elem_vols[elem]
		u_bar = np.matmul(u_elem.transpose(), self.quad_wts_elem*djac).T/vol

		np.seterr(divide='ignore')

		''' Limit scalar '''
		theta = np.abs((u_bar - POS_TOL)/(u_bar - u_elem_faces))
		theta1 = np.amin([1., np.amin(theta)])

		# Rescale
		if theta1 < 1.:
			U = theta1*U + (1. - theta1)*u_bar

		np.seterr(divide='warn')

		return U

