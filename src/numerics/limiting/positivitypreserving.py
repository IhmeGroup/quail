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
#       File : src/numerics/limiting/positivitypreserving.py
#
#       Contains class definitions for positivity-preserving limiters.
#
# ------------------------------------------------------------------------ #
from abc import ABC, abstractmethod
import numpy as np

import errors
import general

import meshing.tools as mesh_tools

import numerics.helpers.helpers as helpers
import numerics.limiting.base as base


POS_TOL = 1.e-10


def trunc(a, decimals=8):
	'''
	This function truncates a float to a specified decimal place.
	Adapted from:
	https://stackoverflow.com/questions/42021972/
	truncating-decimal-digits-numpy-array-of-floats

	Inputs:
	-------
		a: value(s) to truncate
		decimals: truncated decimal place

	Outputs:
	--------
		truncated float
	'''
	return np.trunc(a*10**decimals)/(10**decimals)


class PositivityPreserving(base.LimiterBase):
	'''
	This class corresponds to the positivity-preserving limiter for the
	Euler equations. It inherits from the LimiterBase class. See
	See LimiterBase for detailed comments of attributes and methods. See
	the following references:
		[1] X. Zhang, C.-W. Shu, "On positivity-preserving high order
		discontinuous Galerkin schemes for compressible Euler equations
		on rectangular meshes," Journal of Computational Physics.
		229:8918–8934, 2010.
		[2] C. Wang, X. Zhang, C.-W. Shu, J. Ning, "Robust high order
		discontinuous Galerkin schemes for two-dimensional gaseous
		detonations," Journal of Computational Physics, 231:653-665, 2012.

	Attributes:
	-----------
	var_name1: str
		name of first variable involved in limiting (density)
	var_name2: str
		name of second variable involved in limiting (pressure)
	elem_vols: numpy array
		element volumes
	basis_val_elem_faces: numpy array
		stores basis values for element and faces
	quad_wts_elem: numpy array
		quadrature points for element
	djac_elems: numpy array
		stores Jacobian determinants for each element
	'''
	COMPATIBLE_PHYSICS_TYPES = general.PhysicsType.Euler

	def __init__(self, physics_type):
		super().__init__(physics_type)
		self.var_name1 = "Density"
		self.var_name2 = "Pressure"
		self.elem_vols = np.zeros(0)
		self.basis_val_elem_faces = np.zeros(0)
		self.quad_wts_elem = np.zeros(0)
		self.djac_elems = np.zeros(0)

	def precompute_helpers(self, solver):
		# Unpack
		elem_helpers = solver.elem_helpers
		int_face_helpers = solver.int_face_helpers
		self.elem_vols, _ = mesh_tools.element_volumes(solver.mesh, solver)

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

	def limit_solution(self, solver, Uc):
		# Unpack
		physics = solver.physics
		elem_helpers = solver.elem_helpers
		int_face_helpers = solver.int_face_helpers
		basis = solver.basis

		djac = self.djac_elems

		# Interpolate state at quadrature points over element and on faces
		U_elem_faces = helpers.evaluate_state(Uc, self.basis_val_elem_faces,
				skip_interp=basis.skip_interp)
		nq_elem = self.quad_wts_elem.shape[0]
		U_elem = U_elem_faces[:, :nq_elem, :]
		ne = self.elem_vols.shape[0]
		# Average value of state
		U_bar = helpers.get_element_mean(U_elem, self.quad_wts_elem, djac,
				self.elem_vols)

		# Density and pressure from averaged state
		rho_bar = physics.compute_variable(self.var_name1, U_bar, x=None)
		p_bar = physics.compute_variable(self.var_name2, U_bar, x=None)

		if np.any(rho_bar < 0.) or np.any(p_bar < 0.):
			raise errors.NotPhysicalError

		# Ignore divide-by-zero
		np.seterr(divide='ignore')

		''' Limit density '''
		# Compute density at quadrature points
		rho_elem_faces = physics.compute_variable(self.var_name1,
				U_elem_faces, x=None)
		# Check if limiting is needed
		theta = np.abs((rho_bar - POS_TOL)/(rho_bar - rho_elem_faces))
		# Truncate theta1; otherwise, can get noticeably different
		# results across machines, possibly due to poor conditioning in its
		# calculation
		theta1 = trunc(np.minimum(1., np.min(theta, axis=1)))

		irho = physics.get_state_index(self.var_name1)
		# Get IDs of elements that need limiting
		elem_IDs = np.where(theta1 < 1.)[0]
		# Modify density coefficients
		if basis.MODAL_OR_NODAL == general.ModalOrNodal.Nodal:
			Uc[elem_IDs, :, irho] = theta1[elem_IDs]*Uc[elem_IDs, :, irho] \
					+ (1. - theta1[elem_IDs])*rho_bar[elem_IDs, 0]
		elif basis.MODAL_OR_NODAL == general.ModalOrNodal.Modal:
			Uc[elem_IDs, :, irho] *= theta1[elem_IDs]
			Uc[elem_IDs, 0, irho] += (1. - theta1[elem_IDs, 0])*rho_bar[
					elem_IDs, 0, 0]
		else:
			raise NotImplementedError

		if np.any(theta1 < 1.):
			# Intermediate limited solution
			U_elem_faces = helpers.evaluate_state(Uc,
					self.basis_val_elem_faces,
					skip_interp=basis.skip_interp)

		''' Limit pressure '''
		# Compute pressure at quadrature points
		p_elem_faces = physics.compute_variable(self.var_name2, U_elem_faces, x=None)
		theta[:] = 1.
		# Indices where pressure is negative
		negative_p_indices = np.where(p_elem_faces < 0.)
		elem_IDs = negative_p_indices[0]
		i_neg_p  = negative_p_indices[1]

		theta[elem_IDs, i_neg_p] = (p_bar[elem_IDs, :, 0] - POS_TOL) / (
				p_bar[elem_IDs, :, 0] - p_elem_faces[elem_IDs, i_neg_p, :])

		# Truncate theta2; otherwise, can get noticeably different
		# results across machines, possibly due to poor conditioning in its
		# calculation
		theta2 = trunc(np.min(theta, axis=1))
		# Get IDs of elements that need limiting
		elem_IDs = np.where(theta2 < 1.)[0]
		# Modify coefficients
		if basis.MODAL_OR_NODAL == general.ModalOrNodal.Nodal:
			Uc[elem_IDs] = np.einsum('im, ijk -> ijk', theta2[elem_IDs], 
					Uc[elem_IDs]) + np.einsum('im, ijk -> ijk', 1 - theta2[
					elem_IDs], U_bar[elem_IDs])
		elif basis.MODAL_OR_NODAL == general.ModalOrNodal.Modal:
			Uc[elem_IDs] *= np.expand_dims(theta2[elem_IDs], axis=2)
			Uc[elem_IDs, 0] += np.einsum('im, ijk -> ik', 1 - theta2[
					elem_IDs], U_bar[elem_IDs])
		else:
			raise NotImplementedError

		np.seterr(divide='warn')

		return Uc # [ne, nq, ns]


class PositivityPreservingChem(PositivityPreserving):
	'''
    Class: PPLimiter
    ------------------
    This class contains information about the positivity preserving limiter
    '''

	COMPATIBLE_PHYSICS_TYPES = general.PhysicsType.Chemistry

	def __init__(self, physics_type):
		'''
		Method: __init__
		-------------------
		Initializes PPLimiter object
		'''
		super().__init__(physics_type)
		self.var_name3 = "Mixture"


	def limit_solution(self, solver, Uc):
		# Unpack
		physics = solver.physics
		elem_helpers = solver.elem_helpers
		int_face_helpers = solver.int_face_helpers
		basis = solver.basis

		djac = self.djac_elems

		# Interpolate state at quadrature points over element and on faces
		U_elem_faces = helpers.evaluate_state(Uc, self.basis_val_elem_faces,
				skip_interp=basis.skip_interp)
		nq_elem = self.quad_wts_elem.shape[0]
		U_elem = U_elem_faces[:, :nq_elem, :]

		# Average value of state
		U_bar = helpers.get_element_mean(U_elem, self.quad_wts_elem, djac,
				self.elem_vols)
		ne = self.elem_vols.shape[0]
		# Density and pressure from averaged state
		rho_bar = physics.compute_variable(self.var_name1, U_bar, x=None)
		p_bar = physics.compute_variable(self.var_name2, U_bar)
		rhoY_bar = physics.compute_variable(self.var_name3, U_bar, x=None)

		if np.any(rho_bar < 0.) or np.any(p_bar < 0.) or np.any(
				rhoY_bar < 0.):
			raise errors.NotPhysicalError

		# Ignore divide-by-zero
		np.seterr(divide='ignore')

		''' Limit density '''
		# Compute density
		rho_elem_faces = physics.compute_variable(self.var_name1,
				U_elem_faces, x=None)
		# Check if limiting is needed
		theta = np.abs((rho_bar - POS_TOL)/(rho_bar - rho_elem_faces))
		# Truncate theta1; otherwise, can get noticeably different
		# results across machines, possibly due to poor conditioning in its
		# calculation
		theta1 = trunc(np.minimum(1., np.min(theta, axis=1)))

		irho = physics.get_state_index(self.var_name1)
		# Get IDs of elements that need limiting
		elem_IDs = np.where(theta1 < 1.)[0]
		# Modify density coefficients
		if basis.MODAL_OR_NODAL == general.ModalOrNodal.Nodal:
			Uc[elem_IDs, :, irho] = theta1[elem_IDs]*Uc[elem_IDs, :, irho] \
					+ (1. - theta1[elem_IDs])*rho_bar[elem_IDs, 0]
		elif basis.MODAL_OR_NODAL == general.ModalOrNodal.Modal:
			Uc[elem_IDs, :, irho] *= theta1[elem_IDs]
			Uc[elem_IDs, 0, irho] += (1. - theta1[elem_IDs, 0])*rho_bar[
					elem_IDs, 0, 0]
		else:
			raise NotImplementedError

		if np.any(theta1 < 1.):
			# Intermediate limited solution
			U_elem_faces = helpers.evaluate_state(Uc,
					self.basis_val_elem_faces,
					skip_interp=basis.skip_interp)


		''' Limit mass fraction '''
		rhoY_elem_faces = physics.compute_variable(self.var_name3, U_elem_faces, x=None)
		theta = np.abs(rhoY_bar/(rhoY_bar-rhoY_elem_faces+POS_TOL))
		# Truncate theta2; otherwise, can get noticeably different
		# results across machines, possibly due to poor conditioning in its
		# calculation
		theta2 = trunc(np.minimum(1., np.amin(theta, axis=1)))

		irhoY = physics.get_state_index(self.var_name3)
		# Get IDs of elements that need limiting
		elem_IDs = np.where(theta2 < 1.)[0]
		# Modify density coefficients
		if basis.MODAL_OR_NODAL == general.ModalOrNodal.Nodal:
			Uc[elem_IDs, :, irhoY] = theta2[elem_IDs]*Uc[elem_IDs, :, 
					irhoY] + (1. - theta2[elem_IDs])*rho_bar[elem_IDs, 0]
		elif basis.MODAL_OR_NODAL == general.ModalOrNodal.Modal:
			Uc[elem_IDs, :, irhoY] *= theta2[elem_IDs]
			Uc[elem_IDs, 0, irhoY] += (1. - theta2[elem_IDs, 0])*rho_bar[
					elem_IDs, 0, 0]
		else:
			raise NotImplementedError

		if np.any(theta2 < 1.):
			U_elem_faces = helpers.evaluate_state(Uc,
					self.basis_val_elem_faces,
					skip_interp=basis.skip_interp)

		''' Limit pressure '''
		# Compute pressure at quadrature points
		p_elem_faces = physics.compute_variable(self.var_name2, U_elem_faces, x=None)
		theta[:] = 1.
		# Indices where pressure is negative
		negative_p_indices = np.where(p_elem_faces < 0.)
		elem_IDs = negative_p_indices[0]
		i_neg_p  = negative_p_indices[1]

		theta[elem_IDs, i_neg_p] = p_bar[elem_IDs, :, 0] / (
				p_bar[elem_IDs, :, 0] - p_elem_faces[elem_IDs, i_neg_p])

		# Truncate theta3; otherwise, can get noticeably different
		# results across machines, possibly due to poor conditioning in its
		# calculation
		theta3 = trunc(np.min(theta, axis=1))
		# Get IDs of elements that need limiting
		elem_IDs = np.where(theta3 < 1.)[0]
		# Modify coefficients
		if basis.MODAL_OR_NODAL == general.ModalOrNodal.Nodal:
			Uc[elem_IDs] = np.einsum('im, ijk -> ijk', theta3[elem_IDs], 
					Uc[elem_IDs]) + np.einsum('im, ijk -> ijk', 1 - theta3[
					elem_IDs], U_bar[elem_IDs])
		elif basis.MODAL_OR_NODAL == general.ModalOrNodal.Modal:
			Uc[elem_IDs] *= np.expand_dims(theta3[elem_IDs], axis=2)
			Uc[elem_IDs, 0] += np.einsum('im, ijk -> ik', 1 - theta3[
					elem_IDs], U_bar[elem_IDs])
		else:
			raise NotImplementedError

		np.seterr(divide='warn')

		return Uc # [ne, nq, ns]
