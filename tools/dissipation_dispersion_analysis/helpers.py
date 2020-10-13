import numpy as np

import general
import numerics.basis.tools as basis_tools


def get_matrices_and_basis_vals(solver, mesh, order):
	'''
	Calculate inverse mass matrix, stiffness matrix, basis values
	'''
	# Inverse mass matrix
	iMM = solver.elem_helpers.iMM_elems[0]
	# Stiffness matrix
	SM = basis_tools.get_stiffness_matrix(solver, mesh, order, 0)

	# Basis value at left face
	basis_valL = solver.int_face_helpers.faces_to_basisL[0].transpose()
	# Basis value at right face
	basis_valR = solver.int_face_helpers.faces_to_basisL[1].transpose()
	# Number of basis coefficients
	nb = solver.basis.get_num_basis_coeff(order)

	return iMM, SM, basis_valL, basis_valR, nb


def get_eig_values(iMM, SM, basis_valL, basis_valR, L, p, h, alpha):
	'''
	Calculate eigenvalues
	'''
	A = np.matmul(-h/1.j*iMM, SM + np.matmul(0.5*basis_valL, 
			alpha*basis_valL.transpose() + (2. - alpha)* \
			basis_valR.transpose()*np.exp(-1.j*L*(p+1))) - np.matmul(
			0.5*basis_valR, (2.-alpha)*basis_valR.transpose() + 
			alpha*np.exp(1.j*L*(p+1))*basis_valL.transpose()))

	# Eigenvalues
	Omega, _ = np.linalg.eig(A) 

	Omega_r = np.real(Omega)
	Omega_i = np.imag(Omega)

	return Omega_r, Omega_i