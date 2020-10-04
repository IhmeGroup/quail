# ------------------------------------------------------------------------ #
#
#       File : src/numerics/helpers/helpers.py
#
#       Contains general numerics-related helper functions
#
# ------------------------------------------------------------------------ #
import numpy as np


def get_element_mean(Uq, quad_wts, djac, vol):
	'''
	This function computes the mean over an element.

	Inputs:
	-------
	    Uq: values of state at the quadrature points [nq, sr]
	    quad_wts: quadrature weights [nq, 1]
	    djac: Jacobian determinants [nq, 1]
	    vol: element volume

	Outputs:
	--------
	    U_mean: mean values of state variables [1, sr]
	'''
	U_mean = np.matmul(Uq.transpose(), quad_wts*djac).transpose()/vol

	return U_mean


def evaluate_state(Uc, basis_val, skip_interp=False):
	'''
	This function evaluates the state based on the given basis values.

	Inputs:
	-------
	    Uc: state coefficients [nb, sr]
	    basis_val: basis values [nq, nb]
	    skip_interp: if True, then will simply copy the state coefficients;
	    	useful for a colocated scheme, i.e. quadrature points and
	    	solution nodes (for a nodal basis) are the same

	Outputs:
	--------
	    Uq: values of state [nq, sr]
	'''
	if skip_interp:
		Uq = Uc.copy()
	else:
		# TODO: This is a temporary hack, will have to vectorize some other
		# things to remove this
		if Uc.ndim == 3:
			if basis_val.ndim == 3:
				# For faces, there is a different basis_val for each face
				Up = np.einsum('ijk,ikl->ijl', basis_val, Uc)
			else:
				# For elements, all elements have the same basis_val (for now)
				Up = np.einsum('jk,ikl->ijl', basis_val, Uc)
		else:
			Up = np.matmul(basis_val, Uc)

	return Up
