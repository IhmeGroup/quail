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
	This function computes the mean over n elements.

	Inputs:
	-------
        Uq: values of state at the quadrature points [n, nq, ns]
	    quad_wts: quadrature weights [nq, 1]
	    djac: Jacobian determinants [n, nq, 1]
	    vol: element volume [n]

	Outputs:
	--------
	    U_mean: mean values of state variables [1, ns]
	'''
	U_mean = np.einsum('ijk, jm, ijm, i -> imk', Uq, quad_wts, djac, 1/vol)

	return U_mean


def evaluate_state(Uc, basis_val, skip_interp=False):
	'''
	This function evaluates the state based on the given basis values.

	Inputs:
	-------
	    Uc: state coefficients [n, nb, sr]
	    basis_val: basis values [n, nq, nb]
	    skip_interp: if True, then will simply copy the state coefficients;
	    	useful for a colocated scheme, i.e. quadrature points and
	    	solution nodes (for a nodal basis) are the same

	Outputs:
	--------
	    Uq: values of state [n, nq, sr]
	'''
	if skip_interp:
		Uq = Uc.copy()
	else:
		if basis_val.ndim == 3:
			# For faces, there is a different basis_val for each face
			Uq = np.einsum('ijn, ink -> ijk', basis_val, Uc)
		else:
			# For elements, all elements have the same basis_val (for now)
			Uq = np.einsum('jn, ink -> ijk', basis_val, Uc)

	return Uq
