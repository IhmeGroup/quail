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
	This function computes element averages of the state.

	Inputs:
	-------
        Uq: values of state at the quadrature points [ne, nq, ns]
	    quad_wts: quadrature weights [nq, 1]
	    djac: Jacobian determinants [ne, nq, 1]
	    vol: element volume [ne]

	Outputs:
	--------
	    U_mean: mean values of state variables [ne, 1, ns]
	'''
	U_mean = np.einsum('ijk, jm, ijm, i -> imk', Uq, quad_wts, djac, 1/vol)

	return U_mean # [ne, 1, ns]


def evaluate_state(Uc, basis_val, skip_interp=False):
	'''
	This function evaluates the state based on the given basis values.

	Inputs:
	-------
	    Uc: state coefficients [ne, nb, ns]
	    basis_val: basis values [ne, nq, nb]
	    skip_interp: if True, then will simply copy the state coefficients;
	    	useful for a colocated scheme, i.e. quadrature points and
	    	solution nodes (for a nodal basis) are the same

	Outputs:
	--------
	    Uq: values of state [ne, nq, ns]
	'''
	if skip_interp:
		Uq = Uc.copy()
	else:
		if basis_val.ndim == 3:
			# For faces, there is a different basis_val for each face
			Uq = np.einsum('ijn, ink -> ijk', basis_val, Uc)
		else:
			# For elements, all elements have the same basis_val
			Uq = np.einsum('jn, ink -> ijk', basis_val, Uc)

	return Uq # [ne, nq, ns]


def evaluate_gradient(Uc, basis_phys_grad_elems):
	'''
	This function evaluates the gradient of the state based on the 
	physical gradient of the basis.

	Inputs:
	-------
	    Uc: state coefficients [ne, nb, ns]
	    basis_phys_grad_elems: evaluated gradient of the basis function in
			physical space [nq, nb, ndims]

	Outputs:
	--------
	    gUq: gradient of the state [ne, nq, ns, ndims]
	'''
	gUq = np.einsum('ijml, imk -> ijkl', basis_phys_grad_elems, Uc)

	return gUq # [ne, nq, ns, ndims]