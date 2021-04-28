# ------------------------------------------------------------------------ #
#
#       File : src/numerics/helpers/helpers.py
#
#       Contains general numerics-related helper functions
#
# ------------------------------------------------------------------------ #
import numpy as np
import ctypes



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

	U_mean = np.zeros(Uq.shape[0], Uq.shape[-1])

	lib.get_element_mean(
		ctypes.c_void_p(Uq.ctypes.data),
		ctypes.c_void_p(w.ctypes.data),
		ctypes.c_void_p(dJ.ctypes.data),
		ctypes.c_void_p(vol.ctypes.data),
		ctypes.c_void_p(U_mean.ctypes.data),
		ctypes.c_int(ne_),
		ctypes.c_int(nq),
		ctypes.c_int(ns))


	return U_mean # [ne, 1, ns]


def evaluate_state(Uc, basis_val, skip_interp=False, lib=None):
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

	ne = Uc.shape[0]
	nb = basis_val.shape[-1]
	ns = Uc.shape[-1]
	basis_dim = basis_val.ndim

	if basis_dim == 3:
		nq = basis_val.shape[1]
		Uq = np.zeros([ne, nq, ns])

		lib.evaluate_face_state(
			ctypes.c_void_p(Uc.ctypes.data), 
			ctypes.c_void_p(basis_val.ctypes.data),
			ctypes.c_void_p(Uq.ctypes.data),
			ctypes.c_int(ne), 
			ctypes.c_int(nq), 
			ctypes.c_int(nb),
			ctypes.c_int(ns),
			ctypes.c_int(basis_dim), 
			ctypes.c_bool(skip_interp), 
			)

	else:
		nq = basis_val.shape[0]
		Uq = np.zeros([ne, nq, ns])

		lib.evaluate_elem_state(
			ctypes.c_void_p(Uc.ctypes.data), 
			ctypes.c_void_p(basis_val.ctypes.data),
			ctypes.c_void_p(Uq.ctypes.data),
			ctypes.c_int(ne), 
			ctypes.c_int(nq), 
			ctypes.c_int(nb),
			ctypes.c_int(ns),
			ctypes.c_int(basis_dim), 
			ctypes.c_bool(skip_interp), 
			)

	return Uq # [ne, nq, ns]


# def get_element_mean(Uq, quad_wts, djac, vol):
# 	'''
# 	This function computes element averages of the state.

# 	Inputs:
# 	-------
#         Uq: values of state at the quadrature points [ne, nq, ns]
# 	    quad_wts: quadrature weights [nq, 1]
# 	    djac: Jacobian determinants [ne, nq, 1]
# 	    vol: element volume [ne]

# 	Outputs:
# 	--------
# 	    U_mean: mean values of state variables [ne, 1, ns]
# 	'''
# 	U_mean = np.einsum('ijk, jm, ijm, i -> imk', Uq, quad_wts, djac, 1/vol)

# 	return U_mean # [ne, 1, ns]


# def evaluate_state(Uc, basis_val, skip_interp=False, lib=None):
# 	'''
# 	This function evaluates the state based on the given basis values.

# 	Inputs:
# 	-------
# 	    Uc: state coefficients [ne, nb, ns]
# 	    basis_val: basis values [ne, nq, nb]
# 	    skip_interp: if True, then will simply copy the state coefficients;
# 	    	useful for a colocated scheme, i.e. quadrature points and
# 	    	solution nodes (for a nodal basis) are the same

# 	Outputs:
# 	--------
# 	    Uq: values of state [ne, nq, ns]
# 	'''
# 	if skip_interp:
# 		Uq = Uc.copy()
# 	else:
# 		if basis_val.ndim == 3:
# 			# For faces, there is a different basis_val for each face
# 			Uq = np.einsum('ijn, ink -> ijk', basis_val, Uc)
# 			# import code; code.interact(local=locals())
# 		else:
# 			# For elements, all elements have the same basis_val
# 			Uq = np.einsum('jn, ink -> ijk', basis_val, Uc)

# 	return Uq # [ne, nq, ns]
