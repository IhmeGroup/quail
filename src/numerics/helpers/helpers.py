import numpy as np


def get_element_mean(Uq, quad_wts, djac, vol):
	U_mean = np.matmul(Uq.transpose(), quad_wts*djac).transpose()/vol

	return U_mean


def evaluate_state(Uc, basis_val, skip_interp=False):
	if skip_interp:
		Up = Uc.copy()
	else:
		Up = np.matmul(basis_val, Uc)

	return Up