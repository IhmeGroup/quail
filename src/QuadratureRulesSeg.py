import numpy as np


def get_quadrature_points_weights(order, quad_type):

	# Only Gauss-Legendre for now
	qpts, qwts = get_quadrature_gauss_legendre(order)

	return qpts, qwts



def get_quadrature_gauss_legendre(order):
	# if order is even, increment by 1
	if order % 2 == 0:
		order += 1

	npts = (order + 1)//2

	qpts, qwts = np.polynomial.legendre.leggauss(npts)

	qpts.shape = -1,1
	qwts.shape = -1,1

	return qpts, qwts