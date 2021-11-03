import numpy as np
import pytest
import sys
sys.path.append('../src')

import numerics.basis.basis as basis_defs
import numerics.basis.tools as basis_tools
from numerics.quadrature import segment, quadrilateral, triangle, \
		hexahedron
import general

import meshing.common as mesh_common

rtol = 1e-13
atol = 1e-14


@pytest.mark.parametrize('order',
	# Quadrature order
	range(1, 20)
)
def test_quadrature_weights_sum(order):
	'''
	This test ensures that the quadrature weights sum to the size of the
	reference element (length in 1D, area in 2D, volume in 3D).

	Inputs:
	-------
		order: quadrature order
	'''
	# Quadrature types
	gauss_legendre_type = general.QuadratureType.GaussLegendre
	gauss_lobatto_type = general.QuadratureType.GaussLobatto
	dunavant_type = general.QuadratureType.Dunavant

	''' 1D line segment '''
	segment_length = 2.
	# Gauss-Legendre
	_, quad_wts = segment.get_quadrature_points_weights(order, 
			gauss_legendre_type)
	np.testing.assert_allclose(np.sum(quad_wts), segment_length, rtol, atol)
	# Gauss-Lobatto
	_, quad_wts = segment.get_quadrature_points_weights(order, 
			gauss_lobatto_type)
	np.testing.assert_allclose(np.sum(quad_wts), segment_length, rtol, atol)

	''' 2D quadrilateral '''
	quadrilateral_area = 4.
	# Gauss-Legendre
	_, quad_wts = quadrilateral.get_quadrature_points_weights(order, 
			gauss_legendre_type)
	np.testing.assert_allclose(np.sum(quad_wts), quadrilateral_area, rtol, 
			atol)
	# Gauss-Lobatto
	_, quad_wts = quadrilateral.get_quadrature_points_weights(order, 
			gauss_lobatto_type)
	np.testing.assert_allclose(np.sum(quad_wts), quadrilateral_area, rtol, 
			atol)

	''' 3D hexahedron '''
	hexahedron_volume = 8.
	# Gauss-Legendre
	_, quad_wts = hexahedron.get_quadrature_points_weights(order, 
			gauss_legendre_type)
	np.testing.assert_allclose(np.sum(quad_wts), hexahedron_volume, rtol, 
			atol)
	# Gauss-Lobatto
	_, quad_wts = hexahedron.get_quadrature_points_weights(order, 
			gauss_lobatto_type)
	np.testing.assert_allclose(np.sum(quad_wts), hexahedron_volume, rtol, 
			atol)

	''' 2D triangle '''
	triangle_area = 0.5
	# Gauss-Legendre
	_, quad_wts = triangle.get_quadrature_points_weights(order, 
			gauss_legendre_type)
	np.testing.assert_allclose(np.sum(quad_wts), triangle_area, rtol, atol)
	# Dunavant
	_, quad_wts = triangle.get_quadrature_points_weights(order, 
			dunavant_type)
	np.testing.assert_allclose(np.sum(quad_wts), triangle_area, rtol, atol)


def integrate_basis_functions(basis, order):
	# Get quadrature data
	quad_pts, quad_wts = basis.get_quadrature_data(order)
	# Interpolate basis functions to quadrature points
	basis.get_basis_val_grads(quad_pts, get_val=True)
	basis_val = basis.basis_val
	# Contract with quadrature weights to evaluate integral
	return np.matmul(basis_val.transpose(), quad_wts)


@pytest.mark.parametrize('order',
	# Quadrature order
	range(1, 8)
)
@pytest.mark.parametrize('Basis', [
	# Basis class
	basis_defs.LagrangeSeg, basis_defs.LagrangeQuad, basis_defs.LagrangeHex,
	basis_defs.HierarchicH1Tri
])
def test_quadrature_order_vs_quadrature_order_plus_one(basis, order):
	'''
	This test ensures that quadrature rules of order p and p+1 obtain the
	same value when integrating a polyomial of degree p

	Inputs:
	-------
		modal_basis: Pytest fixture containing the basis object to be tested
		order: polynomial order of Legendre basis being tested
	'''

	if basis.SHAPE_TYPE == general.ShapeType.Triangle:
		''' Gauss-Legendre quadrature '''
		basis.set_elem_quadrature_type("GaussLegendre")
		# Integrate basis functions
		val1 = integrate_basis_functions(basis, order)
		val2 = integrate_basis_functions(basis, order+1)
		np.testing.assert_allclose(val1, val2, rtol, atol)

		''' Dunavant quadrature '''
		basis.set_elem_quadrature_type("Dunavant")
		# Integrate basis functions
		val1 = integrate_basis_functions(basis, order)
		val2 = integrate_basis_functions(basis, order+1)
		np.testing.assert_allclose(val1, val2, rtol, atol)
	else:
		''' Gauss-Legendre quadrature '''
		# Integrate basis functions
		basis.set_elem_quadrature_type("GaussLegendre")
		val1 = integrate_basis_functions(basis, order)
		val2 = integrate_basis_functions(basis, order+1)
		np.testing.assert_allclose(val1, val2, rtol, atol)

		''' Gauss-Lobatto quadrature '''
		# Integrate basis functions
		basis.set_elem_quadrature_type("GaussLobatto")
		val1 = integrate_basis_functions(basis, order)
		val2 = integrate_basis_functions(basis, order+1)
		np.testing.assert_allclose(val1, val2, rtol, atol)

