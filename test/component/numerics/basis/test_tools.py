import numpy as np
import pytest
import sys
sys.path.append('../src')

import numerics.basis.basis as basis_defs
import numerics.basis.tools as basis_tools
import meshing.common as mesh_common

rtol = 1e-14
atol = 1e-14


def test_check_function_set_nodetypes_0():
	'''
	Checks setter function for nodetypes
	'''
	fcn = basis_tools.set_1D_node_calc("Equidistant")
	assert fcn == basis_tools.equidistant_nodes_1D_range


def test_check_function_set_nodetypes_1():
	'''
	Checks setter function for nodetypes
	'''
	fcn = basis_tools.set_1D_node_calc("GaussLobatto")
	assert fcn == basis_tools.gauss_lobatto_nodes_1D_range


def test_check_function_set_nodetypes_err():
	'''
	Checks setter function for nodetypes with an invalid key
	'''
	pytest.raises(KeyError, basis_tools.set_1D_node_calc, 
		node_type="InvalidKey")


def test_1D_nodes_two_points():
	'''
	Tests that two points remain at their locations
	'''
	xnodes = basis_tools.equidistant_nodes_1D_range(0.0, 10.0, 2)
	expected = np.array([0.0, 10.0])
	np.testing.assert_allclose(xnodes, expected, rtol, atol)


def test_1D_nodes_nnodes_not_one():
	'''
	Tests that an error is thrown with nnodes<=1
	'''
	pytest.raises(ValueError, basis_tools.equidistant_nodes_1D_range,
		start = 0.0, stop=1.0, nnodes=1)


def test_1D_nodes_start_greater_than_stop():
	'''
	Fails is the start is greater than the stop
	'''
	pytest.raises(ValueError, basis_tools.equidistant_nodes_1D_range,
		start = 1.0, stop = 0.0, nnodes=2)

def test_1D_nodes_three_points():
	'''
	Checks three points
	'''
	xnodes = basis_tools.equidistant_nodes_1D_range(-1.0, 1.0, 3)
	expected = np.array([-1., 0., 1.])
	np.testing.assert_allclose(xnodes, expected, rtol, atol)


def test_1D_gl_nodes_nnodes_not_one():
	'''
	Tests that an error is thrown with nnodes<=1
	'''
	pytest.raises(ValueError, basis_tools.gauss_lobatto_nodes_1D_range,
		start = 0.0, stop=1.0, nnodes=1)


def test_1D_gl_nodes_start_greater_than_stop():
	'''
	Fails is the start is greater than the stop
	'''
	pytest.raises(ValueError, basis_tools.gauss_lobatto_nodes_1D_range,
		start = 1.0, stop = 0.0, nnodes=2)


def test_1D_nodes_two_points_wrong_start_stop():
	'''
	Tests that two points remain at their locations
	'''
	pytest.raises(ValueError, basis_tools.gauss_lobatto_nodes_1D_range,
		start = 0.0, stop = 10.0, nnodes=2)


def test_1D_nodes_three_points():
	'''
	Checks three points
	'''
	xnodes = basis_tools.gauss_lobatto_nodes_1D_range(-1.0, 1.0, 3)
	expected = np.array([-1., 0., 1.])
	np.testing.assert_allclose(xnodes, expected, rtol, atol)


def test_legendre_basis_1D_p1():
	'''
	Tests the 1D legendre basis for p1
	'''
	xpts = np.array([-1., 0., 1.])
	p = 1

	expected = np.zeros([3, 2])

	expected[0, 0] = 1.0
	expected[0, 1] = -1.0
	expected[1, 0] = 1.0
	expected[1, 1] = 0.0
	expected[2, 0] = 1.0
	expected[2, 1] = 1.0

	basis_val = np.zeros_like(expected)
	basis_tools.get_legendre_basis_1D(xpts, p, basis_val=basis_val)

	np.testing.assert_allclose(basis_val, expected, rtol, atol)


def test_set_basis_type_lagrangeseg():
	'''
	Checks setter function for BasisType
	'''
	basis = basis_tools.set_basis(1, "LagrangeSeg")
	assert basis == basis_defs.LagrangeSeg(1)


def test_set_basis_type_lagrangequad():
	'''
	Checks setter function for BasisType
	'''
	basis = basis_tools.set_basis(1, "LagrangeQuad")
	assert basis == basis_defs.LagrangeQuad(1)


def test_set_basis_type_lagrangequad():
	'''
	Checks setter function for BasisType
	'''
	basis = basis_tools.set_basis(1, "LagrangeQuad")
	assert basis == basis_defs.LagrangeQuad(1)


def test_set_basis_type_lagrangetri():
	'''
	Checks setter function for BasisType
	'''
	basis = basis_tools.set_basis(1, "LagrangeTri")
	assert basis == basis_defs.LagrangeTri(1)


def test_set_basis_type_legendreseg():
	'''
	Checks setter function for BasisType
	'''
	basis = basis_tools.set_basis(1, "LegendreSeg")
	assert basis == basis_defs.LegendreSeg(1)


def test_set_basis_type_legendrequad():
	'''
	Checks setter function for BasisType
	'''
	basis = basis_tools.set_basis(1, "LegendreQuad")
	assert basis == basis_defs.LegendreQuad(1)


def test_set_basis_type_hierarchicH1Tri():
	'''
	Checks setter function for BasisType
	'''
	basis = basis_tools.set_basis(1, "HierarchicH1Tri")
	assert basis == basis_defs.HierarchicH1Tri(1)
