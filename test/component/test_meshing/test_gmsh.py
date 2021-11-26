import numpy as np
import os
import pytest
import sys
sys.path.append('../src')

import numerics.basis.basis as basis_defs
import numerics.basis.tools as basis_tools
import general

import meshing.gmsh as mesh_gmsh

rtol = 1e-15
atol = 1e-15

def test_two_triangles_should_be_loaded_correctly():
	'''
	Make sure that
	'''
	# Path to mesh
	file_path = os.path.dirname(os.path.realpath(__file__)) + '/test_data/two_triangles_v4.msh'
	# Import mesh
	mesh = mesh_gmsh.import_gmsh_mesh(file_path)

	# Check element nodes
	np.testing.assert_array_equal(mesh.elements[0].node_IDs,
			np.array([0, 1, 3]))
	np.testing.assert_array_equal(mesh.elements[1].node_IDs,
			np.array([1, 2, 3]))
	# Check node coordinates
	np.testing.assert_allclose(mesh.node_coords,
			np.array([
			[0, 0],
			[1, 0],
			[1, 1],
			[0, 1]]), rtol, atol)
	# Make sure each boundary has one face
	groups = ['x1', 'x2', 'y1', 'y2']
	for group in groups:
		assert(len(mesh.boundary_groups[group].boundary_faces) == 1)
	# Make sure there is one interior face, between elements 0 and 1
	assert(len(mesh.interior_faces) == 1)
	assert(mesh.interior_faces[0].elemL_ID in [0, 1])
	assert(mesh.interior_faces[0].elemR_ID in [0, 1])


#@pytest.mark.parametrize('order', [
#	# Order of Lagrange basis
#	0, 1, 2, 3, 4, 5,
#])
#@pytest.mark.parametrize('Basis', [
#	# Basis class representing the element geometry
#	basis_defs.LagrangeSeg, basis_defs.LagrangeTri, basis_defs.LagrangeQuad,
#])
#def test_lagrange_basis_should_be_nodal(basis, order):
#	'''
#	This test ensures that the ith Lagrange basis is equal to 1 at the ith
#	node, and 0 at other nodes.
#
#	Inputs:
#	-------
#		basis: Pytest fixture containing the basis object to be tested
#		order: polynomial order of Lagrange basis being tested
#	'''
#	# Evaluate Lagrange basis at nodes
#	phi = basis.get_values(basis.get_nodes(order))
#	# This matrix should be identity
#	expected = np.identity(phi.shape[0])
#	# Assert
#	np.testing.assert_allclose(phi, expected, rtol, atol)
#
#
#@pytest.mark.parametrize('order', [
#	# Order of Lagrange basis
#	0, 1, 2, 3, 4, 5,
#])
#@pytest.mark.parametrize('Basis', [
#	# Basis class representing the element geometry
#	basis_defs.LegendreSeg, basis_defs.LegendreQuad
#])
#def test_legengre_massmatrix_should_be_diagonal(basis, order):
#	'''
#	This test ensures that the mass matrix for a Legendre basis is
#	diagonal
#
#	Inputs:
#	-------
#		modal_basis: Pytest fixture containing the basis object to be tested
#		order: polynomial order of Legendre basis being tested
#	'''
#
#	if basis.NDIMS == 1:
#		mesh = mesh_common.mesh_1D(num_elems=1, xmin=-1., xmax=1.)
#		# Set quadrature
#		basis.set_elem_quadrature_type("GaussLegendre")
#		basis.set_face_quadrature_type("GaussLegendre")
#		mesh.gbasis.set_elem_quadrature_type("GaussLegendre")
#		mesh.gbasis.set_face_quadrature_type("GaussLegendre")
#
#		iMM = basis_tools.get_elem_inv_mass_matrix(mesh, basis, order, -1)
#
#		should_be_zero = np.count_nonzero(np.abs(iMM - np.diag(np.diagonal(iMM))) > 10.*atol)
#
#	elif basis.NDIMS == 2:
#		mesh = mesh_common.mesh_2D(num_elems_x=1, num_elems_y=1, xmin=-1., xmax=1.,
#			 ymin=-1., ymax=1.)
#		# Set quadrature
#		basis.set_elem_quadrature_type("GaussLegendre")
#		basis.set_face_quadrature_type("GaussLegendre")
#		mesh.gbasis.set_elem_quadrature_type("GaussLegendre")
#		mesh.gbasis.set_face_quadrature_type("GaussLegendre")
#
#		iMM = basis_tools.get_elem_inv_mass_matrix(mesh, basis, order, -1)
#
#		should_be_zero = np.count_nonzero(np.abs(iMM - np.diag(np.diagonal(iMM))) > 1e-12)
#
#	np.testing.assert_allclose(should_be_zero, 0, 0, 0)
#
#@pytest.mark.parametrize('order', [
#	# Order of Lagrange basis
#	0, 1, 2, 3, 4, 5,
#])
#@pytest.mark.parametrize('Basis', [
#	# Basis class representing the element geometry
#	basis_defs.LagrangeSeg, basis_defs.LagrangeQuad,
#])
#def test_lagrange_massmatrix_should_be_symmetric(basis, order):
#	'''
#	This test ensures that the mass matrix for a Lagrange basis is
#	symmetric
#
#	Inputs:
#	-------
#		modal_basis: Pytest fixture containing the basis object to be tested
#		order: polynomial order of Legendre basis being tested
#	'''
#
#	if basis.NDIMS == 1:
#		mesh = mesh_common.mesh_1D(num_elems=1, xmin=-1., xmax=1.)
#		# Set quadrature
#		basis.set_elem_quadrature_type("GaussLegendre")
#		basis.set_face_quadrature_type("GaussLegendre")
#		mesh.gbasis.set_elem_quadrature_type("GaussLegendre")
#		mesh.gbasis.set_face_quadrature_type("GaussLegendre")
#
#		iMM = basis_tools.get_elem_inv_mass_matrix(mesh, basis, order, -1)
#		iMM_T = np.transpose(iMM)
#	elif basis.NDIMS == 2:
#		mesh = mesh_common.mesh_2D(num_elems_x=1, num_elems_y=1, xmin=-1., xmax=1.,
#			 ymin=-1., ymax=1.)
#		# Set quadrature
#		basis.set_elem_quadrature_type("GaussLegendre")
#		basis.set_face_quadrature_type("GaussLegendre")
#		mesh.gbasis.set_elem_quadrature_type("GaussLegendre")
#		mesh.gbasis.set_face_quadrature_type("GaussLegendre")
#
#		iMM = basis_tools.get_elem_inv_mass_matrix(mesh, basis, order, -1)
#		iMM_T = np.transpose(iMM)
#
#	should_be_one = np.abs(iMM - iMM_T) + 1.0
#	expected=np.ones_like(iMM)
#
#	np.testing.assert_allclose(should_be_one, expected, 1e-13, 1e-13)
