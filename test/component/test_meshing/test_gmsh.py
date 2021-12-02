import numpy as np
import os
import pytest
import sys
sys.path.append('../src')

import meshing.gmsh as mesh_gmsh

rtol = 1e-15
atol = 1e-15

@pytest.mark.parametrize('mesh_file_name', [
	# Name of gmsh file
	'two_triangles_v4.msh',
	'two_triangles_v2.msh',
])
def test_two_triangles_should_be_loaded_correctly(mesh_file_name):
	'''
	Make sure that a simple Gmsh mesh with two triangles is correctly loaded.
	'''
	# Path to mesh
	file_path = (os.path.dirname(os.path.realpath(__file__)) + '/test_data/' +
			mesh_file_name)
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
