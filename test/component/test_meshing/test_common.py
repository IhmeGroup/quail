import numpy as np
import pytest
import sys
sys.path.append('../src')

import meshing.common as mesh_common

rtol = 1e-15
atol = 1e-15

def test_mesh_1D_for_3_elements():
	'''
	Make sure that a 1D mesh with three elements has the correct nodes,
	neighbors, and faces.
	'''
	# Create mesh
	mesh = mesh_common.mesh_1D(num_elems=3, xmin=0, xmax=3)

	# Check node coordinates
	np.testing.assert_allclose(mesh.node_coords, np.array([[0, 1, 2, 3]]).T, rtol,
			atol)
	# Check node IDs of elements
	np.testing.assert_array_equal(mesh.elem_to_node_IDs,
			np.array([[0, 1], [1, 2], [2, 3]]))
	# Check neighbors
	np.testing.assert_array_equal(mesh.elements[0].face_to_neighbors,
			np.array([-1, 1]))
	np.testing.assert_array_equal(mesh.elements[1].face_to_neighbors,
			np.array([0, 2]))
	np.testing.assert_array_equal(mesh.elements[2].face_to_neighbors,
			np.array([1, -1]))
	# Check faces
	for face_ID in range(2):
		assert(mesh.interior_faces[face_ID].elemL_ID == face_ID)
		assert(mesh.interior_faces[face_ID].elemR_ID == face_ID + 1)
		assert(mesh.interior_faces[face_ID].faceL_ID == 1)
		assert(mesh.interior_faces[face_ID].faceR_ID == 0)
	# Check boundaries
	assert(mesh.boundary_groups['x1'].boundary_faces[0].elem_ID == 0)
	assert(mesh.boundary_groups['x1'].boundary_faces[0].face_ID == 0)
	assert(mesh.boundary_groups['x2'].boundary_faces[0].elem_ID == 2)
	assert(mesh.boundary_groups['x2'].boundary_faces[0].face_ID == 1)

def test_mesh_2D_for_4_elements():
	'''
	Make sure that a 2D mesh with four elements has the correct nodes,
	neighbors, and faces.
	'''
	# Create mesh
	mesh = mesh_common.mesh_2D(num_elems_x=2, num_elems_y=2, xmin=0, xmax=2,
			ymin=0, ymax=2)

	# Check node coordinates
	np.testing.assert_allclose(mesh.node_coords,
			np.array([
				[0, 0], [1, 0], [2, 0],
				[0, 1], [1, 1], [2, 1],
				[0, 2], [1, 2], [2, 2],
				]), rtol, atol)
	# Check node IDs of elements
	np.testing.assert_array_equal(mesh.elem_to_node_IDs,
			np.array([
				[0, 1, 3, 4],
				[1, 2, 4, 5],
				[3, 4, 6, 7],
				[4, 5, 7, 8],
				]))
	# Check neighbors
	np.testing.assert_array_equal(mesh.elements[0].face_to_neighbors,
			np.array([-1, 1, 2, -1]))
	np.testing.assert_array_equal(mesh.elements[1].face_to_neighbors,
			np.array([-1, -1, 3, 0]))
	np.testing.assert_array_equal(mesh.elements[2].face_to_neighbors,
			np.array([0, 3, -1, -1]))
	np.testing.assert_array_equal(mesh.elements[3].face_to_neighbors,
			np.array([1, -1, -1, 2]))
	# Check vertical faces
	for face_ID in range(2):
		assert(mesh.interior_faces[face_ID].elemL_ID == 2*face_ID)
		assert(mesh.interior_faces[face_ID].elemR_ID == 2*face_ID + 1)
		assert(mesh.interior_faces[face_ID].faceL_ID == 1)
		assert(mesh.interior_faces[face_ID].faceR_ID == 3)
	# Check horizontal faces
	for face_ID in range(2):
		assert(mesh.interior_faces[face_ID + 2].elemL_ID == face_ID)
		assert(mesh.interior_faces[face_ID + 2].elemR_ID == face_ID + 2)
		assert(mesh.interior_faces[face_ID + 2].faceL_ID == 2)
		assert(mesh.interior_faces[face_ID + 2].faceR_ID == 0)
	# Check vertical boundaries
	for face_ID in range(2):
		assert(mesh.boundary_groups['x1'].boundary_faces[face_ID].elem_ID ==
				2*face_ID)
		assert(mesh.boundary_groups['x1'].boundary_faces[face_ID].face_ID == 3)
		assert(mesh.boundary_groups['x2'].boundary_faces[face_ID].elem_ID ==
				2*face_ID + 1)
		assert(mesh.boundary_groups['x2'].boundary_faces[face_ID].face_ID == 1)
	# Check horizontal boundaries
	for face_ID in range(2):
		assert(mesh.boundary_groups['y1'].boundary_faces[face_ID].elem_ID ==
				face_ID)
		assert(mesh.boundary_groups['y1'].boundary_faces[face_ID].face_ID == 0)
		assert(mesh.boundary_groups['y2'].boundary_faces[face_ID].elem_ID ==
				face_ID + 2)
		assert(mesh.boundary_groups['y2'].boundary_faces[face_ID].face_ID == 2)

def test_split_quadrils_into_tris_for_1_quad():
	'''
	Make sure that a 2D mesh with one quadrilateral is correctly split into two
	triangles.
	'''
	# Create mesh with 1 quad
	mesh = mesh_common.mesh_2D(num_elems_x=1, num_elems_y=1, xmin=0, xmax=1,
			ymin=0, ymax=1)
	# Split into triangles
	mesh = mesh_common.split_quadrils_into_tris(mesh)

	# Check node coordinates
	np.testing.assert_allclose(mesh.node_coords,
			np.array([
				[0, 0], [1, 0],
				[0, 1], [1, 1],
				]), rtol, atol)
	# Check node IDs of elements
	np.testing.assert_array_equal(mesh.elem_to_node_IDs,
			np.array([
				[0, 1, 2],
				[3, 2, 1],
				]))
	# Check neighbors
	np.testing.assert_array_equal(mesh.elements[0].face_to_neighbors,
			np.array([1, -1, -1]))
	np.testing.assert_array_equal(mesh.elements[1].face_to_neighbors,
			np.array([0, -1, -1]))
	# Check interior face
	assert(mesh.interior_faces[0].elemL_ID == 0)
	assert(mesh.interior_faces[0].elemR_ID == 1)
	assert(mesh.interior_faces[0].faceL_ID == 0)
	assert(mesh.interior_faces[0].faceR_ID == 0)
	# Check vertical boundaries
	assert(mesh.boundary_groups['x1'].boundary_faces[0].elem_ID == 0)
	assert(mesh.boundary_groups['x1'].boundary_faces[0].face_ID == 1)
	assert(mesh.boundary_groups['x2'].boundary_faces[0].elem_ID == 1)
	assert(mesh.boundary_groups['x2'].boundary_faces[0].face_ID == 1)
	# Check horizontal boundaries
	assert(mesh.boundary_groups['y1'].boundary_faces[0].elem_ID == 0)
	assert(mesh.boundary_groups['y1'].boundary_faces[0].face_ID == 2)
	assert(mesh.boundary_groups['y2'].boundary_faces[0].elem_ID == 1)
	assert(mesh.boundary_groups['y2'].boundary_faces[0].face_ID == 2)
