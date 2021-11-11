import numpy as np
import pytest
import sys
sys.path.append('../src')

import meshing.meshbase as mesh_defs

rtol = 1e-15
atol = 1e-15


def test_interior_faces_should_be_constructed_to_zero():
	'''
	Make sure that interior faces are initialized to zero.
	'''
	# Create face
	interior_face = mesh_defs.InteriorFace()
	# Assert
	assert(interior_face.elemL_ID == 0)
	assert(interior_face.elemR_ID == 0)
	assert(interior_face.faceL_ID == 0)
	assert(interior_face.faceR_ID == 0)

def test_boundary_faces_should_be_constructed_to_zero():
	'''
	Make sure that boundary faces are initialized to zero.
	'''
	# Create face
	boundary_face = mesh_defs.BoundaryFace()
	# Assert
	assert(boundary_face.elem_ID == 0)
	assert(boundary_face.face_ID == 0)

def test_boundary_group_should_allocate_five_boundary_faces():
	'''
	Make sure that the boundary group can correctly allocate its boundary
	faces.
	'''
	# Make boundary group
	boundary_group = mesh_defs.BoundaryGroup()
	# Check that it is initialized to be "empty"
	assert(boundary_group.name == '')
	assert(boundary_group.number == -1)
	assert(boundary_group.num_boundary_faces == 0)
	assert(boundary_group.boundary_faces == [])

	# Set number of faces
	boundary_group.num_boundary_faces = 5
	# Allocate boundary faces
	boundary_group.allocate_boundary_faces()
	# Make sure there are 5 boundary faces in this group
	assert(len(boundary_group.boundary_faces) == 5)
	for face in boundary_group.boundary_faces:
		assert(isinstance(face, mesh_defs.BoundaryFace))

def test_elements_should_be_constructed_to_zero():
	'''
	Make sure that elements are initialized to zero.
	'''
	# Make element
	element = mesh_defs.Element(5)
	# Make sure ID is set properly, and everything else is zero
	assert(element.ID == 5)
	assert(element.node_IDs.size == 0)
	assert(element.node_coords.size == 0)
	assert(element.face_to_neighbors.size == 0)

def test_mesh_should_be_constructed_from_inputs(mesh):
	'''
	Make sure that the inputs for the example mesh are set properly in the
	constructor.
	'''
	assert(mesh.ndims == 2)
	assert(mesh.num_nodes == 4)
	assert(mesh.num_elems == 2)
	assert(mesh.gorder == 1)
	assert(mesh.num_nodes_per_elem == 3)

def test_mesh_should_properly_size_elem_to_node_IDs(mesh):
	'''
	Make sure that elem_to_node_IDs is sized correctly.
	'''
	mesh.allocate_elem_to_node_IDs_map()
	assert(mesh.elem_to_node_IDs.shape == (2, 3))

def test_mesh_should_properly_size_interior_faces(mesh):
	'''
	Make sure that interior_faces is sized and set correctly.
	'''
	mesh.allocate_interior_faces()
	assert(len(mesh.interior_faces) == 1)
	for face in mesh.interior_faces:
		assert(isinstance(face, mesh_defs.InteriorFace))

def test_mesh_should_raise_error_for_repeated_boundary_names(mesh):
	'''
	Make sure that repeating boundary names raises a ValueError.
	'''
	# Add a name
	mesh.boundary_groups['DG is great'] = True
	# Make sure adding the same name raises a ValueError
	with pytest.raises(ValueError) as e_info:
		mesh.add_boundary_group('DG is great')

def test_mesh_should_get_face_neighbors_correctly(filled_mesh):
	'''
	Make sure that face neighbors are found correctly when elements are created.
	'''
	# Create elements
	filled_mesh.create_elements()
	assert(len(filled_mesh.elements) == 2)
	# Check neighbors
	np.testing.assert_array_equal(filled_mesh.elements[0].face_to_neighbors,
			np.array([1, -1, -1]))
	np.testing.assert_array_equal(filled_mesh.elements[1].face_to_neighbors,
			np.array([0, -1, -1]))
