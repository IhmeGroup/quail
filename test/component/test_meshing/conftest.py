import pytest
import numpy as np
import sys
sys.path.append('../src')

import meshing.meshbase as mesh_defs

def make_empty_mesh():
	'''
	Create a mesh object with a mocked basis for two Q1 triangles.
	'''
	# Make gbasis mock
	gbasis = BasisMock()
	# Make mesh
	mesh = mesh_defs.Mesh(ndims = 2, num_nodes = 4, num_elems = 2, gbasis =
			gbasis, gorder = 1)
	mesh.num_interior_faces = 1
	return mesh

@pytest.fixture
def mesh():
	'''
	This fixture yields a mesh object with a mocked basis for two Q1 triangles.
	'''
	yield make_empty_mesh()

@pytest.fixture
def filled_mesh():
	'''
	This fixture yields a mesh object with a mocked basis for two Q1 triangles,
	with the coordinates and faces filled out.
	'''
	# Make mesh
	filled_mesh = make_empty_mesh()
	# Add nodes: Two triangles in a unit square from (0, 1) x (0, 1).
	filled_mesh.elem_to_node_IDs = np.array([[0, 1, 2], [3, 2, 1]])
	filled_mesh.node_coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
	filled_mesh.interior_faces = [mesh_defs.InteriorFace()]
	filled_mesh.interior_faces[0].elemL_ID = 0
	filled_mesh.interior_faces[0].elemR_ID = 1
	filled_mesh.interior_faces[0].faceL_ID = 0
	filled_mesh.interior_faces[0].faceR_ID = 0
	yield filled_mesh

class BasisMock:
	'''
	Class used to mock the basis given to the mesh object, for the case of a Q1
	triangle.
	'''
	NFACES = 3
	def get_num_basis_coeff(self, order):
		return 3
