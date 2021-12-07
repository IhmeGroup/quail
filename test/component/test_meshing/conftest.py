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
	# Add nodes: Two triangles in a double unit square from (0, 2) x (0, 2).
	filled_mesh.elem_to_node_IDs = np.array([[0, 1, 2], [3, 2, 1]])
	filled_mesh.node_coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1]]) * 2
	filled_mesh.interior_faces = [mesh_defs.InteriorFace()]
	filled_mesh.interior_faces[0].elemL_ID = 0
	filled_mesh.interior_faces[0].elemR_ID = 1
	filled_mesh.interior_faces[0].faceL_ID = 0
	filled_mesh.interior_faces[0].faceR_ID = 0
	filled_mesh.elements = [ElementMock(), ElementMock()]
	for elem_ID in range(2):
		filled_mesh.elements[elem_ID].node_coords = filled_mesh.node_coords[
				filled_mesh.elem_to_node_IDs[elem_ID]]
	yield filled_mesh

class BasisMock:
	'''
	Class used to mock the basis given to the mesh object, for the case of a
	Q1 triangle. The class being mocked is BasisBase from
	numerics/basis/basis.py.
	'''
	NFACES = 3
	NDIMS = 2
	CENTROID = np.array([[1/3, 1/3]])
	# Basis values at the geometric nodes
	basis_val = np.identity(3)
	def get_num_basis_coeff(*args, **kwargs):
		return 3

	def get_basis_val_grads(self, xref, get_val=True):
		# If there are three points, assume they're the geometric nodes
		if xref.shape[0] == 3:
			self.basis_val = np.identity(3)
		# Else, assume it's the centroid
		else:
			self.basis_val = np.array([[1/3, 1/3, 1/3]])

	def get_quadrature_order(*args, **kwargs):
		return 1

	# Use the geometric nodes as quadrature points. This works for Q1.
	def get_quadrature_data(*args, **kwargs):
		return np.array([[0, 0], [1, 0], [0, 1]]), np.ones((3, 1))/6

	# The bases are: 1 - x - y, x, and y. Therefore, the x derivative is
	# -1, 1, 0, and the y derivative is -1, 0, 1.
	def get_grads(*args, **kwargs):
		grads = np.array([[[-1, -1], [1, 0], [0, 1]]])
		# There are three quadrature points
		return np.tile(grads, (3, 1, 1))

class ElementMock:
	'''
	Class used to mock elements, just containing node coordinates. The class
	being mocked is Element from meshing/meshbase.py.
	'''
	node_coords = None
