import os

import ctypes
from ctypes import POINTER, pointer, byref
import numpy as np

import meshing.meshbase as meshdefs


class MMG5_pMesh(ctypes.c_void_p):
	pass


# File containing shared object library for interface with Mmg
lib_file = os.path.dirname(os.path.realpath(__file__)) + '/libmesh_adapter.so'
lib = ctypes.cdll.LoadLibrary(lib_file)

# -- adapt_mesh -- #
lib.adapt_mesh.restype = MMG5_pMesh
lib.adapt_mesh.argtypes = [
		# Node coords
		np.ctypeslib.ndpointer(dtype=np.float64, ndim=2,
			flags='C_CONTIGUOUS'),
		# Node IDs
		np.ctypeslib.ndpointer(dtype=np.int64, ndim=2,
			flags='C_CONTIGUOUS'),
		# Sizes
		POINTER(ctypes.c_int), POINTER(ctypes.c_int)]
adapt_mesh = lib.adapt_mesh

# -- get_results -- #
lib.get_results.restype = None
lib.get_results.argtypes = [
		# Mesh output from Mmg
		MMG5_pMesh,
		# Node coords
		np.ctypeslib.ndpointer(dtype=np.float64, ndim=2,
			flags='C_CONTIGUOUS'),
		# Node IDs
		np.ctypeslib.ndpointer(dtype=np.int64, ndim=2,
			flags='C_CONTIGUOUS')]
get_results = lib.get_results


class Adapter:

	def __init__(self, solver):
		self.solver = solver

	def adapt(self):
		mesh = self.solver.mesh

		# Run Mmg to do mesh adaptation
		npoints = ctypes.c_int()
		ntris = ctypes.c_int()
		mmgMesh = adapt_mesh(mesh.node_coords, mesh.elem_to_node_IDs,
				byref(npoints), byref(ntris))
		npoints = npoints.value
		ntris = ntris.value

		# Create new arrays with the sizing given by Mmg
		mesh.node_coords = np.empty((npoints, mesh.ndims))
		mesh.elem_to_node_IDs = np.empty((ntris, 3), dtype=np.int64)

		# Extract results from Mmg
		get_results(mmgMesh, mesh.node_coords, mesh.elem_to_node_IDs)
		# TODO: Fix for unfortunate 1-indexing
		mesh.elem_to_node_IDs -= 1

		mesh.num_elems = ntris
		mesh.num_nodes = npoints
		_, N_n, N_k = self.solver.state_coeffs.shape
		self.solver.state_coeffs = np.zeros((ntris, N_n, N_k))

		mesh.elements = []
		mesh.interior_faces = []
		mesh.num_interior_faces = 3*ntris
		for i in range(mesh.num_elems):
			mesh.elements.append(meshdefs.Element())
			elem = mesh.elements[i]
			elem.ID = i
			elem.node_IDs = mesh.elem_to_node_IDs[i]
			elem.node_coords = mesh.node_coords[elem.node_IDs]
			# TODO
			elem.face_to_neighbors = np.zeros(0, dtype=int)

			mesh.interior_faces.append(meshdefs.InteriorFace())
			mesh.interior_faces.append(meshdefs.InteriorFace())
			mesh.interior_faces.append(meshdefs.InteriorFace())
			int_face1, int_face2, int_face3 = mesh.interior_faces[3*i:3*i+3]
			int_face1.elemL_ID = i
			int_face2.elemL_ID = i
			int_face3.elemL_ID = i
			int_face1.faceL_ID = 0
			int_face2.faceL_ID = 1
			int_face3.faceL_ID = 2
