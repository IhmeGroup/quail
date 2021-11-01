import os

import ctypes
import numpy as np

import meshing.meshbase as meshdefs


class Adapter:

	def __init__(self, solver):
		self.solver = solver

	def adapt(self):
		lib_file = os.path.dirname(os.path.realpath(__file__)) + '/libmesh_adapter.so'
		lib = ctypes.cdll.LoadLibrary(lib_file)

		# Make oversized arrays to hold data
		new_node_coords = np.empty(1000 * 2)
		new_node_IDs = np.empty(1000 * 3, dtype=np.int32)

		npoints = ctypes.c_int()
		ntris = ctypes.c_int()
		lib.adapt_mesh.restype = None
		lib.adapt_mesh(
				# Old node coords
				ctypes.c_void_p(self.solver.mesh.node_coords.ctypes.data),
				# Sizes
				ctypes.pointer(npoints), ctypes.pointer(ntris),
				# New (oversized) arrays
				ctypes.c_void_p(new_node_coords.ctypes.data), ctypes.c_void_p(new_node_IDs.ctypes.data))

		# Clip these to the proper size
		self.solver.mesh.node_coords = new_node_coords[:npoints.value*2].reshape(-1, 2)
		# These are zero indexed unfortunately
		self.solver.mesh.elem_to_node_IDs = new_node_IDs[:ntris.value*3].reshape(-1, 3) - 1

		self.solver.mesh.num_elems = ntris.value
		self.solver.mesh.num_nodes = npoints.value
		_, N_n, N_k = self.solver.state_coeffs.shape
		self.solver.state_coeffs = np.zeros((ntris.value, N_n, N_k))

		self.solver.mesh.elements = []
		self.solver.mesh.interior_faces = []
		self.solver.mesh.num_interior_faces = 3*ntris.value
		for i in range(self.solver.mesh.num_elems):
			self.solver.mesh.elements.append(meshdefs.Element())
			elem = self.solver.mesh.elements[i]
			elem.ID = i
			elem.node_IDs = self.solver.mesh.elem_to_node_IDs[i]
			elem.node_coords = self.solver.mesh.node_coords[elem.node_IDs]
			# TODO
			elem.face_to_neighbors = np.zeros(0, dtype=int)

			self.solver.mesh.interior_faces.append(meshdefs.InteriorFace())
			self.solver.mesh.interior_faces.append(meshdefs.InteriorFace())
			self.solver.mesh.interior_faces.append(meshdefs.InteriorFace())
			int_face1, int_face2, int_face3 = self.solver.mesh.interior_faces[3*i:3*i+3]
			int_face1.elemL_ID = i
			int_face2.elemL_ID = i
			int_face3.elemL_ID = i
			int_face1.faceL_ID = 0
			int_face2.faceL_ID = 1
			int_face3.faceL_ID = 2
