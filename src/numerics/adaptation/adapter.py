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
		# Boundary face information
		np.ctypeslib.ndpointer(dtype=np.int64, ndim=2,
			flags='C_CONTIGUOUS'),
		# Sizes
		POINTER(ctypes.c_int), POINTER(ctypes.c_int), POINTER(ctypes.c_int)]
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
			flags='C_CONTIGUOUS'),
		# Face information
		np.ctypeslib.ndpointer(dtype=np.int64, ndim=2,
			flags='C_CONTIGUOUS'),
		# Number of boundary faces in each boundary group
		np.ctypeslib.ndpointer(dtype=np.int64, ndim=1,
			flags='C_CONTIGUOUS'),
		# Information from each boundary edge
		np.ctypeslib.ndpointer(dtype=np.int64, ndim=2,
			flags='C_CONTIGUOUS')]
get_results = lib.get_results


class Adapter:

	def __init__(self, solver):
		self.solver = solver

	def adapt(self):
		mesh = self.solver.mesh
		solver = self.solver

		num_edges = ctypes.c_int(0)
		# Loop over boundary groups
		for bgroup in mesh.boundary_groups.values():
			# Add to total number of edges
			num_edges.value += bgroup.num_boundary_faces

		bface_info = np.empty((num_edges.value, 3), dtype=np.int64)
		edge_idx = 0
		# Loop over boundary groups
		for bgroup in mesh.boundary_groups.values():
			# Loop over boundary faces
			for bface in bgroup.boundary_faces:
				# Get node IDs on this boundary face
				node_nums = mesh.gbasis.get_local_face_node_nums(mesh.gorder,
						bface.face_ID)
				# Get global node IDs
				face_node_IDs = mesh.elem_to_node_IDs[bface.elem_ID, node_nums]
				# Store these, along with the group number
				bface_info[edge_idx] = [face_node_IDs[0], face_node_IDs[1],
						bgroup.number]
				# Increment edge index
				edge_idx += 1;

		# Sizing
		num_nodes = ctypes.c_int(mesh.num_nodes)
		num_elems = ctypes.c_int(mesh.num_elems)

		# Run Mmg to do mesh adaptation
		mmgMesh = adapt_mesh(mesh.node_coords, mesh.elem_to_node_IDs,
				bface_info, byref(num_nodes), byref(num_elems),
				byref(num_edges))
		num_nodes = num_nodes.value
		num_elems = num_elems.value
		num_edges = num_edges.value
		mesh.num_interior_faces = ((num_elems * 3) - num_edges) // 2

		# Create new arrays with the sizing given by Mmg
		mesh.node_coords = np.empty((num_nodes, mesh.ndims))
		mesh.elem_to_node_IDs = np.empty((num_elems, 3), dtype=np.int64)
		face_info = np.empty((mesh.num_interior_faces, 4), dtype=np.int64)
		num_faces_per_bgroup = np.zeros(mesh.num_boundary_groups,
				dtype=np.int64)
		bface_info = np.empty((num_edges, 3), dtype=np.int64)

		# Extract results from Mmg
		get_results(mmgMesh, mesh.node_coords, mesh.elem_to_node_IDs, face_info,
				num_faces_per_bgroup, bface_info)

		# Set sizes
		mesh.num_elems = num_elems
		mesh.num_nodes = num_nodes
		_, N_n, N_k = self.solver.state_coeffs.shape
		self.solver.state_coeffs = np.zeros((num_elems, N_n, N_k))

		# Allocate interior faces
		mesh.allocate_interior_faces()

		# Assign face information
		for face, info in zip(mesh.interior_faces, face_info):
			face.elemL_ID, face.elemR_ID, face.faceL_ID, face.faceR_ID = info

		# Create elements
		mesh.create_elements()

		# Loop over boundary groups
		for bgroup in mesh.boundary_groups.values():
			# Set number of faces
			bgroup.num_boundary_faces = num_faces_per_bgroup[bgroup.number]
			# Allocate boundary faces
			bgroup.allocate_boundary_faces()

		# List of boundary names
		boundary_names = list(mesh.boundary_groups)

		bgroup_counter = np.zeros(mesh.num_boundary_groups, dtype=int)
		# Loop over edges
		for info in bface_info:
			# Get the boundary group of this face
			boundary_group_idx = info[2]
			bgroup = mesh.boundary_groups[boundary_names[boundary_group_idx]]
			# Get the corresponding boundary face
			bface = bgroup.boundary_faces[bgroup_counter[boundary_group_idx]]
			# Assign face information
			bface.elem_ID, bface.face_ID = info[:2]
		breakpoint()

		# Update solver helpers and stepper
		solver.precompute_matrix_helpers()
		solver.stepper.res = np.zeros_like(solver.state_coeffs)
		solver.stepper.dU = np.zeros_like(solver.state_coeffs)
