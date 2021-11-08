import os

import ctypes
from ctypes import POINTER, pointer, byref
import numpy as np
from scipy import optimize

import meshing.meshbase as meshdefs
import meshing.tools as mesh_tools
import solver.tools as solver_tools


class MMG5_pMesh(ctypes.c_void_p):
	pass

class MMG5_pSol(ctypes.c_void_p):
	pass


# File containing shared object library for interface with Mmg
lib_file = os.path.dirname(os.path.realpath(__file__)) + '/libmesh_adapter.so'
lib = ctypes.cdll.LoadLibrary(lib_file)

# -- adapt_mesh -- #
lib.adapt_mesh.restype = None
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
		POINTER(ctypes.c_int), POINTER(ctypes.c_int), POINTER(ctypes.c_int),
		# Mmg structs for the mesh and sol
		MMG5_pMesh, MMG5_pSol]
adapt_mesh = lib.adapt_mesh

# -- get_results -- #
lib.get_results.restype = None
lib.get_results.argtypes = [
		# Mesh and sol output from Mmg
		MMG5_pMesh, MMG5_pSol,
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
		# TODO
		if self.solver.time != .05: return

		mesh = self.solver.mesh
		solver = self.solver

		# Copy the old nodes and solution
		Uc_old = self.solver.state_coeffs.copy()
		elem_node_coords_old = mesh.node_coords[mesh.elem_to_node_IDs].copy()

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
		mmgMesh = MMG5_pMesh()
		mmgSol = MMG5_pSol()
		# TODO: For some reason mesh.node_coords is F contiguous???
		adapt_mesh(np.ascontiguousarray(mesh.node_coords), mesh.elem_to_node_IDs,
				bface_info, byref(num_nodes), byref(num_elems),
				byref(num_edges), byref(mmgMesh), byref(mmgSol))
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
		get_results(mmgMesh, mmgSol, mesh.node_coords, mesh.elem_to_node_IDs,
				face_info, num_faces_per_bgroup, bface_info)

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
			# Increment counter
			bgroup_counter[boundary_group_idx] += 1;

		# Update solver helpers and stepper
		solver.precompute_matrix_helpers()
		solver.stepper.res = np.zeros_like(solver.state_coeffs)
		solver.stepper.dU = np.zeros_like(solver.state_coeffs)

		nq = solver.elem_helpers.quad_pts.shape[0]
		Uq = np.empty((mesh.num_elems, nq, Uc_old.shape[2]))
		# Loop over elements
		for elem_ID in range(mesh.num_elems):
			# Loop over quadrature points
			for j in range(nq):
				quad_pt = solver.elem_helpers.quad_pts[[j]]
				# Get quad point in physical space
				quad_pt_phys = mesh_tools.ref_to_phys(mesh, elem_ID, quad_pt)
				# Get element ID of the element from the old mesh which would
				# have contained this point
				old_elem_ID = get_elem_containing_point(quad_pt_phys[0],
						elem_node_coords_old)
				# Newton-solve to back out the point in reference space which
				# corresponds to this point in physical space on the old element
				def rhs(guess):
					# Get basis values
					mesh.gbasis.get_basis_val_grads(guess, get_val=True)
					# Convert to physical space
					xphys = np.matmul(mesh.gbasis.basis_val,
							elem_node_coords_old[old_elem_ID])
					return xphys - quad_pt_phys
				guess = np.array([[.5, .5]])
				quad_pt_old = optimize.broyden1(rhs, guess)
				# Get basis values at this point
				basis_val = solver.basis.get_values(quad_pt_old)
				# Evaluate solution at this point
				Uq[elem_ID, j] = basis_val @ Uc_old[old_elem_ID]

		# Project solution from old mesh to new
		solver_tools.L2_projection(mesh, solver.elem_helpers.iMM_elems,
				solver.basis, solver.elem_helpers.quad_pts,
				solver.elem_helpers.quad_wts, Uq, solver.state_coeffs)
		breakpoint()

# Find which element contains a point
# TODO: Only works for Q1 triangles
def get_elem_containing_point(p, elem_node_coords):
	result = None
	# Loop over elements
	for elem_ID, (v1, v2, v3) in enumerate(elem_node_coords):
		if point_in_triangle(p, v1, v2, v3):
			result = elem_ID
			break
	if result is None:
		print(f"No elements contain point {p}!")
	else:
		return elem_ID


# Check if a point p is inside a triangle given by vertices v1, v2, v3
def point_in_triangle(p, v1, v2, v3):
	d1 = edge_sign(p, v1, v2)
	d2 = edge_sign(p, v2, v3)
	d3 = edge_sign(p, v3, v1)

	has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
	has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

	return not (has_neg and has_pos)

def edge_sign(p1, p2, p3):
	return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
