import os

import copy
import ctypes
from ctypes import POINTER, pointer, byref
import numpy as np
from scipy import optimize
import time

import numerics.basis.tools  as basis_tools
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
		# Singular vectors
		np.ctypeslib.ndpointer(dtype=np.float64, ndim=3,
			flags='C_CONTIGUOUS'),
		# Singular values
		np.ctypeslib.ndpointer(dtype=np.float64, ndim=2,
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
		skip_iter = 25
		n_adaptation_steps = 100
		if self.solver.itime > 100: skip_iter = 300
		if self.solver.itime % skip_iter != 0: return
		if self.solver.itime >= skip_iter * n_adaptation_steps: return

		# Unpack
		mesh = self.solver.mesh
		solver = self.solver
		ndims = mesh.ndims

		# Copy the old nodes and solution and elements
		Uc_old = self.solver.state_coeffs.copy()
		elem_node_coords_old = mesh.node_coords[mesh.elem_to_node_IDs].copy()
		old_mesh_elements = copy.deepcopy(mesh.elements)

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

		vertices = solver.basis.PRINCIPAL_NODE_COORDS
		nv = vertices.shape[0]
		basis_phys_grad_vertices = np.empty([mesh.num_elems, nv,
			Uc_old.shape[1], ndims])
		for elem_ID in range(mesh.num_elems):
			# Inverse Jacobian at element vertices
			_, _, ijac_vertices = basis_tools.element_jacobian(mesh,
					elem_ID, vertices, get_djac=True, get_jac=True,
					get_ijac=True)
			# Physical gradient at element vertices
			solver.basis.get_basis_val_grads(vertices, get_ref_grad=True,
					get_phys_grad=True, ijac=ijac_vertices)
			basis_phys_grad_vertices[elem_ID] = solver.basis.basis_phys_grad

		# Evaluate solution gradient at element vertices
		grad_Uq = np.einsum('ijnl, ink -> ijkl', basis_phys_grad_vertices, Uc_old)
		# Get momentum gradient tensor
		smom = solver.physics.get_momentum_slice()
		grad_mom = grad_Uq[:, :, smom]
		# Do a singular value decomposition
		u_elems, sigma_elems, _ = np.linalg.svd(grad_mom)

		# Average across equal nodes
		u = np.zeros((mesh.num_nodes, ndims, ndims))
		sigma = np.zeros((mesh.num_nodes, ndims))
		count = np.zeros(mesh.num_nodes, dtype=int)
		for elem_ID in range(mesh.num_elems):
			node_IDs = mesh.elem_to_node_IDs[elem_ID]
			u[node_IDs] += u_elems[elem_ID]
			sigma[node_IDs] += sigma_elems[elem_ID]
			count[node_IDs] += 1
		# Divide by the number of elements that contributed to this node
		u = u / count.reshape(-1, 1, 1)
		sigma = sigma / count.reshape(-1, 1)

		# Sizing
		num_nodes = ctypes.c_int(mesh.num_nodes)
		num_elems = ctypes.c_int(mesh.num_elems)

		# Run Mmg to do mesh adaptation
		mmgMesh = MMG5_pMesh()
		mmgSol = MMG5_pSol()
		# TODO: For some reason mesh.node_coords is F contiguous???
		adapt_mesh(np.ascontiguousarray(mesh.node_coords), mesh.elem_to_node_IDs,
				bface_info, u, sigma, byref(num_nodes), byref(num_elems),
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
		print('Computing matrix helpers...', end='')
		start_time = time.time()
		solver.precompute_matrix_helpers()
		solver.stepper.res = np.zeros_like(solver.state_coeffs)
		solver.stepper.dU = np.zeros_like(solver.state_coeffs)
		print(f'done in {time.time() - start_time} s')

		print('Searching, Newton-solving, and evaluating old solution...', end='')
		start_time = time.time()
		nq = solver.elem_helpers.quad_pts.shape[0]
		Uq = np.empty((mesh.num_elems, nq, Uc_old.shape[2]))
		newton_time_total = 0
		eval_time_total = 0
		search_time_total = 0
		# Loop over elements
		for elem_ID in range(mesh.num_elems):
			# Loop over quadrature points
			for j in range(nq):
				quad_pt = solver.elem_helpers.quad_pts[[j]]
				# Get quad point in physical space
				quad_pt_phys = mesh_tools.ref_to_phys(mesh, elem_ID, quad_pt)
				# Get element ID of the element from the old mesh which would
				# have contained this point
				search_time = time.time()
				old_elem_ID = get_elem_containing_point(quad_pt_phys[0],
						elem_node_coords_old, old_mesh_elements)
				search_time_total += time.time() - search_time
				# Newton-solve to back out the point in reference space which
				# corresponds to this point in physical space on the old element
				def rhs(guess):
					# Get basis values
					#basis_val = mesh.gbasis.get_values(guess)
					# TODO: HACK: Only for Q1. Using get_values is VERY slow for
					# lagrange triangles!
					basis_val = np.array([1 - guess[0, 0] - guess[0, 1],
							guess[0, 0],
							guess[0, 1]])
					# Convert to physical space
					xphys = np.matmul(basis_val,
							elem_node_coords_old[old_elem_ID])
					return xphys - quad_pt_phys
				newton_time = time.time()
				guess = np.array([[.5, .5]])
				quad_pt_old = optimize.broyden1(rhs, guess)
				newton_time_total += time.time() - newton_time
				eval_time = time.time()
				# Get basis values at this point
				basis_val = solver.basis.get_values(quad_pt_old)
				# Evaluate solution at this point
				Uq[elem_ID, j] = basis_val @ Uc_old[old_elem_ID]
				eval_time_total += time.time() - eval_time
		print(f'done in {time.time() - start_time} s')
		print(f'Newton solving took {newton_time_total} s')
		print(f'Solution evaluation took {eval_time_total} s')
		print(f'Searching the mesh took {search_time_total} s')

		# Project solution from old mesh to new
		print('L2-projecting...', end='')
		start_time = time.time()
		solver_tools.L2_projection(mesh, solver.elem_helpers.iMM_elems,
				solver.basis, solver.elem_helpers.quad_pts,
				solver.elem_helpers.quad_wts, Uq, solver.state_coeffs)
		print(f'done in {time.time() - start_time} s')

		# Recompute helpers for the limiters
		if solver.limiters:
			for limiter in solver.limiters:
				limiter.precompute_helpers(solver)

# Find which element contains a point
# TODO: Only works for Q1 triangles
def get_elem_containing_point(p, elem_node_coords, elements):
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

# Find which element contains a point
# TODO: Only works for Q1 triangles
# TODO: This is even slower than searching the whole mesh!!
def get_elem_containing_point_directional(p, elem_node_coords, elements):
	result = None
	# Loop over elements
	#for elem_ID, (v1, v2, v3) in enumerate(elem_node_coords):
	elem_ID = 0
	tried_already = set()
	tried_counter = 1
	for i in range(10 * elem_node_coords.shape[0]):
		# If it already tried this element, skip it, and start from a
		# (potentially) fresh one
		if elem_ID in tried_already:
			elem_ID = tried_counter
			tried_counter += 1
			continue

		# Check if this is the one
		v1, v2, v3 = elem_node_coords[elem_ID]
		if point_in_triangle(p, v1, v2, v3):
			result = elem_ID
			break
		# Otherwise, move in the right direction
		else:
			tried_already.add(elem_ID)
			# Compute unit vector in direction of point
			v = (v1 + v2 + v3) / 3
			vec = p - v
			direction = vec / np.linalg.norm(vec)
			# Get face centers
			c1 = (v2 + v3) / 2
			c2 = (v3 + v1) / 2
			c3 = (v1 + v2) / 2
			# Normalized directions
			dir_c1 = (c1 - v) / np.linalg.norm(c1 - v)
			dir_c2 = (c2 - v) / np.linalg.norm(c2 - v)
			dir_c3 = (c3 - v) / np.linalg.norm(c3 - v)
			# Dot products
			dot1 = np.dot(direction, dir_c1)
			dot2 = np.dot(direction, dir_c2)
			dot3 = np.dot(direction, dir_c3)
			# Sort
			face_IDs_sorted = np.argsort([dot1, dot2, dot3])[::-1]
			# Move that way, avoiding boundaries
			for face_ID in face_IDs_sorted:
				elem_ID = elements[elem_ID].face_to_neighbors[face_ID]
				if elem_ID != -1: break
	if result is None:
		print()
		print(f"No elements contain point {p}!")
		breakpoint()
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
