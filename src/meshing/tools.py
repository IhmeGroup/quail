# ------------------------------------------------------------------------ #
#
#       File : src/meshing/tools.py
#
#       Contains helper functions related to meshes.
#
# ------------------------------------------------------------------------ #
import numpy as np

import meshing.meshbase as mesh_defs
import numerics.basis.tools as basis_tools


TOL = 1.e-10


def ref_to_phys(mesh, elem_ID, xref):
	'''
	This function converts reference space coordinates to physical
	space coordinates.

	Inputs:
	-------
		mesh: mesh object
		elem_ID: element ID
		xref: coordinates in reference space [nq, ndims]

	Outputs:
	--------
		xphys: coordinates in physical space [nq, ndims]
	'''
	gbasis = mesh.gbasis
	gorder = mesh.gorder

	# Get basis values
	gbasis.get_basis_val_grads(xref, get_val=True)

	# Element node coordinates
	elem_coords = mesh.elements[elem_ID].node_coords

	# Convert to physical space
	xphys = np.matmul(gbasis.basis_val, elem_coords)

	return xphys # [nq, ndims]


def element_volumes(mesh, solver=None):
	'''
	This function calculates total and per-element volumes

	Inputs:
	-------
		mesh: mesh object
		solver: solver object (e.g., DG, ADER-DG, etc.)

	Outputs:
	--------
		vol_elems: volume of each element [num_elems]
		domain_vol: total volume of the domain
	'''
	# Check if already calculated
	if solver is not None:
		if hasattr(solver.elem_helpers, "domain_vol") \
				and hasattr(solver.elem_helpers, "vol_elems"):
			return solver.elem_helpers.vol_elems, \
					solver.elem_helpers.domain_vol

	# Allocate, unpack
	vol_elems = np.zeros(mesh.num_elems)
	gorder = mesh.gorder
	gbasis = mesh.gbasis

	# Get quadrature data
	quad_order = gbasis.get_quadrature_order(mesh, gorder)
	quad_pts, quad_wts = gbasis.get_quadrature_data(quad_order)

	# Get element volumes
	for elem_ID in range(mesh.num_elems):
		djac, _, _ = basis_tools.element_jacobian(mesh, elem_ID, quad_pts,
				get_djac=True)
		vol_elems[elem_ID] = np.sum(quad_wts*djac)

	# Get domain volume
	domain_vol = np.sum(vol_elems)

	return vol_elems, domain_vol # [num_elems], [1]


def get_element_centroid(mesh, elem_ID):
	'''
	This function obtains the centroid of an element in physical space.

	Inputs:
	-------
		mesh: mesh object
		elem_ID: element ID

	Outputs:
	--------
		xcentroid: element centroid in physical space [1, ndims]
	'''
	gbasis = mesh.gbasis
	xcentroid = ref_to_phys(mesh, elem_ID, mesh.gbasis.CENTROID)

	return xcentroid # [1, ndims]


def check_face_orientations(mesh):
	'''
	This function checks the face orientations for 2D meshes.

	Inputs:
	-------
		mesh: mesh object

	Notes:
	------
		An error is raised if face orientations don't match up.
	'''
	gbasis = mesh.gbasis
	if mesh.ndims == 1:
		# Don't need to check for 1D
		return

	for interior_face in mesh.interior_faces:
		elemL_ID = interior_face.elemL_ID
		elemR_ID = interior_face.elemR_ID
		faceL_ID = interior_face.faceL_ID
		faceR_ID = interior_face.faceR_ID

		# Get local IDs of element nodes
		elemL_node_IDs = mesh.elements[elemL_ID].node_IDs
		elemR_node_IDs = mesh.elements[elemR_ID].node_IDs

		''' Get global IDs of face nodes '''
		# Local IDs - left
		face_node_IDs = gbasis.get_local_face_principal_node_nums(
				mesh.gorder, faceL_ID)
		# Global IDs - left
		global_node_IDs_L = elemL_node_IDs[face_node_IDs]
		# Local IDs - right
		face_node_IDs = gbasis.get_local_face_principal_node_nums(
				mesh.gorder, faceR_ID)
		# Global IDs - right
		global_node_IDs_R = elemR_node_IDs[face_node_IDs]

		# Node ordering should be reversed between the two elements
		if not np.all(global_node_IDs_L == global_node_IDs_R[::-1]):
			raise Exception("Face orientation for elemL_ID = %d, elemR_ID "
					% (elemL_ID) + "= %d is incorrect" % (elemR_ID))


def verify_periodic_compatibility(mesh, boundary_group, icoord):
	'''
	This function checks whether a boundary is compatible with periodicity.
	Specifically, it verifies that all boundary nodes are located on the same
	plane, up to a given tolerance. It then potentially slightly modifies the
	node coordinates to ensure the exact same value.

	Inputs:
	-------
		mesh: mesh object
		boundary_group: boundary group object
		icoord: which spatial direction to check (0 for x, 1 for y)

	Outputs:
	--------
		mesh: mesh object (coordinates potentially modified)
		coord: position of boundary in the icoord direction
	'''
	coord = np.nan
	gbasis = mesh.gbasis
	for boundary_face in boundary_group.boundary_faces:
		# Physical coordinates of nodes
		coords = mesh.node_coords[boundary_face.node_IDs]

		# Make sure all nodes have same icoord-position (within TOL)
		if np.isnan(coord):
			coord = coords[0, icoord]
		if np.any(np.abs(coords[:, icoord] - coord) > TOL):
			raise ValueError("Boundary %s not compatible with periodicity" %
					(boundary_group.name))

		# Now force each node to have the same exact icoord-position
		coords[:, icoord] = coord

	return coord


def reorder_periodic_boundary_nodes(mesh, b1, b2, icoord,
		old_to_new_node_map, new_to_old_node_map, next_node_ID):
	'''
	This function checks whether a boundary is compatible with periodicity.
	Specifically, it verifies that all boundary nodes are located on the same
	plane, up to a given tolerance. It then potentially slightly modifies the
	node coordinates to ensure the exact same value.

	Inputs:
	-------
		mesh: mesh object
		b1: name of 1st periodic boundary
		b2: name of 2nd periodic boundary
		icoord: spatial direction of periodicity (0 for x, 1 for y)
		old_to_new_node_map: maps old to new node IDs [num_nodes]
		new_to_old_node_map: maps new to old node IDs [num_nodes]
		next_node_ID: next new node ID to assign

	Outputs:
	--------
		boundary_group1: boundary group object for 1st periodic boundary
		boundary_group2: boundary group object for 2nd periodic boundary
		node_pairs: stores pairs of nodes that match each other on opposite
			boundaries [num_node_pairs, 2]
		next_node_ID: next new node ID to assign (modified)
		old_to_new_node_map: maps old to new node IDs (modified) [num_nodes]
		new_to_old_node_map: maps new to old node IDs (modified) [num_nodes]

	Notes:
	------
		node_pairs[i] = np.array([node1_ID, node2_ID]) is the ith node pair,
		where node1_ID is the ID of a node on boundary 1 and node2_ID is the
		ID of the node on boundary 2 that corresponds to node1
	'''
	gbasis = mesh.gbasis

	if b1 is None and b2 is None:
		# Trivial case - no periodicity in given direction
		return None, None, None, next_node_ID
	elif b1 == b2:
		raise ValueError("Duplicate boundaries")

	# Allocate arrays
	node_pairs = np.zeros([mesh.num_nodes, 2], dtype=int) - 1
	idx_in_node_pairs = np.zeros(mesh.num_nodes, dtype=int) - 1
	num_node_pairs = 0
	node2_matched = np.zeros(mesh.num_nodes, dtype=bool)

	# Extract the two boundary_groups
	boundary_group1 = mesh.boundary_groups[b1]
	boundary_group2 = mesh.boundary_groups[b2]

	start_node_ID = next_node_ID

	if icoord < 0 or icoord >= mesh.ndims:
		raise ValueError

	'''
	Make sure each boundary is compatible with periodicity
	Note: the boundary node coordinates may be slightly modified
	to ensure same coordinate in periodic direction
	'''
	if boundary_group1.num_boundary_faces != \
			boundary_group2.num_boundary_faces:
		raise ValueError
	pcoord1 = verify_periodic_compatibility(mesh, boundary_group1, icoord)
	pcoord2 = verify_periodic_compatibility(mesh, boundary_group2, icoord)
	pdiff = np.abs(pcoord1-pcoord2) # distance between the two boundaries

	'''
	Deal with first boundary
	'''
	# Populate node maps for first boundary
	for boundary_face in boundary_group1.boundary_faces:
		# Extract info
		elem_ID = boundary_face.elem_ID
		face_ID = boundary_face.face_ID

		# Local IDs of face nodes
		local_node_IDs = gbasis.get_local_face_principal_node_nums(
				mesh.gorder, face_ID)
		# Global IDs of face nodes
		global_node_IDs = mesh.elem_to_node_IDs[elem_ID][local_node_IDs]

		# Populate node maps
		for node_ID in global_node_IDs:
			if old_to_new_node_map[node_ID] == -1:
				# Node already mapped
				old_to_new_node_map[node_ID] = next_node_ID
				new_to_old_node_map[next_node_ID] = node_ID
				next_node_ID += 1
			if idx_in_node_pairs[node_ID] == -1:
				node_pairs[num_node_pairs, 0] = node_ID
				idx_in_node_pairs[node_ID] = num_node_pairs
					# maps node ID to index in node_pairs
				num_node_pairs += 1

	# Last ID assigned to nodes on boundary 1
	stop_node_ID = next_node_ID

	'''
	Deal with second boundary
	'''
	# Populate node maps for second boundary
	for boundary_face in boundary_group2.boundary_faces:
		# Extract info
		elem_ID = boundary_face.elem_ID
		face_ID = boundary_face.face_ID

		# Local IDs of face nodes
		local_node_IDs = gbasis.get_local_face_principal_node_nums(
				mesh.gorder, face_ID)
		# Global IDs of face nodes
		global_node_IDs = mesh.elem_to_node_IDs[elem_ID][local_node_IDs]

		for node2_ID in global_node_IDs:
			''' Find matching nodes on boundary group 1 '''

			if node2_matched[node2_ID]:
				# this node already matched - skip
				# sanity check
				if old_to_new_node_map[node2_ID] == -1:
					raise ValueError("Node %d already matched" % node2_ID)
				continue

			# Physical coordinates of node
			coord2 = mesh.node_coords[node2_ID]

			match = False
			# Match with a node on boundary 1
			for n in range(num_node_pairs):
				if node_pairs[n, 1] != -1:
					# node1 already paired - skip
					continue

				node1_ID = node_pairs[n, 0]
				coord1 = mesh.node_coords[node1_ID]

				# Find distance between the two nodes
				norm = np.linalg.norm(coord1-coord2, ord=1)

				# Check if distance is equal to pdiff (within TOL)
				if np.abs(norm-pdiff) < TOL:
					match = True
					if old_to_new_node_map[node2_ID] == -1:
						# node2 not reordered yet

						# Populate maps
						# Note: the difference in IDs of matching nodes is
						# equal to (stop_node_ID - start_node_ID)
						node2_ID_new = old_to_new_node_map[node1_ID] + \
								stop_node_ID - start_node_ID
						old_to_new_node_map[node2_ID] = node2_ID_new
						new_to_old_node_map[node2_ID_new] = node2_ID
						next_node_ID = np.amax([next_node_ID, node2_ID_new])

						# Force nodes to match exactly
						for d in range(mesh.ndims):
							if d == icoord:
								# Skip periodic direction
								continue
							coord2[d] = coord1[d]

					# Store node pair
					idx1 = idx_in_node_pairs[node1_ID]
					node_pairs[idx1, 1] = node2_ID
					idx_in_node_pairs[node2_ID] = idx1

					# Flag node2 as matched
					node2_matched[node2_ID] = True
					break

			if not match:
				raise ValueError("Could not find matching boundary node " +
						"for Node %d" % (node2_ID))

	# Modify next node ID
	if start_node_ID != stop_node_ID:
		# This means at least one pair of nodes was matched
		next_node_ID += 1
	# Sanity check
	if next_node_ID != 2*stop_node_ID - start_node_ID:
		raise ValueError

	# Resize node_pairs
	node_pairs = node_pairs[:num_node_pairs, :]

	# Print info
	if icoord == 0:
		s = "x"
	elif icoord == 1:
		s = "y"
	else:
		s = "z"
	print("Reordered periodic boundaries in %s-direction" % (s))

	return boundary_group1, boundary_group2, node_pairs, next_node_ID


def remap_nodes(mesh, old_to_new_node_map, new_to_old_node_map,
		next_node_ID=-1):
	'''
	This function remaps node IDs based on the maps passed in as input
	arguments.

	Inputs:
	-------
		mesh: mesh object
		old_to_new_node_map: maps old to new node IDs [num_nodes]
		new_to_old_node_map: maps new to old node IDs [num_nodes]
		next_node_ID: next new node ID to assign

	Outputs:
	--------
		mesh: mesh object (modified)
		old_to_new_node_map: maps old to new node IDs (modified)
		new_to_old_node_map: maps new to old node IDs (modified)
	'''

	# Fill up node maps with non-periodic nodes
	# Note: non-periodic nodes come after the periodic nodes
	if next_node_ID != -1:
		for node_ID in range(mesh.num_nodes):
			if old_to_new_node_map[node_ID] == -1:
				# Has not been re-ordered yet
				old_to_new_node_map[node_ID] = next_node_ID
				new_to_old_node_map[next_node_ID] = node_ID
				next_node_ID += 1

	# Assign new node IDs
	mesh.node_coords = mesh.node_coords[new_to_old_node_map]

	# New elem_to_node_IDs
	num_elems = mesh.num_elems
	for elem_ID in range(num_elems):
		mesh.elem_to_node_IDs[elem_ID,:] = old_to_new_node_map[
				mesh.elem_to_node_IDs[elem_ID, :]]


def match_boundary_pair(mesh, icoord, boundary_group1, boundary_group2,
		node_pairs, old_to_new_node_map, new_to_old_node_map):
	'''
	This function creates interior faces to match the two periodic
	boundaries.

	Inputs:
	-------
		mesh: mesh object
		icoord: spatial direction of periodicity (0 for x, 1 for y)
		boundary_group1: boundary object of 1st periodic boundary
		boundary_group2: boundary object of 2nd periodic boundary
		node_pairs: stores pairs of nodes that match each other on opposite
			boundaries [num_node_pairs, 2]
		old_to_new_node_map: maps old to new node IDs [num_nodes]
		new_to_old_node_map: maps new to old node IDs [num_nodes]

	Outputs:
	--------
		mesh: mesh object (modified - new interior faces, removed boundary
			groups)
	'''
	gbasis = mesh.gbasis
	interior_faces = mesh.interior_faces

	if boundary_group1 is None and boundary_group2 is None:
		return
	elif boundary_group1 is None or boundary_group2 is None:
		raise ValueError("Only one boundary group provided")

	'''
	Remap node_pairs and idx_in_node_pairs
	'''
	# Allocate
	new_node_pairs = np.zeros_like(node_pairs, dtype=int) - 1
	idx_in_node_pairs = np.zeros(mesh.num_nodes, dtype=int) - 1
	# Remap
	new_node_pairs = old_to_new_node_map[node_pairs]
	n1 = new_node_pairs[:, 0]
	idx_in_node_pairs[n1] = np.arange(n1.shape[0])
	# Replace
	node_pairs = new_node_pairs

	# Sanity check
	if np.amin(new_node_pairs) == -1:
		raise ValueError

	'''
	Identify and create periodic interior_faces
	'''
	for boundary_face1 in boundary_group1.boundary_faces:
		# Extract info
		elem_ID1 = boundary_face1.elem_ID
		face_ID1 = boundary_face1.face_ID

		# Local IDs of face nodes
		local_node_IDs = gbasis.get_local_face_principal_node_nums(
			mesh.gorder, face_ID1)
		# Global IDs of face nodes
		global_node_IDs = mesh.elem_to_node_IDs[elem_ID1][local_node_IDs]
		# Sort for easy comparison later
		global_node_IDs_1 = np.sort(global_node_IDs)

		# Pair each node with corresponding one on other boundary
		for boundary_face2 in boundary_group2.boundary_faces:
			# Extract info
			elem_ID2 = boundary_face2.elem_ID
			face_ID2 = boundary_face2.face_ID

			# Local IDs of face nodes
			local_node_IDs = gbasis.get_local_face_principal_node_nums(
					mesh.gorder, face_ID2)
			# Global IDs of face nodes
			global_node_IDs = mesh.elem_to_node_IDs[elem_ID2][local_node_IDs]
			# Sort for easy comparison later
			global_node_IDs_2 = np.sort(global_node_IDs)

			''' Check for complete match between all nodes '''
			# Get nodes on boundary 2 paired with those in global_node_IDs_1
			idx1 = idx_in_node_pairs[global_node_IDs_1]
			nodes1_partner_IDs = node_pairs[idx1, 1]
			# Sort
			nodes1_partner_IDs_sort = np.sort(nodes1_partner_IDs)

			match = False
			if np.all(global_node_IDs_2 == nodes1_partner_IDs_sort):
				# Matching nodes
				match = True

				# Sanity check
				if not np.all(global_node_IDs_2 == nodes1_partner_IDs):
					raise ValueError("Node ordering on opposite periodic " +
							"faces is different")

				# Create interior face between these two faces
				mesh.num_interior_faces += 1
				interior_faces.append(mesh_defs.InteriorFace())
				interior_face = interior_faces[-1]
				interior_face.elemL_ID = elem_ID1
				interior_face.faceL_ID = face_ID1
				interior_face.elemR_ID = elem_ID2
				interior_face.faceR_ID = face_ID2

				# Decrement number of boundary faces
				boundary_group1.num_boundary_faces -= 1
				boundary_group2.num_boundary_faces -= 1

				break

		if not match:
			raise ValueError("Could not find matching boundary face")

	# Verification
	if boundary_group1.num_boundary_faces != 0 or \
			boundary_group2.num_boundary_faces != 0:
		raise ValueError

	# Remove the 2 boundary groups
	mesh.num_boundary_groups -= 2
	mesh.boundary_groups.pop(boundary_group1.name)
	mesh.boundary_groups.pop(boundary_group2.name)

	# Print info
	if icoord == 0:
		s = "x"
	elif icoord == 1:
		s = "y"
	else:
		s = "z"
	print("Matched periodic boundaries in %s-direction" % (s))


def update_boundary_group_nums(mesh):
	'''
	This function updates the boundary group numbers.

	Inputs:
	-------
		mesh: mesh object

	Outputs:
	--------
		mesh: mesh object (boundary group numbers modified)
	'''
	i = 0
	for boundary_group in mesh.boundary_groups.values():
		boundary_group.number = i
		i += 1


def verify_periodic_mesh(mesh):
	'''
	This function verifies the periodicity of the mesh.

	Inputs:
	-------
		mesh: mesh object
	'''
	# Loop through interior faces
	for interior_face in mesh.interior_faces:
		# Extract info
		elemL_ID = interior_face.elemL_ID
		elemR_ID = interior_face.elemR_ID
		faceL_ID = interior_face.faceL_ID
		faceR_ID = interior_face.faceR_ID
		gbasis = mesh.gbasis
		gorder = mesh.gorder

		''' Get global IDs of face nodes '''
		# Local IDs - left
		local_node_IDs = gbasis.get_local_face_principal_node_nums(
				gorder, faceL_ID)
		# Global IDs - left
		global_node_IDs_L = mesh.elem_to_node_IDs[elemL_ID][local_node_IDs]
		# Local IDs - right
		local_node_IDs = gbasis.get_local_face_principal_node_nums(
			gorder, faceR_ID)
		# Global IDs - right
		global_node_IDs_R = mesh.elem_to_node_IDs[elemR_ID][local_node_IDs]

		''' If exact same global nodes, then this is NOT a periodic face '''
		# Sort for easy comparison
		global_node_IDs_L = np.sort(global_node_IDs_L)
		global_node_IDs_R = np.sort(global_node_IDs_R)
		if np.all(global_node_IDs_L == global_node_IDs_R):
			# Skip non-periodic faces
			continue

		''' Compare distances '''
		coordsL = mesh.node_coords[global_node_IDs_L]
		coordsR = mesh.node_coords[global_node_IDs_R]
		dists = np.linalg.norm(coordsL-coordsR, axis=1)
		if np.abs(np.max(dists) - np.min(dists)) > TOL:
			raise ValueError


def make_periodic_translational(mesh, x1=None, x2=None, y1=None, y2=None):
	'''
	This function imposes translational periodicity on the mesh.

	Inputs:
	-------
		mesh: mesh object
		x1: name of 1st periodic boundary in x-direction
		x2: name of 2nd periodic boundary in x-direction
		y1: name of 1st periodic boundary in y-direction
		y2: name of 2nd periodic boundary in y-direction

	Outputs:
	--------
		mesh: mesh object (modified)
	'''
	print("-------------------------------------------------")
	print("IMPOSING PERIODICITY\n")
	''' Reorder nodes '''
	old_to_new_node_map = np.zeros(mesh.num_nodes, dtype=int) - 1
		# old_to_new_node_map[n] = the new node ID of the nth node
		# (pre-ordering)
	new_to_old_node_map = np.zeros(mesh.num_nodes, dtype=int) - 1
		# new_to_old_node_map[i] = the old node ID of the ith node
		# (post-reordering)
	next_node_ID = 0

	# x
	boundary_group_x1, boundary_group_x2, node_pairs_x, next_node_ID = \
			reorder_periodic_boundary_nodes(mesh, x1, x2, 0,
			old_to_new_node_map, new_to_old_node_map, next_node_ID)
	# y
	boundary_group_y1, boundary_group_y2, node_pairs_y, next_node_ID = \
			reorder_periodic_boundary_nodes(mesh, y1, y2, 1,
			old_to_new_node_map, new_to_old_node_map, next_node_ID)

	''' Apply node remapping '''
	remap_nodes(mesh, old_to_new_node_map, new_to_old_node_map, next_node_ID)

	''' Match pairs of periodic boundary faces '''
	# x
	match_boundary_pair(mesh, 0, boundary_group_x1, boundary_group_x2,
			node_pairs_x, old_to_new_node_map, new_to_old_node_map)
	# y
	match_boundary_pair(mesh, 1, boundary_group_y1, boundary_group_y2,
			node_pairs_y, old_to_new_node_map, new_to_old_node_map)

	''' Update boundary group numbers '''
	update_boundary_group_nums(mesh)

	''' Verify valid mesh '''
	verify_periodic_mesh(mesh)

	''' Update elements '''
	mesh.create_elements()

	print("\nDONE")
	print("-------------------------------------------------")
