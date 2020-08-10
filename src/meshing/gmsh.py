import code
import copy
import numpy as np

import errors
import general

import meshing.meshbase as mesh_defs
import meshing.tools as mesh_tools

import numerics.basis.basis as basis_defs


VERSION2 = "2.2"
VERSION4 = "4.1"
MSH_MAX_NUM = 140 # number of element types


class GmshElementData(object):
	def __init__(self):
		self.num_nodes = -1
		self.gorder = -1
		self.gbasis = -1
		self.node_order = None


def gmsh_node_order_seg(gorder):
	num_nodes = gorder + 1
	nodes = np.arange(num_nodes)
	nodes[1:-1] = nodes[2:]
	nodes[-1] = 1

	return nodes


def populate_nodes_quadril(gorder, start, nodes):
	if gorder == 0:
		return start
	else:
		# principal vertices
		nodes[0, 0] = start
		nodes[0, -1] = start + 1
		nodes[-1, -1] = start + 2
		nodes[-1, 0] = start + 3
		# bottom face
		start += 4
		nodes[0, 1:-1] = np.arange(start, start+gorder-1)
		# right face
		start += gorder-1
		nodes[1:-1, -1] = np.arange(start, start+gorder-1)
		# top face
		start += gorder-1
		nodes[-1, -2:0:-1] = np.arange(start, start+gorder-1)
		# left face
		start += gorder-1
		nodes[-2:0:-1, 0] = np.arange(start, start+gorder-1)
		# interior
		if gorder >= 2:
			# recursively fill the interior nodes
			start += gorder - 1
			nodes[1:-1, 1:-1] = populate_nodes_quadril(gorder-2, start, nodes[1:-1, 1:-1])

	return nodes


def gmsh_node_order_quadril(gorder):

	nodes = populate_nodes_quadril(gorder, 0, np.zeros([gorder+1, gorder+1], dtype=int))
	nodes.shape = -1

	return nodes


def populate_nodes_tri(gorder, start, nodes):
	if gorder == 0:
		return start
	else:
		# principal vertices
		nodes[0,0] = start
		nodes[0,-1] = start+1
		nodes[-1,0] = start+2
		# bottom face
		start += 3
		nodes[0,1:-1] = np.arange(start, start+gorder-1)
		# hypotenuse
		idx = np.arange(1, gorder), np.arange(gorder-1, 0, -1) 
			# indices to access diagonal, excluding corner elements
		start += gorder-1
		nodes[idx] = np.arange(start, start+gorder-1)
		# left face
		start += gorder-1
		nodes[-2:0:-1,0] = np.arange(start, start+gorder-1)
		# interior
		if gorder >= 3:
			# recursively fill the interior nodes
			start += gorder-1
			nodes[1:gorder-1,1:gorder-1] = populate_nodes_tri(gorder-3, start, 
					nodes[1:gorder-1,1:gorder-1])

	return nodes


def gmsh_node_order_tri(gorder):

	# only lower triangular 
	nodes = populate_nodes_tri(gorder, 0, np.zeros([gorder+1, gorder+1], dtype=int)-1)
	nodes = nodes[nodes >= 0]

	return nodes


def create_gmsh_element_database():
	# (NTYPES+1) objects due to 1-indexing
	# gmsh_element_database = [GmshElementData() for n in range(MSH_MAX_NUM+1)]
	gmsh_element_database = {}

	''' 
	Assume most element types are not supported
	Only fill in supported elements
	'''

	# Point
	etype_num = 15
	elem_data = GmshElementData()
	gmsh_element_database.update({etype_num : elem_data})
	elem_data.gorder = 0
	elem_data.gbasis = basis_defs.PointShape() # shape here instead of gbasis
	elem_data.num_nodes = 1
	elem_data.node_order = np.array([0])

	# Line segments (q = 1 to q = 11)
	etype_nums = np.array([1, 8, 26, 27, 28, 62, 63, 64, 65, 66])
	for i in range(etype_nums.shape[0]):
		etype_num = etype_nums[i]
		elem_data = GmshElementData()
		gmsh_element_database.update({etype_num : elem_data})
		gorder = i + 1
		elem_data.gorder = gorder
		elem_data.gbasis = basis_defs.LagrangeSeg(gorder)
		elem_data.num_nodes = gorder + 1
		elem_data.node_order = gmsh_node_order_seg(gorder)

	# Triangles (q = 1 to q = 10)
	etype_nums = np.array([2, 9, 21, 23, 25, 42, 43, 44, 45, 46])
	for i in range(etype_nums.shape[0]):
		etype_num = etype_nums[i]
		elem_data = GmshElementData()
		gmsh_element_database.update({etype_num : elem_data})
		gorder = i + 1
		elem_data.gorder = gorder
		elem_data.gbasis = basis_defs.LagrangeTri(gorder)
		elem_data.num_nodes = (gorder + 1)*(gorder + 2)//2
		elem_data.node_order = gmsh_node_order_tri(gorder)

	# Quadrilaterals (q = 1 to q = 11)
	etype_nums = np.array([3, 10, 36, 37, 38, 47, 48, 49, 50, 51])
	for i in range(etype_nums.shape[0]):
		etype_num = etype_nums[i]
		elem_data = GmshElementData()
		gmsh_element_database.update({etype_num : elem_data})
		gorder = i + 1
		elem_data.gorder = gorder
		elem_data.gbasis = basis_defs.LagrangeQuad(gorder)
		elem_data.num_nodes = (gorder + 1)**2
		elem_data.node_order = gmsh_node_order_quadril(gorder)

	return gmsh_element_database


class PhysicalGroup(object):
	def __init__(self):
		self.dim = 1
		self.boundary_group_num = -1
		self.gmsh_phys_num = -1
		self.name = ""
		self.entity_tags = set()


class FaceInfo(object):
	def __init__(self):
		self.num_adjacent_elems = 0
		self.at_boundary = False
		self.boundary_group_num = 0
		self.elem_id = 0
		self.face_id = 0
		self.num_face_nodes = 0
		self.snodes = None # should be a tuple (hashable)

	def set_info(self, **kwargs):
		for key in kwargs:
			if key is "snodes":
				self.snodes = tuple(kwargs[key]) # make it hashable
			elif hasattr(self, key):
				setattr(self, key, kwargs[key])
			else: 
				raise AttributeError
	# def __eq__(self, other):
	# 	return isinstance(other, FaceInfo) and self.snodes == other.snodes
	# def __hash__(self):
	# 	return hash(self.snodes)


def go_to_line_below_string(fo, string):
	# Start from beginning
	fo.seek(0)

	# Compare line by line
	found = False
	while not found:
		fl = fo.readline()
		if not fl:
			# reached end of file
			break
		if fl.startswith(string):
			found = True

	if not found:
		raise errors.FileReadError


def check_mesh_format(fo):
	# Find beginning of section
	go_to_line_below_string(fo, "$MeshFormat")

	# Get Gmsh version
	fl = fo.readline()
	ver = fl.split()[0]
	if ver != VERSION2 and ver != VERSION4:
		raise errors.FileReadError("Unsupported version")
	file_type = int(fl.split()[1])
	if file_type != 0:
		raise errors.FileReadError("Only ASCII format supported")

	# Verify footer
	fl = fo.readline()
	if not fl.startswith("$EndMeshFormat"):
		raise errors.FileReadError

	return ver


def read_physical_groups(fo, mesh):
	# Find beginning of section
	go_to_line_below_string(fo, "$PhysicalNames")

	# Number of physical names
	num_phys_groups = int(fo.readline())

	# Allocate
	phys_groups = [PhysicalGroup() for i in range(num_phys_groups)]

	# Loop over entities
	for i in range(num_phys_groups):
		phys_group = phys_groups[i]
		fl = fo.readline()
		ls = fl.split()
		phys_group.dim = int(ls[0])
		phys_group.gmsh_phys_num = int(ls[1])
		phys_group.name = ls[2][1:-1]

		if phys_group.dim < mesh.dim-1 or phys_group.dim > mesh.dim:
			raise Exception("Physical groups should be created only for " +
					"elements and boundary faces")

	# Verify footer
	fl = fo.readline()
	if not fl.startswith("$EndPhysicalNames"):
		raise errors.FileReadError

	# Need at least one physical group to correspond to volume elements
	match = False
	for phys_group in phys_groups:
		if phys_group.dim == mesh.dim:
			match = True
			break

	if not match:
		raise Exception("No elements assigned to a physical group")

	return phys_groups, num_phys_groups


def get_nodes_ver2(fo):
	# Number of nodes
	num_nodes = int(fo.readline())
	old_to_new_node_tags = {}
	# Allocate nodes - assume 3D first
	node_coords = np.zeros([num_nodes,3])
	# Extract nodes
	new_node_tag = 0
	for n in range(num_nodes):
		fl = fo.readline()
		ls = fl.split()
		# Explicitly use for loop for compatibility with
		# both Python 2 and Python 3
		old_node_tag = int(ls[0])
		old_to_new_node_tags.update({old_node_tag : new_node_tag})
		for d in range(3):
			node_coords[new_node_tag,d] = float(ls[d+1])
		# Sanity check
		if int(ls[0]) > num_nodes:
			raise errors.FileReadError

		new_node_tag += 1

	return node_coords, old_to_new_node_tags


def get_nodes_ver4(fo):
	fl = fo.readline()
	ls = [int(l) for l in fl.split()]
	num_blocks = ls[0]
	num_nodes = ls[1]
	min_node_tag = ls[2]
	max_node_tag = ls[3]
	# Require continuous node tagging from 1 to num_nodes
	# Note: does not have to be ordered
	old_to_new_node_tags = {}
	# if min_node_tag != 1 or max_node_tag != num_nodes:
	# 	raise ValueError
	# Allocate nodes - assume 3D first
	node_coords = np.zeros([num_nodes,3])

	new_node_tag = 0
	for b in range(num_blocks):
		fl = fo.readline()
		# one block at a time
		ls = [int(l) for l in fl.split()]
		parametric = ls[2]
		num_nodes_in_block = ls[3]
		new_node_tags = np.zeros(num_nodes_in_block, dtype=int)
		# read node tags
		for n in range(num_nodes_in_block):
			# node tag
			fl = fo.readline()
			old_node_tag = int(fl)
			new_node_tags[n] = new_node_tag
			old_to_new_node_tags.update({old_node_tag : new_node_tag})
			new_node_tag += 1
		# read node coordinates
		for n in range(num_nodes_in_block):
			fl = fo.readline()
			inode = new_node_tags[n]
			node_coords[inode] = [float(l) for l in fl.split()[:3]]

	return node_coords, old_to_new_node_tags


def read_nodes(fo, ver, mesh):
	# Find beginning of section
	go_to_line_below_string(fo, "$Nodes")

	if ver == VERSION2:
		node_coords, old_to_new_node_tags = get_nodes_ver2(fo)
	else:
		node_coords, old_to_new_node_tags = get_nodes_ver4(fo)

	# Verify footer
	fl = fo.readline()
	if not fl.startswith("$EndNodes"):
		raise errors.FileReadError
	
	# Change dimension if needed
	ds = [0,1,2]
	for d in ds:
		# Find max perturbation from zero
		diff = np.amax(np.abs(node_coords[:,d]))
		if diff <= general.eps:
			# remove from ds
			ds.remove(d)

	# New dimension
	dim = len(ds)
	node_coords = node_coords[:,ds]

	if dim == 3:
		raise ValueError("3D meshes not supported")

	# Store in mesh
	mesh.node_coords = node_coords
	mesh.num_nodes = node_coords.shape[0]
	mesh.dim = dim

	return mesh, old_to_new_node_tags


def read_mesh_entities(fo, ver, mesh, phys_groups):

	if ver == VERSION2:
		return phys_groups

	# Find beginning of section
	go_to_line_below_string(fo, "$Entities")

	fl = fo.readline()
	ls = [int(l) for l in fl.split()]
	num_points = ls[0]
	num_curves = ls[1]
	num_surfaces = ls[2]

	if mesh.dim == 2:
		# skip points
		for _ in range(num_points):
			fo.readline()
		# curves + surfaces
		for _ in range(num_curves + num_surfaces):
			fl = fo.readline()
			ls = fl.split()
			entity_tag = int(ls[0])
			num_phys_tags = int(ls[7])
			if num_phys_tags == 1:
				phys_num = int(ls[8])
				for phys_group in phys_groups:
					if phys_group.gmsh_phys_num == phys_num:
						break
				phys_group.entity_tags.add(entity_tag)
			elif num_phys_tags > 1:
				raise ValueError("Entity should not be assigned to more " +
						"than one physical group")

	else:
		# add dim = 1 later
		raise NotImplementedError

	return phys_groups


def get_elem_bface_info_ver2(fo, mesh, phys_groups, num_phys_groups,
		gmsh_element_database):
	# Number of entities (cells, faces, edges)
	num_entities = int(fo.readline())
	# Loop over entities
	for _ in range(num_entities):
		fl = fo.readline()
		ls = fl.split()
		# Parse line
		# enum = int(ls[0])
		etype = int(ls[1])
		phys_num = int(ls[3])

		# if phys_num == 0:
		# 	raise ValueError("All elements need to be assigned to a physical group")
		
		found = False
		# for PGidx in range(num_phys_groups):
		# 	phys_group = phys_groups[PGidx]
		for phys_group in phys_groups:
			if phys_group.gmsh_phys_num == phys_num:
				found = True
				break
		if not found:
			raise errors.DoesNotExistError("All elements and boundary" +
					"faces must be assigned to a physical group")

		if phys_group.dim == mesh.dim:
			### Entity is an element
# <<<<<<< Updated upstream
# 			gorder = EntitiesInfo[etype].gorder
# 			gbasis = EntitiesInfo[etype].gbasis

# 			if mesh.num_elems == 0:
# 				mesh.set_params(gbasis=gbasis, gorder=gorder, num_elems=0)
# =======
			gorder = gmsh_element_database[etype].gorder
			gbasis = gmsh_element_database[etype].gbasis
			# if gbasis == -1:
			# 	raise NotImplementedError("Element type not supported")
			if mesh.num_elems == 0:
				mesh.set_params(gbasis=gbasis, gorder=gorder, num_elems=0)
			else:
				if gorder != mesh.gorder or gbasis != mesh.gbasis:
					raise ValueError(">1 element type not supported")
# >>>>>>> Stashed changes
			mesh.num_elems += 1
			# # Check for existing element group
			# found = False
			# for egrp in range(mesh.num_elemsGroup):
			# 	EG = mesh.elem_idGroups[egrp]
			# 	if QOrder == EG.QOrder and QBasis == EG.QBasis:
			# 		found = True
			# 		break
			# if found:
			# 	EG.num_elems += 1
			# else:
			# 	# Need new element group
			# 	mesh.num_elemsGroup += 1
			# 	mesh.elem_idGroups.append(Mesh.elem_idGroup(QBasis=QBasis,QOrder=QOrder))
		elif phys_group.dim == mesh.dim - 1:
			### Boundary entity
			# Check for existing boundary face group
			# found = False
			# for ibfgrp in range(mesh.num_boundary_groups):
			# 	BFG = mesh.boundary_groups[ibfgrp]
			# 	if BFG.name == phys_group.name:
			# 		found = True
			# 		break
			try:
				bgroup = mesh.boundary_groups[phys_group.name]
			except KeyError:
			# if phys_group.name in mesh.boundary_groups:
				# Group has not been assigned yet
				# mesh.num_boundary_groups += 1
				# BFG = mesh_defs.BFaceGroup()
				# mesh.boundary_groups.append(BFG)
				# BFG.name = phys_group.name
				bgroup = mesh.add_boundary_group(phys_group.name)
				phys_group.boundary_group_num = bgroup.number
			bgroup.num_boundary_faces += 1
		# else:
		# 	raise Exception("Mesh error")


def get_elem_bface_info_ver4(fo, mesh, phys_groups, num_phys_groups, gmsh_element_database):
	fl = fo.readline()
	lint = [int(l) for l in fl.split()]
	num_entity_blocks = lint[0]
	num_elems_bfaces = lint[1]

	for _ in range(num_entity_blocks):
		fl = fo.readline()
		lint = [int(l) for l in fl.split()]
		dim = lint[0]
		entity_tag = lint[1]
		etype = lint[2]
		num_in_block = lint[3]

		# find physical boundary group
		found = False
		for phys_group in phys_groups:
			if entity_tag in phys_group.entity_tags and \
					dim == phys_group.dim:
				found = True
				break
		if not found:
			raise errors.DoesNotExistError("All elements and boundary " +
					"faces must be assigned to a physical group")

		if dim == mesh.dim:
			# Element
			# Get element type data
			gorder = gmsh_element_database[etype].gorder
			gbasis = gmsh_element_database[etype].gbasis
			# if QBasis == -1:
			# 	raise NotImplementedError("Element type not supported")

			# Loop
			for _ in range(num_in_block):
				fo.readline()
				if mesh.num_elems == 0:
					mesh.set_params(gbasis=gbasis, gorder=gorder, 
							num_elems=0)
				else:
					if gorder != mesh.gorder or gbasis != mesh.gbasis:
						raise ValueError(">1 element type not supported")
				mesh.num_elems += 1
		elif dim == mesh.dim - 1:

			if phys_group.boundary_group_num >= 0:
				# BFG = mesh.boundary_groups[phys_group.boundary_group_num]
				bgroup = mesh.boundary_groups[phys_group.name]
			else:
				# Group has not been assigned yet
				# mesh.num_boundary_groups += 1
				# BFG = mesh_defs.BFaceGroup()
				# mesh.boundary_groups.append(BFG)
				# BFG.name = phys_group.name
				bgroup = mesh.add_boundary_group(phys_group.name)
				phys_group.boundary_group_num = bgroup.number
			# Loop and increment num_boundary_faces
			for _ in range(num_in_block):
				fo.readline()
				bgroup.num_boundary_faces += 1
		else:
			for _ in range(num_in_block):
				fo.readline()


	return mesh



def read_mesh_elems_boundary_faces(fo, ver, mesh, phys_groups, 
		num_phys_groups, gmsh_element_database):
	# First pass to get sizes
	# Find beginning of section
	go_to_line_below_string(fo, "$Elements")

	if ver == VERSION2:
		get_elem_bface_info_ver2(fo, mesh, phys_groups, num_phys_groups, gmsh_element_database)
	else:
		get_elem_bface_info_ver4(fo, mesh, phys_groups, num_phys_groups, gmsh_element_database)

	# Verify footer
	fl = fo.readline()
	if not fl.startswith("$EndElements"):
		raise errors.FileReadError

	return mesh


def AddFaceToHash(node0_to_faces_info, nfnode, nodes, at_boundary, Group, 
		Elem, Face):

	if nfnode <= 0:
		raise ValueError("Need nfnode > 1")

	# snodes = np.zeros(nfnode, dtype=int)
	# snodes[:] = nodes[:nfnode]

	# Sort nodes and convert to tuple (to make it hashable)
	snodes = tuple(np.sort(nodes[:nfnode]))

	# Check if face already exists in face hash
	Exists = False
	node0 = snodes[0]
	faces_info = node0_to_faces_info[node0]
	if snodes in faces_info:
		Exists = True
		face_info = faces_info[snodes]
		face_info.num_adjacent_elems += 1
	else:
		# If it doesn't exist, then add it
		face_info = FaceInfo()
		faces_info.update({snodes : face_info})
		face_info.set_info(at_boundary=at_boundary, boundary_group_num=Group, elem_id=Elem, 
				face_id=Face, num_face_nodes=nfnode, snodes=snodes)
		if not at_boundary:
			face_info.num_adjacent_elems = 1

	return face_info, Exists


def DeleteFaceFromHash(node0_to_faces_info, nfnode, nodes):

	if nfnode <= 0:
		raise ValueError("Need nfnode > 1")

	snodes = np.zeros(nfnode, dtype=int)
	snodes[:] = nodes[:nfnode]

	# Sort nodes
	snodes = tuple(np.sort(snodes))

	# Check if face already exists in face hash
	n0 = snodes[0]
	faces_info = node0_to_faces_info[n0]
	# if FaceInfos == []:
	# 	raise LookupError

	if snodes in faces_info:
		del faces_info[snodes]


def fill_elems_bfaces_ver2(fo, mesh, phys_groups, num_phys_groups, gmsh_element_database,
		old_to_new_node_tags, bf, node0_to_faces_info):
	# Number of entities
	num_entities = int(fo.readline())
	elem = 0 # elem counter
	# Loop through entities
	for _ in range(num_entities):
		fl = fo.readline()
		ls = fl.split()
		# Parse line
		# enum = int(ls[0])
		etype = int(ls[1])
		phys_num = int(ls[3])
		
		found = False
		# for PGidx in range(num_phys_groups):
		# 	phys_group = phys_groups[PGidx]
		for phys_group in phys_groups:
			if phys_group.gmsh_phys_num == phys_num:
				found = True
				break
		if not found:
			raise errors.DoesNotExistError("Physical group not found!")

		# Check if entity type is supported
		# if not gmsh_element_database[etype].Supported:
		# 	raise Exception("Entity type not supported")

		# Get nodes	
		nTag = int(ls[2]) # number of tags
			# see http://www.manpagez.com/info/gmsh/gmsh-2.2.6/gmsh_63.php
		offsetTag = 3 # 3 integers (including nTag) before tags start
		iStart = nTag + offsetTag # starting index of node numbering
		nn = gmsh_element_database[etype].num_nodes
		elist = ls[iStart:] # list of nodes (string format)
		if len(elist) != nn: 
			raise Exception("Wrong number of nodes")
		nodes = np.zeros(nn, dtype=int)
		for i in range(nn):
			# # Convert to int one-by-one for compatibility with Python 2 and 3
			# nodes[i] = int(elist[i]) - 1 # switch to zero index
			# Convert from old to new tags
			nodes[i] = old_to_new_node_tags[int(elist[i])]

		if phys_group.boundary_group_num >= 0:
			### Boundary
			# Get basic info
# <<<<<<< Updated upstream
# 			gbasis = EntitiesInfo[etype].gbasis
# 			gorder = EntitiesInfo[etype].gorder
# =======
			gbasis = gmsh_element_database[etype].gbasis
			gorder = gmsh_element_database[etype].gorder
# >>>>>>> Stashed changes
			ibfgrp = phys_group.boundary_group_num
			# BFG = mesh.boundary_groups[ibfgrp]
			BFG = mesh.boundary_groups[phys_group.name]
			# Number of q = 1 face nodes
			nfnode = gbasis.get_num_basis_coeff(1)

			# Add q = 1 nodes to hash table
			face_info, Exists = AddFaceToHash(node0_to_faces_info, nfnode, nodes, 
					True, ibfgrp, -1, bf[ibfgrp])
			bf[ibfgrp] += 1
		elif phys_group.boundary_group_num == -1:
			### Interior element
			# Get basic info
# <<<<<<< Updated upstream
# 			gorder = EntitiesInfo[etype].gorder
# 			gbasis = EntitiesInfo[etype].gbasis
# =======
			gorder = gmsh_element_database[etype].gorder
			gbasis = gmsh_element_database[etype].gbasis
# >>>>>>> Stashed changes
			# Check for existing element group
			# found = False
			# for EG in mesh.elem_idGroups:
			# 	if QOrder == EG.QOrder and QBasis == EG.QBasis:
			# 		found = True
			# 		break
			# # Sanity check
			# if not found:
			# 	raise Exception("Can't find element group")
			# Number of element nodes
			nnode = gbasis.get_num_basis_coeff(gorder)
			# Sanity check
			if nnode != gmsh_element_database[etype].num_nodes:
				raise Exception("Check Gmsh entities")
			# Convert node Ordering
			newnodes = nodes[gmsh_element_database[etype].node_order]
			# Store in elem_to_node_ids
			mesh.elem_to_node_ids[elem] = newnodes
			# Increment elem counter
			elem += 1
		else:
			raise ValueError
		 


def fill_elems_bfaces_ver4(fo, mesh, phys_groups, num_phys_groups, gmsh_element_database, 
		old_to_new_node_tags, bf, node0_to_faces_info):	
	fl = fo.readline()
	lint = [int(l) for l in fl.split()]
	num_entity_blocks = lint[0]
	num_elems_bfaces = lint[1]

	elem = 0

	for _ in range(num_entity_blocks):
		fl = fo.readline()
		lint = [int(l) for l in fl.split()]
		dim = lint[0]
		entity_tag = lint[1]
		etype = lint[2]
		num_in_block = lint[3]

		if dim == mesh.dim:
			# Element
			# Get element type data
			gorder = gmsh_element_database[etype].gorder
			gbasis = gmsh_element_database[etype].gbasis

			# Loop
			for _ in range(num_in_block):
				fl = fo.readline()
				lint = [int(l) for l in fl.split()]
				# Convert node Ordering
				nodes = np.array(lint[1:])
				for n in range(len(nodes)):
					nodes[n] = old_to_new_node_tags[nodes[n]]
				newnodes = nodes[gmsh_element_database[etype].node_order]
				# Store in elem_to_node_ids
				mesh.elem_to_node_ids[elem] = newnodes
				# Increment elem counter
				elem += 1
		elif dim == mesh.dim - 1:
			# find physical boundary group
			for phys_group in phys_groups:
				if entity_tag in phys_group.entity_tags:
					if phys_group.dim == dim:
						ibfgrp = phys_group.boundary_group_num
						break
			BFG = mesh.boundary_groups[phys_group.name]
			gbasis = gmsh_element_database[etype].gbasis
			nfnode = gbasis.get_num_basis_coeff(1) 
			# Loop and increment num_boundary_faces
			for _ in range(num_in_block):
				fl = fo.readline()
				lint = [int(l) for l in fl.split()]
				nodes = np.array(lint[1:])
				for n in range(len(nodes)):
					nodes[n] = old_to_new_node_tags[nodes[n]]
				# Add q = 1 nodes to hash table
				face_info, Exists = AddFaceToHash(node0_to_faces_info, nfnode, nodes, True, 
					ibfgrp, -1, bf[ibfgrp])
				bf[ibfgrp] += 1
		else:
			for _ in range(num_in_block):
				fo.readline()


	return mesh

def FillMesh(fo, ver, mesh, phys_groups, num_phys_groups, gmsh_element_database, old_to_new_node_tags):
	# Allocate additional mesh structures
	# for ibfgrp in range(mesh.num_boundary_groups):
	# 	BFG = mesh.boundary_groups[ibfgrp]
	# 	BFG.allocate_boundary_faces()
	for BFG in mesh.boundary_groups.values():
		BFG.allocate_boundary_faces()
	# nFaceMax = 0
	# for EG in mesh.elem_idGroups:
	# 	# also find maximum # faces per elem
	# 	EG.allocate_faces()
	# 	EG.allocate_elem_to_node_ids_map()
	# 	if nFaceMax < EG.nFacePerElem: nFaceMax = EG.nFacePerElem
	# mesh.allocate_faces()
	mesh.allocate_elem_to_node_ids_map()
	nFaceMax = mesh.gbasis.NFACES

	# Over-allocate interior_faces
	mesh.num_interior_faces = mesh.num_elems*nFaceMax
	mesh.allocate_interior_faces()

	# reset num_interior_faces - use as a counter
	mesh.num_interior_faces = 0

	# Dictionary for hashing
	# node0_to_faces_info = {n:FaceInfo() for n in range(mesh.num_nodes)}
	# node0_to_faces_info = {n:[] for n in range(mesh.num_nodes)}
	node0_to_faces_info = [{} for n in range(mesh.num_nodes)] # list of dicts

	# Go to entities section
	go_to_line_below_string(fo, "$Elements")

	bf = [0 for i in range(mesh.num_boundary_groups)] # boundary_face counter

	if ver == VERSION2:
		fill_elems_bfaces_ver2(fo, mesh, phys_groups, num_phys_groups, gmsh_element_database, 
				old_to_new_node_tags, bf, node0_to_faces_info)
	else:
		fill_elems_bfaces_ver4(fo, mesh, phys_groups, num_phys_groups, gmsh_element_database, 
				old_to_new_node_tags, bf, node0_to_faces_info)


	# Verify footer
	fl = fo.readline()
	if not fl.startswith("$EndElements"):
		raise errors.FileReadError

	# Fill boundary and interior face info
	# for egrp in range(mesh.num_elemsGroup):
	# 	EG = mesh.elem_idGroups[egrp]
	for elem in range(mesh.num_elems):
		for face in range(mesh.gbasis.NFACES):
			# Local q = 1 nodes on face
			gbasis = mesh.gbasis
			fnodes = gbasis.get_local_face_principal_node_nums(mesh.gorder, face)
			nfnode = fnodes.shape[0]

			# Convert to global nodes
			fnodes = mesh.elem_to_node_ids[elem][fnodes]

			# Add to hash table
			face_info, Exists = AddFaceToHash(node0_to_faces_info, nfnode, fnodes, False, 
				-1, elem, face)

			if Exists:
				# Face already exists in hash table
				# if face_info.nVisit != 2:
				# 	raise ValueError("More than two elements share a face " + 
				# 		"or a boundary face is referenced by more than one element")

				# Link elem to boundary_face or IFace
				if face_info.at_boundary:
					if face_info.num_adjacent_elems != 1:
						raise ValueError("More than one element adjacent " +
								"to boundary face")
					# boundary face
					# Store in BFG
					# BFG = mesh.boundary_groups[face_info.boundary_group_num]
					found = False
					# Make this cleaner later
					for phys_group in phys_groups:
						if phys_group.boundary_group_num == face_info.boundary_group_num:
							found = True
							break
					if not found: raise Exception
					BFG = mesh.boundary_groups[phys_group.name]
					# try:
					# 	boundary_face = BFG.boundary_faces[face_info.face_id]
					# except:
					# 	code.interact(local=locals())
					boundary_face = BFG.boundary_faces[face_info.face_id]
					boundary_face.elem_id = elem; boundary_face.face_id = face
					# Store in Face
					# Face = mesh.face_ids[elem][face]
					# Face.boundary_group_num = face_info.boundary_group_num
					# Face.gmsh_phys_num = face_info.face_id
				else:
					# interior face
					if face_info.num_adjacent_elems != 2:
						raise ValueError("More than two elements adjacent " +
								"to interior face")
					# Store in IFace
					IFace = mesh.interior_faces[mesh.num_interior_faces]
					IFace.elemL_id = face_info.elem_id
					IFace.faceL_id = face_info.face_id
					IFace.elemR_id = elem
					IFace.faceR_id = face
					# Store in left Face
					# Face = mesh.face_ids[face_info.elem_id][face_info.face_id]
					# Face.boundary_group_num = general.INTERIORFACE
					# Face.gmsh_phys_num = mesh.num_interior_faces
					# # Store in right face
					# Face = mesh.face_ids[elem][face]
					# Face.boundary_group_num = general.INTERIORFACE
					# Face.gmsh_phys_num = mesh.num_interior_faces
					# Increment IFace counter
					mesh.num_interior_faces += 1

				DeleteFaceFromHash(node0_to_faces_info, nfnode, fnodes)

	# Make sure no faces left in hash
	nleft = 0
	for n in range(mesh.num_nodes):
		faces_info = node0_to_faces_info[n]
		for snodes in faces_info.keys():
			print(snodes)
			# for node in snodes:
			# 	print(int(node+1))
			nleft += 1

	if nleft != 0:
		raise ValueError("Mesh connectivity error: the above %d " % (nleft) +
			"face(s) remain(s) in the hash")

	# Resize IFace
	if mesh.num_interior_faces > mesh.num_elems*nFaceMax:
		raise ValueError
	mesh.interior_faces = mesh.interior_faces[:mesh.num_interior_faces]

	# mesh.fill_faces()
	mesh.create_elements()

	# Check face orientations
	mesh_tools.check_face_orientations(mesh)



def ReadGmshFile(FileName):
	# Check file extension
	if FileName[-4:] != ".msh":
		raise errors.FileReadError("Wrong file type")

	# Open file
	fo = open(FileName, "r")

	# Mesh object
	mesh = mesh_defs.Mesh(num_elems=0)

	# Object that stores Gmsh entity info
	gmsh_element_database = create_gmsh_element_database()

	# Read sections one-by-one
	ver = check_mesh_format(fo)
	mesh, old_to_new_node_tags = read_nodes(fo, ver, mesh)
	phys_groups, num_phys_groups = read_physical_groups(fo, mesh)
	phys_groups = read_mesh_entities(fo, ver, mesh, phys_groups)
	mesh = read_mesh_elems_boundary_faces(fo, ver, mesh, phys_groups, num_phys_groups, gmsh_element_database)
	# code.interact(local=locals())

	# Create rest of mesh
	FillMesh(fo, ver, mesh, phys_groups, num_phys_groups, gmsh_element_database, old_to_new_node_tags)

	# Print some stats
	print("%d elements in the mesh" % (mesh.num_elems))
	
	# Done with file
	fo.close()

	return mesh
