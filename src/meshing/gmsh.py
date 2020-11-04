# ------------------------------------------------------------------------ #
#
#       File : src/meshing/gmsh.py
#
#       Contains functions for processing Gmsh files.
#      
# ------------------------------------------------------------------------ #
import numpy as np

import errors
import general

import meshing.meshbase as mesh_defs
import meshing.tools as mesh_tools

import numerics.basis.basis as basis_defs


VERSION2 = "2.2" 
	# see http://www.manpagez.com/info/gmsh/gmsh-2.2.6/gmsh_63.php
VERSION4 = "4.1"
	# see https://gmsh.info/doc/texinfo/gmsh.html#MSH-file-format


class GmshElementData(object):
	'''
	This class provides information about a given Gmsh element.

	Attributes:
	-----------
	num_nodes: int
		number of nodes
	gorder: int
		order of geometry interpolation
	gbasis: int
		object for geometry basis
	node_order: numpy array (int)
		maps quail node ordering to gmsh node ordering
	'''
	def __init__(self):
		self.num_nodes = -1
		self.gorder = -1
		self.gbasis = -1
		self.node_order = np.array(0, dtype=int)


def gmsh_node_order_seg(gorder):
	'''
	This function maps quail node ordering to gmsh node ordering for segments

	Inputs:
	------- 
	    gorder: order of geometry interpolation

	Outputs:
	--------
	    nodes: maps quail node ordering to gmsh node ordering
	'''
	num_nodes = gorder + 1
	nodes = np.arange(num_nodes)
	nodes[1:-1] = nodes[2:]
	nodes[-1] = 1

	return nodes


def populate_nodes_quadril(gorder, start, nodes):
	'''
	Helper function for mapping quail node ordering to gmsh node ordering for
	quadrilaterals. Recursion is employed.

	Inputs: 
	    gorder: order of geometry interpolation
	    start: starting node
	    nodes: array for mapping node orders

	Outputs:
	    nodes: maps quail node ordering to gmsh node ordering
	    	(modified)
	'''
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
			nodes[1:-1, 1:-1] = populate_nodes_quadril(gorder-2, start, 
				nodes[1:-1, 1:-1])

	return nodes


def gmsh_node_order_quadril(gorder):
	'''
	This function maps quail node ordering to gmsh node ordering for 
	quadrilaterals.

	Inputs: 
	    gorder: order of geometry interpolation

	Outputs:
	    nodes: maps quail node ordering to gmsh node ordering
	'''
	nodes = populate_nodes_quadril(gorder, 0, np.zeros([gorder+1, gorder+1], 
			dtype=int))
	nodes.shape = -1

	return nodes


def populate_nodes_tri(gorder, start, nodes):
	'''
	Helper function for mapping quail node ordering to gmsh node ordering for
	triangles. Recursion is employed.

	Inputs: 
	    gorder: order of geometry interpolation
	    start: starting node
	    nodes: array for mapping node orders

	Outputs:
	    nodes: maps quail node ordering to gmsh node ordering
	    	(modified)
	'''
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
			nodes[1:gorder-1,1:gorder-1] = populate_nodes_tri(gorder-3, 
				start, nodes[1:gorder-1,1:gorder-1])

	return nodes


def gmsh_node_order_tri(gorder):
	'''
	This function maps quail node ordering to gmsh node ordering for 
	triangles.

	Inputs: 
	    gorder: order of geometry interpolation

	Outputs:
	    nodes: maps quail node ordering to gmsh node ordering
	'''
	nodes = populate_nodes_tri(gorder, 0, np.zeros([gorder+1, gorder+1], 
			dtype=int)-1)
	# only lower triangular 
	nodes = nodes[nodes >= 0]

	return nodes


def create_gmsh_element_database():
	'''
	This function creates a database that holds information about all
	supported Gmsh elements.

	Outputs:
	--------
	    gmsh_element_database: Gmsh element database
	'''
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
	'''
	This class holds information about a given Gmsh physical group.

	Attributes:
	-----------
	ndims: int
	    number of spatial dimensions
	boundary_group_num: int
	    boundary group number (in quail)
	gmsh_phys_num: int
	    physical group number assigned by Gmsh
	name: str
	    name of physical group
	entity_tags: set
	    tags of entities associated with this physical group
	'''
	def __init__(self):
		self.ndims = 1
		self.boundary_group_num = -1
		self.gmsh_phys_num = -1
		self.name = ""
		self.entity_tags = set()


class FaceInfo(object):
	'''
	This class holds information about a given interior or boundary face.

	Attributes:
	-----------
	num_adjacent_elems: int
	    number of elements adjacent to given face
	at_boundary: bool
	    True if boundary face, False if interior face
	boundary_group_num: int
	    boundary group number (in quail)
	elem_ID: int
	    ID of current adjacent element
	face_ID: int
	    local ID of face from perspective of current adjacent element
	num_face_nodes: int
	    number of face nodes (q = 1)
	node_IDs_sort: tuple
		global IDs of face nodes sorted in ascending order 
    
    Methods:
    ---------
    set_info
        sets attributes
	'''
	def __init__(self):
		self.num_adjacent_elems = 0
		self.at_boundary = False
		self.boundary_group_num = 0
		self.elem_ID = 0
		self.face_ID = 0
		self.num_face_nodes = 0
		self.node_IDs_sort = tuple() # should be a tuple (hashable)

	def set_info(self, **kwargs):
		'''
		This method is a wrapper for setting all of this class's attributes.
		Note that node_IDs_sort can be passed as a numpy array, list, etc.; 
		it will be converted to a tuple, which is hashable and can therefore
		be used as a dictionary key.
		'''
		for key in kwargs:
			if key == "node_IDs_sort":
				self.node_IDs_sort = tuple(kwargs[key]) # make it hashable
			elif hasattr(self, key):
				setattr(self, key, kwargs[key])
			else: 
				raise AttributeError


def go_to_line_below_string(fo, string):
	'''
	This function sets a given file's current position to the line after 
	an input string.

	Inputs:
	-------
	    fo: file object
	    string: input string

	Outputs:
	--------
		fo: file object (current position is modified)
	'''
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
	'''
	This function checks the Gmsh file format and version for compatibility.

	Inputs:
	-------
	    fo: file object

	Outputs:
	--------
		fo: file object (current position is modified)
		ver: Gmsh file version
	'''
	# Find beginning of section
	go_to_line_below_string(fo, "$MeshFormat")

	# Get Gmsh version
	fl = fo.readline()
	ver = fl.split()[0]
	if ver != VERSION2 and ver != VERSION4:
		raise errors.FileReadError("Unsupported version! " + 
				"Only versions 2.2 and 4.1 are supported.")
	file_type = int(fl.split()[1])
	if file_type != 0:
		raise errors.FileReadError("Only ASCII format supported")

	# Verify footer
	fl = fo.readline()
	if not fl.startswith("$EndMeshFormat"):
		raise errors.FileReadError

	return ver


def import_physical_groups(fo, mesh):
	'''
	This function imports information about Gmsh physical groups.

	Inputs:
	-------
	    fo: file object
	    mesh: mesh object

	Outputs:
	-------
		fo: file object (current position is modified)
	    phys_groups: list of physical group objects
	    num_phys_groups: number of physical groups
	'''
	# Find beginning of section
	go_to_line_below_string(fo, "$PhysicalNames")

	# Number of physical names
	num_phys_groups = int(fo.readline())

	# Allocate
	phys_groups = [PhysicalGroup() for i in range(num_phys_groups)]

	# Loop over physical groups
	for i in range(num_phys_groups):
		phys_group = phys_groups[i]
		fl = fo.readline()
		ls = fl.split()
		phys_group.ndims = int(ls[0])
		phys_group.gmsh_phys_num = int(ls[1])
		phys_group.name = ls[2][1:-1]

		if phys_group.ndims < mesh.ndims-1 or phys_group.ndims > mesh.ndims:
			raise Exception("Physical groups should be created only for " +
					"elements and boundary faces")

	# Verify footer
	fl = fo.readline()
	if not fl.startswith("$EndPhysicalNames"):
		raise errors.FileReadError

	# Need at least one physical group to correspond to volume elements
	match = False
	for phys_group in phys_groups:
		if phys_group.ndims == mesh.ndims:
			match = True
			break

	if not match:
		raise Exception("No elements assigned to a physical group")

	return phys_groups, num_phys_groups


def get_nodes_ver2(fo):
	'''
	This function imports node information for Gmsh 2.2.

	Inputs:
	-------
	    fo: file object

	Outputs:
	--------
		fo: file object (current position is modified)
		node_coords: node coordinates
		old_to_new_node_IDs: maps Gmsh-assigned (old) node IDs to new IDs

	Notes:
	------
		New IDs are assigned sequentially from 0 to num_nodes-1 in the order 
		in which nodes are read. Gmsh does not necessarily follow this 
		convention, especially with the newer versions.
	'''
	# Number of nodes
	num_nodes = int(fo.readline())
	if num_nodes == 0:
		raise ValueError("No nodes to import!")
	old_to_new_node_IDs = {}
	# Allocate nodes - assume 3D first
	node_coords = np.zeros([num_nodes, 3])

	# Extract nodes
	new_node_ID = 0
	for n in range(num_nodes):
		fl = fo.readline()
		ls = fl.split()
		# Node ID
		old_node_ID = int(ls[0])
		old_to_new_node_IDs.update({old_node_ID : new_node_ID})
		# Node coordinates
		for d in range(3):
			node_coords[new_node_ID, d] = float(ls[d+1])
		# Sanity check
		if int(ls[0]) > num_nodes:
			raise errors.FileReadError

		new_node_ID += 1

	return node_coords, old_to_new_node_IDs


def get_nodes_ver4(fo):
	'''
	This function imports node information for Gmsh 4.1.

	Inputs:
	-------
	    fo: file object

	Outputs:
	--------
		fo: file object (current position is modified)
		node_coords: node coordinates
		old_to_new_node_IDs: maps Gmsh-assigned (old) node IDs to new IDs

	Notes:
	------
		New IDs are assigned sequentially from 0 to num_nodes-1 in the order 
		in which nodes are read. Gmsh does not necessarily follow this 
		convention, especially with the newer versions.
	'''
	fl = fo.readline()
	ls = [int(l) for l in fl.split()]
	num_blocks = ls[0]
	num_nodes = ls[1]
	if num_nodes == 0:
		raise ValueError("No nodes to import!")

	old_to_new_node_IDs = {}
	# Allocate nodes - assume 3D first
	node_coords = np.zeros([num_nodes, 3])

	# Extract nodes
	new_node_ID = 0
	for b in range(num_blocks):
		# One block at a time
		fl = fo.readline()
		ls = [int(l) for l in fl.split()]
		num_nodes_in_block = ls[3]
		new_node_IDs = np.zeros(num_nodes_in_block, dtype=int)
		# Node IDs
		for n in range(num_nodes_in_block):
			fl = fo.readline()
			old_node_ID = int(fl)
			new_node_IDs[n] = new_node_ID
			old_to_new_node_IDs.update({old_node_ID : new_node_ID})
			new_node_ID += 1
		# Node coordinates
		for n in range(num_nodes_in_block):
			fl = fo.readline()
			inode = new_node_IDs[n]
			node_coords[inode] = [float(l) for l in fl.split()[:3]]

	return node_coords, old_to_new_node_IDs


def import_nodes(fo, ver, mesh):
	'''
	This function imports and processes node information.

	Inputs:
	-------
	    fo: file object
	    ver: Gmsh file version
	    mesh: mesh object

	Outputs:
	--------
		fo: file object (current position is modified)
		mesh: mesh object (modified)
		old_to_new_node_IDs: maps Gmsh-assigned (old) node IDs to new IDs

	Notes:
	------
		New IDs are assigned sequentially from 0 to num_nodes-1 in the order 
		in which nodes are read. Gmsh does not necessarily follow this 
		convention, especially with the newer versions.
	'''
	# Find beginning of section
	go_to_line_below_string(fo, "$Nodes")

	# Import nodes
	if ver == VERSION2:
		node_coords, old_to_new_node_IDs = get_nodes_ver2(fo)
	else:
		node_coords, old_to_new_node_IDs = get_nodes_ver4(fo)

	# Verify footer
	fl = fo.readline()
	if not fl.startswith("$EndNodes"):
		raise errors.FileReadError
	
	# Number of spatial dimensions
	ds_all = [0, 1, 2]
	ds = []
	for d in ds_all:
		# Find max perturbation from zero
		diff = np.amax(np.abs(node_coords[:, d]))
		if diff > general.eps:
			# keep this spatial dimension
			ds.append(ds_all[d])

	# New dimension
	ndims = len(ds)
	node_coords = node_coords[:, ds]

	if ndims == 3:
		raise ValueError("3D meshes not supported")

	# Store in mesh
	mesh.node_coords = node_coords
	mesh.num_nodes = node_coords.shape[0]
	mesh.ndims = ndims

	return mesh, old_to_new_node_IDs


def import_mesh_entities(fo, ver, mesh, phys_groups):
	'''
	This function imports entity information for Gmsh 4.1.

	Inputs:
	-------
	    fo: file object
	    ver: Gmsh file version
	    mesh: mesh object
	    phys_groups: list of physical group objects

	Outputs:
	--------
		fo: file object (current position is modified)
		phys_groups: list of physical group objects (modified)
	'''
	def get_entity_tag(fo, phys_groups, num_phys_tags_idx, phys_num_idx):
		'''
		This inner function obtains the tag of a given entity and stores it
		in the corresponding physical group

		Inputs:
		-------
		    fo: file object
		    phys_groups: list of physical group objects
		    num_phys_tags_idx: index to obtain number of physical tags
		    phys_num_idx: index to obtain physical group number

		Outputs:
		--------
			fo: file object (current position is modified)
			phys_groups: list of physical group objects (modified)
		'''
		fl = fo.readline()
		ls = fl.split()
		entity_tag = int(ls[0])
		num_phys_tags = int(ls[num_phys_tags_idx])
		if num_phys_tags == 1:
			phys_num = int(ls[phys_num_idx])
			for phys_group in phys_groups:
				if phys_group.gmsh_phys_num == phys_num:
					break
			phys_group.entity_tags.add(entity_tag)
		elif num_phys_tags > 1:
			raise ValueError("Entity should not be assigned to more " +
					"than one physical group")

	if ver == VERSION2:
		return phys_groups

	# Find beginning of section
	go_to_line_below_string(fo, "$Entities")

	fl = fo.readline()
	ls = [int(l) for l in fl.split()]
	num_points = ls[0]
	num_curves = ls[1]
	num_surfaces = ls[2]
	num_volumes = ls[3]

	# Read entities
	if mesh.ndims == 2:
		# Skip points
		for _ in range(num_points):
			fo.readline()
		# Curves + surfaces
		for _ in range(num_curves + num_surfaces):
			get_entity_tag(fo, phys_groups, 7, 8)
		# Skip volumes
		for _ in range(num_volumes):
			fo.readline()
	else:
		# 1D
		# Points
		for _ in range(num_points):
			get_entity_tag(fo, phys_groups, 4, 5)
		# Curves
		for _ in range(num_curves):
			get_entity_tag(fo, phys_groups, 7, 8)
		# Skip surfaces and volumes
		for _ in range(num_surfaces + num_volumes):
			fl = fo.readline()

	# Verify footer
	fl = fo.readline()
	if not fl.startswith("$EndEntities"):
		raise errors.FileReadError

	return phys_groups


def get_elem_bface_info_ver2(fo, mesh, phys_groups, num_phys_groups,
		gmsh_element_database):
	'''
	This function imports element and boundary face info for Gmsh 2.2.

	Inputs:
	-------
	    fo: file object
	    mesh: mesh object
	    phys_groups: list of physical group objects
	    num_phys_groups: number of physical groups
	    gmsh_element_database: database on Gmsh elements

	Outputs:
	--------
		fo: file object (current position is modified)
		mesh: mesh object (modified)
	'''
	# Number of elements and boundary faces
	num_elems_bfaces = int(fo.readline())
	if num_elems_bfaces == 0:
		raise ValueError("No elements or boundary faces to import")

	# Loop 
	for _ in range(num_elems_bfaces):
		fl = fo.readline()
		ls = fl.split()
		etype = int(ls[1]) # Gmsh element type
		phys_num = int(ls[3])
		
		# Extract physical group
		found = False
		for phys_group in phys_groups:
			if phys_group.gmsh_phys_num == phys_num:
				found = True
				break
		if not found:
			raise errors.DoesNotExistError("All elements and boundary" +
					"faces must be assigned to a physical group")

		# Process
		if phys_group.ndims == mesh.ndims:
			# Element
			# Make sure only one type of volume element in type
			gorder = gmsh_element_database[etype].gorder
			gbasis = gmsh_element_database[etype].gbasis

			if mesh.num_elems == 0:
				mesh.set_params(gbasis=gbasis, gorder=gorder, num_elems=0)
			else:
				if gorder != mesh.gorder or gbasis != mesh.gbasis:
					raise ValueError(">1 element type not supported")

			# Increment number of elements
			mesh.num_elems += 1

		elif phys_group.ndims == mesh.ndims - 1:
			# Boundary face
			try:
				bgroup = mesh.boundary_groups[phys_group.name]
			except KeyError:
				# Add new boundary group
				bgroup = mesh.add_boundary_group(phys_group.name)
				phys_group.boundary_group_num = bgroup.number

			# Increment number of boundary faces
			bgroup.num_boundary_faces += 1


def get_elem_bface_info_ver4(fo, mesh, phys_groups, num_phys_groups, 
		gmsh_element_database):
	'''
	This function imports element and boundary face info for Gmsh 4.1.

	Inputs:
	-------
	    fo: file object
	    mesh: mesh object
	    phys_groups: list of physical group objects
	    num_phys_groups: number of physical groups
	    gmsh_element_database: database on Gmsh elements

	Outputs:
	--------
		fo: file object (current position is modified)
		mesh: mesh object (modified)
	'''
	fl = fo.readline()
	lint = [int(l) for l in fl.split()]
	num_entity_blocks = lint[0]
	num_elems_bfaces = lint[1]
	if num_elems_bfaces == 0:
		raise ValueError("No elements or boundary faces to import")

	for _ in range(num_entity_blocks):
		fl = fo.readline()
		lint = [int(l) for l in fl.split()]
		ndims = lint[0]
		entity_tag = lint[1]
		etype = lint[2] # Gmsh element type
		num_in_block = lint[3]

		# Find physical boundary group
		found = False
		for phys_group in phys_groups:
			if entity_tag in phys_group.entity_tags and \
					ndims == phys_group.ndims:
				found = True
				break
		if not found:
			raise errors.DoesNotExistError("All elements and boundary " +
					"faces must be assigned to a physical group")

		if ndims == mesh.ndims:
			# Element
			# Get element type data
			gorder = gmsh_element_database[etype].gorder
			gbasis = gmsh_element_database[etype].gbasis

			# Loop
			for _ in range(num_in_block):
				# Make sure only one type of volume element in type
				fo.readline()
				if mesh.num_elems == 0:
					mesh.set_params(gbasis=gbasis, gorder=gorder, 
							num_elems=0)
				else:
					if gorder != mesh.gorder or gbasis != mesh.gbasis:
						raise ValueError(">1 element type not supported")

				# Increment number of elements
				mesh.num_elems += 1

		elif ndims == mesh.ndims - 1:
			# Boundary face

			if phys_group.boundary_group_num >= 0:
				bgroup = mesh.boundary_groups[phys_group.name]
			else:
				# Group has not been assigned yet
				bgroup = mesh.add_boundary_group(phys_group.name)
				phys_group.boundary_group_num = bgroup.number

			# Loop and increment number of boundary faces
			for _ in range(num_in_block):
				fo.readline()
				bgroup.num_boundary_faces += 1
		else:
			# Skip
			for _ in range(num_in_block):
				fo.readline()

	return mesh


def import_mesh_elems_boundary_faces(fo, ver, mesh, phys_groups, 
		num_phys_groups, gmsh_element_database):
	'''
	This function imports element and boundary face info.

	Inputs:
	-------
	    fo: file object
	    ver: Gmsh file version
	    mesh: mesh object
	    phys_groups: list of physical group objects
	    num_phys_groups: number of physical groups
	    gmsh_element_database: database on Gmsh elements

	Outputs:
	--------
		fo: file object (current position is modified)
		mesh: mesh object (modified)
	'''
	# Find beginning of section
	go_to_line_below_string(fo, "$Elements")

	# Get element and boundary face info
	if ver == VERSION2:
		get_elem_bface_info_ver2(fo, mesh, phys_groups, num_phys_groups, 
				gmsh_element_database)
	else:
		get_elem_bface_info_ver4(fo, mesh, phys_groups, num_phys_groups, 
				gmsh_element_database)

	# Verify footer
	fl = fo.readline()
	if not fl.startswith("$EndElements"):
		raise errors.FileReadError

	return mesh


def add_face_info_to_table(node0_to_faces_info, num_face_nodes, node_IDs, 
	at_boundary, group_num, elem_ID, face_ID):
	'''
	This function adds face (boundary or interior) info to the 
	node0_to_faces_info table.

	Inputs:
	-------
	    node0_to_faces_info: list of dicts, where, for a given dict, each 
	    	key is a tuple containing the global node IDs defining a face 
	    	(in ascending order) and the corresponding value is an object
	    	containing relevant info about the face; all the keys for a 
	    	given dict have the same node0, which is the smallest global 
	    	node ID of the keys (tuples); the list is indexed by node0;
	    	this is used to connect faces to elements
	    num_face_nodes: number of (q = 1) face nodes
	    node_IDs: global IDs of nodes defining the face
	    at_boundary: True if boundary face; False if interior face
	    group_num: number of physical group if boundary face; -1 if 
	    	interior face
	    elem_ID: element ID of current adjacent element
	    face_ID: local face ID from perspective of current adjacent element

	Outputs:
	--------
		face_info: object containing relevant face info
		already_added: True if face has already been added to table; False
			if otherwise
		node0_to_faces_info: see above (modified)
	'''
	if num_face_nodes <= 0:
		raise ValueError("Need num_face_nodes > 1")

	# Sort nodes and convert to tuple (to make it hashable)
	node_IDs_sort = tuple(np.sort(node_IDs[:num_face_nodes]))

	# Extract correct faces_info dict
	node0 = node_IDs_sort[0]
	faces_info = node0_to_faces_info[node0]

	# Check if face already exists in table
	already_added = False
	if node_IDs_sort in faces_info:
		already_added = True
		face_info = faces_info[node_IDs_sort]
		face_info.num_adjacent_elems += 1
	else:
		# Not yet added, so add now
		face_info = FaceInfo()
		faces_info.update({node_IDs_sort : face_info})
		face_info.set_info(at_boundary=at_boundary, 
				boundary_group_num=group_num, elem_ID=elem_ID, 
				face_ID=face_ID, num_face_nodes=num_face_nodes, 
				node_IDs_sort=node_IDs_sort)
		if not at_boundary:
			face_info.num_adjacent_elems = 1

	return face_info, already_added


def delete_face_info_from_table(node0_to_faces_info, num_face_nodes, 
		node_IDs):
	'''
	This function deletes face (boundary or interior) info from the 
	node0_to_faces_info table.

	Inputs:
	-------
	    node0_to_faces_info: see above description of add_face_info_to_table
	    num_face_nodes: number of (q = 1) face nodes
	    node_IDs: global IDs of nodes defining the face

	Outputs:
	--------
		node0_to_faces_info: see above (modified)
	'''
	if num_face_nodes <= 0:
		raise ValueError("Need num_face_nodes > 1")

	# Sort nodes and convert to tuple (to make it hashable)
	node_IDs_sort = tuple(np.sort(node_IDs[:num_face_nodes]))

	# Extract correct faces_info dict
	n0 = node_IDs_sort[0]
	faces_info = node0_to_faces_info[n0]

	# Delete
	if node_IDs_sort in faces_info:
		del faces_info[node_IDs_sort]


def process_elems_bfaces_ver2(fo, mesh, phys_groups, num_phys_groups, 
		gmsh_element_database, old_to_new_node_IDs, num_bfaces_per_bgroup, 
		node0_to_faces_info):
	'''
	This function processes element and boundary face info for Gmsh 2.2.

	Inputs:
	-------
		fo: file object
		mesh: mesh object
		phys_groups: list of physical group objects
		num_phys_groups: number of physical groups
		gmsh_element_database: Gmsh element database
		old_to_new_node_IDs: maps Gmsh-assigned (old) node IDs to new IDs
		num_bfaces_per_bgroup: number of boundary faces per boundary group
	    node0_to_faces_info: see above description of add_face_info_to_table

	Outputs:
	--------
		fo: file object (current position is modified)
		mesh: mesh object (modified)
	    node0_to_faces_info: see above description of add_face_info_to_table
	    	(modified)
	'''
	num_elems_bfaces = int(fo.readline())
	num_elems = 0 # counter for number of elements

	# Loop through entities
	for _ in range(num_elems_bfaces):
		fl = fo.readline()
		ls = fl.split()
		etype = int(ls[1]) # Gmsh element type
		phys_num = int(ls[3])
		
		# Find physical group
		found = False
		for phys_group in phys_groups:
			if phys_group.gmsh_phys_num == phys_num:
				found = True
				break
		if not found:
			raise errors.DoesNotExistError("Physical group not found!")

		# Get nodes	
		num_tags = int(ls[2]) # number of tags
		tag_offset = 3 # 3 integers (including num_tags) before tags start
		istart = num_tags + tag_offset # starting index of node numbering
		num_nodes = gmsh_element_database[etype].num_nodes
		elist = ls[istart:] # list of nodes (string format)
		if len(elist) != num_nodes: 
			raise Exception("Wrong number of nodes")

		# Convert nodes IDs
		node_IDs = np.zeros(num_nodes, dtype=int)
		for i in range(num_nodes):
			node_IDs[i] = old_to_new_node_IDs[int(elist[i])]

		if phys_group.boundary_group_num >= 0:
			# This is a boundary face

			# Get info
			gbasis = gmsh_element_database[etype].gbasis
			gorder = gmsh_element_database[etype].gorder
			bgroup_num = phys_group.boundary_group_num
			bgroup = mesh.boundary_groups[phys_group.name]
			num_face_nodes = gbasis.get_num_basis_coeff(1)

			# Add face info to table
			_, _ = add_face_info_to_table(node0_to_faces_info, 
					num_face_nodes, node_IDs, True, bgroup_num, -1, 
					num_bfaces_per_bgroup[bgroup_num])

			# Increment number of boundary faces
			num_bfaces_per_bgroup[bgroup_num] += 1

		elif phys_group.boundary_group_num == -1:
			# This is a volume element

			# Get info
			gorder = gmsh_element_database[etype].gorder
			gbasis = gmsh_element_database[etype].gbasis
			num_nodes = gbasis.get_num_basis_coeff(gorder)
			# Sanity check
			if num_nodes != gmsh_element_database[etype].num_nodes:
				raise Exception("Number of nodes doesn't match up")

			# Convert from Gmsh node ordering to quail node ordering and
			# store
			new_node_IDs = node_IDs[gmsh_element_database[etype].node_order]
			mesh.elem_to_node_IDs[num_elems] = new_node_IDs

			# Increment elem counter
			num_elems += 1

		else:
			raise ValueError


def process_elems_bfaces_ver4(fo, mesh, phys_groups, num_phys_groups, 
		gmsh_element_database, old_to_new_node_IDs, num_bfaces_per_bgroup, 
		node0_to_faces_info):
	'''
	This function processes element and boundary face info for Gmsh 4.1.

	Inputs:
	-------
		fo: file object
		mesh: mesh object
		phys_groups: list of physical group objects
		num_phys_groups: number of physical groups
		gmsh_element_database: Gmsh element database
		old_to_new_node_IDs: maps Gmsh-assigned (old) node IDs to new IDs
		num_bfaces_per_bgroup: number of boundary faces per boundary group
	    node0_to_faces_info: see above description of add_face_info_to_table

	Outputs:
	--------
		fo: file object (current position is modified)
		mesh: mesh object (modified)
	    node0_to_faces_info: see above description of add_face_info_to_table
	    	(modified)
	'''
	fl = fo.readline()
	lint = [int(l) for l in fl.split()]
	num_entity_blocks = lint[0]
	num_elems_bfaces = lint[1]
	num_elems = 0 # counter for number of elements

	for _ in range(num_entity_blocks):
		fl = fo.readline()
		lint = [int(l) for l in fl.split()]
		ndims = lint[0]
		entity_tag = lint[1]
		etype = lint[2] # Gmsh element type
		num_in_block = lint[3]

		if ndims == mesh.ndims:
			# Volume element

			# Get info
			gorder = gmsh_element_database[etype].gorder
			gbasis = gmsh_element_database[etype].gbasis

			# Extract and process nodes
			for _ in range(num_in_block):
				fl = fo.readline()
				lint = [int(l) for l in fl.split()]

				# Convert from Gmsh to quail node ordering and store
				nodes = np.array(lint[1:])
				for n in range(len(nodes)):
					nodes[n] = old_to_new_node_IDs[nodes[n]]
				new_node_IDs = nodes[gmsh_element_database[etype].node_order]
				mesh.elem_to_node_IDs[num_elems] = new_node_IDs

				# Increment number of elements
				num_elems += 1

		elif ndims == mesh.ndims - 1:
			# Boundary face

			# Find physical boundary group
			found = False
			for phys_group in phys_groups:
				if entity_tag in phys_group.entity_tags:
					if phys_group.ndims == ndims:
						bgroup_num = phys_group.boundary_group_num
						found = True
						break
			if not found:
				raise errors.DoesNotExistError("Physical boundary group " +
						"not found")
			bgroup = mesh.boundary_groups[phys_group.name]

			# Get info
			gbasis = gmsh_element_database[etype].gbasis
			num_face_nodes = gbasis.get_num_basis_coeff(1) 

			# Loop
			for _ in range(num_in_block):
				fl = fo.readline()
				lint = [int(l) for l in fl.split()]

				# Convert node IDs
				nodes = np.array(lint[1:])
				for n in range(len(nodes)):
					nodes[n] = old_to_new_node_IDs[nodes[n]]

				# Add face info to table
				_, _ = add_face_info_to_table(node0_to_faces_info,
						num_face_nodes, nodes, True, bgroup_num, -1, 
						num_bfaces_per_bgroup[bgroup_num])

				# Increment number of boundary faces
				num_bfaces_per_bgroup[bgroup_num] += 1

		else:
			# Skip
			for _ in range(num_in_block):
				fo.readline()


def fill_mesh(fo, ver, mesh, phys_groups, num_phys_groups, 
		gmsh_element_database, old_to_new_node_IDs):
	'''
	This function fills the mesh.

	Inputs:
	-------
		fo: file object
		ver: Gmsh file version
		mesh: mesh object
		phys_groups: list of physical group objects
		num_phys_groups: number of physical groups
		gmsh_element_database: Gmsh element database
		old_to_new_node_IDs: maps Gmsh-assigned (old) node IDs to new IDs

	Outputs:
	--------
		fo: file object (current position is modified)
		mesh: mesh object (modified)
	'''
	# Allocate boundary groups and faces
	for bgroup in mesh.boundary_groups.values():
		bgroup.allocate_boundary_faces()
	# Allocate element-to-node_IDs map
	mesh.allocate_elem_to_node_IDs_map()
	# Over-allocate interior_faces since we haven't distinguished
	# between interior and boundary faces yet
	num_faces_per_elem = mesh.gbasis.NFACES
	mesh.num_interior_faces = mesh.num_elems*num_faces_per_elem
	mesh.allocate_interior_faces()

	# reset num_interior_faces - use as a counter
	mesh.num_interior_faces = 0

	# Table to store face info and connect elements to faces
	node0_to_faces_info = [{} for n in range(mesh.num_nodes)] # list of dicts

	# Go to Gmsh elements section
	go_to_line_below_string(fo, "$Elements")

	# Boundary face counter
	num_bfaces_per_bgroup = [0 for i in range(mesh.num_boundary_groups)] 

	# Process elements and boundary faces
	if ver == VERSION2:
		process_elems_bfaces_ver2(fo, mesh, phys_groups, num_phys_groups, 
				gmsh_element_database, old_to_new_node_IDs, 
				num_bfaces_per_bgroup, node0_to_faces_info)
	else:
		process_elems_bfaces_ver4(fo, mesh, phys_groups, num_phys_groups, 
				gmsh_element_database, old_to_new_node_IDs, 
				num_bfaces_per_bgroup, node0_to_faces_info)

	# Verify footer
	fl = fo.readline()
	if not fl.startswith("$EndElements"):
		raise errors.FileReadError

	# Fill boundary and interior face info
	for elem_ID in range(mesh.num_elems):
		for face_ID in range(mesh.gbasis.NFACES):
			# Get local q = 1 face nodes
			gbasis = mesh.gbasis
			local_node_nums = gbasis.get_local_face_principal_node_nums(
					mesh.gorder, face_ID)
			num_face_nodes = local_node_nums.shape[0]

			# Convert to global node IDs
			global_node_nums = mesh.elem_to_node_IDs[elem_ID][
					local_node_nums]

			# Add to face info table
			face_info, already_added = add_face_info_to_table(
					node0_to_faces_info, num_face_nodes, global_node_nums, 
					False, -1, elem_ID, face_ID)

			if already_added:
				# Face was already added to table previously

				# Identify element as either boundary face or interior face
				if face_info.at_boundary:
					# Boundary face

					if face_info.num_adjacent_elems != 1:
						raise ValueError("More than one element adjacent " +
								"to boundary face")

					# Find physical group
					found = False
					for phys_group in phys_groups:
						if phys_group.boundary_group_num == \
								face_info.boundary_group_num:
							found = True
							break
					if not found: 
						raise errors.DoesNotExistError("Physical boundary " +
								"group not found")
					boundary_group = mesh.boundary_groups[phys_group.name]

					# Fill in info
					boundary_face = boundary_group.boundary_faces[
							face_info.face_ID]
					boundary_face.elem_ID = elem_ID
					boundary_face.face_ID = face_ID
				else:
					# Interior face

					if face_info.num_adjacent_elems != 2:
						raise ValueError("More than two elements adjacent " +
								"to interior face")
					
					# Fill in info
					int_face = mesh.interior_faces[mesh.num_interior_faces]
					int_face.elemL_ID = face_info.elem_ID
					int_face.faceL_ID = face_info.face_ID
					int_face.elemR_ID = elem_ID
					int_face.faceR_ID = face_ID

					# Increment number of interior faces
					mesh.num_interior_faces += 1

				delete_face_info_from_table(node0_to_faces_info, 
						num_face_nodes, global_node_nums)

	# Any faces not accounted for?
	num_faces_left = 0
	for n in range(mesh.num_nodes):
		faces_info = node0_to_faces_info[n]
		for node_IDs_sort in faces_info.keys():
			print(node_IDs_sort)
			num_faces_left += 1

	if num_faces_left != 0:
		raise ValueError("Above %d faces not identified" % (num_faces_left) +
			" as valid boundary or interior faces")

	# Make sure number of interior faces makes sense
	if mesh.num_interior_faces > mesh.num_elems*num_faces_per_elem:
		raise ValueError
	# Remove superfluous (empty) interior faces
	mesh.interior_faces = mesh.interior_faces[:mesh.num_interior_faces]

	# Create elements
	mesh.create_elements()


def import_gmsh_mesh(file_name):
	'''
	This function reads a Gmsh file to create a mesh.

	Inputs:
	-------
		file_name: name of Gmsh file

	Outputs:
	--------
		mesh: mesh object
	'''
	# Check file extension
	if file_name[-4:] != ".msh":
		raise errors.FileReadError("Wrong file type")

	# Open file
	fo = open(file_name, "r")

	# Mesh object
	mesh = mesh_defs.Mesh(num_elems=0)

	# Create Gmsh element database
	gmsh_element_database = create_gmsh_element_database()

	# Read sections one-by-one and process
	ver = check_mesh_format(fo)
	mesh, old_to_new_node_IDs = import_nodes(fo, ver, mesh)
	phys_groups, num_phys_groups = import_physical_groups(fo, mesh)
	phys_groups = import_mesh_entities(fo, ver, mesh, phys_groups)
	mesh = import_mesh_elems_boundary_faces(fo, ver, mesh, phys_groups, 
			num_phys_groups, gmsh_element_database)

	# Create rest of mesh
	fill_mesh(fo, ver, mesh, phys_groups, num_phys_groups, 
			gmsh_element_database, old_to_new_node_IDs)

	# Ensure valid mesh
	mesh_tools.check_face_orientations(mesh)

	# Print some stats
	print("%d elements in the mesh" % (mesh.num_elems))
	
	# Done with file
	fo.close()

	return mesh
