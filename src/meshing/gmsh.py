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
		self.nNode = -1
		self.gorder = -1
		self.gbasis = -1
		self.NodeOrder = None


def gmsh_node_order_seg(gorder):
	nNode = gorder+1
	nodes = np.arange(nNode)
	nodes[1:-1] = nodes[2:]
	nodes[-1] = 1

	return nodes

def populate_nodes_quadril(gorder, start, nodes):
	if gorder == 0:
		return start
	else:
		# principal vertices
		nodes[0,0] = start
		nodes[0,-1] = start+1
		nodes[-1,-1] = start+2
		nodes[-1,0] = start+3
		# bottom face
		start += 4
		nodes[0,1:-1] = np.arange(start, start+gorder-1)
		# right face
		start += gorder-1
		nodes[1:-1,-1] = np.arange(start, start+gorder-1)
		# top face
		start += gorder-1
		nodes[-1,-2:0:-1] = np.arange(start, start+gorder-1)
		# left face
		start += gorder-1
		nodes[-2:0:-1,0] = np.arange(start, start+gorder-1)
		# interior
		if gorder >= 2:
			# recursively fill the interior nodes
			start += gorder-1
			nodes[1:-1,1:-1] = populate_nodes_quadril(gorder-2, start, nodes[1:-1,1:-1])

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


def CreateGmshElementDataBase():
	# (NTYPES+1) objects due to 1-indexing
	# gmsh_element_database = [GmshElementData() for n in range(MSH_MAX_NUM+1)]
	gmsh_element_database = {}

	''' 
	Assume most element types are not supported
	Only fill in supported elements
	'''

# <<<<<<< Updated upstream
# 	# Linear line segments
# 	EntityInfo = EntitiesInfo[1]
# 	EntityInfo.nNode = 2
# 	EntityInfo.gorder = 1
# 	EntityInfo.gbasis = Basis.LagrangeEqSeg(EntityInfo.gorder)
# 	EntityInfo.shape = Basis.SegShape()
# 	EntityInfo.Supported = True
# 	EntityInfo.NodeOrder = np.array([0, 1])

# 	# Linear triangle
# 	EntityInfo = EntitiesInfo[2]
# 	EntityInfo.nNode = 3
# 	EntityInfo.gorder = 1
# 	EntityInfo.gbasis = Basis.LagrangeEqTri(EntityInfo.gorder)
# 	EntityInfo.shape = Basis.TriShape()
# 	EntityInfo.Supported = True
# 	EntityInfo.NodeOrder = np.array([0, 1, 2])

# 	# Linear quadrilateral
# 	EntityInfo = EntitiesInfo[3]
# 	EntityInfo.nNode = 4
# 	EntityInfo.gorder = 1
# 	EntityInfo.gbasis = Basis.LagrangeEqQuad(EntityInfo.gorder)
# 	EntityInfo.shape = Basis.QuadShape()
# 	EntityInfo.Supported = True
# 	EntityInfo.NodeOrder = np.array([0, 1, 3, 2])

# 	# Quadratic line segment
# 	EntityInfo = EntitiesInfo[8]
# 	EntityInfo.nNode = 3
# 	EntityInfo.gorder = 2
# 	EntityInfo.gbasis = Basis.LagrangeEqSeg(EntityInfo.gorder)
# 	EntityInfo.shape = Basis.SegShape()
# 	EntityInfo.Supported = True
# 	EntityInfo.NodeOrder = np.array([0, 2, 1])

# 	# Quadratic triangle
# 	EntityInfo = EntitiesInfo[9]
# 	EntityInfo.nNode = 6
# 	EntityInfo.gorder = 2
# 	EntityInfo.gbasis = Basis.LagrangeEqTri(EntityInfo.gorder)
# 	EntityInfo.shape = Basis.TriShape()
# 	EntityInfo.Supported = True
# 	EntityInfo.NodeOrder = np.array([0, 3, 1, 5, 4, 2])

# 	# Quadratic quadrilateral
# 	EntityInfo = EntitiesInfo[10]
# 	EntityInfo.nNode = 9
# 	EntityInfo.gorder = 2
# 	EntityInfo.gbasis = Basis.LagrangeEqQuad(EntityInfo.gorder)
# 	EntityInfo.shape = Basis.QuadShape()
# 	EntityInfo.Supported = True
# 	EntityInfo.NodeOrder = np.array([0, 4, 1, 7, 8, 5, 3, 6, 2])

# 	# Point
# 	EntityInfo = EntitiesInfo[15]
# 	EntityInfo.nNode = 1
# 	EntityInfo.gorder = 0
# 	EntityInfo.shape = Basis.PointShape()
# 	EntityInfo.Supported = True

# 	# Cubic triangle
# 	EntityInfo = EntitiesInfo[21]
# 	EntityInfo.nNode = 10
# 	EntityInfo.gorder = 3
# 	EntityInfo.gbasis = Basis.LagrangeEqTri(EntityInfo.gorder)
# 	EntityInfo.shape = Basis.TriShape()
# 	EntityInfo.Supported = True
# 	EntityInfo.NodeOrder = np.array([0, 3, 4, 1, 8, 9, 5, 7, 6, 2])

# 	# Quartic triangle
# 	EntityInfo = EntitiesInfo[23]
# 	EntityInfo.nNode = 15
# 	EntityInfo.gorder = 4
# 	EntityInfo.gbasis = Basis.LagrangeEqTri(EntityInfo.gorder)
# 	EntityInfo.shape = Basis.TriShape()
# 	EntityInfo.Supported = True
# 	EntityInfo.NodeOrder = np.array([0, 3, 4, 5, 1, 11, 12, 13, 6, 
# 									10, 14, 7, 9, 8, 2])

# 	# Cubic line segment
# 	EntityInfo = EntitiesInfo[26]
# 	EntityInfo.nNode = 4
# 	EntityInfo.gorder = 3
# 	EntityInfo.gbasis = Basis.LagrangeEqSeg(EntityInfo.gorder)
# 	EntityInfo.shape = Basis.SegShape()
# 	EntityInfo.Supported = True
# 	EntityInfo.NodeOrder = np.array([0, 2, 3, 1])

# 	# Quartic line segment
# 	EntityInfo = EntitiesInfo[27]
# 	EntityInfo.nNode = 5
# 	EntityInfo.gorder = 4
# 	EntityInfo.gbasis = Basis.LagrangeEqSeg(EntityInfo.gorder)
# 	EntityInfo.shape = Basis.SegShape()
# 	EntityInfo.Supported = True
# 	EntityInfo.NodeOrder = np.array([0, 2, 3, 4, 1])

# 	# Quintic line segment
# 	EntityInfo = EntitiesInfo[28]
# 	EntityInfo.nNode = 6
# 	EntityInfo.gorder = 5
# 	EntityInfo.gbasis = Basis.LagrangeEqSeg(EntityInfo.gorder)
# 	EntityInfo.shape = Basis.SegShape()
# 	EntityInfo.Supported = True
# 	EntityInfo.NodeOrder = np.array([0, 2, 3, 4, 5, 1])

# 	# Cubic quadrilateral
# 	EntityInfo = EntitiesInfo[36]
# 	EntityInfo.nNode = 16
# 	EntityInfo.gorder = 3
# 	EntityInfo.gbasis = Basis.LagrangeEqQuad(EntityInfo.gorder)
# 	EntityInfo.shape = Basis.QuadShape()
# 	EntityInfo.Supported = True
# 	EntityInfo.NodeOrder = np.array([0, 4, 5, 1, 11, 12, 13, 6, 10, 15, 14, 
# 									7, 3, 9, 8, 2])

# 	# Quartic quadrilateral
# 	EntityInfo = EntitiesInfo[37]
# 	EntityInfo.nNode = 25
# 	EntityInfo.gorder = 4
# 	EntityInfo.gbasis = Basis.LagrangeEqQuad(EntityInfo.gorder)
# 	EntityInfo.shape = Basis.QuadShape()
# 	EntityInfo.Supported = True
# 	EntityInfo.NodeOrder = np.array([0, 4, 5, 6, 1, 15, 16, 20, 17, 7,
# 								    14, 23, 24, 21, 8, 13, 19, 22, 18, 9,
# 								    3, 12, 11, 10, 2])

# 	return EntitiesInfo
# =======


	# Point
	etype_num = 15
	elem_data = GmshElementData()
	gmsh_element_database.update({etype_num : elem_data})
	elem_data.gorder = 0
	elem_data.gbasis = basis_defs.PointShape() # shape here instead of gbasis
	elem_data.nNode = 1
	elem_data.NodeOrder = np.array([0])

	# Line segments (q = 1 to q = 11)
	etype_nums = np.array([1, 8, 26, 27, 28, 62, 63, 64, 65, 66])
	for i in range(etype_nums.shape[0]):
		etype_num = etype_nums[i]
		elem_data = GmshElementData()
		gmsh_element_database.update({etype_num : elem_data})
		gorder = i + 1
		elem_data.gorder = gorder
		elem_data.gbasis = basis_defs.LagrangeSeg(gorder)
		elem_data.nNode = gorder + 1
		elem_data.NodeOrder = gmsh_node_order_seg(gorder)

	# Triangles (q = 1 to q = 10)
	etype_nums = np.array([2, 9, 21, 23, 25, 42, 43, 44, 45, 46])
	for i in range(etype_nums.shape[0]):
		etype_num = etype_nums[i]
		elem_data = GmshElementData()
		gmsh_element_database.update({etype_num : elem_data})
		gorder = i + 1
		elem_data.gorder = gorder
		elem_data.gbasis = basis_defs.LagrangeTri(gorder)
		elem_data.nNode = (gorder + 1)*(gorder + 2)//2
		elem_data.NodeOrder = gmsh_node_order_tri(gorder)

	# Quadrilaterals (q = 1 to q = 11)
	etype_nums = np.array([3, 10, 36, 37, 38, 47, 48, 49, 50, 51])
	for i in range(etype_nums.shape[0]):
		etype_num = etype_nums[i]
		elem_data = GmshElementData()
		gmsh_element_database.update({etype_num : elem_data})
		gorder = i + 1
		elem_data.gorder = gorder
		elem_data.gbasis = basis_defs.LagrangeQuad(gorder)
		elem_data.nNode = (gorder + 1)**2
		elem_data.NodeOrder = gmsh_node_order_quadril(gorder)

	return gmsh_element_database
# >>>>>>> Stashed changes


class PhysicalGroup(object):
	def __init__(self):
		self.Dim = 1
		self.Group = -1
		self.Number = -1
		self.Name = ""
		self.entity_tags = set()


class FaceInfo(object):
	def __init__(self):
		self.nVisit = 1
		self.BFlag = 0
		self.Group = 0
		self.Elem = 0
		self.Face = 0
		self.nfnode = 0
		self.snodes = None # should be a tuple (hashable)
	def Set(self, **kwargs):
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


def FindLineAfterString(fo, string):
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


def ReadMeshFormat(fo):
	# Find beginning of section
	FindLineAfterString(fo, "$MeshFormat")
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


def ReadPhysicalGroups(fo, mesh):
	# Find beginning of section
	FindLineAfterString(fo, "$PhysicalNames")
	# Number of physical names
	nPGroup = int(fo.readline())
	# Allocate
	PGroups = [PhysicalGroup() for i in range(nPGroup)]
	# Loop over entities
	for i in range(nPGroup):
		PGroup = PGroups[i]
		fl = fo.readline()
		ls = fl.split()
		PGroup.Dim = int(ls[0])
		PGroup.Number = int(ls[1])
		PGroup.Name = ls[2][1:-1]

		if PGroup.Dim < mesh.Dim-1 or PGroup.Dim > mesh.Dim:
			raise Exception("Physical groups should be created only for " +
					"elements and boundary faces")

	# Verify footer
	fl = fo.readline()
	if not fl.startswith("$EndPhysicalNames"):
		raise errors.FileReadError

	# Need at least one physical group to correspond to volume elements
	match = False
	for PGroup in PGroups:
		if PGroup.Dim == mesh.Dim:
			match = True
			break

	if not match:
		raise Exception("No elements assigned to a physical group")

	return PGroups, nPGroup


def get_nodes_ver2(fo):
	# Number of nodes
	nNode = int(fo.readline())
	old_to_new_node_tags = {}
	# Allocate nodes - assume 3D first
	Nodes = np.zeros([nNode,3])
	# Extract nodes
	new_node_tag = 0
	for n in range(nNode):
		fl = fo.readline()
		ls = fl.split()
		# Explicitly use for loop for compatibility with
		# both Python 2 and Python 3
		old_node_tag = int(ls[0])
		old_to_new_node_tags.update({old_node_tag : new_node_tag})
		for d in range(3):
			Nodes[new_node_tag,d] = float(ls[d+1])
		# Sanity check
		if int(ls[0]) > nNode:
			raise errors.FileReadError

		new_node_tag += 1

	return Nodes, old_to_new_node_tags


def get_nodes_ver4(fo):
	fl = fo.readline()
	ls = [int(l) for l in fl.split()]
	num_blocks = ls[0]
	nNode = ls[1]
	min_node_tag = ls[2]
	max_node_tag = ls[3]
	# Require continuous node tagging from 1 to nNode
	# Note: does not have to be ordered
	old_to_new_node_tags = {}
	# if min_node_tag != 1 or max_node_tag != nNode:
	# 	raise ValueError
	# Allocate nodes - assume 3D first
	Nodes = np.zeros([nNode,3])

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
			Nodes[inode] = [float(l) for l in fl.split()[:3]]

	return Nodes, old_to_new_node_tags


def ReadNodes(fo, ver, mesh):
	# Find beginning of section
	FindLineAfterString(fo, "$Nodes")


	if ver == VERSION2:
		Nodes, old_to_new_node_tags = get_nodes_ver2(fo)
	else:
		Nodes, old_to_new_node_tags = get_nodes_ver4(fo)

	# Verify footer
	fl = fo.readline()
	if not fl.startswith("$EndNodes"):
		raise errors.FileReadError
	
	# Change dimension if needed
	ds = [0,1,2]
	for d in ds:
		# Find max perturbation from zero
		diff = np.amax(np.abs(Nodes[:,d]))
		if diff <= general.eps:
			# remove from ds
			ds.remove(d)

	# New dimension
	dim = len(ds)
	Nodes = Nodes[:,ds]

	if dim == 3:
		raise ValueError("3D meshes not supported")

	# Store in mesh
	mesh.Coords = Nodes
	mesh.nNode = Nodes.shape[0]
	mesh.Dim = dim

	return mesh, old_to_new_node_tags


def ReadMeshEntities(fo, ver, mesh, PGroups):

	if ver == VERSION2:
		return PGroups

	# Find beginning of section
	FindLineAfterString(fo, "$Entities")

	fl = fo.readline()
	ls = [int(l) for l in fl.split()]
	num_points = ls[0]
	num_curves = ls[1]
	num_surfaces = ls[2]

	if mesh.Dim == 2:
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
				phys_tag = int(ls[8])
				for PGroup in PGroups:
					if PGroup.Number == phys_tag:
						break
				PGroup.entity_tags.add(entity_tag)
			elif num_phys_tags > 1:
				raise ValueError("Entity should not be assigned to >1 physical groups")

	else:
		# add dim = 1 later
		raise NotImplementedError

	return PGroups


def get_elem_bface_info_ver2(fo, mesh, PGroups, nPGroup, gmsh_element_database):
	# Number of entities (cells, faces, edges)
	nEntity = int(fo.readline())
	# Loop over entities
	for n in range(nEntity):
		fl = fo.readline()
		ls = fl.split()
		# Parse line
		enum = int(ls[0])
		etype = int(ls[1])
		PGnum = int(ls[3])

		# if PGnum == 0:
		# 	raise ValueError("All elements need to be assigned to a physical group")
		
		found = False
		for PGidx in range(nPGroup):
			PGroup = PGroups[PGidx]
			if PGroup.Number == PGnum:
				found = True
				break
		if not found:
			raise errors.DoesNotExistError("All elements and boundary faces must " +
					"be assigned to a physical group")

		if PGroup.Dim == mesh.Dim:
			# Assume only one element type - need to check for this later
			### Entity is an element
# <<<<<<< Updated upstream
# 			gorder = EntitiesInfo[etype].gorder
# 			gbasis = EntitiesInfo[etype].gbasis

# 			if mesh.nElem == 0:
# 				mesh.SetParams(gbasis=gbasis, gorder=gorder, nElem=0)
# =======
			gorder = gmsh_element_database[etype].gorder
			gbasis = gmsh_element_database[etype].gbasis
			# if gbasis == -1:
			# 	raise NotImplementedError("Element type not supported")
			if mesh.nElem == 0:
				mesh.SetParams(gbasis=gbasis, gorder=gorder, nElem=0)
			else:
				if gorder != mesh.gorder or gbasis != mesh.gbasis:
					raise ValueError(">1 element type not supported")
# >>>>>>> Stashed changes
			mesh.nElem += 1
			# # Check for existing element group
			# found = False
			# for egrp in range(mesh.nElemGroup):
			# 	EG = mesh.ElemGroups[egrp]
			# 	if QOrder == EG.QOrder and QBasis == EG.QBasis:
			# 		found = True
			# 		break
			# if found:
			# 	EG.nElem += 1
			# else:
			# 	# Need new element group
			# 	mesh.nElemGroup += 1
			# 	mesh.ElemGroups.append(Mesh.ElemGroup(QBasis=QBasis,QOrder=QOrder))
		elif PGroup.Dim == mesh.Dim - 1:
			### Boundary entity
			# Check for existing boundary face group
			# found = False
			# for ibfgrp in range(mesh.nBFaceGroup):
			# 	BFG = mesh.BFaceGroups[ibfgrp]
			# 	if BFG.Name == PGroup.Name:
			# 		found = True
			# 		break
			try:
				BFG = mesh.BFaceGroups[PGroup.Name]
			except KeyError:
			# if PGroup.Name in mesh.BFaceGroups:
				# Group has not been assigned yet
				# mesh.nBFaceGroup += 1
				# BFG = mesh_defs.BFaceGroup()
				# mesh.BFaceGroups.append(BFG)
				# BFG.Name = PGroup.Name
				BFG = mesh.add_bface_group(PGroup.Name)
				PGroup.Group = BFG.number
			BFG.nBFace += 1
		# else:
		# 	raise Exception("Mesh error")


def get_elem_bface_info_ver4(fo, mesh, PGroups, nPGroup, gmsh_element_database):
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
		for PGroup in PGroups:
			if entity_tag in PGroup.entity_tags and dim == PGroup.Dim:
				found = True
				break
		if not found:
			raise errors.DoesNotExistError("All elements and boundary faces must " +
					"be assigned to a physical group")

		if dim == mesh.Dim:
			# Element
			# Get element type data
			gorder = gmsh_element_database[etype].gorder
			gbasis = gmsh_element_database[etype].gbasis
			# if QBasis == -1:
			# 	raise NotImplementedError("Element type not supported")

			# Loop
			for _ in range(num_in_block):
				fo.readline()
				if mesh.nElem == 0:
					mesh.SetParams(gbasis=gbasis, gorder=gorder, nElem=0)
				else:
					if gorder != mesh.gorder or gbasis != mesh.gbasis:
						raise ValueError(">1 element type not supported")
				mesh.nElem += 1
		elif dim == mesh.Dim - 1:

			if PGroup.Group >= 0:
				# BFG = mesh.BFaceGroups[PGroup.Group]
				BFG = mesh.BFaceGroups[PGroup.Name]
			else:
				# Group has not been assigned yet
				# mesh.nBFaceGroup += 1
				# BFG = mesh_defs.BFaceGroup()
				# mesh.BFaceGroups.append(BFG)
				# BFG.Name = PGroup.Name
				BFG = mesh.add_bface_group(PGroup.Name)
				PGroup.Group = BFG.number
			# Loop and increment nBFace
			for _ in range(num_in_block):
				fo.readline()
				BFG.nBFace += 1
		else:
			for _ in range(num_in_block):
				fo.readline()


	return mesh



def ReadMeshElemsBFaces(fo, ver, mesh, PGroups, nPGroup, gmsh_element_database):
	# First pass to get sizes
	# Find beginning of section
	FindLineAfterString(fo, "$Elements")

	if ver == VERSION2:
		get_elem_bface_info_ver2(fo, mesh, PGroups, nPGroup, gmsh_element_database)
	else:
		get_elem_bface_info_ver4(fo, mesh, PGroups, nPGroup, gmsh_element_database)

	# Verify footer
	fl = fo.readline()
	if not fl.startswith("$EndElements"):
		raise errors.FileReadError

	return mesh


def AddFaceToHash(Node2FaceTable, nfnode, nodes, BFlag, Group, Elem, Face):

	if nfnode <= 0:
		raise ValueError("Need nfnode > 1")

	snodes = np.zeros(nfnode, dtype=int)
	snodes[:] = nodes[:nfnode]

	# Sort nodes and convert to tuple (to make it hashable)
	snodes = tuple(np.sort(snodes))

	# Check if face already exists in face hash
	Exists = False
	n0 = snodes[0]
	FaceInfoDict = Node2FaceTable[n0]
	if snodes in FaceInfoDict:
		Exists = True
		FInfo = FaceInfoDict[snodes]
		FInfo.nVisit += 1
	else:
		# If it doesn't exist, then add it
		FInfo = FaceInfo()
		FaceInfoDict.update({snodes : FInfo})
		FInfo.Set(BFlag=BFlag, Group=Group, Elem=Elem, Face=Face,
				nfnode=nfnode, snodes=snodes)


	# for FInfo in FaceInfoDict:
	# 	if np.array_equal(snodes, FInfo.snodes):
	# 		Exists = True
	# 		# increment number of visits
	# 		FInfo.nVisit += 1
	# 		break

	# if not Exists:
	# 	# If it doesn't exist, then add it
	# 	FInfo = FaceInfo()
	# 	FaceInfoDict.append(FInfo)
	# 	FInfo.Set(BFlag=BFlag, Group=Group, Elem=Elem, Face=Face,
	# 			nfnode=nfnode, snodes=snodes)

	return FInfo, Exists


def DeleteFaceFromHash(Node2FaceTable, nfnode, nodes):

	if nfnode <= 0:
		raise ValueError("Need nfnode > 1")

	snodes = np.zeros(nfnode, dtype=int)
	snodes[:] = nodes[:nfnode]

	# Sort nodes
	snodes = tuple(np.sort(snodes))

	# Check if face already exists in face hash
	n0 = snodes[0]
	FaceInfoDict = Node2FaceTable[n0]
	# if FaceInfos == []:
	# 	raise LookupError

	if snodes in FaceInfoDict:
		del FaceInfoDict[snodes]


	# DelIdx = [] # for storing which indices to delete
	# for i in range(len(FaceInfos)):
	# 	FInfo = FaceInfos[i]
	# 	found = False
	# 	if np.array_equal(snodes, FInfo.snodes):
	# 		found = True

	# 	# If found, store for deletion later
	# 	if found:
	# 		DelIdx.append(i)

	# # Delete
	# for i in DelIdx:
	# 	del FaceInfos[i]


def fill_elems_bfaces_ver2(fo, mesh, PGroups, nPGroup, gmsh_element_database, 
		old_to_new_node_tags, bf, Node2FaceTable):
	# Number of entities
	nEntity = int(fo.readline())
	elem = 0 # elem counter
	# Loop through entities
	for e in range(nEntity):
		fl = fo.readline()
		ls = fl.split()
		# Parse line
		enum = int(ls[0])
		etype = int(ls[1])
		PGnum = int(ls[3])
		
		found = False
		for PGidx in range(nPGroup):
			PGroup = PGroups[PGidx]
			if PGroup.Number == PGnum:
				found = True
				break
		if not found:
			raise Exception("Physical group not found!")

		# Check if entity type is supported
		# if not gmsh_element_database[etype].Supported:
		# 	raise Exception("Entity type not supported")

		# Get nodes	
		nTag = int(ls[2]) # number of tags
			# see http://www.manpagez.com/info/gmsh/gmsh-2.2.6/gmsh_63.php
		offsetTag = 3 # 3 integers (including nTag) before tags start
		iStart = nTag + offsetTag # starting index of node numbering
		nn = gmsh_element_database[etype].nNode
		elist = ls[iStart:] # list of nodes (string format)
		if len(elist) != nn: 
			raise Exception("Wrong number of nodes")
		nodes = np.zeros(nn, dtype=int)
		for i in range(nn):
			# # Convert to int one-by-one for compatibility with Python 2 and 3
			# nodes[i] = int(elist[i]) - 1 # switch to zero index
			# Convert from old to new tags
			nodes[i] = old_to_new_node_tags[int(elist[i])]

		if PGroup.Group >= 0:
			### Boundary
			# Get basic info
# <<<<<<< Updated upstream
# 			gbasis = EntitiesInfo[etype].gbasis
# 			gorder = EntitiesInfo[etype].gorder
# =======
			gbasis = gmsh_element_database[etype].gbasis
			gorder = gmsh_element_database[etype].gorder
# >>>>>>> Stashed changes
			ibfgrp = PGroup.Group
			# BFG = mesh.BFaceGroups[ibfgrp]
			BFG = mesh.BFaceGroups[PGroup.Name]
			# Number of q = 1 face nodes
			nfnode = gbasis.get_num_basis_coeff(1)

			# Add q = 1 nodes to hash table
			FInfo, Exists = AddFaceToHash(Node2FaceTable, nfnode, nodes, True, 
				ibfgrp, -1, bf[ibfgrp])
			bf[ibfgrp] += 1
		elif PGroup.Group == -1:
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
			# for EG in mesh.ElemGroups:
			# 	if QOrder == EG.QOrder and QBasis == EG.QBasis:
			# 		found = True
			# 		break
			# # Sanity check
			# if not found:
			# 	raise Exception("Can't find element group")
			# Number of element nodes
			nnode = gbasis.get_num_basis_coeff(gorder)
			# Sanity check
			if nnode != gmsh_element_database[etype].nNode:
				raise Exception("Check Gmsh entities")
			# Convert node Ordering
			newnodes = nodes[gmsh_element_database[etype].NodeOrder]
			# Store in Elem2Nodes
			mesh.Elem2Nodes[elem] = newnodes
			# Increment elem counter
			elem += 1
		else:
			raise ValueError
		 


def fill_elems_bfaces_ver4(fo, mesh, PGroups, nPGroup, gmsh_element_database, 
		old_to_new_node_tags, bf, Node2FaceTable):	
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

		if dim == mesh.Dim:
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
				newnodes = nodes[gmsh_element_database[etype].NodeOrder]
				# Store in Elem2Nodes
				mesh.Elem2Nodes[elem] = newnodes
				# Increment elem counter
				elem += 1
		elif dim == mesh.Dim - 1:
			# find physical boundary group
			for PGroup in PGroups:
				if entity_tag in PGroup.entity_tags:
					if PGroup.Dim == dim:
						ibfgrp = PGroup.Group
						break
			BFG = mesh.BFaceGroups[PGroup.Name]
			gbasis = gmsh_element_database[etype].gbasis
			nfnode = gbasis.get_num_basis_coeff(1) 
			# Loop and increment nBFace
			for _ in range(num_in_block):
				fl = fo.readline()
				lint = [int(l) for l in fl.split()]
				nodes = np.array(lint[1:])
				for n in range(len(nodes)):
					nodes[n] = old_to_new_node_tags[nodes[n]]
				# Add q = 1 nodes to hash table
				FInfo, Exists = AddFaceToHash(Node2FaceTable, nfnode, nodes, True, 
					ibfgrp, -1, bf[ibfgrp])
				bf[ibfgrp] += 1
		else:
			for _ in range(num_in_block):
				fo.readline()


	return mesh

def FillMesh(fo, ver, mesh, PGroups, nPGroup, gmsh_element_database, old_to_new_node_tags):
	# Allocate additional mesh structures
	# for ibfgrp in range(mesh.nBFaceGroup):
	# 	BFG = mesh.BFaceGroups[ibfgrp]
	# 	BFG.allocate_bfaces()
	for BFG in mesh.BFaceGroups.values():
		BFG.allocate_bfaces()
	# nFaceMax = 0
	# for EG in mesh.ElemGroups:
	# 	# also find maximum # faces per elem
	# 	EG.allocate_faces()
	# 	EG.allocate_elem_to_nodes()
	# 	if nFaceMax < EG.nFacePerElem: nFaceMax = EG.nFacePerElem
	# mesh.allocate_faces()
	mesh.allocate_elem_to_nodes()
	nFaceMax = mesh.gbasis.NFACES

	# Over-allocate IFaces
	mesh.nIFace = mesh.nElem*nFaceMax
	mesh.allocate_ifaces()

	# reset nIFace - use as a counter
	mesh.nIFace = 0

	# Dictionary for hashing
	# Node2FaceTable = {n:FaceInfo() for n in range(mesh.nNode)}
	# Node2FaceTable = {n:[] for n in range(mesh.nNode)}
	Node2FaceTable = [{} for n in range(mesh.nNode)] # list of dicts

	# Go to entities section
	FindLineAfterString(fo, "$Elements")

	bf = [0 for i in range(mesh.nBFaceGroup)] # BFace counter

	if ver == VERSION2:
		fill_elems_bfaces_ver2(fo, mesh, PGroups, nPGroup, gmsh_element_database, 
				old_to_new_node_tags, bf, Node2FaceTable)
	else:
		fill_elems_bfaces_ver4(fo, mesh, PGroups, nPGroup, gmsh_element_database, 
				old_to_new_node_tags, bf, Node2FaceTable)


	# Verify footer
	fl = fo.readline()
	if not fl.startswith("$EndElements"):
		raise errors.FileReadError

	# Fill boundary and interior face info
	# for egrp in range(mesh.nElemGroup):
	# 	EG = mesh.ElemGroups[egrp]
	for elem in range(mesh.nElem):
		for face in range(mesh.gbasis.NFACES):
			# Local q = 1 nodes on face
			gbasis = mesh.gbasis
			fnodes, nfnode = gbasis.local_q1_face_nodes(mesh.gorder, face)

			# Convert to global nodes
			fnodes = mesh.Elem2Nodes[elem][fnodes]

			# Add to hash table
			FInfo, Exists = AddFaceToHash(Node2FaceTable, nfnode, fnodes, False, 
				-1, elem, face)

			if Exists:
				# Face already exists in hash table
				if FInfo.nVisit != 2:
					raise ValueError("More than two elements share a face " + 
						"or a boundary face is referenced by more than one element")

				# Link elem to BFace or IFace
				if FInfo.BFlag:
					# boundary face
					# Store in BFG
					# BFG = mesh.BFaceGroups[FInfo.Group]
					found = False
					# Make this cleaner later
					for PGroup in PGroups:
						if PGroup.Group == FInfo.Group:
							found = True
							break
					if not found: raise Exception
					BFG = mesh.BFaceGroups[PGroup.Name]
					# try:
					# 	BFace = BFG.BFaces[FInfo.Face]
					# except:
					# 	code.interact(local=locals())
					BFace = BFG.BFaces[FInfo.Face]
					BFace.Elem = elem; BFace.face = face
					# Store in Face
					# Face = mesh.Faces[elem][face]
					# Face.Group = FInfo.Group
					# Face.Number = FInfo.Face
				else:
					# interior face
					# Store in IFace
					IFace = mesh.IFaces[mesh.nIFace]
					IFace.ElemL = FInfo.Elem
					IFace.faceL = FInfo.Face
					IFace.ElemR = elem
					IFace.faceR = face
					# Store in left Face
					# Face = mesh.Faces[FInfo.Elem][FInfo.Face]
					# Face.Group = general.INTERIORFACE
					# Face.Number = mesh.nIFace
					# # Store in right face
					# Face = mesh.Faces[elem][face]
					# Face.Group = general.INTERIORFACE
					# Face.Number = mesh.nIFace
					# Increment IFace counter
					mesh.nIFace += 1

				DeleteFaceFromHash(Node2FaceTable, nfnode, fnodes)

	# Make sure no faces left in hash
	nleft = 0
	for n in range(mesh.nNode):
		FaceInfoDict = Node2FaceTable[n]
		for snodes in FaceInfoDict.keys():
			print(snodes)
			# for node in snodes:
			# 	print(int(node+1))
			nleft += 1

	if nleft != 0:
		raise ValueError("Mesh connectivity error: the above %d " % (nleft) +
			"face(s) remain(s) in the hash")

	# Resize IFace
	if mesh.nIFace > mesh.nElem*nFaceMax:
		raise ValueError
	mesh.IFaces = mesh.IFaces[:mesh.nIFace]

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
	mesh = mesh_defs.Mesh(nElem=0)

	# Object that stores Gmsh entity info
	gmsh_element_database = CreateGmshElementDataBase()

	# Read sections one-by-one
	ver = ReadMeshFormat(fo)
	mesh, old_to_new_node_tags = ReadNodes(fo, ver, mesh)
	PGroups, nPGroup = ReadPhysicalGroups(fo, mesh)
	PGroups = ReadMeshEntities(fo, ver, mesh, PGroups)
	mesh = ReadMeshElemsBFaces(fo, ver, mesh, PGroups, nPGroup, gmsh_element_database)
	# code.interact(local=locals())

	# Create rest of mesh
	FillMesh(fo, ver, mesh, PGroups, nPGroup, gmsh_element_database, old_to_new_node_tags)

	# Print some stats
	print("%d elements in the mesh" % (mesh.nElem))
	
	# Done with file
	fo.close()

	return mesh
