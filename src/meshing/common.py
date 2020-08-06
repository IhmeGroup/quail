import code
import copy
import numpy as np

import general

import meshing.meshbase as mesh_defs
import numerics.basis.basis as basis_defs

def mesh_1D(node_coords=None, nElem=10, Uniform=True, xmin=-1., xmax=1., Periodic=True):
	'''
	Function: mesh_1D
	-------------------
	This function creates a 1D mesh.

	INPUTS:
	    node_coords: x-coordinates
	    Uniform: True for a uniform mesh (will be set to False if node_coords is not None)
	    nElem: number of elements (only relevant for Uniform=True)
	    xmin: minimum coordinate (only relevant for Uniform=True)
	    xmax: maximum coordinate (only relevant for Uniform=True)
	    Periodic: True for a periodic mesh

	OUTPUTS:
	    mesh: Mesh object that stores relevant mesh info
	'''
	if node_coords is None and not Uniform:
		raise Exception("Input error")

	### Create mesh
	if node_coords is None:
		num_nodes = nElem + 1
		mesh = mesh_defs.Mesh(dim=1, num_nodes=num_nodes)
		mesh.node_coords = np.zeros([mesh.num_nodes,mesh.dim])
		mesh.node_coords[:,0] = np.linspace(xmin,xmax,mesh.num_nodes)
	else:
		Uniform = False
		node_coords.shape = -1,1
		num_nodes = node_coords.shape[0]
		nElem = num_nodes - 1
		mesh = mesh_defs.Mesh(dim=1, num_nodes=num_nodes)
		mesh.node_coords = node_coords

	# IFaces
	if Periodic:
		mesh.nIFace = mesh.num_nodes - 1
		mesh.allocate_ifaces()
		for i in range(mesh.nIFace):
			IFace_ = mesh.IFaces[i]
			IFace_.elemL_id = i-1
			IFace_.faceL_id = 1
			IFace_.elemR_id = i
			IFace_.faceR_id = 0
		# Leftmost face
		mesh.IFaces[0].elemL_id = nElem - 1
	# Rightmost face
	# mesh.IFaces[-1].elemR_id = 0
	else:
		mesh.nIFace = nElem - 1
		mesh.allocate_ifaces()
		for i in range(mesh.nIFace):
			IFace_ = mesh.IFaces[i]
			IFace_.elemL_id = i
			IFace_.faceL_id = 1
			IFace_.elemR_id = i+1
			IFace_.faceR_id = 0
		# Boundary groups
		# mesh.nBFaceGroup = 2
		# mesh.allocate_bface_groups()
		# for i in range(mesh.nBFaceGroup):
		# 	BFG = mesh.BFaceGroups[i]
		# 	BFG.nBFace = 1
		# 	BFG.allocate_bfaces()
		# 	BF = BFG.BFaces[0]
		# 	if i == 0:
		# 		BFG.Name = "Left"
		# 		BF.elem_id = 0
		# 		BF.face_id = 0
		# 	else:
		# 		BFG.Name = "Right"
		# 		BF.elem_id = nElem - 1
		# 		BF.face_id = 1
		# left
		BFG = mesh.add_bface_group("Left")
		BFG.nBFace = 1
		BFG.allocate_bfaces()
		BF = BFG.BFaces[0]
		BF.elem_id = 0
		BF.face_id = 0
		# right
		BFG = mesh.add_bface_group("Right")
		BFG.nBFace = 1
		BFG.allocate_bfaces()
		BF = BFG.BFaces[0]
		BF.elem_id = nElem - 1
		BF.face_id = 1

	mesh.SetParams(gbasis=basis_defs.LagrangeSeg(1), gorder=1, nElem=nElem)
	# mesh.allocate_faces()
	# interior elements
	# for elem in range(mesh.nElem):
	# 	for i in range(mesh.nFacePerElem):
	# 		Face_ = mesh.Faces[elem][i]
	# 		Face_.Type = general.INTERIORFACE
	# 		Face_.Number = elem + i
	# 		if not Periodic:
	# 			if elem == 0 and i == 0:
	# 				Face_.Type = general.NULLFACE
	# 				Face_.Number = 0
	# 			elif elem == mesh.nElem-1 and i == 1:
	# 				Face_.Type = general.NULLFACE
	# 				Face_.Number = 1
	# 			else:
	# 				Face_.Number = elem + i - 1


	mesh.allocate_elem_to_nodes()
	for elem in range(mesh.nElem):
		for i in range(mesh.num_nodes_per_elem):
			mesh.Elem2Nodes[elem][i] = elem + i

	# mesh.fill_faces()
	mesh.create_elements()

	return mesh


# def refine_uniform_1D(node_coords_old):
# 	'''
# 	Function: refine_uniform_1D
# 	-------------------
# 	This function uniformly refines a set of coordinates

# 	INPUTS:
# 	    node_coords_old: coordinates to refine

# 	OUTPUTS:
# 	    node_coords: refined coordinates
# 	'''
# 	num_nodes_old = len(node_coords_old)
# 	nElem_old = num_nodes_old-1

# 	nElem = nElem_old*2
# 	num_nodes = nElem+1

# 	node_coords = np.zeros([num_nodes,1])

# 	for n in range(num_nodes_old-1):
# 		node_coords[2*n] = node_coords_old[n]
# 		node_coords[2*n+1] = np.mean(node_coords_old[n:n+2])
# 	node_coords[-1] = node_coords_old[-1]

# 	return node_coords


def mesh_2D(xcoords=None, ycoords=None, nElem_x=10, nElem_y = 10, Uniform=True, xmin=-1., xmax=1., 
	ymin=-1., ymax=1., Periodic=True):
	'''
	Function: mesh_2D
	-------------------
	This function creates a 2D mesh.

	INPUTS:
	    node_coords: x-coordinates
	    Uniform: True for a uniform mesh (will be set to False if node_coords is not None)
	    nElem: number of elements (only relevant for Uniform=True)
	    xmin: minimum coordinate (only relevant for Uniform=True)
	    xmax: maximum coordinate (only relevant for Uniform=True)
	    Periodic: True for a periodic mesh

	OUTPUTS:
	    mesh: Mesh object that stores relevant mesh info
	'''

	### Create mesh
	if xcoords is None and ycoords is None:
		# Uniform
		num_nodes_x = nElem_x + 1
		num_nodes_y = nElem_y + 1
		xcoords = np.linspace(xmin, xmax, num_nodes_x)
		ycoords = np.linspace(ymin, ymax, num_nodes_y)
	elif xcoords is not None and ycoords is not None:
		Uniform = False
		num_nodes_x = len(xcoords)
		num_nodes_y = len(ycoords)
		nElem_x = num_nodes_x - 1
		nElem_y = num_nodes_y - 1
	else:
		raise Exception("Input error")

	X, Y = np.meshgrid(xcoords, ycoords)
	xp = np.array([np.reshape(X,-1),np.reshape(Y,-1)]).transpose()

	mesh = mesh_defs.Mesh(dim=2, num_nodes=xp.shape[0], nElem=nElem_x*nElem_y, gbasis=basis_defs.LagrangeQuad(1),
		gorder=1)

	mesh.node_coords = xp

	### Elems
	mesh.allocate_elem_to_nodes()
	elem = 0
	for ny in range(nElem_y):
		for nx in range(nElem_x):
			mesh.Elem2Nodes[elem][0] = num_nodes_x*ny + nx
			mesh.Elem2Nodes[elem][1] = num_nodes_x*ny + nx + 1
			mesh.Elem2Nodes[elem][2] = num_nodes_x*(ny+1) + nx
			mesh.Elem2Nodes[elem][3] = num_nodes_x*(ny+1) + nx + 1
			elem += 1

	# mesh.allocate_faces()

	### BFGs
	# mesh.nBFaceGroup = 4
	# mesh.allocate_bface_groups()
	# for i in range(mesh.nBFaceGroup):
	# 	BFG = mesh.BFaceGroups[i]
	# 	if i == 0:
	# 		BFG.Name = "x1"
	# 		BFG.nBFace = nElem_y
	# 	if i == 1:
	# 		BFG.Name = "x2"
	# 		BFG.nBFace = nElem_y
	# 	if i == 2:
	# 		BFG.Name = "y1"
	# 		BFG.nBFace = nElem_x
	# 	if i == 3:
	# 		BFG.Name = "y2"
	# 		BFG.nBFace = nElem_x
	# 	BFG.allocate_bfaces()

	# x1
	# BFG = mesh.BFaceGroups[0]
	BFG = mesh.add_bface_group("x1")
	BFG.nBFace = nElem_y
	BFG.allocate_bfaces()
	n = 0
	for BF in BFG.BFaces:
		BF.elem_id = nElem_x*n
		BF.face_id = 3
		n += 1
	# x2
	# BFG = mesh.BFaceGroups[1]
	BFG = mesh.add_bface_group("x2")
	BFG.nBFace = nElem_y
	BFG.allocate_bfaces()
	n = 0
	for BF in BFG.BFaces:
		BF.elem_id = nElem_x*(n + 1) - 1
		BF.face_id = 1
		n += 1
	# y1
	# BFG = mesh.BFaceGroups[2]
	BFG = mesh.add_bface_group("y1")
	BFG.nBFace = nElem_x
	BFG.allocate_bfaces()
	n = 0
	for BF in BFG.BFaces:
		BF.elem_id = n
		BF.face_id = 0
		n += 1
	# y2
	# BFG = mesh.BFaceGroups[3]
	BFG = mesh.add_bface_group("y2")
	BFG.nBFace = nElem_x
	BFG.allocate_bfaces()
	n = 0
	for BF in BFG.BFaces:
		BF.elem_id = mesh.nElem - nElem_x + n
		BF.face_id = 2
		n += 1


	### IFaces
	mesh.nIFace = nElem_y*(nElem_x-1) + nElem_x*(nElem_y-1)
	mesh.allocate_ifaces()
	
	# x direction
	n = 0
	for ny in range(nElem_y):
		for nx in range(nElem_x-1):
			IF = mesh.IFaces[n]
			IF.elemL_id = nElem_x*ny + nx
			IF.faceL_id = 1
			IF.elemR_id = nElem_x*ny + nx + 1
			IF.faceR_id = 3
			n += 1

	# y direction
	for nx in range(nElem_x):
		for ny in range(nElem_y-1):
			IF = mesh.IFaces[n]
			IF.elemL_id = nElem_x*ny + nx
			IF.faceL_id = 2
			IF.elemR_id = nElem_x*(ny + 1) + nx
			IF.faceR_id = 0
			n += 1

	# mesh.fill_faces()
	mesh.create_elements()

	return mesh


def split_quadrils_into_tris(mesh_old):
	nElem_old = mesh_old.nElem 
	nElem = nElem_old*2

	mesh = copy.deepcopy(mesh_old)

	mesh.SetParams(nElem=nElem, gbasis=basis_defs.LagrangeTri(1))

	def reorder_nodes(QOrder, num_nodes_per_quad, num_nodes_per_tri):
		num_nodes_per_face = QOrder + 1
		if num_nodes_per_face != np.sqrt(num_nodes_per_quad):
			return ValueError
		quadril_nodes = np.arange(num_nodes_per_quad)
		quadril_nodes.shape = num_nodes_per_face, num_nodes_per_face
		# faces 0 and 3 become faces 0 and 2 of tri1
		tri1_nodes = np.arange(num_nodes_per_tri)
		# faces 1 and 2 become faces 2 and 0 of tri2
		tri2_nodes = np.copy(tri1_nodes)

		n = 0
		for j in range(num_nodes_per_face):
			tri1_nodes[n:n+num_nodes_per_face-j] = quadril_nodes[j,:num_nodes_per_face-j]
			if j == 0:
				tri2_nodes[n:n+num_nodes_per_face-j] = quadril_nodes[num_nodes_per_face-1,::-1]
			else:
				tri2_nodes[n:n+num_nodes_per_face-j] = quadril_nodes[num_nodes_per_face-(j+1),
						num_nodes_per_face-1:j-1:-1]
			n += num_nodes_per_face-j

		return tri1_nodes, tri2_nodes

	tri1_nodes, tri2_nodes = reorder_nodes(mesh.gorder, mesh_old.num_nodes_per_elem, mesh.num_nodes_per_elem)


	# Elems
	mesh.allocate_elem_to_nodes()
	for elem_id in range(nElem_old):
		# First triangle
		mesh.Elem2Nodes[elem_id] = mesh_old.Elem2Nodes[elem_id, tri1_nodes]
		# Second triangle
		mesh.Elem2Nodes[elem_id+nElem_old] = mesh_old.Elem2Nodes[elem_id, tri2_nodes]


	old_to_new_face = np.array([2, 1, 2, 1])

	def modify_face_info(elem, face):
		if face == 1 or face == 2:
			elem += nElem_old
		face = old_to_new_face[face]
		return elem, face

	# BFGs
	for BFG in mesh.BFaceGroups.values():
		for BF in BFG.BFaces:
			BF.elem_id, BF.face_id = modify_face_info(BF.elem_id, BF.face_id)


	# Modify IFaces
	for IF in mesh.IFaces:
		IF.elemL_id, IF.faceL_id = modify_face_info(IF.elemL_id, IF.faceL_id)
		IF.elemR_id, IF.faceR_id = modify_face_info(IF.elemR_id, IF.faceR_id)

	# New IFaces
	# code.interact(local=locals())
	# nIFace_old = mesh.nIFace
	# IFaces_old = mesh.IFaces
	mesh.nIFace += nElem_old
	# mesh.allocate_ifaces()
	# mesh.IFaces[:nIFace_old] = IFaces_old
	# for IF in mesh.IFaces[nIFace_old:]:
	for elem_id in range(nElem_old):
		IF = mesh_defs.IFace()
		IF.elemL_id = elem_id
		IF.faceL_id = 0
		IF.elemR_id = elem_id + nElem_old
		IF.faceR_id = 0
		mesh.IFaces.append(IF)

	# mesh.allocate_faces()
	# mesh.fill_faces()
	mesh.create_elements()

	return mesh












