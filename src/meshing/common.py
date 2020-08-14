import code
import copy
import numpy as np

import general

import meshing.meshbase as mesh_defs
import numerics.basis.basis as basis_defs


def mesh_1D(num_elems=10, xmin=-1., xmax=1.):
	'''
	This function creates a 1D uniform mesh.

	Inputs:
	-------
	    num_elems: number of mesh elements
	    xmin: minimum x-coordinate
	    xmax: maximum x-coordinate

	Outputs:
	--------
	    mesh: mesh object
	'''
	''' Create mesh and set node coordinates '''
	num_nodes = num_elems + 1
	mesh = mesh_defs.Mesh(dim=1, num_nodes=num_nodes)
	mesh.node_coords = np.zeros([mesh.num_nodes, mesh.dim])
	mesh.node_coords[:,0] = np.linspace(xmin, xmax, mesh.num_nodes)
	# Set parameters
	mesh.set_params(gbasis=basis_defs.LagrangeSeg(1), gorder=1, num_elems=num_elems)

	''' Interior faces '''
	mesh.num_interior_faces = num_elems - 1
	mesh.allocate_interior_faces()
	for i in range(mesh.num_interior_faces):
		interior_face = mesh.interior_faces[i]
		interior_face.elemL_id = i
		interior_face.faceL_id = 1
		interior_face.elemR_id = i+1
		interior_face.faceR_id = 0

	''' Boundary groups and faces '''
	# Left
	boundary_group = mesh.add_boundary_group("Left")
	boundary_group.num_boundary_faces = 1
	boundary_group.allocate_boundary_faces()
	boundary_face = boundary_group.boundary_faces[0]
	boundary_face.elem_id = 0
	boundary_face.face_id = 0
	# Right
	boundary_group = mesh.add_boundary_group("Right")
	boundary_group.num_boundary_faces = 1
	boundary_group.allocate_boundary_faces()
	boundary_face = boundary_group.boundary_faces[0]
	boundary_face.elem_id = num_elems - 1
	boundary_face.face_id = 1

	''' Create element-to-node-ID map '''
	mesh.allocate_elem_to_node_ids_map()
	for elem_id in range(mesh.num_elems):
		for n in range(mesh.num_nodes_per_elem):
			mesh.elem_to_node_ids[elem_id][n] = elem_id + n

	''' Create element objects '''
	mesh.create_elements()

	return mesh


def mesh_2D(num_elems_x=10, num_elems_y =10, xmin=-1., xmax=1., 
		ymin=-1., ymax=1.):
	'''
	This function creates a uniform 2D quadrilateral mesh.

	Inputs:
	-------
	    num_elems_x: number of elements in x-direction
	    num_elems_y: number of elements in y-direction
	    xmin: minimum x-coordinate
	    xmax: maximum x-coordinate
	    ymin: minimum y-coordinate
	    ymax: maximum y-coordinate

	Outputs:
	--------
	    mesh: mesh object
	'''
	''' Create mesh and set node coordinates '''
	# Number of nodes
	num_nodes_x = num_elems_x + 1
	num_nodes_y = num_elems_y + 1
	# xy-coordinates
	xcoords = np.linspace(xmin, xmax, num_nodes_x)
	ycoords = np.linspace(ymin, ymax, num_nodes_y)
	xgrid, ygrid = np.meshgrid(xcoords, ycoords)
	xp = np.array([np.reshape(xgrid,-1),np.reshape(ygrid,-1)]).transpose()
	# Create mesh
	mesh = mesh_defs.Mesh(dim=2, num_nodes=xp.shape[0], 
			num_elems=num_elems_x*num_elems_y, 
			gbasis=basis_defs.LagrangeQuad(1),
			gorder=1)
	# Store coordinates
	mesh.node_coords = xp

	''' Interior faces '''
	# Number of interior faces
	mesh.num_interior_faces = num_elems_y*(num_elems_x-1) + \
			num_elems_x*(num_elems_y-1)
	mesh.allocate_interior_faces()
	# x-direction
	n = 0
	for ny in range(num_elems_y):
		for nx in range(num_elems_x-1):
			interior_face = mesh.interior_faces[n]
			interior_face.elemL_id = num_elems_x*ny + nx
			interior_face.faceL_id = 1
			interior_face.elemR_id = num_elems_x*ny + nx + 1
			interior_face.faceR_id = 3
			n += 1
	# y-direction
	for nx in range(num_elems_x):
		for ny in range(num_elems_y-1):
			interior_face = mesh.interior_faces[n]
			interior_face.elemL_id = num_elems_x*ny + nx
			interior_face.faceL_id = 2
			interior_face.elemR_id = num_elems_x*(ny + 1) + nx
			interior_face.faceR_id = 0
			n += 1

	''' Boundary groups and faces '''
	# x1
	boundary_group = mesh.add_boundary_group("x1")
	boundary_group.num_boundary_faces = num_elems_y
	boundary_group.allocate_boundary_faces()
	n = 0
	for boundary_face in boundary_group.boundary_faces:
		boundary_face.elem_id = num_elems_x*n
		boundary_face.face_id = 3
		n += 1
	# x2
	boundary_group = mesh.add_boundary_group("x2")
	boundary_group.num_boundary_faces = num_elems_y
	boundary_group.allocate_boundary_faces()
	n = 0
	for boundary_face in boundary_group.boundary_faces:
		boundary_face.elem_id = num_elems_x*(n + 1) - 1
		boundary_face.face_id = 1
		n += 1
	# y1
	boundary_group = mesh.add_boundary_group("y1")
	boundary_group.num_boundary_faces = num_elems_x
	boundary_group.allocate_boundary_faces()
	n = 0
	for boundary_face in boundary_group.boundary_faces:
		boundary_face.elem_id = n
		boundary_face.face_id = 0
		n += 1
	# y2
	boundary_group = mesh.add_boundary_group("y2")
	boundary_group.num_boundary_faces = num_elems_x
	boundary_group.allocate_boundary_faces()
	n = 0
	for boundary_face in boundary_group.boundary_faces:
		boundary_face.elem_id = mesh.num_elems - num_elems_x + n
		boundary_face.face_id = 2
		n += 1


	### Elems
	mesh.allocate_elem_to_node_ids_map()
	elem = 0
	for ny in range(num_elems_y):
		for nx in range(num_elems_x):
			mesh.elem_to_node_ids[elem][0] = num_nodes_x*ny + nx
			mesh.elem_to_node_ids[elem][1] = num_nodes_x*ny + nx + 1
			mesh.elem_to_node_ids[elem][2] = num_nodes_x*(ny+1) + nx
			mesh.elem_to_node_ids[elem][3] = num_nodes_x*(ny+1) + nx + 1
			elem += 1

	''' Create element objects '''
	mesh.create_elements()

	return mesh


def split_quadrils_into_tris(mesh_old):
	num_elems_old = mesh_old.num_elems 
	num_elems = num_elems_old*2

	mesh = copy.deepcopy(mesh_old)

	mesh.set_params(num_elems=num_elems, gbasis=basis_defs.LagrangeTri(1))

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
	mesh.allocate_elem_to_node_ids_map()
	for elem_id in range(num_elems_old):
		# First triangle
		mesh.elem_to_node_ids[elem_id] = mesh_old.elem_to_node_ids[elem_id, tri1_nodes]
		# Second triangle
		mesh.elem_to_node_ids[elem_id+num_elems_old] = mesh_old.elem_to_node_ids[elem_id, tri2_nodes]


	old_to_new_face = np.array([2, 1, 2, 1])

	def modify_face_info(elem, face):
		if face == 1 or face == 2:
			elem += num_elems_old
		face = old_to_new_face[face]
		return elem, face

	# BFGs
	for BFG in mesh.boundary_groups.values():
		for BF in BFG.boundary_faces:
			BF.elem_id, BF.face_id = modify_face_info(BF.elem_id, BF.face_id)


	# Modify interior_faces
	for IF in mesh.interior_faces:
		IF.elemL_id, IF.faceL_id = modify_face_info(IF.elemL_id, IF.faceL_id)
		IF.elemR_id, IF.faceR_id = modify_face_info(IF.elemR_id, IF.faceR_id)

	# New interior_faces
	# code.interact(local=locals())
	# num_interior_faces_old = mesh.num_interior_faces
	# interior_faces_old = mesh.interior_faces
	mesh.num_interior_faces += num_elems_old
	# mesh.allocate_interior_faces()
	# mesh.interior_faces[:num_interior_faces_old] = interior_faces_old
	# for IF in mesh.interior_faces[num_interior_faces_old:]:
	for elem_id in range(num_elems_old):
		IF = mesh_defs.InteriorFace()
		IF.elemL_id = elem_id
		IF.faceL_id = 0
		IF.elemR_id = elem_id + num_elems_old
		IF.faceR_id = 0
		mesh.interior_faces.append(IF)

	# mesh.allocate_faces()
	# mesh.fill_faces()
	mesh.create_elements()

	return mesh












