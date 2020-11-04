# ------------------------------------------------------------------------ #
#
#       File : src/meshing/common.py
#
#       Contains functions for creating and modifying structured meshes.
#      
# ------------------------------------------------------------------------ #
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

	Notes:
	------
		Two boundary groups are created:
		x1: located at x = xmin
		x2: located at x = xmax
	'''
	''' Create mesh and set node coordinates '''
	num_nodes = num_elems + 1
	mesh = mesh_defs.Mesh(dim=1, num_nodes=num_nodes)
	mesh.node_coords = np.zeros([mesh.num_nodes, mesh.dim])
	mesh.node_coords[:,0] = np.linspace(xmin, xmax, mesh.num_nodes)
	# Set parameters
	mesh.set_params(gbasis=basis_defs.LagrangeSeg(1), gorder=1, 
			num_elems=num_elems)

	''' Interior faces '''
	mesh.num_interior_faces = num_elems - 1
	mesh.allocate_interior_faces()
	for i in range(mesh.num_interior_faces):
		interior_face = mesh.interior_faces[i]
		interior_face.elemL_ID = i
		interior_face.faceL_ID = 1
		interior_face.elemR_ID = i+1
		interior_face.faceR_ID = 0

	''' Boundary groups and faces '''
	# Left
	boundary_group = mesh.add_boundary_group("x1")
	boundary_group.num_boundary_faces = 1
	boundary_group.allocate_boundary_faces()
	boundary_face = boundary_group.boundary_faces[0]
	boundary_face.elem_ID = 0
	boundary_face.face_ID = 0
	# Right
	boundary_group = mesh.add_boundary_group("x2")
	boundary_group.num_boundary_faces = 1
	boundary_group.allocate_boundary_faces()
	boundary_face = boundary_group.boundary_faces[0]
	boundary_face.elem_ID = num_elems - 1
	boundary_face.face_ID = 1

	''' Create element-to-node-ID map '''
	mesh.allocate_elem_to_node_IDs_map()
	for elem_ID in range(mesh.num_elems):
		for n in range(mesh.num_nodes_per_elem):
			mesh.elem_to_node_IDs[elem_ID][n] = elem_ID + n

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

	Notes:
	------
		Four boundary groups are created:
		x1: located along the line x = xmin
		x2: located along the line x = xmax
		y1: located along the line y = ymin
		y2: located along the line y = ymax
	'''
	''' Create mesh and set node coordinates '''
	# Number of nodes
	num_nodes_x = num_elems_x + 1
	num_nodes_y = num_elems_y + 1
	# xy-coordinates
	xcoords = np.linspace(xmin, xmax, num_nodes_x)
	ycoords = np.linspace(ymin, ymax, num_nodes_y)
	xgrid, ygrid = np.meshgrid(xcoords, ycoords)
	xp = np.array([np.reshape(xgrid, -1), np.reshape(ygrid, -1)]).transpose()
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
			interior_face.elemL_ID = num_elems_x*ny + nx
			interior_face.faceL_ID = 1
			interior_face.elemR_ID = num_elems_x*ny + nx + 1
			interior_face.faceR_ID = 3
			n += 1
	# y-direction
	for nx in range(num_elems_x):
		for ny in range(num_elems_y-1):
			interior_face = mesh.interior_faces[n]
			interior_face.elemL_ID = num_elems_x*ny + nx
			interior_face.faceL_ID = 2
			interior_face.elemR_ID = num_elems_x*(ny + 1) + nx
			interior_face.faceR_ID = 0
			n += 1

	''' Boundary groups and faces '''
	# x1
	boundary_group = mesh.add_boundary_group("x1")
	boundary_group.num_boundary_faces = num_elems_y
	boundary_group.allocate_boundary_faces()
	n = 0
	for boundary_face in boundary_group.boundary_faces:
		boundary_face.elem_ID = num_elems_x*n
		boundary_face.face_ID = 3
		n += 1
	# x2
	boundary_group = mesh.add_boundary_group("x2")
	boundary_group.num_boundary_faces = num_elems_y
	boundary_group.allocate_boundary_faces()
	n = 0
	for boundary_face in boundary_group.boundary_faces:
		boundary_face.elem_ID = num_elems_x*(n + 1) - 1
		boundary_face.face_ID = 1
		n += 1
	# y1
	boundary_group = mesh.add_boundary_group("y1")
	boundary_group.num_boundary_faces = num_elems_x
	boundary_group.allocate_boundary_faces()
	n = 0
	for boundary_face in boundary_group.boundary_faces:
		boundary_face.elem_ID = n
		boundary_face.face_ID = 0
		n += 1
	# y2
	boundary_group = mesh.add_boundary_group("y2")
	boundary_group.num_boundary_faces = num_elems_x
	boundary_group.allocate_boundary_faces()
	n = 0
	for boundary_face in boundary_group.boundary_faces:
		boundary_face.elem_ID = mesh.num_elems - num_elems_x + n
		boundary_face.face_ID = 2
		n += 1

	''' Create element-to-node-ID map '''
	mesh.allocate_elem_to_node_IDs_map()
	elem = 0
	for ny in range(num_elems_y):
		for nx in range(num_elems_x):
			mesh.elem_to_node_IDs[elem][0] = num_nodes_x*ny + nx
			mesh.elem_to_node_IDs[elem][1] = num_nodes_x*ny + nx + 1
			mesh.elem_to_node_IDs[elem][2] = num_nodes_x*(ny+1) + nx
			mesh.elem_to_node_IDs[elem][3] = num_nodes_x*(ny+1) + nx + 1
			elem += 1

	''' Create element objects '''
	mesh.create_elements()

	return mesh


def split_quadrils_into_tris(mesh_old):
	'''
	This function splits each quadrilateral element of a mesh into two
	triangular elements.

	Inputs:
	-------
	    mesh_old: quadrilateral mesh object to modify

	Outputs:
	--------
	    mesh: new triangular mesh object
	'''
	def convert_nodes(gorder, num_nodes_per_quad, num_nodes_per_tri):
		'''
		This nested function converts the local node IDs of a quadrilateral 
		element	into two sets of local node IDs for the resulting two
		triangular elements.

		Inputs:
		-------
		    gorder: order of geometry interpolation
		    num_nodes_per_quad: number of nodes per quadrilateral
		    num_nodes_per_tri: number of nodes per triangle

		Outputs:
		--------
		    tri1_node_IDs: local node IDs of first triangle
		    tri2_node_IDs: local node IDs of second triangle
		'''
		# Number of nodes per face
		num_nodes_per_face = gorder + 1
		if num_nodes_per_face != np.sqrt(num_nodes_per_quad):
			return ValueError

		# Local quadrilateral node IDs
		quad_node_IDs = np.arange(num_nodes_per_quad)
		quad_node_IDs.shape = num_nodes_per_face, num_nodes_per_face

		''' Local triangle node IDs '''
		# faces 0 and 3 become faces 0 and 2 of tri1
		tri1_node_IDs = np.arange(num_nodes_per_tri)
		# faces 1 and 2 become faces 2 and 0 of tri2
		tri2_node_IDs = np.copy(tri1_node_IDs)

		n = 0
		for j in range(num_nodes_per_face):
			tri1_node_IDs[n:n+num_nodes_per_face-j] = quad_node_IDs[j,
					:num_nodes_per_face-j]
			if j == 0:
				tri2_node_IDs[n:n+num_nodes_per_face-j] = quad_node_IDs[
						num_nodes_per_face-1, ::-1]
			else:
				tri2_node_IDs[n:n+num_nodes_per_face-j] = quad_node_IDs[
						num_nodes_per_face-(j+1),
						num_nodes_per_face-1:j-1:-1]
			n += num_nodes_per_face-j

		return tri1_node_IDs, tri2_node_IDs

	def modify_element_face_info(elem_ID, face_ID):
		'''
		This nested function modifies element and local face IDs.

		Inputs:
		-------
		    elem_ID: element ID
		    face_ID: local face ID

		Outputs:
		--------
		    elem_ID: element ID (modified)
		    face_ID: local face ID (modified)
		'''
		if face_ID == 1 or face_ID == 2:
			elem_ID += num_elems_old
		face_ID = old_to_new_face[face_ID]
		return elem_ID, face_ID

	# New number of elements
	num_elems_old = mesh_old.num_elems 
	num_elems = num_elems_old*2

	# Create deep copy of mesh
	mesh = copy.deepcopy(mesh_old)
	mesh.set_params(num_elems=num_elems, gbasis=basis_defs.LagrangeTri(1))

	# Convert local quadrilateral node IDs to triangle node IDs
	tri1_node_IDs, tri2_node_IDs = convert_nodes(mesh.gorder, 
			mesh_old.num_nodes_per_elem, mesh.num_nodes_per_elem)

	# Array to maps old (quadrilateral) local face ID to new (triangle) 
	# local face ID
	old_to_new_face = np.array([2, 1, 2, 1])

	# Boundary groups
	for bgroup in mesh.boundary_groups.values():
		for bface in bgroup.boundary_faces:
			bface.elem_ID, bface.face_ID = modify_element_face_info(
					bface.elem_ID, bface.face_ID)

	# Modify existing interior faces
	for int_face in mesh.interior_faces:
		int_face.elemL_ID, int_face.faceL_ID = modify_element_face_info(
				int_face.elemL_ID, int_face.faceL_ID)
		int_face.elemR_ID, int_face.faceR_ID = modify_element_face_info(
				int_face.elemR_ID, int_face.faceR_ID)

	# Create new interior_faces ("diagonal" faces)
	mesh.num_interior_faces += num_elems_old
	for elem_ID in range(num_elems_old):
		int_face = mesh_defs.InteriorFace()
		int_face.elemL_ID = elem_ID
		int_face.faceL_ID = 0
		int_face.elemR_ID = elem_ID + num_elems_old
		int_face.faceR_ID = 0
		mesh.interior_faces.append(int_face)

	# Element-to-node-ID map
	mesh.allocate_elem_to_node_IDs_map()
	for elem_ID in range(num_elems_old):
		# First triangle
		mesh.elem_to_node_IDs[elem_ID] = mesh_old.elem_to_node_IDs[elem_ID, 
				tri1_node_IDs]
		# Second triangle
		mesh.elem_to_node_IDs[elem_ID+num_elems_old] = \
				mesh_old.elem_to_node_IDs[elem_ID, tri2_node_IDs]

	# Create element objects
	mesh.create_elements()

	return mesh