# ------------------------------------------------------------------------ #
#
#       File : src/meshing/meshbase.py
#
#       Contains class definitions for mesh structures.
#
# ------------------------------------------------------------------------ #
from enum import Enum, auto
import numpy as np

from general import ShapeType
import numerics.basis.basis as basis_defs


class InteriorFace(object):
	'''
	This class provides information about a given interior face.

	Attributes:
	-----------
	elemL_ID : int
		ID of "left" element
	faceL_ID : int
		local ID of face from perspective of left element
	elemR_ID : int
		ID of "right" element
	faceR_ID : int
		local ID of face from perspective of right element
	children : list
		either an empty list if a leaf face or a list of two subfaces created
		from refinement
	refQ1nodes_L: numpy array
		coordinates of the face Q1 nodes in the left element's reference space
		[num_Q1_nodes, ndims]
	refQ1nodes_R: numpy array
		coordinates of the face Q1 nodes in the right element's reference space
		[num_Q1_nodes, ndims]
	'''
	def __init__(self, elemL_ID = 0, faceL_ID = 0, elemR_ID = 0, faceR_ID = 0):
		self.elemL_ID = elemL_ID
		self.faceL_ID = faceL_ID
		self.elemR_ID = elemR_ID
		self.faceR_ID = faceR_ID
		self.children = []
		self.refQ1nodes_L = np.array([])
		self.refQ1nodes_R = np.array([])


class BoundaryFace(object):
	'''
	This class provides information about a given boundary face.

	Attributes:
	-----------
	elem_ID : int
		ID of adjacent element
	face_ID : int
		local ID of face from perspective of adjacent element
	children : list
		either an empty list if a leaf face or a list of two subfaces created
		from refinement
	refQ1nodes: numpy array
		coordinates of the face Q1 nodes in the element's reference space
		[num_Q1_nodes, ndims]
	name : str
		boundary name
	'''
	def __init__(self, boundary_name = None):
		self.elem_ID = 0
		self.face_ID = 0
		self.children = []
		self.refQ1nodes = np.array([])
		self.name = boundary_name


class BoundaryGroup(object):
	'''
	This class stores boundary face objects for a given boundary group.

	Attributes:
	-----------
	name : str
		boundary name
	number : int
		boundary number
	num_boundary_faces : int
		number of faces in boundary group
	boundary_faces : list
		list of BoundaryFace objects

	Methods:
	---------
	allocate_boundary_faces
		allocates list of BoundaryFace objects
	'''
	def __init__(self):
		self.name = ""
		self.number = -1
		self.num_boundary_faces = 0
		self.boundary_faces = []

	def allocate_boundary_faces(self):
		'''
		This method allocates the list of boundary_face objects

		Outputs:
		--------
			self.boundary_faces
		'''
		self.boundary_faces = [BoundaryFace(self.name) for i in
				range(self.num_boundary_faces)]


class Element(object):
	'''
	This class provides information about a given element.

	Attributes:
	-----------
	ID: int
		element ID
	node_coords: numpy array
		coordinates of the element nodes [num_nodes, ndims]
	faces: list
		list of faces (both InteriorFaces and BoundaryFaces) which border this
		element, ordered by local face ID
	'''
	def __init__(self, elem_ID=-1):
		self.ID = elem_ID
		self.node_coords = np.zeros(0)
		self.faces = np.empty([], dtype=object)


class Mesh(object):
	'''
	This class stores information about the mesh.

	Attributes:
	-----------
	ndims : int
		number of spatial dimensions
	num_nodes : int
		total number of nodes
	node_coords : numpy array
		coordinates of nodes [num_nodes, ndims]
	num_interior_faces : int
		number of interior faces
	interior_faces : list
		list of interior face objects
	num_boundary_groups : int
		number of boundary face groups
	boundary_groups : dict
		dict whose keys are boundary names and values are BoundaryGroup
		objects
	gbasis : Basis class
		object for geometry basis
	gorder : int
		order of geometry interpolation
	num_elems : int
		total number of elements in mesh
	num_nodes_per_elem : int
		number of nodes per element
	elem_to_node_IDs : numpy array
		maps element ID to global node IDs
		[num_elems, num_nodes_per_elem]
	elements : list
		list of Element objects

	Methods:
	---------
	set_params
		sets certain mesh parameters
	allocate_elem_to_node_IDs_map
		allocates self.elem_to_node_IDs
	allocate_interior_faces
		allocates self.interior_faces
	add_boundary_group
		appends new boundary group to self.boundary_groups
	create_elements
		creates self.elements
	'''
	def __init__(self, ndims=1, num_nodes=1, num_elems=1, gbasis=None,
			gorder=1):
		if gbasis is None:
			gbasis = basis_defs.LagrangeSeg(1)

		self.ndims = ndims
		self.num_nodes = num_nodes
		self.node_coords = None
		self.num_interior_faces = 0
		self.interior_faces = []
		self.num_boundary_groups = 0
		self.boundary_groups = {}
		self.gbasis = gbasis
		self.gorder = gorder
		self.num_elems = num_elems
		self.num_nodes_per_elem = gbasis.get_num_basis_coeff(gorder)
		self.elem_to_node_IDs = np.zeros(0, dtype=int)
		self.elements = []

	def set_params(self, gbasis, gorder=1, num_elems=1):
		'''
		This method sets certain mesh parameters

		Inputs:
		-------
			gbasis: geometry basis object
			gorder: [OPTIONAL] order of geometry interpolation
			num_elems: [OPTIONAL] total number of elements in mesh

		Outputs:
		--------
			self.gbasis: geometry basis object
			self.gorder: order of geometry interpolation
			self.num_elems: total number of elements in mesh
			self.num_nodes_per_elem: number of nodes per element
		'''
		self.gbasis = gbasis
		self.gorder = gorder
		self.num_elems = num_elems
		self.num_nodes_per_elem = gbasis.get_num_basis_coeff(gorder)

	def allocate_elem_to_node_IDs_map(self):
		'''
		This method allocates self.elem_to_node_IDs

		Outputs:
		--------
			self.elem_to_node_IDs: maps element ID to global node IDs
				[num_elems, num_nodes_per_elem]

		Notes:
		------
			elem_to_node_IDs[elem_ID][i] = ith node of elem_ID,
				where i = 1, 2, ..., num_nodes_per_elem
		'''
		self.elem_to_node_IDs = np.zeros([self.num_elems,
				self.num_nodes_per_elem], dtype=int)

	def allocate_interior_faces(self):
		'''
		This method allocates self.interior_faces

		Outputs:
		--------
			self.interior_faces: list of InteriorFace objects
		'''
		self.interior_faces = [InteriorFace() for i in range(
				self.num_interior_faces)]

	def add_boundary_group(self, bname):
		'''
		This method appends a new boundary group to self.boundary_groups

		Inputs:
		-------
			bname: name of boundary

		Outputs:
		--------
			bgroup: new boundary group
			self.boundary groups: updated to contain bgroup
		'''
		if bname in self.boundary_groups:
			raise ValueError("Repeated boundary names")
		bgroup = BoundaryGroup()
		self.boundary_groups[bname] = bgroup
		bgroup.name = bname
		self.num_boundary_groups = len(self.boundary_groups)
		bgroup.number = self.num_boundary_groups - 1

		return bgroup

	def create_elements(self):
		'''
		This method creates self.elements

		Outputs:
		--------
			self.elements: list of Element objects
		'''
		# Allocate
		self.elements = [Element() for i in range(self.num_elems)]

		# Fill in information for each element
		for elem_ID in range(self.num_elems):
			elem = self.elements[elem_ID]

			elem.ID = elem_ID
			elem.node_coords = self.node_coords[self.elem_to_node_IDs[elem_ID]]
			elem.faces = np.empty(self.gbasis.NFACES, dtype=object)

		# Add interior faces to each element
		for int_face in self.interior_faces:
			elemL_ID = int_face.elemL_ID
			faceL_ID = int_face.faceL_ID
			elemR_ID = int_face.elemR_ID
			faceR_ID = int_face.faceR_ID

			elemL = self.elements[elemL_ID]
			elemR = self.elements[elemR_ID]

			elemL.faces[faceL_ID] = int_face
			elemR.faces[faceR_ID] = int_face

		# Add boundary faces to each element
		for bgroup in self.boundary_groups.values():
			for boundary_face in bgroup.boundary_faces:
				elem_ID = boundary_face.elem_ID
				face_ID = boundary_face.face_ID
				elem = self.elements[elem_ID]
				elem.faces[face_ID] = boundary_face


	def get_face_coords(self):
		'''
		Create an array of physical node coordinates for each InteriorFace and
		BoundaryFace.

		Outputs:
		--------
			self.interior_faces[:].node_IDs: Array of node IDs for each
					interior face
			self.boundary_groups[:].boundary_faces[:].node_IDs: Array of node
					IDs for each boundary face
		'''
		# Loop over InteriorFaces
		for interior_face in self.interior_faces:
			# Get node numbers on the element to the left of this face
			node_nums = self.gbasis.get_local_face_node_nums(self.gorder,
					interior_face.faceL_ID)
			# Coordinates in physical space associated with the node numbers on
			# the element
			node_coords = self.elements[interior_face.elemL_ID].node_coords[
					node_nums]
			# Coordinates in physical space
			interior_face.node_coords = node_coords
			# Get reference space Q1 nodes of face
			ref_nodes = self.gbasis.get_nodes(self.gorder)
			node_indices_L = self.gbasis.get_local_face_principal_node_nums(
					self.gorder, interior_face.faceL_ID)
			interior_face.refQ1nodes_L = ref_nodes[node_indices_L]
			node_indices_R = self.gbasis.get_local_face_principal_node_nums(
					self.gorder, interior_face.faceR_ID)
			interior_face.refQ1nodes_R = ref_nodes[node_indices_R][::-1]

		# Loop over BoundaryGroups
		for bgroup in self.boundary_groups.values():
			# Loop over BoundaryFaces
			for boundary_face in bgroup.boundary_faces:
				# Get node numbers on the element adjacent to this face
				node_nums = self.gbasis.get_local_face_node_nums(self.gorder,
						boundary_face.face_ID)
				# Coordinates in physical space associated with the node
				# numbers on the element
				node_coords = self.elements[boundary_face.elem_ID].node_coords[
						node_nums]
				# Coordinates in physical space
				boundary_face.node_coords = node_coords
				# Get reference space Q1 nodes of face
				ref_nodes = self.gbasis.get_nodes(self.gorder)
				node_indices = self.gbasis.get_local_face_principal_node_nums(
						self.gorder, boundary_face.face_ID)
				boundary_face.refQ1nodes = ref_nodes[node_indices]
