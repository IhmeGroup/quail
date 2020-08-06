# ------------------------------------------------------------------------ #
#
#       File : src/meshing/meshbase.py
#
#       Contains class definitions for mesh structures.
#
#       Authors: Eric Ching and Brett Bornhoft
#
#       Created: January 2020
#      
# ------------------------------------------------------------------------ #

import code
from enum import Enum, auto
import numpy as np

from general import ShapeType
import numerics.basis.basis as basis_defs


class InteriorFace(object):
    '''
    This class provides information about a given interior face.

    Attributes:
    -----------
    elemL_id : int
        ID of "left" element
    faceL_id : int
        local ID of face from perspective of left element
    elemR_id : int
        ID of "right" element
    faceR_id : int
        local ID of face from perspective of right element
    '''
    def __init__(self):
        '''
        Attributes:
        -----------
        elemL_id : int
            ID of "left" element
        faceL_id : int
            local ID of face from perspective of left element
        elemR_id : int
            ID of "right" element
        faceR_id : int
            local ID of face from perspective of right element
        '''
        self.elemL_id = 0 
        self.faceL_id = 0 
        self.elemR_id = 0 
        self.faceR_id = 0 


class BoundaryFace(object):
    '''
    This class provides information about a given boundary face.

    Attributes:
    -----------
    elem_id : int
        ID of adjacent element
    face_id : int
        local ID of face from perspective of adjacent element
    '''
    def __init__(self):
        '''
        Method: __init__
        -------------------
        This method initializes the object
        '''
        self.elem_id = 0 
        self.face_id = 0 


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
        '''
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
        '''
        self.name = ""
        self.number = -1
        self.num_boundary_faces = 0 
        self.boundary_faces = []

    def allocate_boundary_faces(self):
        '''
        This method allocates the list of boundary_face objects

        OUTPUTS:
            self.boundary_faces
        '''
        self.boundary_faces = [BoundaryFace() for i in \
                range(self.num_boundary_faces)]


class Element(object):
    '''
    This class provides information about a given element.

    Attributes:
    -----------
    id: int
        element ID
    node_ids: numpy array
        global IDs of the element nodes
    node_coords: numpy array
        coordinates of the element nodes [num_nodes, dim]
    face_to_neighbors: numpy array
        maps local face ID to element ID of 
        neighbor across said face [num_faces]
    '''
    def __init__(self, elem_id=-1):
        '''
        Attributes:
        -----------
        id: int
            element ID
        node_ids : numpy array
            global IDs of the element nodes
        node_coords : numpy array
            coordinates of the element nodes [num_elem_nodes, dim]
        face_to_neighbors : numpy array
            maps local face ID to element ID of 
            neighbor across said face [num_faces]
        '''
        self.id = elem_id
        self.node_ids = np.zeros(0, dtype=int)
        self.node_coords = np.zeros(0)
        self.face_to_neighbors = np.zeros(0, dtype=int)


class Mesh(object):
    '''
    This class stores information about the mesh

    Attributes:
    -----------
    dim : int
        number of spatial dimensions
    num_nodes : int
        total number of nodes
    node_coords : numpy array
        coordinates of nodes [num_nodes, dim]
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
    elem_to_node_ids : numpy array
        maps element ID to global node IDs 
        [num_elems, num_nodes_per_elem]
    elements : list
        list of Element objects
    
    Methods:
    ---------
    set_params
        sets certain mesh parameters
    allocate_elem_to_node_ids_map
        allocates self.elem_to_node_ids
    allocate_interior_faces
        allocates self.interior_faces
    add_boundary_group
        appends new boundary group to self.boundary_groups
    create_elements
        creates self.elements
    '''
    def __init__(self, dim=1, num_nodes=1, num_elems=1, gbasis=None, 
            gorder=1):
        '''
        Attributes:
        -----------
        dim : int
            number of spatial dimensions
        num_nodes : int
            total number of nodes
        node_coords : numpy array
            coordinates of nodes [num_nodes, dim]
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
            geometry basis object
        gorder : int
            order of geometry interpolation
        num_elems : int
            total number of elements in mesh
        num_nodes_per_elem : int
            number of nodes per element
        elem_to_node_ids : numpy array
            maps element ID to global node IDs 
            [num_elems, num_nodes_per_elem]
        elements : list
            list of Element objects
        '''
        if gbasis is None:
            gbasis = basis_defs.LagrangeSeg(1)

        self.dim = dim
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
        self.elem_to_node_ids = np.zeros(0, dtype=int)
        self.elements = []

    def set_params(self, gbasis, gorder=1, num_elems=1):
        '''
        This method sets certain mesh parameters

        INPUTS:
            gbasis: geometry basis object
            gorder: [OPTIONAL] order of geometry interpolation
            num_elems: [OPTIONAL] total number of elements in mesh

        OUTPUTS:
            self.gbasis: geometry basis object
            self.gorder: order of geometry interpolation
            self.num_elems: total number of elements in mesh
            self.num_nodes_per_elem: number of nodes per element
        '''
        self.gbasis = gbasis
        self.gorder = gorder
        self.num_elems = num_elems
        # self.nFacePerElem = gbasis.nfaceperelem
        self.num_nodes_per_elem = gbasis.get_num_basis_coeff(gorder)

    def allocate_elem_to_node_ids_map(self):
        '''
        This method allocates self.elem_to_node_ids

        OUTPUTS:
            self.elem_to_node_ids: maps element ID to global node IDs 
                [num_elems, num_nodes_per_elem]

        NOTES:
            elem_to_node_ids[elem_id][i] = ith node of elem_id, 
                where i = 1,2,...,num_nodes_per_elem
        '''
        self.elem_to_node_ids = np.zeros([self.num_elems,
                self.num_nodes_per_elem], dtype=int)

    def allocate_interior_faces(self):
        '''
        This method allocates self.interior_faces

        OUTPUTS:
            self.interior_faces: list of InteriorFace objects
        '''
        self.interior_faces = [InteriorFace() for i in range(self.num_interior_faces)]

    def add_boundary_group(self, bname):
        '''
        This method appends a new boundary group to self.boundary_groups

        INPUTS:
            bname: name of boundary

        OUTPUTS:
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

        OUTPUTS:
            self.elements: list of Element objects
        '''
        # allocate
        self.elements = [Element() for i in range(self.num_elems)]

        # fill in information for each element
        for elem_id in range(self.num_elems):
            elem = self.elements[elem_id]

            elem.id = elem_id
            elem.node_ids = self.elem_to_node_ids[elem_id]
            elem.node_coords = self.node_coords[elem.node_ids]
            elem.face_to_neighbors = np.full(self.gbasis.NFACES, -1)

        # fill in information about neighbors
        for int_face in self.interior_faces:
            elemL_id = int_face.elemL_id
            elemR_id = int_face.elemR_id
            faceL_id = int_face.faceL_id
            faceR_id = int_face.faceR_id

            elemL = self.elements[elemL_id]
            elemR = self.elements[elemR_id]

            elemL.face_to_neighbors[faceL_id] = elemR_id
            elemR.face_to_neighbors[faceR_id] = elemL_id