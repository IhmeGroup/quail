# ------------------------------------------------------------------------ #
#
#       File : meshing/meshbase.py
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

# def iface_normal(mesh, IFace, quad_pts, NData=None):
#     '''
#     Function: iface_normal
#     -------------------
#     This function obtains the outward-pointing normals from the 
#     perspective of element on the "left" of IFace

#     INPUTS:
#         mesh: Mesh object
#         IFace: interior face object
#         quad_pts: points in reference space at which to calculate normals

#     OUTPUTS:
#         NData: normal data object
#     '''
#     elemL = IFace.elemL_id
#     elemR = IFace.elemR_id
#     gorderL = mesh.gorder
#     gorderR = mesh.gorder

#     nq = quad_pts.shape[0]

#     gbasis = mesh.gbasis
#     # if NData is None: 
#         # NData = NormalData()

#     if gorderL <= gorderR:
#         nvec = gbasis.calculate_normals(gbasis, mesh, elemL, IFace.faceL, quad_pts)
#     else:
#         nvec = gbasis.calculate_normals(gbasis, mesh, elemR, IFace.faceR_id, quad_pts)
#         nvec *= -1.

#     return nvec


# def bface_normal(mesh, BFace, quad_pts, NData=None):
#     '''
#     Function: bface_normal
#     -------------------
#     This function obtains the outward-pointing normals at a
#     boundary face

#     INPUTS:
#         mesh: Mesh object
#         BFace: boundary face object
#         quad_pts: points in reference space at which to calculate normals

#     OUTPUTS:
#         NData: normal data object
#     '''
#     elem = BFace.elem_id
#     gorder = mesh.gorder
#     gbasis = mesh.gbasis

#     nq = quad_pts.shape[0]

#     # if NData is None:
#     #     NData = NormalData()

#     nvec = gbasis.calculate_normals(gbasis, mesh, elem, BFace.face_id, quad_pts)

#     return nvec

# class FaceType(Enum):
#     '''
#     Class: FaceType
#     -------------------
#     Enumeration of face types

#     ATTRIBUTES:
#         Interior: interior face
#         Boundary: boundary face
#     '''
#     Interior = auto()
#     Boundary = auto()


# class Face(object):
#     '''
#     Class: Face
#     -------------------
#     This class provides information about a given face.

#     NOTES:
#         Not used for now

#     ATTRIBUTES:
#         Type: face type (interior or boundary)
#         Number: Global number of face of given type
#     '''
#     def __init__(self):
#         '''
#         Method: __init__
#         -------------------
#         This method initializes the object
#         '''
#         self.Type = FaceType.Interior
#         self.Group = -1
#         self.Number = 0 


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
    number : int
        boundary number
    '''
    '''
    Class: BoundaryGroup
    -------------------
    This class stores boundary face objects for a given boundary group

    ATTRIBUTES:
        Name: name of boundary face group
        num_boundary_faces: number of boundary faces within this group
        BFaces: list of BFace objects
    '''
    def __init__(self):
        '''
        Method: __init__
        -------------------
        This method initializes the object
        '''
        self.name = ""
        self.number = -1
        self.num_boundary_faces = 0 
        self.BFaces = None

    def allocate_bfaces(self):
        '''
        Method: allocate_bfaces
        ------------------------
        This method allocates the list of BFace objects

        OUTPUTS:
            self.BFaces
        '''
        self.BFaces = [BoundaryFace() for i in range(self.num_boundary_faces)]


'''
Dictionary: Shape2nFace
-------------------
This dictionary stores the number of faces per element
for each shape type

USAGE:
    Shape2nFace[shape] = number of faces per element of shape
'''
# Shape2nFace = {
#     ShapeType.Point : 0, 
#     ShapeType.Segment : 2,
#     ShapeType.Triangle : 3,
#     ShapeType.Quadrilateral : 4
# }


# '''

# class PeriodicGroup(object):
#     '''
#     Class: PeriodicGroup
#     -------------------
#     This class stores information about periodic groups

#     NOTES:
#         Not used for now

#     ATTRIBUTES:
#         nPeriodicNode: number of periodic nodes in group
#         PeriodicNodes: periodic nodes
#     '''
#     def __init__(self):
#         '''
#         Method: __init__
#         -------------------
#         This method initializes the object
#         '''
#         self.nPeriodicNode = 0
#         self.PeriodicNodes = None


class Element(object):
    '''
    Class: Element
    -------------------
    This class provides information about a given element.

    ATTRIBUTES:
        id: element ID
        node_ids: global IDs of the element nodes
        node_coords: coordinates of the element nodes [num_nodes, dim]
        face_to_neighbors: maps local face ID to element IDs of 
            neighbors [num_faces]
    '''
    def __init__(self, elem_id=-1):
        self.id = elem_id
        self.node_ids = np.zeros(0, dtype=int)
        self.node_coords = np.zeros(0)
        self.face_to_neighbors = np.zeros(0, dtype=int)


class Mesh(object):
    '''
    Class: Mesh
    -------------------
    This class stores the important mesh information

    ATTRIBUTES:
        Dim: dimension of mesh
        num_nodes: total number of nodes
        node_coords: coordinates of nodes
        num_interior_faces: number of interior faces
        interior_faces: list of interior face objects
        num_boundary_groups: number of boundary face groups
        boundary_groups: list of boundary face groups
        BFGNames: list of BFaceGroup names (for easy access)
        num_elemss: list of number of elements in each element group (for easy access)
        num_elems_tot: total number of elements in mesh
        nPeriodicGroup: number of periodic groups
        PeriodicGroups: list of periodic groups
    '''
    def __init__(self, dim=1, num_nodes=1, num_elems=1, gbasis=None, gorder=1):
        '''
        Method: __init__
        -------------------
        This method initializes the object

        INPUTS:
            dim: dimension of mesh
            num_nodes: total number of nodes
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
        # self.nFacePerElem = gbasis.nfaceperelem 
        # self.Faces = None
        self.num_nodes_per_elem = gbasis.get_num_basis_coeff(gorder)
        self.elem_to_node_ids = np.zeros(0, dtype=int)
            # elem_to_node_ids[elem_id][i] = ith node of elem_id, where i = 1,2,...,num_nodes_per_elem
        self.elements = []

    def set_params(self, gbasis, gorder=1, num_elems=1):

        self.gbasis = gbasis
        self.gorder = gorder
        self.num_elems = num_elems
        # self.nFacePerElem = gbasis.nfaceperelem
        self.num_nodes_per_elem = gbasis.get_num_basis_coeff(gorder)

    # def allocate_faces(self):
    #     '''
    #     Method: allocate_faces
    #     -----------------------
    #     This method allocates the list of Face objects

    #     OUTPUTS:
    #         self.Faces
    #     '''
    #     self.Faces = [[Face() for j in range(self.nFacePerElem)] for i in range(self.num_elems)]

    def allocate_elem_to_node_ids_map(self):
        '''
        Method: allocate_elem_to_node_ids_map
        -------------------
        This method allocates elem_to_node_ids

        OUTPUTS:
            self.elem_to_node_ids
        '''
        self.elem_to_node_ids = np.zeros([self.num_elems,self.num_nodes_per_elem], dtype=int)


    def allocate_interior_faces(self):
        '''
        Method: allocate_interior_faces
        -------------------
        This method allocates interior_faces

        OUTPUTS:
            self.interior_faces
        '''
        self.interior_faces = [InteriorFace() for i in range(self.num_interior_faces)]

    # def allocate_bface_groups(self):
    #     '''
    #     Method: allocate_bface_groups
    #     -------------------
    #     This method allocates boundary_groups

    #     OUTPUTS:
    #         self.boundary_groups
    #     '''
    #     self.boundary_groups = [BFaceGroup() for i in range(self.num_boundary_groups)]

    def add_boundary_group(self, bname):
        if bname in self.boundary_groups:
            raise ValueError
        BFG = BoundaryGroup()
        self.boundary_groups[bname] = BFG
        BFG.name = bname
        self.num_boundary_groups = len(self.boundary_groups)
        BFG.number = self.num_boundary_groups - 1

        return BFG

    def create_elements(self):
        self.elements = [Element() for i in range(self.num_elems)]

        for elem_id in range(self.num_elems):
            elem = self.elements[elem_id]

            elem.id = elem_id
            elem.node_ids = self.elem_to_node_ids[elem_id]
            elem.node_coords = self.node_coords[elem.node_ids]
            elem.face_to_neighbors = np.full(self.gbasis.NFACES, -1)

        # neighbors
        for iif in range(self.num_interior_faces):
            int_face = self.interior_faces[iif]
            elemL_id = int_face.elemL_id
            elemR_id = int_face.elemR_id
            faceL_id = int_face.faceL_id
            faceR_id = int_face.faceR_id

            elemL = self.elements[elemL_id]
            elemR = self.elements[elemR_id]

            elemL.face_to_neighbors[faceL_id] = elemR_id
            elemR.face_to_neighbors[faceR_id] = elemL_id



    # def fill_faces(self):
    #     for iiface in range(self.num_interior_faces):
    #         IFace = self.interior_faces[iiface]
    #         elemL = IFace.elemL_id
    #         elemR = IFace.elemR_id
    #         faceL = IFace.faceL
    #         faceR_id = IFace.faceR_id

    #         FaceL = self.Faces[elemL][faceL]
    #         FaceR = self.Faces[elemR][faceR_id]

    #         FaceL.Type = FaceType.Interior
    #         FaceR.Type = FaceType.Interior

    #         FaceL.Number = iiface
    #         FaceR.Number = iiface

    #     # for ibfgrp in range(self.num_boundary_groups):
    #     #     BFG = self.boundary_groups[ibfgrp]

    #     for BFG in self.boundary_groups.values():
            
    #         for ibface in range(BFG.num_boundary_faces):
    #             BFace = BFG.BFaces[ibface]
    #             elem = BFace.elem_id
    #             face = BFace.face_id

    #             Face = self.Faces[elem][face]

    #             Face.Type = FaceType.Boundary
    #             Face.Number = ibface
    #             # Face.Group = ibfgrp
    #             Face.Group = BFG.number








