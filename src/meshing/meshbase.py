import code
from enum import Enum, auto
import numpy as np

from general import ShapeType
import numerics.basis.basis as basis_defs

def ref_to_phys(mesh, elem, PhiData, xref, xphys=None):
    '''
    Function: ref_to_phys
    -------------------
    This function converts reference space coordinates to physical
    space coordinates

    INPUTS:
        mesh: Mesh object
        elem: element 
        PhiData: basis data
        npoint: number of coordinates to convert
        xref: coordinates in reference space
        xphys: pre-allocated storage for physical coordinates (optional) 

    OUTPUTS:
        xphys: coordinates in physical space
    '''
    gbasis = mesh.gbasis
    gorder = mesh.gorder
    #PhiData = gbasis.basis_val
    npoint = xref.shape[0]

    # if PhiData is None:
    #     PhiData = Basis.BasisData(QBasis,QOrder,mesh)
    #     PointsChanged = True
    # if PointsChanged or PhiData.basis != QBasis or PhiData.order != QOrder:
    #     PhiData.get_basis_val_grads(xref, get_val=True)

    gbasis.get_basis_val_grads(xref, get_val=True)

    Phi= gbasis.basis_val
    dim = mesh.Dim
    Coords = mesh.Coords
    # Phi = PhiData.Phi
    nb = gbasis.basis_val.shape[1]
    if nb != mesh.nNodePerElem:
        raise Exception("Wrong number of nodes per element")

    ElemNodes = mesh.Elem2Nodes[elem]

    if xphys is None:
        xphys = np.zeros([npoint,dim])
    else:
        xphys[:] = 0.

    # for ipoint in range(npoint):
    #     for n in range(nn):
    #         nodeNum = ElemNodes[n]
    #         val = Phi[ipoint][n]
    #         for d in range(dim):
    #             xphys[ipoint][d] += val*Coords[nodeNum][d]

    xphys[:] = np.matmul(Phi, Coords[ElemNodes])

    return xphys, PhiData

def iface_normal(mesh, IFace, quad_pts, NData=None):
    '''
    Function: iface_normal
    -------------------
    This function obtains the outward-pointing normals from the 
    perspective of element on the "left" of IFace

    INPUTS:
        mesh: Mesh object
        IFace: interior face object
        quad_pts: points in reference space at which to calculate normals

    OUTPUTS:
        NData: normal data object
    '''
    elemL = IFace.ElemL
    elemR = IFace.ElemR
    gorderL = mesh.gorder
    gorderR = mesh.gorder

    nq = quad_pts.shape[0]

    gbasis = mesh.gbasis
    # if NData is None: 
        # NData = NormalData()

    if gorderL <= gorderR:
        nvec = gbasis.calculate_normals(gbasis, mesh, elemL, IFace.faceL, quad_pts)
    else:
        nvec = gbasis.calculate_normals(gbasis, mesh, elemR, IFace.faceR, quad_pts)
        nvec *= -1.

    return nvec


def bface_normal(mesh, BFace, quad_pts, NData=None):
    '''
    Function: bface_normal
    -------------------
    This function obtains the outward-pointing normals at a
    boundary face

    INPUTS:
        mesh: Mesh object
        BFace: boundary face object
        quad_pts: points in reference space at which to calculate normals

    OUTPUTS:
        NData: normal data object
    '''
    elem = BFace.Elem
    gorder = mesh.gorder
    gbasis = mesh.gbasis

    nq = quad_pts.shape[0]

    # if NData is None:
    #     NData = NormalData()

    nvec = gbasis.calculate_normals(gbasis, mesh, elem, BFace.face, quad_pts)

    return nvec

class FaceType(Enum):
    '''
    Class: FaceType
    -------------------
    Enumeration of face types

    ATTRIBUTES:
        Interior: interior face
        Boundary: boundary face
    '''
    Interior = auto()
    Boundary = auto()


class Face(object):
    '''
    Class: Face
    -------------------
    This class provides information about a given face.

    NOTES:
        Not used for now

    ATTRIBUTES:
        Type: face type (interior or boundary)
        Number: Global number of face of given type
    '''
    def __init__(self):
        '''
        Method: __init__
        -------------------
        This method initializes the object
        '''
        self.Type = FaceType.Interior
        self.Group = -1
        self.Number = 0 


class IFace(object):
    '''
    Class: IFace
    -------------------
    This class provides information about a given interior face.

    ATTRIBUTES:
        ElemL: left element
        faceL: local face number from ElemL's perspective
        ElemR: right element
        faceR: local face number from ElemR's perspective
    '''
    def __init__(self):
        '''
        Method: __init__
        -------------------
        This method initializes the object
        '''
        self.ElemL = 0 
        self.faceL = 0 
        self.ElemR = 0 
        self.faceR = 0 


class BFace(object):
    '''
    Class: BFace
    -------------------
    This class provides information about a given boundary face.

    ATTRIBUTES:
        Elem: adjacent element
        face: local face number from Elem's perspective
    '''
    def __init__(self):
        '''
        Method: __init__
        -------------------
        This method initializes the object
        '''
        self.Elem = 0 
        self.face = 0 


class BFaceGroup(object):
    '''
    Class: BFaceGroup
    -------------------
    This class stores BFace objects for a given boundary face group

    ATTRIBUTES:
        Name: name of boundary face group
        nBFace: number of boundary faces within this group
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
        self.nBFace = 0 
        self.BFaces = None

    def allocate_bfaces(self):
        '''
        Method: allocate_bfaces
        ------------------------
        This method allocates the list of BFace objects

        OUTPUTS:
            self.BFaces
        '''
        self.BFaces = [BFace() for i in range(self.nBFace)]


'''
Dictionary: Shape2nFace
-------------------
This dictionary stores the number of faces per element
for each shape type

USAGE:
    Shape2nFace[shape] = number of faces per element of shape
'''
Shape2nFace = {
    ShapeType.Point : 0, 
    ShapeType.Segment : 2,
    ShapeType.Triangle : 3,
    ShapeType.Quadrilateral : 4
}


'''
Dictionary: Shape2nNodeQ1
-------------------
This dictionary stores the number of Q1 nodes per element
for each shape type

USAGE:
    Shape2nFace[shape] = number of nodes per element of shape
'''
Shape2nNodeQ1 = {
    ShapeType.Point : 1,
    ShapeType.Segment : 2,
    ShapeType.Triangle : 3,
    ShapeType.Quadrilateral : 4
}

class PeriodicGroup(object):
    '''
    Class: PeriodicGroup
    -------------------
    This class stores information about periodic groups

    NOTES:
        Not used for now

    ATTRIBUTES:
        nPeriodicNode: number of periodic nodes in group
        PeriodicNodes: periodic nodes
    '''
    def __init__(self):
        '''
        Method: __init__
        -------------------
        This method initializes the object
        '''
        self.nPeriodicNode = 0
        self.PeriodicNodes = None


class Element(object):
    def __init__(self, number=-1):
        self.number = number
        self.node_nums = np.zeros(0, dtype=int)
        self.node_coords = np.zeros(0)
        self.face_to_neighbors = np.zeros(0, dtype=int)




class Mesh(object):
    '''
    Class: Mesh
    -------------------
    This class stores the important mesh information

    ATTRIBUTES:
        Dim: dimension of mesh
        nNode: total number of nodes
        Coords: coordinates of nodes
        nIFace: number of interior faces
        IFaces: list of interior face objects
        nBFaceGroup: number of boundary face groups
        BFaceGroups: list of boundary face groups
        BFGNames: list of BFaceGroup names (for easy access)
        nElems: list of number of elements in each element group (for easy access)
        nElemTot: total number of elements in mesh
        nPeriodicGroup: number of periodic groups
        PeriodicGroups: list of periodic groups
    '''
    def __init__(self,dim=1,nNode=1,nElem=1,gbasis=None,gorder=1):
        '''
        Method: __init__
        -------------------
        This method initializes the object

        INPUTS:
            dim: dimension of mesh
            nNode: total number of nodes
        '''
        if gbasis is None:
            gbasis = basis_defs.LagrangeSeg(1)

        self.Dim = dim
        self.nNode = nNode
        self.Coords = None
        self.nIFace = 0
        self.IFaces = []
        self.nBFaceGroup = 0
        self.BFaceGroups = {}
        self.gbasis = gbasis
        self.gorder = gorder
        self.nElem = nElem
        # self.nFacePerElem = gbasis.nfaceperelem 
        # self.Faces = None
        self.nNodePerElem = gbasis.get_num_basis_coeff(gorder)
        self.Elem2Nodes = None
        self.elements = []
            # Elem2Nodes[elem][i] = ith node of elem, where i = 1,2,...,nNodePerElem

    def SetParams(self, gbasis, gorder=1, nElem=1):

        self.gbasis = gbasis
        self.gorder = gorder
        self.nElem = nElem
        # self.nFacePerElem = gbasis.nfaceperelem
        self.nNodePerElem = gbasis.get_num_basis_coeff(gorder)

    # def allocate_faces(self):
    #     '''
    #     Method: allocate_faces
    #     -----------------------
    #     This method allocates the list of Face objects

    #     OUTPUTS:
    #         self.Faces
    #     '''
    #     self.Faces = [[Face() for j in range(self.nFacePerElem)] for i in range(self.nElem)]

    def allocate_elem_to_nodes(self):
        '''
        Method: allocate_elem_to_nodes
        -------------------
        This method allocates Elem2Nodes

        OUTPUTS:
            self.Elem2Nodes
        '''
        self.Elem2Nodes = np.zeros([self.nElem,self.nNodePerElem], dtype=int)


    def allocate_ifaces(self):
        '''
        Method: allocate_ifaces
        -------------------
        This method allocates IFaces

        OUTPUTS:
            self.IFaces
        '''
        self.IFaces = [IFace() for i in range(self.nIFace)]

    # def allocate_bface_groups(self):
    #     '''
    #     Method: allocate_bface_groups
    #     -------------------
    #     This method allocates BFaceGroups

    #     OUTPUTS:
    #         self.BFaceGroups
    #     '''
    #     self.BFaceGroups = [BFaceGroup() for i in range(self.nBFaceGroup)]

    def add_bface_group(self, bname):
        if bname in self.BFaceGroups:
            raise ValueError
        BFG = BFaceGroup()
        self.BFaceGroups[bname] = BFG
        BFG.name = bname
        self.nBFaceGroup = len(self.BFaceGroups)
        BFG.number = self.nBFaceGroup - 1

        return BFG

    def create_elements(self):
        self.elements = [Element() for i in range(self.nElem)]

        for ielem in range(self.nElem):
            elem = self.elements[ielem]

            elem.number = ielem
            elem.node_nums = self.Elem2Nodes[ielem]
            elem.node_coords = self.Coords[elem.node_nums]
            elem.face_to_neighbors = np.full(self.gbasis.NFACES, -1)

        # neighbors
        for iif in range(self.nIFace):
            int_face = self.IFaces[iif]
            ielemL = int_face.ElemL
            ielemR = int_face.ElemR
            faceL = int_face.faceL
            faceR = int_face.faceR

            elemL = self.elements[ielemL]
            elemR = self.elements[ielemR]

            elemL.face_to_neighbors[faceL] = ielemR
            elemR.face_to_neighbors[faceR] = ielemL



    # def fill_faces(self):
    #     for iiface in range(self.nIFace):
    #         IFace = self.IFaces[iiface]
    #         elemL = IFace.ElemL
    #         elemR = IFace.ElemR
    #         faceL = IFace.faceL
    #         faceR = IFace.faceR

    #         FaceL = self.Faces[elemL][faceL]
    #         FaceR = self.Faces[elemR][faceR]

    #         FaceL.Type = FaceType.Interior
    #         FaceR.Type = FaceType.Interior

    #         FaceL.Number = iiface
    #         FaceR.Number = iiface

    #     # for ibfgrp in range(self.nBFaceGroup):
    #     #     BFG = self.BFaceGroups[ibfgrp]

    #     for BFG in self.BFaceGroups.values():
            
    #         for ibface in range(BFG.nBFace):
    #             BFace = BFG.BFaces[ibface]
    #             elem = BFace.Elem
    #             face = BFace.face

    #             Face = self.Faces[elem][face]

    #             Face.Type = FaceType.Boundary
    #             Face.Number = ibface
    #             # Face.Group = ibfgrp
    #             Face.Group = BFG.number








