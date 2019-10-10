import numpy as np
from enum import IntEnum
from General import *
import Basis


def GetEntityDim(mesh, entity):
    '''
    Function: GetEntityDim
    -------------------
    This function returns the dimension of a given entity

    INPUTS:
        mesh: Mesh object
        entity: Element, IFace, or BFace

    OUTPUTS:
        dim: entity dimension
    '''
    if entity == EntityType.Element:
        dim = mesh.Dim 
    else:
        dim = mesh.Dim - 1

    return dim


def Ref2Phys(mesh, egrp, elem, PhiData, npoint, xref, xphys=None):
    '''
    Function: Ref2Phys
    -------------------
    This function converts reference space coordinates to physical
        space coordinates

    INPUTS:
        mesh: Mesh object
        egrp: element group
        elem: element 
        PhiData: basis data
        npoint: number of coordinates to convert
        xref: coordinates in reference space
        xphys: pre-allocated storage for physical coordinates (optional) 

    OUTPUTS:
        xphys: coordinates in physical space
    '''
    EGroup = mesh.ElemGroups[egrp]
    QBasis = EGroup.QBasis
    QOrder = EGroup.QOrder
    if PhiData is None or PhiData.Basis != QBasis or PhiData.Order != QOrder:
        PhiData = Basis.BasisData(QBasis,QOrder,npoint,mesh)
        PhiData.EvalBasis(xref, True, False, False, None)
        # PhiData = Basis.BasisData(egrp,QOrder,EntityType.Element,npoint,xref,mesh,True,False)

    dim = mesh.Dim
    Coords = mesh.Coords
    Phi = PhiData.Phi
    nn = PhiData.nn

    if nn != EGroup.nNodePerElem:
        raise Exception("Wrong number of nodes per element")

    ElemNodes = EGroup.Elem2Nodes[elem]

    if xphys is None:
        xphys = np.zeros([npoint,dim])
    else:
        xphys[:] = 0.

    for ipoint in range(npoint):
        for n in range(nn):
            nodeNum = ElemNodes[n]
            val = Phi[ipoint][n]
            for d in range(dim):
                xphys[ipoint][d] += val*Coords[nodeNum][d]

    return xphys


def RefFace2Elem(Shape, face, nq, xface, xelem=None):
    '''
    Function: RefFace2Elem
    -------------------
    This function converts coordinates in face reference space to
        element reference space

    INPUTS:
        Shape: element shape
        face: local face number
        nq: number of points to convert 
        xface: coordinates in face reference space
        xelem: pre-allocated storage for output coordinates (optional)

    OUTPUTS:
        xelem: coordinates in element reference space
    '''
    if Shape == ShapeType.Segment:
        if xelem is None: xelem = np.zeros([1,1])
        if face == 0: xelem[0] = 0.
        elif face == 1: xelem[0] = 1.
        else: raise Exception("Face error")
    else:
        raise Exception("Shape error")

    return xelem


def IFaceNormal(mesh, IFace, nq, xq):
    '''
    Function: IFaceNormal
    -------------------
    This function obtains the outward-pointing normals from the 
        perspective of element on the "left" of IFace

    INPUTS:
        mesh: Mesh object
        IFace: interior face object
        nq: number of points at which to calculate normals
        xq: points in reference space at which to calculate normals

    OUTPUTS:
        NData: normal data object
    '''
    egrpL = IFace.ElemGroupL
    egrpR = IFace.ElemGroupR
    elemL = IFace.ElemL
    elemR = IFace.ElemR
    QOrderL = mesh.ElemGroups[egrpL].QOrder
    QOrderR = mesh.ElemGroups[egrpR].QOrder

    if QOrderL <= QOrderR:
        NData = NormalData(mesh, egrpL, elemL, IFace.faceL, nq, xq)
    else:
        NData = NormalData(mesh, egrpR, elemR, IFace.faceR, nq, xq)
        NData.nvec *= -1.

    return NData


def BFaceNormal(mesh, BFace, nq, xq):
    '''
    Function: BFaceNormal
    -------------------
    This function obtains the outward-pointing normals at a
        boundary face

    INPUTS:
        mesh: Mesh object
        BFace: boundary face object
        nq: number of points at which to calculate normals
        xq: points in reference space at which to calculate normals

    OUTPUTS:
        NData: normal data object
    '''
    egrp = BFace.ElemGroup
    elem = BFace.Elem
    QOrder = mesh.ElemGroups[egrp].QOrder

    NData = NormalData(mesh, egrp, elem, BFace.face, nq, xq)

    return NData


class NormalData(object):
    '''
    Class: NormalData
    -------------------
    This class contains information about normal vectors

    ATTRIBUTES:
        nq: number of points at which normals are calculated
        dim: dimension of mesh
        nvec: normals [nq, dim]
    '''
    def __init__(self, mesh, egrp, elem, face, nq, xq):
        '''
        Method: __init__
        -------------------
        This method initializes the object

        INPUTS:
            mesh: Mesh object
            egrp: element group
            elem: element
            face: local face number from elem's perspective
            nq: number of points at which to calculate normals
            xq: points in reference space at which to calculate normals
        '''
        self.nq = nq
        self.dim = mesh.Dim

        QBasis = mesh.ElemGroups[egrp].QBasis
        Shape = QBasis

        if nq != 1:
            raise Exception("nq should be 1")

        nvec = np.zeros([nq,mesh.Dim])

        # only segment for now
        if face == 0:
            nvec[0] = -1.
        elif face == 1:
            nvec[0] = 1.
        else:
            raise Exception("Wrong face")

        self.nvec = nvec


class FaceType(IntEnum):
    '''
    Class: FaceType
    -------------------
    Enumeration of face types

    ATTRIBUTES:
        Interior: interior face
        Boundary: boundary face
    '''
    Interior = 0
    Boundary = 1


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
        self.Number = 0 


class IFace(object):
    '''
    Class: IFace
    -------------------
    This class provides information about a given interior face.

    ATTRIBUTES:
        ElemGroupL: element group of right element
        ElemL: left element
        faceL: local face number from ElemL's perspective
        ElemGroupR: element group of right element
        ElemR: right element
        faceR: local face number from ElemR's perspective
    '''
    def __init__(self):
        '''
        Method: __init__
        -------------------
        This method initializes the object
        '''
        self.ElemGroupL = 0
        self.ElemL = 0 
        self.faceL = 0 
        self.ElemGroupR = 0
        self.ElemR = 0 
        self.faceR = 0 


class BFace(object):
    '''
    Class: BFace
    -------------------
    This class provides information about a given boundary face.

    ATTRIBUTES:
        ElemGroup: element group of adjacent element
        Elem: adjacent element
        face: local face number from Elem's perspective
    '''
    def __init__(self):
        '''
        Method: __init__
        -------------------
        This method initializes the object
        '''
        self.ElemGroup = 0
        self.Elem = 0 
        self.face = 0 


class BFaceGroup(object):
    '''
    Class: BFaceGroup
    -------------------
    This class stores BFace objects for a given boundary face group

    ATTRIBUTES:
        Title: title of boundary face group
        nBFace: number of boundary faces within this group
        BFaces: list of BFace objects
    '''
    def __init__(self):
        '''
        Method: __init__
        -------------------
        This method initializes the object
        '''
        self.Title = "" 
        self.nBFace = 0 
        self.BFaces = None

    def AllocBFaces(self):
        '''
        Method: AllocBFaces
        -------------------
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
Dictionary: Shape2nNode
-------------------
This dictionary stores the number of nodes per element
    for each shape type

USAGE:
    Shape2nFace[shape] = number of nodes per element of shape
'''
Shape2nNode = {
    ShapeType.Point : 1,
    ShapeType.Segment : 2,
    ShapeType.Triangle : 3,
    ShapeType.Quadrilateral : 4
}


class ElemGroup(object):
    '''
    Class: ElemGroup
    -------------------
    This class stores information about a given element group

    ATTRIBUTES:
        QBasis: Basis used for geometry representation 
        QOrder: Order used for geometry representation
        nElem: number of elements
        nFacePerElem: number of faces per element
        Faces:
    '''
    def __init__(self,QBasis=BasisType.SegLagrange,QOrder=1,nElem=1):
        '''
        Method: __init__
        -------------------
        This method initializes the object
        '''
        self.QBasis = QBasis
        self.QOrder = QOrder
        self.nElem = nElem
        self.nFacePerElem = Shape2nFace[Basis.Basis2Shape[BasisType.SegLagrange]] 
        self.Faces = None
        self.nNodePerElem = Shape2nNode[Basis.Basis2Shape[BasisType.SegLagrange]]
        self.Elem2Nodes = None

    def SetParams(self,QBasis=BasisType.SegLagrange,QOrder=1,nElem=1):
        self.QBasis = QBasis
        self.QOrder = QOrder
        self.nElem = nElem
        self.nFacePerElem = Shape2nFace[Basis.Basis2Shape[BasisType.SegLagrange]] 
        self.nNodePerElem = Shape2nNode[Basis.Basis2Shape[BasisType.SegLagrange]]

    def AllocFaces(self):
        '''
        Method: AllocFaces
        -------------------
        This method allocates the list of Face objects

        OUTPUTS:
            self.Faces
        '''
        self.Faces = [[Face() for j in range(self.nFacePerElem)] for i in range(self.nElem)]

    def AllocElem2Nodes(self):
        '''
        Method: AllocElem2Nodes
        -------------------
        This method allocates Elem2Nodes

        OUTPUTS:
            self.Elem2Nodes
        '''
        self.Elem2Nodes = np.zeros([self.nElem,self.nNodePerElem], dtype=int)


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
        BFGTitles: list of BFaceGroup titles (for easy access)
        nElemGroup: number of element groups
        ElemGroups: list of element groups
        nElems: list of number of elements in each element group (for easy access)
        nElemTot: total number of elements in mesh
        nPeriodicGroup: number of periodic groups
        PeriodicGroups: list of periodic groups
    '''
    def __init__(self,dim=1,nNode=1,nElemGroup=1):
        '''
        Method: __init__
        -------------------
        This method initializes the object

        INPUTS:
            dim: dimension of mesh
            nNode: total number of nodes
            nElemGroup: number of element groups
        '''
        self.Dim = dim
        self.nNode = nNode
        self.Coords = None
        self.nIFace = 0
        self.IFaces = None
        self.nBFaceGroup = 0
        self.BFaceGroups = None
        self.BFGTitles = []
        self.nElemGroup = nElemGroup
        self.ElemGroups = None
        self.nElems = None
        self.nElemTot = 0
        self.nPeriodicGroup = 0
        self.PeriodicGroups = None

    def Finalize(self):
        '''
        Method: Finalize
        -------------------
        This method creates some final mesh structures

        OUTPUTS:
            self.nElems
            self.nElemTot
            self.BFGTitles
        '''
        self.nElems = [self.ElemGroups[i].nElem for i in range(self.nElemGroup)]
        self.nElemTot = sum(self.nElems)

        for i in range(self.nBFaceGroup):
            self.BFGTitles.append(self.BFaceGroups[i].Title)


    def AllocIFaces(self):
        '''
        Method: AllocIFaces
        -------------------
        This method allocates IFaces

        OUTPUTS:
            self.IFaces
        '''
        self.IFaces = [IFace() for i in range(self.nIFace)]

    def AllocBFaceGroups(self):
        '''
        Method: AllocBFaceGroups
        -------------------
        This method allocates BFaceGroups

        OUTPUTS:
            self.BFaceGroups
        '''
        self.BFaceGroups = [BFaceGroup() for i in range(self.nBFaceGroup)]

    def AllocElemGroups(self):
        '''
        Method: AllocElemGroups
        -------------------
        This method allocates the list of element groups

        OUTPUTS:
            self.ElemGroups
        '''
        self.ElemGroups = [ElemGroup() for i in range(self.nElemGroup)]


