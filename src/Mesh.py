import numpy as np
from enum import IntEnum
from General import *
import Basis
import code


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


def Ref2Phys(mesh, egrp, elem, PhiData, npoint, xref, xphys=None, PointsChanged=False):
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
    if PhiData is None:
        PhiData = Basis.BasisData(QBasis,QOrder,npoint,mesh)
        PointsChanged = True
    if PointsChanged or PhiData.Basis != QBasis or PhiData.Order != QOrder:
        PhiData.EvalBasis(xref, Get_Phi=True)
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

    return xphys, PhiData


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
        if face == 0: xelem[0] = -1.
        elif face == 1: xelem[0] = 1.
        else: raise ValueError
    elif Shape == ShapeType.Quadrilateral:
        if xelem is None: xelem = np.zeros([nq,2])
        # local q = 1 nodes on face
        fnodes, nfnode = Basis.LocalQ1FaceNodes(BasisType.QuadLagrange, 1, face)
        # swap for reversed faces
        # if face >= 2: fnodes = fnodes[[1,0]]
        # coordinates of local q = 1 nodes on face
        x0 = Basis.RefQ1Coords[BasisType.QuadLagrange][fnodes[0]]
        x1 = Basis.RefQ1Coords[BasisType.QuadLagrange][fnodes[1]]
        for i in range(nq):
            if face == 0:
                xelem[i,0] = (xface[i]*x1[0] - xface[i]*x0[0])/2.
                xelem[i,1] = -1.
            elif face == 1:
                xelem[i,1] = (xface[i]*x1[1] - xface[i]*x0[1])/2.
                xelem[i,0] = 1.
            elif face == 2:
                xelem[i,0] = (xface[i]*x1[0] - xface[i]*x0[0])/2.
                xelem[i,1] = 1.
            else:
                xelem[i,1] = (xface[i]*x1[1] - xface[i]*x0[1])/2.
                xelem[i,0] = -1.
        #code.interact(local=locals())
    elif Shape == ShapeType.Triangle:
        if xelem is None: xelem = np.zeros([nq,2])
        xf = np.zeros(nq)
        xf = xf.reshape((nq,1))
        # local q = 1 nodes on face
        fnodes, nfnode = Basis.LocalQ1FaceNodes(BasisType.TriLagrange, 1, face)
        # coordinates of local q = 1 nodes on face
        x0 = Basis.RefQ1Coords[BasisType.TriLagrange][fnodes[0]]
        x1 = Basis.RefQ1Coords[BasisType.TriLagrange][fnodes[1]]
        for i in range(nq):
            xf[i] = (xface[i] + 1.)/2.
            xelem[i,:] = (1. - xf[i])*x0 + xf[i]*x1
    else:
        raise NotImplementedError

    return xelem


def IFaceNormal(mesh, IFace, nq, xq, NData=None):
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

    if NData is None: 
        NData = NormalData()

    if QOrderL <= QOrderR:
        NData.CalculateNormals(mesh, egrpL, elemL, IFace.faceL, nq, xq)
    else:
        NData.CalculateNormals(mesh, egrpR, elemR, IFace.faceR, nq, xq)
        NData.nvec *= -1.

    return NData


def BFaceNormal(mesh, BFace, nq, xq, NData=None):
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

    if NData is None:
        NData = NormalData()

    NData.CalculateNormals(mesh, egrp, elem, BFace.face, nq, xq)

    return NData


class NormalData(object):
    '''
    Class: NormalData
    -------------------
    This class contains information about normal vectors

    ATTRIBUTES:
        nq: number of points at which normals are calculated
        nvec: normals [nq, dim]
        fnodes: for easy storage of local face nodes
    '''
    def __init__(self):
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
        self.nq = 0
        self.nvec = None
        self.fnodes = None
        self.GPhi = None
        self.x_s = None

    def CalculateNormals(self, mesh, egrp, elem, face, nq, xq):

        EG = mesh.ElemGroups[egrp]
        QBasis = EG.QBasis
        Shape = Basis.Basis2Shape[QBasis]
        QOrder = EG.QOrder

        if QOrder == 1:
            nq = 1

        if self.nvec is None or self.nvec.shape != (nq,mesh.Dim):
            self.nvec = np.zeros([nq,mesh.Dim])
        
        nvec = self.nvec
        self.nq = nq

        #1D normals calculation
        if Shape == ShapeType.Segment:
            if face == 0:
                nvec[0] = -1.
            elif face == 1:
                nvec[0] = 1.
            else:
                raise ValueError
        #2D normals calculation
        elif Shape == ShapeType.Quadrilateral or Shape == ShapeType.Triangle:
            ElemNodes = EG.Elem2Nodes[elem]
            if QOrder == 1:
                self.fnodes, nfnode = Basis.LocalQ1FaceNodes(QBasis, QOrder, face, self.fnodes)
                x0 = mesh.Coords[ElemNodes[self.fnodes[0]]]
                x1 = mesh.Coords[ElemNodes[self.fnodes[1]]]

                nvec[0,0] =  (x1[1]-x0[1])/2.;
                nvec[0,1] = -(x1[0]-x0[0])/2.;
            # Calculate normals for curved meshes (quads)
            else:
                if self.x_s is None or self.x_s.shape != self.nvec.shape:
                    self.x_s = np.zeros_like(self.nvec)
                x_s = self.x_s
                self.fnodes, nfnode = Basis.LocalFaceNodes(QBasis, QOrder, face, self.fnodes)
                self.GPhi = Basis.GetGrads(BasisType.SegLagrange, QOrder, 1, nq, xq, self.GPhi)
                Coords = mesh.Coords[ElemNodes[self.fnodes]]

                # Face Jacobian (gradient of (x,y) w.r.t reference coordinate)
                x_s[:] = np.matmul(Coords.transpose(), self.GPhi).reshape(x_s.shape)
                nvec[:,0] = x_s[:,1]
                nvec[:,1] = -x_s[:,0]

                # for iq in range(nq):
                #     GPhi = self.GPhi[iq]

                #     # Face Jacobian (gradient of (x,y) w.r.t reference coordinate)
                #     x_s = np.matmul(Coords.transpose(), GPhi)

                #     # Cross product between tangent vector and out-of-plane vector
                #     nvec[iq,0] = x_s[1]
                #     nvec[iq,1] = -x_s[0]
        else:
            raise NotImplementedError



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
        self.Group = -1
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
        self.Name = "" 
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
        Faces: list of Face objects
        nNodePerElem: number of nodes per element
        Elem2Nodes: element-to-global-node mapping
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
        self.nFacePerElem = Shape2nFace[Basis.Basis2Shape[QBasis]] 
        self.Faces = None
        self.nNodePerElem = Basis.Order2nNode(QBasis, QOrder)
        self.Elem2Nodes = None
            # Elem2Nodes[elem][i] = ith node of elem, where i = 1,2,...,nNodePerElem

    def SetParams(self,QBasis=BasisType.SegLagrange,QOrder=1,nElem=1):
        self.QBasis = QBasis
        self.QOrder = QOrder
        self.nElem = nElem
        self.nFacePerElem = Shape2nFace[Basis.Basis2Shape[QBasis]] 
        self.nNodePerElem = Basis.Order2nNode(QBasis, QOrder)

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
        BFGNames: list of BFaceGroup names (for easy access)
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
        self.IFaces = []
        self.nBFaceGroup = 0
        self.BFaceGroups = []
        self.BFGNames = []
        self.nElemGroup = nElemGroup
        self.ElemGroups = []
        self.nElems = None
        self.nElemTot = 0
        self.nPeriodicGroup = 0
        self.PeriodicGroups = []

    def AllocHelpers(self):
        '''
        Method: AllocHelpers
        -------------------
        This method creates some helper mesh structures

        OUTPUTS:
            self.nElems
            self.nElemTot
            self.BFGNames
        '''
        self.nElems = [self.ElemGroups[i].nElem for i in range(self.nElemGroup)]
        self.nElemTot = sum(self.nElems)

        for i in range(self.nBFaceGroup):
            self.BFGNames.append(self.BFaceGroups[i].Name)


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

    def FillFaces(self):
        for iiface in range(self.nIFace):
            IFace = self.IFaces[iiface]
            egrpL = IFace.ElemGroupL
            egrpR = IFace.ElemGroupR
            elemL = IFace.ElemL
            elemR = IFace.ElemR
            faceL = IFace.faceL
            faceR = IFace.faceR

            FaceL = self.ElemGroups[egrpL].Faces[elemL][faceL]
            FaceR = self.ElemGroups[egrpR].Faces[elemR][faceR]

            FaceL.Type = FaceType.Interior
            FaceR.Type = FaceType.Interior

            FaceL.Number = iiface
            FaceR.Number = iiface

        for ibfgrp in range(self.nBFaceGroup):
            BFG = self.BFaceGroups[ibfgrp]
            
            for ibface in range(BFG.nBFace):
                BFace = BFG.BFaces[ibface]
                egrp = BFace.ElemGroup
                elem = BFace.Elem
                face = BFace.face

                Face = self.ElemGroups[egrp].Faces[elem][face]

                Face.Type = FaceType.Boundary
                Face.Number = ibface
                Face.Group = ibfgrp








