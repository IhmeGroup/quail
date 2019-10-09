import numpy as np
from enum import IntEnum
from General import *
import Basis


def GetEntityDim(entity, mesh):
    if entity == EntityType.Element:
        dim = mesh.Dim 
    else:
        dim = mesh.Dim - 1

    return dim


def Ref2Phys(mesh, egrp, elem, PhiData, npoint, xref, xphys=None):
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

def RefFace2Elem(Shape, face, nq, xface, xelem):
    if Shape == ShapeType.Segment:
        if xelem is None: xelem = np.zeros([1,1])
        if face == 0: xelem[0] = 0.
        elif face == 1: xelem[0] = 1.
        else: raise Exception("Face error")
    else:
        raise Exception("Shape error")

    return xelem


def IFaceNormal(mesh, IFace, nq, xq):
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
    egrp = BFace.ElemGroup
    elem = BFace.Elem
    QOrder = mesh.ElemGroups[egrp].QOrder

    NData = NormalData(mesh, egrp, elem, BFace.face, nq, xq)

    return NData


class NormalData(object):
    def __init__(self, mesh, egrp, elem, face, nq, xq):
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
    INTERIOR = 0
    BOUNDARY = 1


class Face(object):
    '''
    Class: Face
    --------------------------------------------------------------------------
    This is a class defined to encapsulate the temperature table with the 
    relevant methods
    '''
    def __init__(self):
        '''
        Method: __init__
        --------------------------------------------------------------------------
        This method initializes the temperature table. The table uses a
        piecewise linear function for the constant pressure specific heat 
        coefficients. The coefficients are selected to retain the exact 
        enthalpies at the table points.
        '''
        self.Type = INTERIORFACE
        self.Number = 0 # Global number of face of given type


class IFace(object):
    '''
    Class: IFace
    --------------------------------------------------------------------------
    This is a class defined to encapsulate the temperature table with the 
    relevant methods
    '''
    def __init__(self):
        '''
        Method: __init__
        --------------------------------------------------------------------------
        This method initializes the temperature table. The table uses a
        piecewise linear function for the constant pressure specific heat 
        coefficients. The coefficients are selected to retain the exact 
        enthalpies at the table points.
        '''
        self.ElemGroupL = 0
        self.ElemL = 0 # Global element number on left
        self.faceL = 0 # Local face number from ElemL's perspective
        self.ElemGroupR = 0
        self.ElemR = 0 # Global element number on right
        self.faceR = 0 # Local face number from ElemR's perspective


class BFace(object):
    '''
    Class: IFace
    --------------------------------------------------------------------------
    This is a class defined to encapsulate the temperature table with the 
    relevant methods
    '''
    def __init__(self):
        '''
        Method: __init__
        --------------------------------------------------------------------------
        This method initializes the temperature table. The table uses a
        piecewise linear function for the constant pressure specific heat 
        coefficients. The coefficients are selected to retain the exact 
        enthalpies at the table points.
        '''
        self.ElemGroup = 0
        self.Elem = 0 # Global element number 
        self.face = 0 # Local face number from Elem's perspective


class BFaceGroup(object):
    '''
    Class: IFace
    --------------------------------------------------------------------------
    This is a class defined to encapsulate the temperature table with the 
    relevant methods
    '''
    def __init__(self):
        '''
        Method: __init__
        --------------------------------------------------------------------------
        This method initializes the temperature table. The table uses a
        piecewise linear function for the constant pressure specific heat 
        coefficients. The coefficients are selected to retain the exact 
        enthalpies at the table points.
        '''
        self.Title = "" 
        self.nBFace = 0 
        self.BFaces = None

    def CreateBFaces(self):
        self.BFaces = [BFace() for i in range(self.nBFace)]


Shape2nFace = {
    ShapeType.Point : 0,
    ShapeType.Segment : 2,
    ShapeType.Triangle : 3,
    ShapeType.Quadrilateral : 4
}

Shape2nNode = {
    ShapeType.Point : 1,
    ShapeType.Segment : 2,
    ShapeType.Triangle : 3,
    ShapeType.Quadrilateral : 4
}


class ElemGroup(object):
    '''
    Class: IFace
    --------------------------------------------------------------------------
    This is a class defined to encapsulate the temperature table with the 
    relevant methods
    '''
    def __init__(self,QBasis=BasisType.SegLagrange,QOrder=1,nElem=1):
        '''
        Method: __init__
        --------------------------------------------------------------------------
        This method initializes the temperature table. The table uses a
        piecewise linear function for the constant pressure specific heat 
        coefficients. The coefficients are selected to retain the exact 
        enthalpies at the table points.
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

    def CreateFaces(self):
        self.Faces = [[Face() for j in range(self.nFacePerElem)] for i in range(self.nElem)]

    def CreateElem2Nodes(self):
        self.Elem2Nodes = np.zeros([self.nElem,self.nNodePerElem], dtype=int)


class PeriodicGroup(object):
    '''
    Class: IFace
    --------------------------------------------------------------------------
    This is a class defined to encapsulate the temperature table with the 
    relevant methods
    '''
    def __init__(self):
        '''
        Method: __init__
        --------------------------------------------------------------------------
        This method initializes the temperature table. The table uses a
        piecewise linear function for the constant pressure specific heat 
        coefficients. The coefficients are selected to retain the exact 
        enthalpies at the table points.
        '''
        self.nPeriodicNode = 0
        self.PeriodicNodes = None


class Mesh(object):
    '''
    Class: IFace
    --------------------------------------------------------------------------
    This is a class defined to encapsulate the temperature table with the 
    relevant methods
    '''
    def __init__(self,dim=1,nNode=1,nElemGroup=1):
        '''
        Method: __init__
        --------------------------------------------------------------------------
        This method initializes the temperature table. The table uses a
        piecewise linear function for the constant pressure specific heat 
        coefficients. The coefficients are selected to retain the exact 
        enthalpies at the table points.
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
        self.nElems = [self.ElemGroups[i].nElem for i in range(self.nElemGroup)]
        self.nElemTot = sum(self.nElems)

        for i in range(self.nBFaceGroup):
            self.BFGTitles.append(self.BFaceGroups[i].Title)


    def CreateIFaces(self):
        self.IFaces = [IFace() for i in range(self.nIFace)]

    def CreateBFaceGroups(self):
        self.BFaceGroups = [BFaceGroup() for i in range(self.nBFaceGroup)]

    def CreateElemGroups(self):
        self.ElemGroups = [ElemGroup() for i in range(self.nElemGroup)]


