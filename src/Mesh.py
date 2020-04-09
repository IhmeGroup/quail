import numpy as np
from enum import IntEnum
from General import *
import Basis
import code


def get_entity_dim(mesh, entity):
    '''
    Function: get_entity_dim
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


def ref_to_phys(mesh, elem, PhiData, xref, xphys=None, PointsChanged=False):
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
    QBasis = mesh.QBasis
    QOrder = mesh.QOrder

    npoint = xref.shape[0]

    if PhiData is None:
        PhiData = Basis.BasisData(QBasis,QOrder,mesh)
        PointsChanged = True
    if PointsChanged or PhiData.basis != QBasis or PhiData.order != QOrder:
        PhiData.eval_basis(xref, Get_Phi=True)

    dim = mesh.Dim
    Coords = mesh.Coords
    Phi = PhiData.Phi
    nb = PhiData.Phi.shape[1]
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

def ref_to_phys_time(mesh, elem, time, dt, PhiData, xref, tphys=None, PointsChanged=False):
    '''
    Function: ref_to_phys_time
    ------------------------------
    This function converts reference time coordinates to physical
    time coordinates

    INPUTS:
        mesh: Mesh object
        elem: element 
        PhiData: basis data
        npoint: number of coordinates to convert
        xref: coordinates in reference space
        tphys: pre-allocated storage for physical time coordinates (optional) 

    OUTPUTS:
        tphys: coordinates in temporal space
    '''
    QBasis = BasisType.LagrangeQuad
    QOrder = mesh.QOrder

    npoint = xref.shape[0]

    if PhiData is None:
        PhiData = Basis.BasisData(QBasis,QOrder,mesh)
        PointsChanged = True
    if PointsChanged or PhiData.basis != QBasis or PhiData.order != QOrder:
        PhiData.eval_basis(xref, Get_Phi=True)

    dim = mesh.Dim
    
    Phi = PhiData.Phi

    if tphys is None:
        tphys = np.zeros([npoint,dim])
    else:
        tphys[:] = time
    for ipoint in range(npoint):
        #for n in range(nn):
            #nodeNum = ElemNodes[n]
            #val = Phi[ipoint][n]
            #for d in range(dim):
        tphys[ipoint] = (time/2.)*(1-xref[ipoint,dim])+((time+dt)/2.0)*(1+xref[ipoint,dim])

    # tphys = (time/2.)*(1-xref)+((time+dt)/2.0)*(1+xref)

    return tphys, PhiData

def ref_face_to_elem(Shape, face, nq, xface, xelem=None):
    '''
    Function: ref_face_to_elem
    ----------------------------
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
        fnodes, nfnode = Basis.local_q1_face_nodes(BasisType.LagrangeQuad, 1, face)
        # swap for reversed faces
        # if face >= 2: fnodes = fnodes[[1,0]]
        # coordinates of local q = 1 nodes on face
        x0 = Basis.RefQ1Coords[BasisType.LagrangeQuad][fnodes[0]]
        x1 = Basis.RefQ1Coords[BasisType.LagrangeQuad][fnodes[1]]
        # for i in range(nq):
        #     if face == 0:
        #         xelem[i,0] = (xface[i]*x1[0] - xface[i]*x0[0])/2.
        #         xelem[i,1] = -1.
        #     elif face == 1:
        #         xelem[i,1] = (xface[i]*x1[1] - xface[i]*x0[1])/2.
        #         xelem[i,0] = 1.
        #     elif face == 2:
        #         xelem[i,0] = (xface[i]*x1[0] - xface[i]*x0[0])/2.
        #         xelem[i,1] = 1.
        #     else:
        #         xelem[i,1] = (xface[i]*x1[1] - xface[i]*x0[1])/2.
        #         xelem[i,0] = -1.
        if face == 0:
            xelem[:,0] = np.reshape((xface*x1[0] - xface*x0[0])/2., nq)
            xelem[:,1] = -1.
        elif face == 1:
            xelem[:,1] = np.reshape((xface*x1[1] - xface*x0[1])/2., nq)
            xelem[:,0] = 1.
        elif face == 2:
            xelem[:,0] = np.reshape((xface*x1[0] - xface*x0[0])/2., nq)
            xelem[:,1] = 1.
        else:
            xelem[:,1] = np.reshape((xface*x1[1] - xface*x0[1])/2., nq)
            xelem[:,0] = -1.
        #code.interact(local=locals())
    elif Shape == ShapeType.Triangle:
        if xelem is None: xelem = np.zeros([nq,2])
        xf = np.zeros(nq)
        xf = xf.reshape((nq,1))
        # local q = 1 nodes on face
        fnodes, nfnode = Basis.local_q1_face_nodes(BasisType.LagrangeTri, 1, face)
        # coordinates of local q = 1 nodes on face
        x0 = Basis.RefQ1Coords[BasisType.LagrangeTri][fnodes[0]]
        x1 = Basis.RefQ1Coords[BasisType.LagrangeTri][fnodes[1]]
        # for i in range(nq):
        #     xf[i] = (xface[i] + 1.)/2.
        #     xelem[i,:] = (1. - xf[i])*x0 + xf[i]*x1
        xf = (xface + 1.)/2.
        xelem[:] = (1. - xf)*x0 + xf*x1
    else:
        raise NotImplementedError

    return xelem


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
    QOrderL = mesh.QOrder
    QOrderR = mesh.QOrder

    nq = quad_pts.shape[0]

    if NData is None: 
        NData = NormalData()

    if QOrderL <= QOrderR:
        NData.calculate_normals(mesh, elemL, IFace.faceL, quad_pts)
    else:
        NData.calculate_normals(mesh, elemR, IFace.faceR, quad_pts)
        NData.nvec *= -1.

    return NData


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
    QOrder = mesh.QOrder

    nq = quad_pts.shape[0]

    if NData is None:
        NData = NormalData()

    NData.calculate_normals(mesh, elem, BFace.face, quad_pts)

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
        GPhi: evaluated gradient of basis
        x_s: ???
    '''
    def __init__(self):
        '''
        Method: __init__
        -------------------
        This method initializes the object
        '''
        self.nq = 0
        self.nvec = None
        self.fnodes = None
        self.GPhi = None
        self.x_s = None

    def calculate_normals(self, mesh, elem, face, quad_pts):
        '''
        Method: calculate_normals
        -------------------
        Calculate the normals

        INPUTS:
            mesh: Mesh object
            elem: element index
            face: face index
            nq: number of points at which to calculate normals
            quad_pts: points in reference space at which to calculate normals
        '''

        QBasis = mesh.QBasis
        Shape = Basis.Basis2Shape[QBasis]
        QOrder = mesh.QOrder

        nq = quad_pts.shape[0]

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
            ElemNodes = mesh.Elem2Nodes[elem]
            if QOrder == 1:
                self.fnodes, nfnode = Basis.local_q1_face_nodes(QBasis, QOrder, face, self.fnodes)
                x0 = mesh.Coords[ElemNodes[self.fnodes[0]]]
                x1 = mesh.Coords[ElemNodes[self.fnodes[1]]]

                nvec[0,0] =  (x1[1]-x0[1])/2.;
                nvec[0,1] = -(x1[0]-x0[0])/2.;
            # Calculate normals for curved meshes (quads)
            else:
                if self.x_s is None or self.x_s.shape != self.nvec.shape:
                    self.x_s = np.zeros_like(self.nvec)
                x_s = self.x_s
                self.fnodes, nfnode = Basis.local_face_nodes(QBasis, QOrder, face, self.fnodes)
                self.GPhi = Basis.get_grads(BasisType.LagrangeSeg, QOrder, 1, quad_pts, self.GPhi)
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
        self.Name = "" 
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


# class ElemGroup(object):
#     '''
#     Class: ElemGroup
#     -------------------
#     This class stores information about a given element group

#     ATTRIBUTES:
#         QBasis: Basis used for geometry representation 
#         QOrder: Order used for geometry representation
#         nElem: number of elements
#         nFacePerElem: number of faces per element
#         Faces: list of Face objects
#         nNodePerElem: number of nodes per element
#         Elem2Nodes: element-to-global-node mapping
#     '''
#     def __init__(self,QBasis=BasisType.LagrangeSeg,QOrder=1,nElem=1):
#         '''
#         Method: __init__
#         -------------------
#         This method initializes the object
#         '''
#         self.QBasis = QBasis
#         self.QOrder = QOrder
#         self.nElem = nElem
#         self.nFacePerElem = Shape2nFace[Basis.Basis2Shape[QBasis]] 
#         self.Faces = None
#         self.nNodePerElem = Basis.order_to_num_basis_coeff(QBasis, QOrder)
#         self.Elem2Nodes = None
#             # Elem2Nodes[elem][i] = ith node of elem, where i = 1,2,...,nNodePerElem

#     def SetParams(self,QBasis=BasisType.LagrangeSeg,QOrder=1,nElem=1):
#         self.QBasis = QBasis
#         self.QOrder = QOrder
#         self.nElem = nElem
#         self.nFacePerElem = Shape2nFace[Basis.Basis2Shape[QBasis]] 
#         self.nNodePerElem = Basis.order_to_num_basis_coeff(QBasis, QOrder)

#     def allocate_faces(self):
#         '''
#         Method: allocate_faces
#         -------------------
#         This method allocates the list of Face objects

#         OUTPUTS:
#             self.Faces
#         '''
#         self.Faces = [[Face() for j in range(self.nFacePerElem)] for i in range(self.nElem)]

#     def allocate_elem_to_nodes(self):
#         '''
#         Method: allocate_elem_to_nodes
#         -------------------
#         This method allocates Elem2Nodes

#         OUTPUTS:
#             self.Elem2Nodes
#         '''
#         self.Elem2Nodes = np.zeros([self.nElem,self.nNodePerElem], dtype=int)


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
        nElems: list of number of elements in each element group (for easy access)
        nElemTot: total number of elements in mesh
        nPeriodicGroup: number of periodic groups
        PeriodicGroups: list of periodic groups
    '''
    def __init__(self,dim=1,nNode=1,nElem=1,QBasis=BasisType.LagrangeSeg,QOrder=1):
        '''
        Method: __init__
        -------------------
        This method initializes the object

        INPUTS:
            dim: dimension of mesh
            nNode: total number of nodes
        '''
        self.Dim = dim
        self.nNode = nNode
        self.Coords = None
        self.nIFace = 0
        self.IFaces = []
        self.nBFaceGroup = 0
        self.BFaceGroups = []
        self.BFGNames = []
        self.QBasis = QBasis
        self.QOrder = QOrder
        self.nElem = nElem
        self.nFacePerElem = Shape2nFace[Basis.Basis2Shape[QBasis]] 
        self.Faces = None
        self.nNodePerElem = Basis.order_to_num_basis_coeff(QBasis, QOrder)
        self.Elem2Nodes = None
            # Elem2Nodes[elem][i] = ith node of elem, where i = 1,2,...,nNodePerElem

    def SetParams(self,QBasis=BasisType.LagrangeSeg,QOrder=1,nElem=1):
        self.QBasis = QBasis
        self.QOrder = QOrder
        self.nElem = nElem
        self.nFacePerElem = Shape2nFace[Basis.Basis2Shape[QBasis]] 
        self.nNodePerElem = Basis.order_to_num_basis_coeff(QBasis, QOrder)

    def allocate_faces(self):
        '''
        Method: allocate_faces
        -----------------------
        This method allocates the list of Face objects

        OUTPUTS:
            self.Faces
        '''
        self.Faces = [[Face() for j in range(self.nFacePerElem)] for i in range(self.nElem)]

    def allocate_elem_to_nodes(self):
        '''
        Method: allocate_elem_to_nodes
        -------------------
        This method allocates Elem2Nodes

        OUTPUTS:
            self.Elem2Nodes
        '''
        self.Elem2Nodes = np.zeros([self.nElem,self.nNodePerElem], dtype=int)

    def allocate_helpers(self):
        '''
        Method: allocate_helpers
        -------------------
        This method creates some helper mesh structures

        OUTPUTS:
            self.nElems
            self.nElemTot
            self.BFGNames
        '''

        for i in range(self.nBFaceGroup):
            self.BFGNames.append(self.BFaceGroups[i].Name)


    def allocate_ifaces(self):
        '''
        Method: allocate_ifaces
        -------------------
        This method allocates IFaces

        OUTPUTS:
            self.IFaces
        '''
        self.IFaces = [IFace() for i in range(self.nIFace)]

    def allocate_bface_groups(self):
        '''
        Method: allocate_bface_groups
        -------------------
        This method allocates BFaceGroups

        OUTPUTS:
            self.BFaceGroups
        '''
        self.BFaceGroups = [BFaceGroup() for i in range(self.nBFaceGroup)]

    def fill_faces(self):
        for iiface in range(self.nIFace):
            IFace = self.IFaces[iiface]
            elemL = IFace.ElemL
            elemR = IFace.ElemR
            faceL = IFace.faceL
            faceR = IFace.faceR

            FaceL = self.Faces[elemL][faceL]
            FaceR = self.Faces[elemR][faceR]

            FaceL.Type = FaceType.Interior
            FaceR.Type = FaceType.Interior

            FaceL.Number = iiface
            FaceR.Number = iiface

        for ibfgrp in range(self.nBFaceGroup):
            BFG = self.BFaceGroups[ibfgrp]
            
            for ibface in range(BFG.nBFace):
                BFace = BFG.BFaces[ibface]
                elem = BFace.Elem
                face = BFace.face

                Face = self.Faces[elem][face]

                Face.Type = FaceType.Boundary
                Face.Number = ibface
                Face.Group = ibfgrp








