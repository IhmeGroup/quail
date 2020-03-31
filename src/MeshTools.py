import numpy as np 
import Data
import Quadrature
from Basis import *
from Mesh import *
import code


def ElementVolumes(mesh, solver=None):
    # Check if already calculated
    if solver is not None:
        if hasattr(solver.DataSet, "TotalVolume") \
            and hasattr(solver.DataSet, "ElemVolumes"):
                return solver.DataSet.TotalVolume, solver.DataSet.ElemVolumes

    # ElemVol = ArrayList(SimilarArray=EqnSet.U).Arrays
    # ElemVolumes = Data.ArrayList(nArray=mesh.nElemGroup,ArrayDims=[mesh.nElems])
    ElemVolumes = np.zeros(mesh.nElem)
    TotalVolume = 0.
    quadData = None
    JData = JacobianData(mesh)

    Order = mesh.QOrder

    QuadOrder,QuadChanged = Quadrature.get_gaussian_quadrature_elem(mesh, mesh.QBasis, Order, 
        quadData=quadData)
    if QuadChanged:
        quadData = Quadrature.QuadData(mesh, mesh.QBasis, EntityType.Element, QuadOrder)

    nq = quadData.nquad
    xq = quadData.quad_pts
    wq = quadData.quad_wts

    for elem in range(mesh.nElem):
        JData.ElemJacobian(elem,nq,xq,mesh,get_djac=True)

        for iq in range(nq):
            ElemVolumes[elem] += wq[iq] * JData.djac[iq*(JData.nq != 1)]

        TotalVolume += ElemVolumes[elem]

    if solver is not None:
        solver.DataSet.TotalVolume = TotalVolume
        solver.DataSet.ElemVolumes = ElemVolumes

    return TotalVolume, ElemVolumes


def NeighborAcrossFace(mesh, elem, face):
    Face = mesh.Faces[elem][face]

    if Face.Type == FaceType.Interior:
        iiface = Face.Number
        eN  = mesh.IFaces[iiface].ElemR
        faceN = mesh.IFaces[iiface].faceR

        if eN == elem:
            eN  = mesh.IFaces[iiface].ElemL
            faceN = mesh.IFaces[iiface].faceL
    else:
        eN    = -1
        faceN = -1

    return eN, faceN


def CheckFaceOrientations(mesh):

    if mesh.Dim == 1:
        # don't need to check for 1D
        return

    for IFace in mesh.IFaces:
        elemL = IFace.ElemL
        elemR = IFace.ElemR
        faceL = IFace.faceL
        faceR = IFace.faceR

        # Get local q=1 nodes on face for left element
        lfnodes, nfnode = LocalQ1FaceNodes(mesh.QBasis, mesh.QOrder, faceL)
        # Convert to global node numbering
        gfnodesL = mesh.Elem2Nodes[elemL][lfnodes]

        # Get local q=1 nodes on face for right element
        lfnodes, nfnode = LocalQ1FaceNodes(mesh.QBasis, mesh.QOrder, faceR)
        # Convert to global node numbering
        gfnodesR = mesh.Elem2Nodes[elemR][lfnodes]

        # Node Ordering should be reversed between the two elements
        if not np.all(gfnodesL == gfnodesR[::-1]):
            raise Exception("Face orientation for elemL = %d, elemR = %d \\ is incorrect"
                % (elemL, elemR))









