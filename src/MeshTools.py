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
    ElemVolumes = Data.ArrayList(nArray=mesh.nElemGroup,ArrayDims=[mesh.nElems])
    TotalVolume = 0.
    quadData = None
    JData = JacobianData(mesh)
    for egrp in range(mesh.nElemGroup):

        EGroup = mesh.ElemGroups[egrp]
        Order = EGroup.QOrder

        QuadOrder,QuadChanged = Quadrature.GetQuadOrderElem(mesh, egrp, EGroup.QBasis, Order, 
            quadData=quadData)
        if QuadChanged:
            quadData = Quadrature.QuadData(mesh, mesh.ElemGroups[egrp].QBasis, EntityType.Element, QuadOrder)

        nq = quadData.nquad
        xq = quadData.xquad
        wq = quadData.wquad

        # PhiData = BasisData(egrp,Order,entity,nq,xq,mesh,True,True)
        # PhiData = BasisData(ShapeType.Segment,Order,nq,mesh)
        # PhiData.EvalBasis(xq, True, False, False, None)

        for elem in range(mesh.nElems[egrp]):
            JData.ElemJacobian(egrp,elem,nq,xq,mesh,get_djac=True)

            for iq in range(nq):
                ElemVolumes.Arrays[egrp][elem] += wq[iq] * JData.djac[iq*(JData.nq != 1)]

            TotalVolume += ElemVolumes.Arrays[egrp][elem]

    if solver is not None:
        solver.DataSet.TotalVolume = TotalVolume
        solver.DataSet.ElemVolumes = ElemVolumes

    return TotalVolume, ElemVolumes


def NeighborAcrossFace(mesh, egrp, elem, face):
    Face = mesh.ElemGroups[egrp].Faces[elem][face]

    if Face.Type == FaceType.Interior:
        iiface = Face.Number
        egN = mesh.IFaces[iiface].ElemGroupR
        eN  = mesh.IFaces[iiface].ElemR
        faceN = mesh.IFaces[iiface].faceR

        if egN == egrp and eN == elem:
            egN = mesh.IFaces[iiface].ElemGroupL
            eN  = mesh.IFaces[iiface].ElemL
            faceN = mesh.IFaces[iiface].faceL
    else:
        egN   = -1
        eN    = -1
        faceN = -1

    return egN, eN, faceN


def CheckFaceOrientations(mesh):

    if mesh.Dim == 1:
        # don't need to check for 1D
        return

    for IFace in mesh.IFaces:
        egrpL = IFace.ElemGroupL
        egrpR = IFace.ElemGroupR
        elemL = IFace.ElemL
        elemR = IFace.ElemR
        faceL = IFace.faceL
        faceR = IFace.faceR
        EGL = mesh.ElemGroups[egrpL]
        EGR = mesh.ElemGroups[egrpR]

        # Get local q=1 nodes on face for left element
        lfnodes, nfnode = LocalQ1FaceNodes(EGL.QBasis, EGL.QOrder, faceL)
        # Convert to global node numbering
        gfnodesL = EGL.Elem2Nodes[elemL][lfnodes]

        # Get local q=1 nodes on face for right element
        lfnodes, nfnode = LocalQ1FaceNodes(EGR.QBasis, EGR.QOrder, faceR)
        # Convert to global node numbering
        gfnodesR = EGR.Elem2Nodes[elemR][lfnodes]

        # Node ordering should be reversed between the two elements
        if not np.all(gfnodesL == gfnodesR[::-1]):
            raise Exception("Face orientation for elemL = %d, elemR = %d \\ is incorrect"
                % (elemL, elemR))









