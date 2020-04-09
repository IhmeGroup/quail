import numpy as np 
import Data
import Quadrature
from Basis import *
from Mesh import *
import code


def element_volumes(mesh, solver=None):
    '''
    Method: element_volumes
    --------------------------
    Calculates total and per element volumes

    INPUTS:
        mesh: mesh object
        solver: type of solver (i.e. DG, ADER-DG, etc...)
    
    OUTPUTS:
        TotalVolume: total volume in the mesh
        ElemVolumes: volume at each element
    '''
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

    xq = quadData.quad_pts
    wq = quadData.quad_wts
    nq = xq.shape[0]

    for elem in range(mesh.nElem):
        JData.element_jacobian(mesh,elem,xq,get_djac=True)

        # for iq in range(nq):
        #     ElemVolumes[elem] += wq[iq] * JData.djac[iq*(JData.nq != 1)]
        ElemVolumes[elem] = np.sum(wq*JData.djac)

        TotalVolume += ElemVolumes[elem]

    if solver is not None:
        solver.DataSet.TotalVolume = TotalVolume
        solver.DataSet.ElemVolumes = ElemVolumes

    return TotalVolume, ElemVolumes


def neighbor_across_face(mesh, elem, face):
    '''
    Method: neighbor_across_face
    ------------------------------
    Identifies neighbor elements across each face

    INPUTS:
        mesh: mesh object
        elem: element index
        face: face index w.r.t. the element in ref space
    
    OUTPUTS:
        eN: element index of the neighboring face
        faceN: face index w.r.t. the neighboring element in ref space
    '''
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


def check_face_orientations(mesh):
    '''
    Method: check_face_orientations
    --------------------------------
    Checks the face orientations for 2D meshes

    INPUTS:
        mesh: mesh object
    
    NOTES:
        only returns a message if an error exists
    '''

    if mesh.Dim == 1:
        # don't need to check for 1D
        return

    for IFace in mesh.IFaces:
        elemL = IFace.ElemL
        elemR = IFace.ElemR
        faceL = IFace.faceL
        faceR = IFace.faceR

        # Get local q=1 nodes on face for left element
        lfnodes, nfnode = local_q1_face_nodes(mesh.QBasis, mesh.QOrder, faceL)
        # Convert to global node numbering
        gfnodesL = mesh.Elem2Nodes[elemL][lfnodes]

        # Get local q=1 nodes on face for right element
        lfnodes, nfnode = local_q1_face_nodes(mesh.QBasis, mesh.QOrder, faceR)
        # Convert to global node numbering
        gfnodesR = mesh.Elem2Nodes[elemR][lfnodes]

        # Node Ordering should be reversed between the two elements
        if not np.all(gfnodesL == gfnodesR[::-1]):
            raise Exception("Face orientation for elemL = %d, elemR = %d \\ is incorrect"
                % (elemL, elemR))









