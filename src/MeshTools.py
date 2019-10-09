import numpy as np 
import Data
from Quadrature import *
from Basis import *
from Mesh import *
import code


def ElementVolumes(mesh, solver=None):
    # ElemVol = ArrayList(SimilarArray=EqnSet.U).Arrays
    ElemVol = Data.ArrayList(nArray=mesh.nElemGroup,ArrayDims=[mesh.nElems])
    TotVol = 0.
    quadData = None
    JData = JacobianData(mesh)
    for egrp in range(mesh.nElemGroup):

    	EGroup = mesh.ElemGroups[egrp]
        Order = EGroup.QOrder

        QuadOrder,QuadChanged = GetQuadOrderElem(egrp, Order, EGroup.QBasis, mesh, quadData=quadData)
        if QuadChanged:
            quadData = QuadData(QuadOrder, EntityType.Element, egrp, mesh)

        nq = quadData.nquad
        xq = quadData.xquad
        wq = quadData.wquad

        # PhiData = BasisData(egrp,Order,entity,nq,xq,mesh,True,True)
        # PhiData = BasisData(ShapeType.Segment,Order,nq,mesh)
        # PhiData.EvalBasis(xq, True, False, False, None)

        for elem in range(mesh.nElems[egrp]):
            JData.ElemJacobian(egrp,elem,nq,xq,mesh,Get_detJ=True)

            for iq in range(nq):
                ElemVol.Arrays[egrp][elem] += wq[iq] * JData.detJ[iq*(JData.nq != 1)]

            TotVol += ElemVol.Arrays[egrp][elem]

    if solver is not None:
        solver.DataSet.ElemVol = ElemVol
        solver.DataSet.TotVol = TotVol
    
    return TotVol, ElemVol