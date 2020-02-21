import numpy as np
from enum import IntEnum
from General import *
from Quadrature import *
from Math import *
import Mesh
import code
from Data import ArrayList, GenericData


Basis2Shape = {
    BasisType.SegLagrange : ShapeType.Segment,
    BasisType.QuadLagrange : ShapeType.Quadrilateral,
    BasisType.TriLagrange : ShapeType.Triangle,
    BasisType.SegLegendre : ShapeType.Segment,
    BasisType.QuadLegendre : ShapeType.Quadrilateral,
}


Shape2Dim = {
    ShapeType.Point : 0,
    ShapeType.Segment : 1,
    ShapeType.Quadrilateral : 2,
    ShapeType.Triangle : 2,
}


FaceShape = {
    ShapeType.Segment : ShapeType.Point,
    ShapeType.Quadrilateral : ShapeType.Segment,
    ShapeType.Triangle : ShapeType.Segment,
}


RefQ1Coords = {
    BasisType.SegLagrange : np.array([[-1.],[1.]]),
    BasisType.QuadLagrange : np.array([[-1.,-1.],[1.,-1.],
                                [-1.,1.],[1.,1.]]),
    BasisType.TriLagrange : np.array([[0.,0.],[1.,0.],
                                [0.,1.]]),
    BasisType.SegLegendre : np.array([[-1.],[1.]]),
    BasisType.QuadLegendre : np.array([[-1.,-1.],[1.,-1.],
                                [-1.,1.],[1.,1.]]),
}


def Order2nNode(basis, p):
    Shape = Basis2Shape[basis]
    if Shape == ShapeType.Point:
    	nn = 1
    elif Shape == ShapeType.Segment:
        nn = p + 1
    elif Shape == ShapeType.Triangle:
        nn = (p + 1)*(p + 2)/2
    elif Shape == ShapeType.Quadrilateral:
        nn = (p + 1)**2
    else:
    	raise Exception("Shape not supported")

    return nn


def LocalQ1FaceNodes(basis, p, face, fnodes=None):
    if basis == BasisType.SegLagrange: 
        nfnode = 1
        if fnodes is None: fnodes = np.zeros(nfnode, dtype=int)
        if face == 0:
            fnodes[0] = 0
        elif face == 1:
            fnodes[0] = p
        else:
            raise IndexError
    elif basis == BasisType.QuadLagrange:
        nfnode = 2
        if fnodes is None: fnodes = np.zeros(nfnode, dtype=int)
        if face == 0:
            fnodes[0] = 0; fnodes[1] = p
        elif face == 1:
            fnodes[0] = p; fnodes[1] = (p+2)*p
        elif face == 2:
            fnodes[0] = (p+2)*p; fnodes[1] = (p+1)*p
        elif face == 3:
            fnodes[0] = (p+1)*p; fnodes[1] = 0
        else:
             raise IndexError
    elif basis == BasisType.TriLagrange:
        nfnode = 2
        if fnodes is None: fnodes = np.zeros(nfnode, dtype=int)
        if face == 0:
            fnodes[0] = p; fnodes[1] = (p+1)*(p+2)/2-1
        elif face == 1:
            fnodes[0] = (p+1)*(p+2)/2-1; fnodes[1] = 0
        elif face == 2:
            fnodes[0] = 0; fnodes[1] = p
        else:
            raise IndexError
    else:
        raise NotImplementedError

    return fnodes, nfnode


def LocalFaceNodes(basis, p, face, fnodes=None):
    if basis == BasisType.QuadLagrange:
        nfnode = p+1
        if fnodes is None: fnodes = np.zeros(nfnode, dtype=int)
        if face == 0:
            i0 = 0;       d =    1
        elif face == 1:
            i0 = p;       d =  p+1
        elif face == 2:
            i0 = p*(p+2); d =   -1
        elif face == 3:
            i0 = p*(p+1); d = -p-1
        else:
             raise IndexError

        for i in range(p+1):
            fnodes[i] = i0+i*d
    elif basis == BasisType.TriLagrange:
        nfnode = p+1
        if fnodes is None: fnodes = np.zeros(nfnode, dtype=int)
        if face == 0:
            i0 = p; d0 = p; d1 = -1
        elif face == 1:
            i0 = (p+1)*(p+2)/2-1; d0 = -2; d1 = -1
        elif face == 2:
            i0 = 0;  d0 = 1; d1 = 0
        else:
            raise IndexError

        fnodes[0] = i0
        d = d0
        for i in range(1, p+1):
            fnodes[i] = fnodes[i-1] + d
            d += d1
    else:
        raise NotImplementedError

    return fnodes, nfnode


def EquidistantNodes(basis, p, xn=None):
    nn = Order2nNode(basis, p)

    dim = Shape2Dim[Basis2Shape[basis]]

    adim = nn,dim
    if xn is None or xn.shape != adim:
        xn = np.zeros(adim)

    if p == 0:
        xn[:] = 0.0 # 0.5
        return xn, nn

    if basis == BasisType.SegLagrange or basis == BasisType.SegLegendre:
        xn[:,0] = np.linspace(-1.,1.,p+1)
    elif basis == BasisType.QuadLagrange or basis == BasisType.QuadLegendre:
        xseg = np.linspace(-1.,1.,p+1)
        n = 0
        for i in range(p+1):
            xn[n:n+p+1,0] = xseg
            xn[n:n+p+1,1] = xseg[i]
            n += p+1
    elif basis == BasisType.TriLagrange:
        n = 0
        xseg = np.linspace(0.,1.,p+1)
        for j in range(p+1):
            xn[n:n+p+1-j,0] = xseg[:p+1-j]
            xn[n:n+p+1-j,1] = xseg[j]
            n += p+1-j
        # for j in range(p):
        #     for i in range(p-j):
        #         xn[k][0] = float(i)/float(p)
        #         xn[k][1] = float(j)/float(p)

    return xn, nn
  

# def Shape2Dim(Shape):
# 	if Shape == ShapeType.Point:
# 		dim = 0
# 	elif Shape == ShapeType.Segment:
# 		dim = 1
# 	else:
# 		raise Exception("Shape not supported")

# 	return dim


# def FaceShape(ElemShape):
#     if ElemShape == ShapeType.Segment:
#         FShape = ShapeType.Point
#     else:
#         raise Exception("Shape not supported")

#     return FShape


def GetElemMassMatrix(mesh, basis, Order, PhysicalSpace=False, egrp=-1, elem=-1, StaticData=None):
    if StaticData is None:
        pnq = -1
        quadData = None
        PhiData = None
        JData = JacobianData(mesh)
        StaticData = GenericData()
    else:
        nq = StaticData.pnq
        quadData = StaticData.quadData
        PhiData = StaticData.PhiData
        JData = StaticData.JData

    if PhysicalSpace:
        QuadOrder,QuadChanged = GetQuadOrderElem(mesh, egrp, mesh.ElemGroups[egrp].QBasis, Order*2, quadData=quadData)
    else:
        QuadOrder = Order*2
        QuadChanged = True

    if QuadChanged:
        quadData = QuadData(mesh, basis, EntityType.Element, QuadOrder)

    nq = quadData.nquad
    xq = quadData.xquad
    wq = quadData.wquad

    if QuadChanged:
        # PhiData = BasisData(egrp=0,Order,entity=EntityType.Element,nq,xq,mesh,GetPhi=True,GetGPhi=False)
        PhiData = BasisData(basis,Order,nq,mesh)
        PhiData.EvalBasis(xq, Get_Phi=True)

    if PhysicalSpace:
        JData.ElemJacobian(egrp,elem,nq,xq,mesh,Get_detJ=True)
        if JData.nq == 1:
            detJ = np.full(nq, JData.detJ[0])
        else:
            detJ = JData.detJ
    else:
        detJ = np.full(nq, 1.)

    nn = PhiData.nn

    phi = PhiData.Phi
    MM = np.zeros([nn,nn])
    for i in range(nn):
        for j in range(nn):
            t = 0.
            for iq in range(nq):
                t += phi[iq,i]*phi[iq,j]*wq[iq]*detJ[iq] # JData.detJ[iq*(JData.nq != 1)]
            MM[i,j] = t
    StaticData.pnq = nq
    StaticData.quadData = quadData
    StaticData.PhiData = PhiData
    StaticData.JData = JData

    return MM, StaticData

def GetElemADERMatrix(mesh, basis1, basis2, order, PhysicalSpace=False, egrp=-1, elem=-1, StaticData=None):

    #Stiffness matrix in space
    gradDir = 0
    SMS,_= GetStiffnessMatrixADER(gradDir,mesh, order, egrp=0, elem=0, basis=basis1)
    SMS = np.transpose(SMS)
    #Stiffness matrix in time
    gradDir = 1
    SMT,_= GetStiffnessMatrixADER(gradDir,mesh, order, egrp=0, elem=0, basis=basis1)

    #Calculate flux matrices in time at tau=1 (L) and tau=-1 (R)
    FTL,_= GetTemporalFluxADER(mesh, basis1, basis1, order, PhysicalSpace=False, egrp=0, elem=0, StaticData=None)
    FTR,_= GetTemporalFluxADER(mesh, basis1, basis2, order, PhysicalSpace=False, egrp=0, elem=0, StaticData=None)


    A1 = np.subtract(FTL,SMT)
    A = np.add(A1,SMS)

    return A, FTR, StaticData

def GetElemInvMassMatrix(mesh, basis, Order, PhysicalSpace=False, egrp=-1, elem=-1, StaticData=None):
    MM, StaticData = GetElemMassMatrix(mesh, basis, Order, PhysicalSpace, egrp, elem, StaticData)
    
    MMinv = np.linalg.inv(MM) 

    return MMinv, StaticData

def GetElemInvADERMatrix(mesh, basis1, basis2, Order, PhysicalSpace=False, egrp=-1, elem=-1, StaticData=None):
    ADER, FTR, StaticData = GetElemADERMatrix(mesh, basis1, basis2, Order, PhysicalSpace, egrp, elem, StaticData)

    ADERinv = np.linalg.solve(ADER,FTR)
    
    return ADERinv, StaticData

def GetStiffnessMatrix(mesh, egrp, elem, basis, Order, StaticData=None):
    if StaticData is None:
        pnq = -1
        quadData = None
        PhiData = None
        JData = JacobianData(mesh)
        StaticData = GenericData()
    else:
        nq = StaticData.pnq
        quadData = StaticData.quadData
        PhiData = StaticData.PhiData
        JData = StaticData.JData

    QuadOrder,QuadChanged = GetQuadOrderElem(mesh, egrp, mesh.ElemGroups[egrp].QBasis, Order*2, quadData=quadData)
    if QuadChanged:
        quadData = QuadData(mesh, mesh.ElemGroups[egrp].QBasis, EntityType.Element, QuadOrder)

    nq = quadData.nquad
    xq = quadData.xquad
    wq = quadData.wquad

    if QuadChanged:
        # PhiData = BasisData(egrp=0,Order,entity=EntityType.Element,nq,xq,mesh,GetPhi=True,GetGPhi=False)
        PhiData = BasisData(basis,Order,nq,mesh)
        PhiData.EvalBasis(xq, Get_Phi=True, Get_GPhi=True)

    JData.ElemJacobian(egrp,elem,nq,xq,mesh,Get_detJ=True,Get_iJ=True)
    PhiData.EvalBasis(xq, Get_gPhi=True, JData=JData)
    nn = PhiData.nn

    phi = PhiData.Phi
    gPhi = PhiData.gPhi
    SM = np.zeros([nn,nn])
    for i in range(nn):
        for j in range(nn):
            t = 0.
            for iq in range(nq):
                t += gPhi[iq,i,0]*phi[iq,j]*wq[iq]*JData.detJ[iq*(JData.nq != 1)]
            SM[i,j] = t

    StaticData.pnq = nq
    StaticData.quadData = quadData
    StaticData.PhiData = PhiData
    StaticData.JData = JData

    return SM, StaticData


def GetStiffnessMatrixADER(gradDir,mesh, Order, egrp, elem, basis, StaticData=None):
    if StaticData is None:
        pnq = -1
        quadData = None
        PhiData = None
        # JData = JacobianData(mesh)
        StaticData = GenericData()
    else:
        nq = StaticData.pnq
        quadData = StaticData.quadData
        PhiData = StaticData.PhiData
        # JData = StaticData.JData

    QuadOrder,QuadChanged = GetQuadOrderElem(mesh, egrp, basis, Order*2., quadData=quadData)
    #Add one to QuadOrder to adjust the mesh.Dim addition in GetQuadOrderElem.
    QuadOrder+=1
    if QuadChanged:
        quadData = QuadDataADER(mesh, basis, EntityType.Element, QuadOrder)
    nq = quadData.nquad
    xq = quadData.xquad
    wq = quadData.wquad

    if QuadChanged:
        # PhiData = BasisData(egrp=0,Order,entity=EntityType.Element,nq,xq,mesh,GetPhi=True,GetGPhi=False)
        PhiData = BasisData(basis,Order,nq,mesh)
        PhiData.EvalBasis(xq, Get_Phi=True, Get_GPhi=True)

    # JData.ElemJacobian(egrp,elem,nq,xq,mesh,Get_detJ=True,Get_iJ=True)
    #PhiData.EvalBasis(xq, Get_gPhi=True, JData=None)
    nn = PhiData.nn

    phi = PhiData.Phi
    GPhi = PhiData.GPhi
    SM = np.zeros([nn,nn])
    for i in range(nn):
        for j in range(nn):
            t = 0.
            for iq in range(nq):
                t += GPhi[iq,i,gradDir]*phi[iq,j]*wq[iq]
            SM[i,j] = t

    StaticData.pnq = nq
    StaticData.quadData = quadData
    StaticData.PhiData = PhiData
    #StaticData.JData = JData
    return SM, StaticData

def GetTemporalFluxADER(mesh, basis1, basis2, Order, PhysicalSpace=False, egrp=-1, elem=-1, StaticData=None):

    if StaticData is None:
        pnq = -1
        quadData = None
        PhiData = None
        PsiData = None
        StaticData = GenericData()
    else:
        nq = StaticData.pnq
        quadData = StaticData.quadData
        PhiData = StaticData.PhiData
        PsiData = StaticData.PsiData

    if basis1 == basis2:
        face =2 
    else:
        face = 0
    #QuadOrderTest,QuadChangedTest = GetQuadOrderIFace(mesh, face, mesh.ElemGroups[egrp].QBasis, Order, EqnSet=None, quadData=quadData)
    QuadOrder,QuadChanged = GetQuadOrderElem(mesh, egrp, mesh.ElemGroups[egrp].QBasis, Order*2, quadData=quadData)
    #Add one to QuadOrder to adjust the mesh.Dim addition in GetQuadOrderElem.
    #QuadOrder+=1

    if QuadChanged:
        quadData = QuadData(mesh, mesh.ElemGroups[egrp].QBasis, EntityType.Element, QuadOrder)
        #quadDataTest = QuadData(mesh, mesh.ElemGroups[egrp].QBasis, EntityType.Element, QuadOrderTest)

    nq = quadData.nquad
    xq = quadData.xquad
    wq = quadData.wquad

    if QuadChanged:
        if basis1 == basis2:
            face = 2
            basis = basis1
            PhiData = BasisData(basis,Order,nq,mesh)
            PsiData = PhiData
            xelem = np.zeros([nq,mesh.Dim+1])
            PhiData.EvalBasisOnFaceADER(mesh, basis, egrp, face, xq, xelem, Get_Phi=True)
            PsiData.EvalBasisOnFaceADER(mesh, basis, egrp, face, xq, xelem, Get_Phi=True)
        else:
            face = 0
            PhiData = BasisData(basis1,Order,nq,mesh)
            PsiData = BasisData(basis2,Order,nq,mesh)
            xelemPhi = np.zeros([nq,mesh.Dim+1])
            xelemPsi = np.zeros([nq,mesh.Dim])
            PhiData.EvalBasisOnFaceADER(mesh, basis1, egrp, face, xq, xelemPhi, Get_Phi=True)
            #PsiData.EvalBasisOnFaceADER(mesh, basis2, egrp, face, xq, xelemPsi, Get_Phi=True)
            PsiData.EvalBasis(xq, Get_Phi=True, Get_GPhi=False)


    nn1 = PhiData.nn
    nn2 = PsiData.nn

    phi = PhiData.Phi
    psi = PsiData.Phi

    MM = np.zeros([nn1,nn2])
    for i in range(nn1):
        for j in range(nn2):
            t = 0.
            for iq in range(nq):
                t += phi[iq,i]*psi[iq,j]*wq[iq]
            MM[i,j] = t
    StaticData.pnq = nq
    StaticData.quadData = quadData
    StaticData.PhiData = PhiData
 
    return MM, StaticData

def GetProjectionMatrix(mesh, basis_old, Order_old, basis, Order, MMinv):
    QuadOrder = np.amax([Order_old+Order, 2*Order])
    quadData = QuadData(mesh, basis, EntityType.Element, QuadOrder)

    nq = quadData.nquad
    xq = quadData.xquad
    wq = quadData.wquad

    PhiData_old = BasisData(basis_old, Order_old, nq, mesh)
    PhiData_old.EvalBasis(xq, Get_Phi=True)
    nn_old = PhiData_old.nn
    phi_old = PhiData_old.Phi

    PhiData = BasisData(basis, Order, nq, mesh)
    PhiData.EvalBasis(xq, Get_Phi=True)
    nn = PhiData.nn
    phi = PhiData.Phi

    A = np.zeros([nn,nn_old])
    for i in range(nn):
        for j in range(nn_old):
            t = 0.
            for iq in range(nq):
                t += phi[iq,i]*phi_old[iq,j]*wq[iq] # JData.detJ[iq*(JData.nq != 1)]
            A[i,j] = t

    PM = np.matmul(MMinv,A)

    return PM


def GetInvStiffnessMatrix(mesh, egrp, elem, basis, Order, StaticData=None):
    SM, StaticData = GetStiffnessMatrix(mesh, egrp, elem, basis, Order, StaticData)

    MMinv = np.linalg.inv(SM) 

    return SM, StaticData

def ComputeInvADERMatrices(mesh, EqnSet, solver=None):
    ## Allocat ADERinv_all
    # Calculate inverse mass matrix for every single element,
    # even if uniform mesh
    #Hard code basisType to Quads (currently only designed for 1D)
    basis1 = BasisType.QuadLegendre
    basis2 = BasisType.SegLegendre
    
    ArrayDims = [None]*mesh.nElemGroup
    for egrp in range(mesh.nElemGroup):
        Order = EqnSet.Orders[egrp]
        nn1 = Order2nNode(basis1, Order)
        nn2 = Order2nNode(basis2, Order)
        ArrayDims[egrp] = [mesh.nElems[egrp], nn1, nn2]
    ADERinv_all = ArrayList(nArray=mesh.nElemGroup,ArrayDims=ArrayDims)

    StaticData = None

    # Uniform mesh?
    ReCalcMM = True
    if solver is not None:
        ReCalcMM = not solver.Params["UniformMesh"]

    for egrp in range(mesh.nElemGroup):
        EGroup = mesh.ElemGroups[egrp]
        #basis = EqnSet.Bases[egrp]
        Order = EqnSet.Orders[egrp]
        for elem in range(EGroup.nElem):
            if elem == 0 or ReCalcMM:
                # Only recalculate if not using uniform mesh
                ADERinv,StaticData = GetElemInvADERMatrix(mesh, basis1, basis2, Order, False, egrp, elem, StaticData)
            ADERinv_all.Arrays[egrp][elem] = ADERinv

    if solver is not None:
        solver.DataSet.ADERinv_all = ADERinv_all

    return ADERinv_all

def ComputeInvMassMatrices(mesh, EqnSet, solver=None):
    ## Allocate MMinv_all
    # Currently calculating inverse mass matrix for every single element,
    # even if uniform mesh
    ArrayDims = [None]*mesh.nElemGroup
    for egrp in range(mesh.nElemGroup):
        basis = EqnSet.Bases[egrp]
        Order = EqnSet.Orders[egrp]
        nn = Order2nNode(basis, Order)
        ArrayDims[egrp] = [mesh.nElems[egrp], nn, nn]
    MMinv_all = ArrayList(nArray=mesh.nElemGroup,ArrayDims=ArrayDims)

    StaticData = None

    # Uniform mesh?
    ReCalcMM = True
    if solver is not None:
        ReCalcMM = not solver.Params["UniformMesh"]

    for egrp in range(mesh.nElemGroup):
        EGroup = mesh.ElemGroups[egrp]
        basis = EqnSet.Bases[egrp]
        Order = EqnSet.Orders[egrp]
        for elem in range(EGroup.nElem):
            if elem == 0 or ReCalcMM:
                # Only recalculate if not using uniform mesh
                MMinv,StaticData = GetElemInvMassMatrix(mesh, basis, Order, True, egrp, elem, StaticData)
            MMinv_all.Arrays[egrp][elem] = MMinv

    if solver is not None:
        solver.DataSet.MMinv_all = MMinv_all

    return MMinv_all


def GetShapes(basis, Order, nq, xq, phi=None):
    nn = Order2nNode(basis, Order)

    if phi is None or phi.shape != (nq,nn):
        phi = np.zeros([nq,nn])
    else:
        phi[:] = 0.

    if basis == BasisType.SegLagrange:
        for iq in range(nq): 
            Shape_TensorLagrange(1, Order, xq[iq], phi[iq,:])
    elif basis == BasisType.QuadLagrange:
        for iq in range(nq): 
            Shape_TensorLagrange(2, Order, xq[iq], phi[iq,:])
    elif basis == BasisType.TriLagrange:
        for iq in range(nq): 
            Shape_TriLagrange(Order, xq[iq], phi[iq,:])
    elif basis == BasisType.SegLegendre:
        for iq in range(nq):
            Shape_TensorLegendre(1, Order, xq[iq], phi[iq,:])
    elif basis == BasisType.QuadLegendre:
        for iq in range(nq):
            Shape_TensorLegendre(2, Order, xq[iq], phi[iq,:])
    else:
        raise Exception("Basis not supported")

    return phi


def GetGrads(basis, Order, dim, nq, xq, GPhi=None):
    nn = Order2nNode(basis, Order)

    if GPhi is None or GPhi.shape != (nq,nn,dim):
        GPhi = np.zeros([nq,nn,dim])
    else: 
        GPhi[:] = 0.

    if basis == BasisType.SegLagrange:
        for iq in range(nq): 
            Grad_TensorLagrange(1, Order, xq[iq], GPhi[iq,:,:])
    elif basis == BasisType.QuadLagrange:
        for iq in range(nq): 
            Grad_TensorLagrange(2, Order, xq[iq], GPhi[iq,:,:])
    elif basis == BasisType.TriLagrange:
        for iq in range(nq): 
            Grad_TriLagrange(Order, xq[iq], GPhi[iq,:,:])
    elif basis == BasisType.SegLegendre:
        for iq in range(nq):
            Grad_TensorLegendre(1, Order, xq[iq], GPhi[iq,:,:])
    elif basis == BasisType.QuadLegendre:
        for iq in range(nq):
            Grad_TensorLegendre(2, Order, xq[iq], GPhi[iq,:,:])
    else:
        raise Exception("Basis not supported")

    return GPhi


def BasisLagrange1D(x, xnode, nnode, phi, gphi):
    for j in range(nnode):
        if phi is not None:
            pj = 1.
            for i in range(j): pj *= (x-xnode[i])/(xnode[j]-xnode[i])
            for i in range(j+1,nnode): pj *= (x-xnode[i])/(xnode[j]-xnode[i])
            phi[j] = pj
        if gphi is not None:
            gphi[j] = 0.0;
            for i in range(nnode):
                if i != j:
                    g = 1./(xnode[j] - xnode[i])
                    for k in range(nnode):
                        if k != i and k != j:
                            g *= (x - xnode[k])/(xnode[j] - xnode[k])
                    gphi[j] += g

def BasisLagrange2D(x, xnode, nnode, phi, gphi):
    if gphi is not None:
        gphix = 0.*xnode; gphiy = 0.*xnode
    else:
        gphix = None; gphiy = None
    # Always need phi
    phix = 0.*xnode; phiy = 0.*xnode

    BasisLagrange1D(x[0], xnode, nnode, phix, gphix)
    BasisLagrange1D(x[1], xnode, nnode, phiy, gphiy)

    if phi is not None:
        phi[:] = np.reshape(np.outer(phix, phiy), (-1,), 'F')
    if gphi is not None:
        gphi[:,0] = np.reshape(np.outer(gphix, phiy), (-1,), 'F')
        gphi[:,1] = np.reshape(np.outer(phix, gphiy), (-1,), 'F')


def Shape_TensorLagrange(dim, p, x, phi):
    if p == 0:
    	phi[:] = 1.
    	return

    nnode = p + 1
    xnode = np.zeros(nnode)
    dx = 2./float(p)
    for i in range(nnode): xnode[i] = -1. + float(i)*dx
    if dim == 1:
        BasisLagrange1D(x, xnode, nnode, phi, None)
    elif dim == 2:
        BasisLagrange2D(x, xnode, nnode, phi, None)


def Shape_TriLagrange(p, xi, phi):
    x = xi[0]; y = xi[1]

    if p == 0:
        phi[:] = 1.
    elif p == 1:
        phi[0] = 1-x-y
        phi[1] = x  
        phi[2] = y
    elif p == 2:
        phi[0] = 1.0-3.0*x-3.0*y+2.0*x*x+4.0*x*y+2.0*y*y
        phi[2] = -x+2.0*x*x
        phi[5] = -y+2.0*y*y
        phi[4] = 4.0*x*y
        phi[3] = 4.0*y-4.0*x*y-4.0*y*y
        phi[1] = 4.0*x-4.0*x*x-4.0*x*y
    elif p == 3:
        phi[0] = 1.0-11.0/2.0*x-11.0/2.0*y+9.0*x*x+18.0*x*y+9.0*y*y-9.0/2.0*x*x*x-27.0/2.0*x*x*y-27.0/2.0*x*y*y-9.0/2.0*y*y*y
        phi[3] = x-9.0/2.0*x*x+9.0/2.0*x*x*x
        phi[9] = y-9.0/2.0*y*y+9.0/2.0*y*y*y
        phi[6] = -9.0/2.0*x*y+27.0/2.0*x*x*y
        phi[8] = -9.0/2.0*x*y+27.0/2.0*x*y*y
        phi[7] = -9.0/2.0*y+9.0/2.0*x*y+18.0*y*y-27.0/2.0*x*y*y-27.0/2.0*y*y*y
        phi[4] = 9.0*y-45.0/2.0*x*y-45.0/2.0*y*y+27.0/2.0*x*x*y+27.0*x*y*y+27.0/2.0*y*y*y
        phi[1] = 9.0*x-45.0/2.0*x*x-45.0/2.0*x*y+27.0/2.0*x*x*x+27.0*x*x*y+27.0/2.0*x*y*y
        phi[2] = -9.0/2.0*x+18.0*x*x+9.0/2.0*x*y-27.0/2.0*x*x*x-27.0/2.0*x*x*y
        phi[5] = 27.0*x*y-27.0*x*x*y-27.0*x*y*y
    elif p == 4:
        phi[ 0] = 1.0-25.0/3.0*x-25.0/3.0*y+70.0/3.0*x*x+140.0/3.0*x*y+70.0/3.0*y*y-80/3.0*x*x*x-80.0*x*x*y-80.0*x*y*y-80.0/3.0*y*y*y \
        +32.0/3.0*x*x*x*x+128/3.0*x*x*x*y+64.0*x*x*y*y+128.0/3.0*x*y*y*y+32.0/3.0*y*y*y*y 
        phi[ 4] = -x+22.0/3.0*x*x-16.0*x*x*x+32.0/3.0*x*x*x*x 
        phi[14] = -y+22.0/3.0*y*y-16.0*y*y*y+32.0/3.0*y*y*y*y 
        phi[ 8] = 16.0/3.0*x*y-32.0*x*x*y+128.0/3.0*x*x*x*y 
        phi[11] = 4.0*x*y-16.0*x*x*y-16.0*x*y*y+64.0*x*x*y*y 
        phi[13] = 16.0/3.0*x*y-32.0*x*y*y+128.0/3.0*x*y*y*y 
        phi[12] = 16.0/3.0*y-16.0/3.0*x*y-112.0/3.0*y*y+32.0*x*y*y+224.0/3.0*y*y*y-128.0/3.0*x*y*y*y-128.0/3.0*y*y*y*y 
        phi[ 9] = -12.0*y+28.0*x*y+76.0*y*y-16.0*x*x*y-144.0*x*y*y-128.0*y*y*y+64.0*x*x*y*y+128.0*x*y*y*y+64.0*y*y*y*y 
        phi[ 5] = 16.0*y-208.0/3.0*x*y-208.0/3.0*y*y+96.0*x*x*y+192.0*x*y*y+96.0*y*y*y-128.0/3.0*x*x*x*y- 128*x*x*y*y-128.0*x*y*y*y-128.0/3.0*y*y*y*y 
        phi[ 1] = 16.0*x-208.0/3.0*x*x-208.0/3.0*x*y+96.0*x*x*x+192.0*x*x*y+96.0*x*y*y-128.0/3.0*x*x*x*x- 128*x*x*x*y-128.0*x*x*y*y-128.0/3.0*x*y*y*y 
        phi[ 2] = -12.0*x+76.0*x*x+28.0*x*y-128.0*x*x*x-144.0*x*x*y-16.0*x*y*y+64.0*x*x*x*x+128.0*x*x*x*y+64.0*x*x*y*y 
        phi[ 3] = 16.0/3.0*x-112.0/3.0*x*x-16.0/3.0*x*y+224.0/3.0*x*x*x+32.0*x*x*y-128.0/3.0*x*x*x*x-128.0/3.0*x*x*x*y 
        phi[ 6] = 96.0*x*y-224.0*x*x*y-224.0*x*y*y+128.0*x*x*x*y+256.0*x*x*y*y+128.0*x*y*y*y 
        phi[ 7] = -32.0*x*y+160.0*x*x*y+32.0*x*y*y-128.0*x*x*x*y-128.0*x*x*y*y 
        phi[10] = -32.0*x*y+32.0*x*x*y+160.0*x*y*y-128.0*x*x*y*y-128.0*x*y*y*y
    elif p == 5:
        phi[ 0]  = 1.0-137.0/12.0*x-137.0/12.0*y+375.0/8.0*x*x+375.0/4.0*x*y+375.0/8.0*y*y-2125.0/24.0*x*x*x-2125.0/8.0*x*x*y-2125.0/8.0*x*y*y \
        -2125.0/24.0*y*y*y+ 625.0/8.0*x*x*x*x+625.0/2.0*x*x*x*y+1875.0/4.0*x*x*y*y+625.0/2.0*x*y*y*y+625.0/8.0*y*y*y*y-625.0/24.0*x*x*x*x*x \
        -3125.0/24.0*x*x*x*x*y-3125.0/12.0*x*x*x*y*y-3125.0/12.0*x*x*y*y*y-3125.0/24.0*x*y*y*y*y-625.0/24.0*y*y*y*y*y
        phi[ 5]  = x-125.0/12.0*x*x+875.0/24.0*x*x*x-625.0/12.0*x*x*x*x+625.0/24.0*x*x*x*x*x
        phi[20]  = y-125.0/12.0*y*y+875.0/24.0*y*y*y-625.0/12.0*y*y*y*y+625.0/24.0*y*y*y*y*y
        phi[10]  = -25.0/4.0*x*y+1375.0/24.0*x*x*y-625.0/4.0*x*x*x*y+3125.0/24.0*x*x*x*x*y
        phi[14]  = -25.0/6.0*x*y+125.0/4.0*x*x*y+125.0/6.0*x*y*y-625.0/12.0*x*x*x*y-625.0/4.0*x*x*y*y+3125.0/12.0*x*x*x*y*y
        phi[17]  = -25.0/6.0*x*y+125.0/6.0*x*x*y+125.0/4.0*x*y*y-625.0/4.0*x*x*y*y-625.0/12.0*x*y*y*y+3125.0/12.0*x*x*y*y*y
        phi[19]  = -25.0/4.0*x*y+1375.0/24.0*x*y*y-625.0/4.0*x*y*y*y+3125.0/24.0*x*y*y*y*y
        phi[18]  = -25.0/4.0*y+25.0/4.0*x*y+1525.0/24.0*y*y-1375.0/24.0*x*y*y-5125.0/24.0*y*y*y+625.0/4.0*x*y*y*y+6875.0/24.0*y*y*y*y \
        -3125.0/24.0*x*y*y*y*y-3125.0/24.0*y*y*y*y*y
        phi[15]  = 50.0/3.0*y-75.0/2.0*x*y-325.0/2.0*y*y+125.0/6.0*x*x*y+3875.0/12.0*x*y*y+6125.0/12.0*y*y*y-625.0/4.0*x*x*y*y-3125.0/4.0*x*y*y*y \
        -625.0*y*y*y*y+3125.0/12.0*x*x*y*y*y+3125.0/6.0*x*y*y*y*y+3125.0/12.0*y*y*y*y*y
        phi[11]  = -25.0*y+1175.0/12.0*x*y+2675.0/12.0*y*y-125.0*x*x*y-8875.0/12.0*x*y*y-7375.0/12.0*y*y*y+625.0/12.0*x*x*x*y+3125.0/4.0*x*x*y*y \
        +5625.0/4.0*x*y*y*y+8125.0/12.0*y*y*y*y-3125.0/12.0*x*x*x*y*y-3125.0/4.0*x*x*y*y*y-3125.0/4.0*x*y*y*y*y-3125.0/12.0*y*y*y*y*y 
        phi[ 6] = 25.0*y-1925.0/12.0*x*y-1925.0/12.0*y*y+8875.0/24.0*x*x*y+8875.0/12.0*x*y*y+8875.0/24.0*y*y*y-4375.0/12.0*x*x*x*y \
        -4375.0/4.0*x*x*y*y-4375.0/4.0*x*y*y*y-4375.0/12.0*y*y*y*y+3125.0/24.0*x*x*x*x*y+ 3125.0/6.0*x*x*x*y*y+3125.0/4.0*x*x*y*y*y \
        +3125.0/6.0*x*y*y*y*y+3125.0/24.0*y*y*y*y*y
        phi[ 1] = 25.0*x-1925.0/12.0*x*x-1925.0/12.0*x*y+8875.0/24.0*x*x*x+8875.0/12.0*x*x*y+8875.0/24.0*x*y*y-4375.0/12.0*x*x*x*x-4375.0/4.0*x*x*x*y \
        -4375.0/4.0*x*x*y*y-4375.0/12.0*x*y*y*y+3125.0/24.0*x*x*x*x*x+3125.0/6.0*x*x*x*x*y+3125.0/4.0*x*x*x*y*y+3125.0/6.0*x*x*y*y*y+3125.0/24.0*x*y*y*y*y
        phi[ 2] = -25.0*x+2675.0/12.0*x*x+1175.0/12.0*x*y-7375.0/12.0*x*x*x-8875.0/12.0*x*x*y-125.0*x*y*y+8125.0/12.0*x*x*x*x \
        +5625.0/4.0*x*x*x*y+3125.0/4.0*x*x*y*y+625.0/12.0*x*y*y*y-3125.0/12.0*x*x*x*x*x- 3125.0/4.0*x*x*x*x*y-3125.0/4.0*x*x*x*y*y-3125.0/12.0*x*x*y*y*y
        phi[ 3] = 50.0/3.0*x-325.0/2.0*x*x-75.0/2.0*x*y+6125.0/12.0*x*x*x+3875.0/12.0*x*x*y+125.0/6.0*x*y*y-625.0*x*x*x*x \
        -3125.0/4.0*x*x*x*y-625.0/4.0*x*x*y*y+3125.0/12.0*x*x*x*x*x+3125.0/6.0*x*x*x*x*y+3125.0/12.0*x*x*x*y*y
        phi[ 4] = -25.0/4.0*x+1525.0/24.0*x*x+25.0/4.0*x*y-5125.0/24.0*x*x*x-1375.0/24.0*x*x*y+6875.0/24.0*x*x*x*x+625.0/4.0*x*x*x*y \
        -3125.0/24.0*x*x*x*x*x-3125.0/24.0*x*x*x*x*y
        phi[ 7] = 250.0*x*y-5875.0/6.0*x*x*y-5875.0/6.0*x*y*y+1250.0*x*x*x*y+2500.0*x*x*y*y+1250.0*x*y*y*y-3125.0/6.0*x*x*x*x*y \
        -3125.0/2.0*x*x*x*y*y-3125.0/2.0*x*x*y*y*y-3125.0/6.0*x*y*y*y*y
        phi[ 8] = -125.0*x*y+3625.0/4.0*x*x*y+1125.0/4.0*x*y*y-3125.0/2.0*x*x*x*y-6875.0/4.0*x*x*y*y-625.0/4.0*x*y*y*y+3125.0/4.0*x*x*x*x*y \
        +3125.0/2.0*x*x*x*y*y+3125.0/4.0*x*x*y*y*y
        phi[ 9] = 125.0/3.0*x*y-2125.0/6.0*x*x*y-125.0/3.0*x*y*y+2500.0/3.0*x*x*x*y+625.0/2.0*x*x*y*y-3125.0/6.0*x*x*x*x*y-3125.0/6.0*x*x*x*y*y
        phi[12] = -125.0*x*y+1125.0/4.0*x*x*y+3625.0/4.0*x*y*y-625.0/4.0*x*x*x*y-6875.0/4.0*x*x*y*y-3125.0/2.0*x*y*y*y+3125.0/4.0*x*x*x*y*y \
        +3125.0/2.0*x*x*y*y*y+3125.0/4.0*x*y*y*y*y
        phi[13] = 125.0/4.0*x*y-375.0/2.0*x*x*y-375.0/2.0*x*y*y+625.0/4.0*x*x*x*y+4375.0/4.0*x*x*y*y+625.0/4.0*x*y*y*y-3125.0/4.0*x*x*x*y*y \
        -3125.0/4.0*x*x*y*y*y
        phi[16] = 125.0/3.0*x*y-125.0/3.0*x*x*y-2125.0/6.0*x*y*y+625.0/2.0*x*x*y*y+2500.0/3.0*x*y*y*y-3125.0/6.0*x*x*y*y*y-3125.0/6.0*x*y*y*y*y


def Grad_TensorLagrange(dim, p, x, gphi):
    if p == 0:
    	gphi[:,:] = 0.
    	return 

    nnode = p + 1
    xnode = np.zeros(nnode)
    dx = 2./float(p)
    for i in range(nnode): xnode[i] = -1. + float(i)*dx
    if dim == 1:
        BasisLagrange1D(x, xnode, nnode, None, gphi)
    if dim == 2:
        BasisLagrange2D(x, xnode, nnode, None, gphi)


def Grad_TriLagrange(p, xi, gphi):
    x = xi[0]; y = xi[1]

    if p == 0:
        gphi[:] = 0.
    elif p == 1:
        gphi[0,0] =  -1.0
        gphi[1,0] =  1.0
        gphi[2,0] =  0.0
        gphi[0,1] =  -1.0
        gphi[1,1] =  0.0
        gphi[2,1] =  1.0
    elif p == 2:
        gphi[0,0] =  -3.0+4.0*x+4.0*y
        gphi[2,0] =  -1.0+4.0*x
        gphi[5,0] =  0.0
        gphi[4,0] =  4.0*y
        gphi[3,0] =  -4.0*y
        gphi[1,0] =  4.0-8.0*x-4.0*y
        gphi[0,1] =  -3.0+4.0*x+4.0*y
        gphi[2,1] =  0.0
        gphi[5,1] =  -1.0+4.0*y
        gphi[4,1] =  4.0*x
        gphi[3,1] =  4.0-4.0*x-8.0*y
        gphi[1,1] =  -4.0*x
    elif p == 3:
        gphi[0,0] =  -11.0/2.0+18.0*x+18.0*y-27.0/2.0*x*x-27.0*x*y-27.0/2.0*y*y
        gphi[3,0] =  1.0-9.0*x+27.0/2.0*x*x
        gphi[9,0] =  0.0
        gphi[6,0] =  -9.0/2.0*y+27.0*x*y
        gphi[8,0] =  -9.0/2.0*y+27.0/2.0*y*y
        gphi[7,0] =  9.0/2.0*y-27.0/2.0*y*y
        gphi[4,0] =  -45.0/2.0*y+27.0*x*y+27.0*y*y
        gphi[1,0] =  9.0-45.0*x-45.0/2.0*y+81.0/2.0*x*x+54.0*x*y+27.0/2.0*y*y
        gphi[2,0] =  -9.0/2.0+36.0*x+9.0/2.0*y-81.0/2.0*x*x-27.0*x*y
        gphi[5,0] =  27.0*y-54.0*x*y-27.0*y*y
        gphi[0,1] =  -11.0/2.0+18.0*x+18.0*y-27.0/2.0*x*x-27.0*x*y-27.0/2.0*y*y
        gphi[3,1] =  0.0
        gphi[9,1] =  1.0-9.0*y+27.0/2.0*y*y
        gphi[6,1] =  -9.0/2.0*x+27.0/2.0*x*x
        gphi[8,1] =  -9.0/2.0*x+27.0*x*y
        gphi[7,1] =  -9.0/2.0+9.0/2.0*x+36.0*y-27.0*x*y-81.0/2.0*y*y
        gphi[4,1] =  9.0-45.0/2.0*x-45.0*y+27.0/2.0*x*x+54.0*x*y+81.0/2.0*y*y
        gphi[1,1] =  -45.0/2.0*x+27.0*x*x+27.0*x*y
        gphi[2,1] =  9.0/2.0*x-27.0/2.0*x*x
        gphi[5,1] =  27.0*x-27.0*x*x-54.0*x*y
    elif p == 4:
        gphi[ 0,0] =  -25.0/3.0+140.0/3.0*x+140.0/3.0*y-80.0*x*x-160.0*x*y-80.0*y*y+128.0/3.0*x*x*x+128.0*x*x*y+128.0*x*y*y+128.0/3.0*y*y*y
        gphi[ 4,0] =  -1.0+44.0/3.0*x-48.0*x*x+128.0/3.0*x*x*x
        gphi[14,0] =  0.0
        gphi[ 8,0] =  16.0/3.0*y-64.0*x*y+128.0*x*x*y
        gphi[11,0] =  4.0*y-32.0*x*y-16.0*y*y+128.0*x*y*y
        gphi[13,0] =  16.0/3.0*y-32.0*y*y+128.0/3.0*y*y*y
        gphi[12,0] =  -16.0/3.0*y+32.0*y*y-128.0/3.0*y*y*y
        gphi[ 9,0] =  28.0*y-32.0*x*y-144.0*y*y+128.0*x*y*y+128.0*y*y*y
        gphi[ 5,0] =  -208.0/3.0*y+192.0*x*y+192.0*y*y-128.0*x*x*y-256.0*x*y*y-128.0*y*y*y
        gphi[ 1,0] =  16.0-416.0/3.0*x-208.0/3.0*y+288.0*x*x+384.0*x*y+96.0*y*y-512.0/3.0*x*x*x-384.0*x*x*y-256.0*x*y*y-128.0/3.0*y*y*y
        gphi[ 2,0] =  -12.0+152.0*x+28.0*y-384.0*x*x-288.0*x*y-16.0*y*y+256.0*x*x*x+384.0*x*x*y+128.0*x*y*y
        gphi[ 3,0] =  16.0/3.0-224.0/3.0*x-16.0/3.0*y+224.0*x*x+64.0*x*y-512.0/3.0*x*x*x-128.0*x*x*y
        gphi[ 6,0] =  96.0*y-448.0*x*y-224.0*y*y+384.0*x*x*y+512.0*x*y*y+128.0*y*y*y
        gphi[ 7,0] =  -32.0*y+320.0*x*y+32.0*y*y-384.0*x*x*y-256.0*x*y*y
        gphi[10,0] =  -32.0*y+64.0*x*y+160.0*y*y-256.0*x*y*y-128.0*y*y*y
        gphi[ 0,1] =  -25.0/3.0+140.0/3.0*x+140.0/3.0*y-80.0*x*x-160.0*x*y-80.0*y*y+128.0/3.0*x*x*x+128.0*x*x*y+128.0*x*y*y+128.0/3.0*y*y*y
        gphi[ 4,1] =  0.0
        gphi[14,1] =  -1.0+44.0/3.0*y-48.0*y*y+128.0/3.0*y*y*y
        gphi[ 8,1] =  16.0/3.0*x-32.0*x*x+128.0/3.0*x*x*x
        gphi[11,1] =  4.0*x-16.0*x*x-32.0*x*y+128.0*x*x*y
        gphi[13,1] =  16.0/3.0*x-64.0*x*y+128.0*x*y*y
        gphi[12,1] =  16.0/3.0-16.0/3.0*x-224.0/3.0*y+64.0*x*y+224.0*y*y-128.0*x*y*y-512.0/3.0*y*y*y
        gphi[ 9,1] =  -12.0+28.0*x+152.0*y-16.0*x*x-288.0*x*y-384.0*y*y+128.0*x*x*y+384.0*x*y*y+256.0*y*y*y
        gphi[ 5,1] =  16.0-208.0/3.0*x-416.0/3.0*y+96.0*x*x+384.0*x*y+288.0*y*y-128.0/3.0*x*x*x-256.0*x*x*y-384.0*x*y*y-512.0/3.0*y*y*y
        gphi[ 1,1] =  -208.0/3.0*x+192.0*x*x+192.0*x*y-128.0*x*x*x-256.0*x*x*y-128.0*x*y*y
        gphi[ 2,1] =  28.0*x-144.0*x*x-32.0*x*y+128.0*x*x*x+128.0*x*x*y
        gphi[ 3,1] =  -16.0/3.0*x+32.0*x*x-128.0/3.0*x*x*x
        gphi[ 6,1] =  96.0*x-224.0*x*x-448.0*x*y+128.0*x*x*x+512.0*x*x*y+384.0*x*y*y
        gphi[ 7,1] =  -32.0*x+160.0*x*x+64.0*x*y-128.0*x*x*x-256.0*x*x*y
        gphi[10,1] =  -32.0*x+32.0*x*x+320.0*x*y-256.0*x*x*y-384.0*x*y*y
    elif p == 5:
        gphi[ 0,0] =  -137.0/12.0+375.0/4.0*x+375.0/4.0*y-2125.0/8.0*x*x-2125.0/4.0*x*y-2125.0/8.0*y*y+625.0/2.0*x*x*x+1875.0/2.0*x*x*y \
        +1875.0/2.0*x*y*y+625.0/2.0*y*y*y-3125.0/24.0*x*x*x*x-3125.0/6.0*x*x*x*y-3125.0/4.0*x*x*y*y-3125.0/6.0*x*y*y*y-3125.0/24.0*y*y*y*y
        gphi[ 5,0] =  1.0-125.0/6.0*x+875.0/8.0*x*x-625.0/3.0*x*x*x+3125.0/24.0*x*x*x*x
        gphi[20,0] =  0.0
        gphi[10,0] =  -25.0/4.0*y+1375.0/12.0*x*y-1875.0/4.0*x*x*y+3125.0/6.0*x*x*x*y
        gphi[14,0] =  -25.0/6.0*y+125.0/2.0*x*y+125.0/6.0*y*y-625.0/4.0*x*x*y-625.0/2.0*x*y*y+3125.0/4.0*x*x*y*y
        gphi[17,0] =  -25.0/6.0*y+125.0/3.0*x*y+125.0/4.0*y*y-625.0/2.0*x*y*y-625.0/12.0*y*y*y+3125.0/6.0*x*y*y*y
        gphi[19,0] =  -25.0/4.0*y+1375.0/24.0*y*y-625.0/4.0*y*y*y+3125.0/24.0*y*y*y*y
        gphi[18,0] =  25.0/4.0*y-1375.0/24.0*y*y+625.0/4.0*y*y*y-3125.0/24.0*y*y*y*y
        gphi[15,0] =  -75.0/2.0*y+125.0/3.0*x*y+3875.0/12.0*y*y-625.0/2.0*x*y*y-3125.0/4.0*y*y*y+3125.0/6.0*x*y*y*y+3125.0/6.0*y*y*y*y
        gphi[11,0] =  1175.0/12.0*y-250.0*x*y-8875.0/12.0*y*y+625.0/4.0*x*x*y+3125.0/2.0*x*y*y+5625.0/4.0*y*y*y-3125.0/4.0*x*x*y*y \
        -3125.0/2.0*x*y*y*y-3125.0/4.0*y*y*y*y
        gphi[ 6,0] =  -1925.0/12.0*y+8875.0/12.0*x*y+8875.0/12.0*y*y-4375.0/4.0*x*x*y-4375.0/2.0*x*y*y-4375.0/4.0*y*y*y+3125.0/6.0*x*x*x*y \
        +3125.0/2.0*x*x*y*y+3125.0/2.0*x*y*y*y+3125.0/6.0*y*y*y*y
        gphi[ 1,0] =  25.0-1925.0/6.0*x-1925.0/12.0*y+8875.0/8.0*x*x+8875.0/6.0*x*y+8875.0/24.0*y*y-4375.0/3.0*x*x*x-13125.0/4.0*x*x*y \
        -4375.0/2.0*x*y*y-4375.0/12.0*y*y*y+15625.0/24.0*x*x*x*x+6250.0/3.0*x*x*x*y+9375.0/4.0*x*x*y*y+3125.0/3.0*x*y*y*y+3125.0/24.0*y*y*y*y
        gphi[ 2,0] =  -25.0+2675.0/6.0*x+1175.0/12.0*y-7375.0/4.0*x*x-8875.0/6.0*x*y-125.0*y*y+8125.0/3.0*x*x*x+16875.0/4.0*x*x*y \
        +3125.0/2.0*x*y*y+625.0/12.0*y*y*y-15625.0/12.0*x*x*x*x-3125.0*x*x*x*y-9375.0/4.0*x*x*y*y-3125.0/6.0*x*y*y*y
        gphi[ 3,0] =  50.0/3.0-325.0*x-75.0/2.0*y+6125.0/4.0*x*x+3875.0/6.0*x*y+125.0/6.0*y*y-2500.0*x*x*x-9375.0/4.0*x*x*y \
        -625.0/2.0*x*y*y+15625.0/12.0*x*x*x*x+6250.0/3.0*x*x*x*y+3125.0/4.0*x*x*y*y
        gphi[ 4,0] =  -25.0/4.0+1525.0/12.0*x+25.0/4.0*y-5125.0/8.0*x*x-1375.0/12.0*x*y+6875.0/6.0*x*x*x+1875.0/4.0*x*x*y \
        -15625.0/24.0*x*x*x*x-3125.0/6.0*x*x*x*y
        gphi[ 7,0] =  250.0*y-5875.0/3.0*x*y-5875.0/6.0*y*y+3750.0*x*x*y+5000.0*x*y*y+1250.0*y*y*y-6250.0/3.0*x*x*x*y \
        -9375.0/2.0*x*x*y*y-3125.0*x*y*y*y-3125.0/6.0*y*y*y*y
        gphi[ 8,0] =  -125.0*y+3625.0/2.0*x*y+1125.0/4.0*y*y-9375.0/2.0*x*x*y-6875.0/2.0*x*y*y-625.0/4.0*y*y*y+3125.0*x*x*x*y \
        +9375.0/2.0*x*x*y*y+3125.0/2.0*x*y*y*y
        gphi[ 9,0] =  125.0/3.0*y-2125.0/3.0*x*y-125.0/3.0*y*y+2500.0*x*x*y+625.0*x*y*y-6250.0/3.0*x*x*x*y-3125.0/2.0*x*x*y*y
        gphi[12,0] =  -125.0*y+1125.0/2.0*x*y+3625.0/4.0*y*y-1875.0/4.0*x*x*y-6875.0/2.0*x*y*y-3125.0/2.0*y*y*y+9375.0/4.0*x*x*y*y \
        +3125.0*x*y*y*y+3125.0/4.0*y*y*y*y
        gphi[13,0] =  125.0/4.0*y-375.0*x*y-375.0/2.0*y*y+1875.0/4.0*x*x*y+4375.0/2.0*x*y*y+625.0/4.0*y*y*y-9375.0/4.0*x*x*y*y \
        -3125.0/2.0*x*y*y*y
        gphi[16,0] =  125.0/3.0*y-250.0/3.0*x*y-2125.0/6.0*y*y+625.0*x*y*y+2500.0/3.0*y*y*y-3125.0/3.0*x*y*y*y-3125.0/6.0*y*y*y*y
        gphi[ 0,1] =  -137.0/12.0+375.0/4.0*x+375.0/4.0*y-2125.0/8.0*x*x-2125.0/4.0*x*y-2125.0/8.0*y*y+625.0/2.0*x*x*x \
        +1875.0/2.0*x*x*y+1875.0/2.0*x*y*y+625.0/2.0*y*y*y-3125.0/24.0*x*x*x*x-3125.0/6.0*x*x*x*y-3125.0/4.0*x*x*y*y-3125.0/6.0*x*y*y*y-3125.0/24.0*y*y*y*y 
        gphi[ 5,1] =  0.0
        gphi[20,1] =  1.0-125.0/6.0*y+875.0/8.0*y*y-625.0/3.0*y*y*y+3125.0/24.0*y*y*y*y
        gphi[10,1] =  -25.0/4.0*x+1375.0/24.0*x*x-625.0/4.0*x*x*x+3125.0/24.0*x*x*x*x
        gphi[14,1] =  -25.0/6.0*x+125.0/4.0*x*x+125.0/3.0*x*y-625.0/12.0*x*x*x-625.0/2.0*x*x*y+3125.0/6.0*x*x*x*y
        gphi[17,1] =  -25.0/6.0*x+125.0/6.0*x*x+125.0/2.0*x*y-625.0/2.0*x*x*y-625.0/4.0*x*y*y+3125.0/4.0*x*x*y*y
        gphi[19,1] =  -25.0/4.0*x+1375.0/12.0*x*y-1875.0/4.0*x*y*y+3125.0/6.0*x*y*y*y
        gphi[18,1] =  -25.0/4.0+25.0/4.0*x+1525.0/12.0*y-1375.0/12.0*x*y-5125.0/8.0*y*y+1875.0/4.0*x*y*y+6875.0/6.0*y*y*y \
        -3125.0/6.0*x*y*y*y-15625.0/24.0*y*y*y*y
        gphi[15,1] =  50.0/3.0-75.0/2.0*x-325.0*y+125.0/6.0*x*x+3875.0/6.0*x*y+6125.0/4.0*y*y-625.0/2.0*x*x*y-9375.0/4.0*x*y*y \
        -2500.0*y*y*y+3125.0/4.0*x*x*y*y+6250.0/3.0*x*y*y*y+15625.0/12.0*y*y*y*y
        gphi[11,1] =  -25.0+1175.0/12.0*x+2675.0/6.0*y-125.0*x*x-8875.0/6.0*x*y-7375.0/4.0*y*y+625.0/12.0*x*x*x+3125.0/2.0*x*x*y \
        +16875.0/4.0*x*y*y+8125.0/3.0*y*y*y-3125.0/6.0*x*x*x*y-9375.0/4.0*x*x*y*y-3125.0*x*y*y*y-15625.0/12.0*y*y*y*y
        gphi[ 6,1] =  25.0-1925.0/12.0*x-1925.0/6.0*y+8875.0/24.0*x*x+8875.0/6.0*x*y+8875.0/8.0*y*y-4375.0/12.0*x*x*x-4375.0/2.0*x*x*y \
        -13125.0/4.0*x*y*y-4375.0/3.0*y*y*y+3125.0/24.0*x*x*x*x+3125.0/3.0*x*x*x*y+9375.0/4.0*x*x*y*y+6250.0/3.0*x*y*y*y+15625.0/24.0*y*y*y*y
        gphi[ 1,1] =  -1925.0/12.0*x+8875.0/12.0*x*x+8875.0/12.0*x*y-4375.0/4.0*x*x*x-4375.0/2.0*x*x*y-4375.0/4.0*x*y*y \
        +3125.0/6.0*x*x*x*x+3125.0/2.0*x*x*x*y+3125.0/2.0*x*x*y*y+3125.0/6.0*x*y*y*y
        gphi[ 2,1] =  1175.0/12.0*x-8875.0/12.0*x*x-250.0*x*y+5625.0/4.0*x*x*x+3125.0/2.0*x*x*y+625.0/4.0*x*y*y-3125.0/4.0*x*x*x*x \
        -3125.0/2.0*x*x*x*y-3125.0/4.0*x*x*y*y
        gphi[ 3,1] =  -75.0/2.0*x+3875.0/12.0*x*x+125.0/3.0*x*y-3125.0/4.0*x*x*x-625.0/2.0*x*x*y+3125.0/6.0*x*x*x*x+3125.0/6.0*x*x*x*y
        gphi[ 4,1] =  25.0/4.0*x-1375.0/24.0*x*x+625.0/4.0*x*x*x-3125.0/24.0*x*x*x*x
        gphi[ 7,1] =  250.0*x-5875.0/6.0*x*x-5875.0/3.0*x*y+1250.0*x*x*x+5000.0*x*x*y+3750.0*x*y*y-3125.0/6.0*x*x*x*x \
        -3125.0*x*x*x*y-9375.0/2.0*x*x*y*y-6250.0/3.0*x*y*y*y
        gphi[ 8,1] =  -125.0*x+3625.0/4.0*x*x+1125.0/2.0*x*y-3125.0/2.0*x*x*x-6875.0/2.0*x*x*y-1875.0/4.0*x*y*y+3125.0/4.0*x*x*x*x \
        +3125.0*x*x*x*y+9375.0/4.0*x*x*y*y
        gphi[ 9,1] =  125.0/3.0*x-2125.0/6.0*x*x-250.0/3.0*x*y+2500.0/3.0*x*x*x+625.0*x*x*y-3125.0/6.0*x*x*x*x-3125.0/3.0*x*x*x*y
        gphi[12,1] =  -125.0*x+1125.0/4.0*x*x+3625.0/2.0*x*y-625.0/4.0*x*x*x-6875.0/2.0*x*x*y-9375.0/2.0*x*y*y+3125.0/2.0*x*x*x*y \
        +9375.0/2.0*x*x*y*y+3125.0*x*y*y*y
        gphi[13,1] =  125.0/4.0*x-375.0/2.0*x*x-375.0*x*y+625.0/4.0*x*x*x+4375.0/2.0*x*x*y+1875.0/4.0*x*y*y-3125.0/2.0*x*x*x*y-9375.0/4.0*x*x*y*y
        gphi[16,1] =  125.0/3.0*x-125.0/3.0*x*x-2125.0/3.0*x*y+625.0*x*x*y+2500.0*x*y*y-3125.0/2.0*x*x*y*y-6250.0/3.0*x*y*y*y



def Shape_TensorLegendre(dim, p, x, phi):
    if p == 0:
        phi[:] = 1.
        return

    if dim == 1:
        BasisLegendre1D(x, p, phi, None)
    elif dim == 2:
        BasisLegendre2D(x, p, phi, None)

def Grad_TensorLegendre(dim, p, x, gphi):
    if p == 0:
        gphi[:,:] = 0.
        return 
    if dim == 1:
        BasisLegendre1D(x, p, None, gphi)
    if dim == 2:
        BasisLegendre2D(x, p, None, gphi)

def BasisLegendre2D(x, p, phi, gphi):
    if gphi is not None:
        gphix = np.zeros(p+1); gphiy = np.zeros(p+1)
    else:
        gphix = None; gphiy = None
    # Always need phi
    phix = np.zeros(p+1); phiy = np.zeros(p+1)


    BasisLegendre1D(x[0], p, phix, gphix)
    BasisLegendre1D(x[1], p, phiy, gphiy)

    if phi is not None:
        phi[:] = np.reshape(np.outer(phix, phiy), (-1,), 'F')
    if gphi is not None:
        gphi[:,0] = np.reshape(np.outer(gphix, phiy), (-1,), 'F')
        gphi[:,1] = np.reshape(np.outer(phix, gphiy), (-1,), 'F')

def BasisLegendre1D(x, p, phi, gphi):

    if phi is not None:
        if p >= 0:
            phi[0]  = 1.
        if p>=1:
            phi[1]  = x
        if p>=2:
            phi[2]  = 0.5*(3.*x*x - 1.)
        if p>=3:
            phi[3]  = 0.5*(5.*x*x*x - 3.*x)
        if p>=4:
            phi[4]  = 0.125*(35.*x*x*x*x - 30.*x*x + 3.)
        if p>=5:
            phi[5]  = 0.125*(63.*x*x*x*x*x - 70.*x*x*x + 15.*x)
        if p>=6:
            phi[6]  = 0.0625*(231.*x*x*x*x*x*x - 315.*x*x*x*x + 105.*x*x -5.)
        if p==7:
            phi[7]  = 0.0625*(429.*x*x*x*x*x*x*x - 693.*x*x*x*x*x + 315.*x*x*x - 35.*x)
        if p>7:
            raise NotImplementedError("Legendre Polynomial > 7 not supported")

    if gphi is not None:
        if p >= 0:
            gphi[0] = 0.
        if p>=1:
            gphi[1] = 1.
        if p>=2:
            gphi[2] = 3.*x
        if p>=3:
            gphi[3] = 0.5*(15.*x*x - 3.)
        if p>=4:
            gphi[4] = 0.125*(35.*4.*x*x*x - 60.*x)
        if p>=5:
            gphi[5] = 0.125*(63.*5.*x*x*x*x - 210.*x*x + 15.)
        if p>=6:
            gphi[6] = 0.0625*(231.*6.*x*x*x*x*x - 315.*4.*x*x*x + 210.*x)
        if p==7:
            gphi[7] = 0.0625*(429.*7.*x*x*x*x*x*x - 693.*5.*x*x*x*x + 315.*3.*x*x - 35.)
        if p>7:
            raise NotImplementedError("Legendre Polynomial > 7 not supported")

class BasisData(object):
    '''
    Class: IFace
    --------------------------------------------------------------------------
    This is a class defined to encapsulate the temperature table with the 
    relevant methods
    '''
    def __init__(self,basis,Order,nq=0,mesh=None):
        '''
        Method: __init__
        --------------------------------------------------------------------------
        This method initializes the temperature table. The table uses a
        piecewise linear function for the constant pressure specific heat 
        coefficients. The coefficients are selected to retain the exact 
        enthalpies at the table points.
        '''
        self.Basis = basis
        self.Order = Order
        self.nn = Order2nNode(self.Basis, self.Order)
        self.nq = nq
        self.nnqmax = self.nn * self.nq
        self.dim = Shape2Dim[Basis2Shape[self.Basis]]
        self.Phi = None
        self.GPhi = None
        self.gPhi = None
        self.face = -1

    def PhysicalGrad(self, JData):
        nq = self.nq
        if nq != JData.nq and JData.nq != 1:
            raise Exception("Quadrature doesn't match")
        dim = JData.dim
        if dim != self.dim:
            raise Exception("Dimensions don't match")
        nn = self.nn
        GPhi = self.GPhi 
        if GPhi is None:
            raise Exception("GPhi is an empty list")

        if self.gPhi is None or self.gPhi.shape != (nq,nn,dim):
            self.gPhi = np.zeros([nq,nn,dim])
        else:
            self.gPhi *= 0.

        gPhi = self.gPhi

        if gPhi.shape != GPhi.shape:
            raise Exception("gPhi and GPhi are different sizes")

        for iq in range(nq):
            G = GPhi[iq,:,:] # [nn,dim]
            g = gPhi[iq,:,:] # [nn,dim]
            iJ = JData.iJ[iq*(JData.nq != 1),:,:] # [dim,dim]
            g[:] = np.transpose(np.matmul(iJ.transpose(),G.transpose()))

        return gPhi

    def EvalBasis(self, xq, Get_Phi=True, Get_GPhi=False, Get_gPhi=False, JData=None):
        if Get_Phi:
            self.Phi = GetShapes(self.Basis, self.Order, self.nq, xq, self.Phi)
        if Get_GPhi:
            self.GPhi = GetGrads(self.Basis, self.Order, self.dim, self.nq, xq, self.GPhi)
        if Get_gPhi:
            if not JData:
                raise Exception("Need jacobian data")
            self.gPhi = self.PhysicalGrad(JData)

    def EvalBasisOnFace(self, mesh, egrp, face, xq, xelem=None, Get_Phi=True, Get_GPhi=False, Get_gPhi=False, JData=False):
        self.face = face
        basis = mesh.ElemGroups[egrp].QBasis
        if xelem is None or xelem.shape != (self.nq, mesh.Dim):
            xelem = np.zeros([self.nq, mesh.Dim])
        xelem = Mesh.RefFace2Elem(Basis2Shape[basis], face, self.nq, xq, xelem)
        self.EvalBasis(xelem, Get_Phi, Get_GPhi, Get_gPhi, JData)

        return xelem

    def EvalBasisOnFaceADER(self, mesh, basis, egrp, face, xq, xelem=None, Get_Phi=True, Get_GPhi=False, Get_gPhi=False, JData=False):
        self.face = face
        #basis = mesh.ElemGroups[egrp].QBasis
        if Shape2Dim[Basis2Shape[basis]] == ShapeType.Quadrilateral:
            if xelem is None or xelem.shape != (self.nq, mesh.Dim+1):
                xelem = np.zeros([self.nq, mesh.Dim+1])
            xelem = Mesh.RefFace2Elem(Basis2Shape[basis], face, self.nq, xq, xelem)
        elif Shape2Dim[Basis2Shape[basis]] == ShapeType.Segment:
            if xelem is None or xelem.shape != (self.nq, mesh.Dim):
                xelem = np.zeros([self.nq, mesh.Dim])
            xelem = Mesh.RefFace2Elem(Basis2Shape[basis], face, self.nq, xq, xelem)
        self.EvalBasis(xelem, Get_Phi, Get_GPhi, Get_gPhi, JData)

        return xelem
    	

class JacobianData(object):
    '''
    Class: IFace
    --------------------------------------------------------------------------
    This is a class defined to encapsulate the temperature table with the 
    relevant methods
    '''
    def __init__(self,mesh):
        '''
        Method: __init__
        --------------------------------------------------------------------------
        This method initializes the temperature table. The table uses a
        piecewise linear function for the constant pressure specific heat 
        coefficients. The coefficients are selected to retain the exact 
        enthalpies at the table points.
        '''
        self.nq = 0
        self.dim = mesh.Dim
        self.detJ = None
        self.J = None
        self.iJ = None
        self.A = None
        self.GPhi = None

        # self.ElemJacobian(self,egrp,elem,nq,xq,mesh,Get_detJ,Get_J,Get_iJ)

    def ElemJacobian(self,egrp,elem,nq,xq,mesh,Get_detJ=False,Get_J=False,Get_iJ=False,
            UniformJacobian=False):
        EGroup = mesh.ElemGroups[egrp]
        basis = EGroup.QBasis
        Order = EGroup.QOrder
        Shape = Basis2Shape[basis]
        if Order == 1 and Shape != ShapeType.Quadrilateral:
            nq = 1
        # if UniformJacobian:
        #     if nq == 1:
        #         # Already calculated
        #         return
        #     else:
        #         # Need to calculate
        #         nq = 1

        nn = Order2nNode(basis, Order)
        dim = Shape2Dim[Basis2Shape[basis]]

        ## Check if we need to resize or recalculate 
        if self.dim != dim or self.nq != nq: Resize = True
        else: Resize = False

        # if self.GPhi.shape != (nq,nn,dim):
        #     self.GPhi = GetGrads(Basis, Order, dim, nq, xq)
        self.GPhi = GetGrads(basis, Order, dim, nq, xq, self.GPhi)
        GPhi = self.GPhi

        self.dim = dim
        if dim != mesh.Dim:
            raise Exception("Dimensions don't match")

        self.nq = nq

        if Get_J and (Resize or self.J is None): 
            self.J = np.zeros([nq,dim,dim])
        if Get_detJ and (Resize or self.detJ is None): 
            self.detJ = np.zeros([nq,1])
        if Get_iJ and (Resize or self.iJ is None): 
            self.iJ = np.zeros([nq,dim,dim])

        if Resize or self.A is None:
            self.A = np.zeros([dim,dim])

        A = self.A
        Elem2Nodes = EGroup.Elem2Nodes[elem]
        for iq in range(nq):
            G = GPhi[iq,:,:]
            A[:] = 0.
            for i in range(nn):
                for j in range(dim):
                    for k in range(dim):
                        A[j,k] += mesh.Coords[Elem2Nodes[i],j]*G[i,k]
            if Get_J:
                self.J[iq,:] = A[:]
                # for i in range(dim):
                #     for j in range(dim):
                #         self.J[iq,i,j] = A[i,j]
            if Get_detJ: detJ_ = self.detJ[iq]
            else: detJ_ = None
            if Get_iJ: iJ_ = self.iJ[iq,:,:]
            else: iJ_ = None 
            MatDetInv(A, dim, detJ_, iJ_)

            if detJ_ is not None and detJ_ <= 0.:
                raise Exception("Nonpositive Jacobian (egrp = %d, elem = %d)" % (egrp, elem))



