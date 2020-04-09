import numpy as np
from enum import IntEnum
from General import *
from Quadrature import *
from Math import *
import Mesh
import code
from Data import ArrayList, GenericData


Basis2Shape = {
    BasisType.LagrangeSeg : ShapeType.Segment,
    BasisType.LagrangeQuad : ShapeType.Quadrilateral,
    BasisType.LagrangeTri : ShapeType.Triangle,
    BasisType.LegendreSeg : ShapeType.Segment,
    BasisType.LegendreQuad : ShapeType.Quadrilateral,
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
    BasisType.LagrangeSeg : np.array([[-1.],[1.]]),
    BasisType.LagrangeQuad : np.array([[-1.,-1.],[1.,-1.],
                                [-1.,1.],[1.,1.]]),
    BasisType.LagrangeTri : np.array([[0.,0.],[1.,0.],
                                [0.,1.]]),
    BasisType.LegendreSeg : np.array([[-1.],[1.]]),
    BasisType.LegendreQuad : np.array([[-1.,-1.],[1.,-1.],
                                [-1.,1.],[1.,1.]]),
}

def order_to_num_basis_coeff(basis,p):
    '''
    Method: order_to_num_basis_coeff
    -------------------
    This method specifies the number of basis coefficients

    INPUTS:
        basis: type of basis function
        p: order of polynomial space

    OUTPUTS: 
        nb: number of basis coefficients
    '''
    Shape = Basis2Shape[basis]
    if Shape == ShapeType.Point:
    	nb = 1
    elif Shape == ShapeType.Segment:
        nb = p + 1
    elif Shape == ShapeType.Triangle:
        nb = (p + 1)*(p + 2)//2
    elif Shape == ShapeType.Quadrilateral:
        nb = (p + 1)**2
    else:
    	raise Exception("Shape not supported")

    return nb


def local_q1_face_nodes(basis, p, face, fnodes=None):
    '''
    Method: local_q1_face_nodes
    -------------------
    ???

    INPUTS:
        basis: type of basis function
        p: order of polynomial space
        face: face value in ref space

    OUTPUTS: 
        fnodes: index of face nodes
        nfnode: number of face nodes
    '''
    if basis == BasisType.LagrangeSeg: 
        nfnode = 1
        if fnodes is None: fnodes = np.zeros(nfnode, dtype=int)
        if face == 0:
            fnodes[0] = 0
        elif face == 1:
            fnodes[0] = p
        else:
            raise IndexError
    elif basis == BasisType.LagrangeQuad:
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
    elif basis == BasisType.LagrangeTri:
        nfnode = 2
        if fnodes is None: fnodes = np.zeros(nfnode, dtype=int)
        if face == 0:
            fnodes[0] = p; fnodes[1] = (p+1)*(p+2)//2-1
        elif face == 1:
            fnodes[0] = (p+1)*(p+2)//2-1; fnodes[1] = 0
        elif face == 2:
            fnodes[0] = 0; fnodes[1] = p
        else:
            raise IndexError
    else:
        raise NotImplementedError

    return fnodes, nfnode


def local_face_nodes(basis, p, face, fnodes=None):
    '''
    Method: local_q1_face_nodes
    -------------------
    ???

    INPUTS:
        basis: type of basis function
        p: order of polynomial space
        face: face value in ref space

    OUTPUTS: 
        fnodes: index of face nodes
        nfnode: number of face nodes
    '''
    if p < 1:
        raise ValueError

    if basis == BasisType.LagrangeQuad:
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

        fnodes[:] = i0 + np.arange(p+1, dtype=int)*d
    elif basis == BasisType.LagrangeTri:
        nfnode = p+1
        if fnodes is None: fnodes = np.zeros(nfnode, dtype=int)
        if face == 0:
            i0 = p; d0 = p; d1 = -1
        elif face == 1:
            i0 = (p+1)*(p+2)//2-1; d0 = -2; d1 = -1
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


def equidistant_nodes_1D_range(start, stop, nnode):
    '''
    Method: equidistant_nodes_1D_range
    -----------------------------------
    Calculate the 1D coordinates in ref space

    INPUTS:
        start: start of ref space (default: -1)
        stop:  end of ref space (default: 1)
        nnode: num of nodes in 1D ref space

    OUTPUS: 
        xnode: coordinates of nodes in 1D ref space
    '''
    if nnode <= 1:
        raise ValueError
    if stop <= start:
        raise ValueError
    # Note: this is faster than linspace unless p is large
    xnode = np.zeros(nnode)
    dx = (stop-start)/float(nnode-1)
    for i in range(nnode): xnode[i] = start + float(i)*dx

    return xnode

def equidistant_nodes(basis, p, xn=None):
    '''
    Method: equidistant_nodes
    --------------------------
    Calculate the coordinates in ref space

    INPUTS:
        basis: type of basis function
        p: order of polynomial space
        
    OUTPUTS: 
        xn: coordinates of nodes in ref space
    '''
    nb = order_to_num_basis_coeff(basis, p)

    shape = Basis2Shape[basis]
    dim = Shape2Dim[shape]

    adim = nb,dim
    if xn is None or xn.shape != adim:
        xn = np.zeros(adim)

    if p == 0:
        xn[:] = 0.0 # 0.5
        return xn, nb

    if shape == ShapeType.Segment:
        xn[:,0] = equidistant_nodes_1D_range(-1., 1., nb)
    elif shape == ShapeType.Quadrilateral:
        xseg = equidistant_nodes_1D_range(-1., 1., p+1)
        # n = 0
        # for i in range(p+1):
        #     xn[n:n+p+1,0] = xseg
        #     xn[n:n+p+1,1] = xseg[i]
        #     n += p+1
        xn[:,0] = np.tile(xseg, (p+1,1)).reshape(-1)
        xn[:,1] = np.repeat(xseg, p+1, axis=0).reshape(-1)
    elif shape == ShapeType.Triangle:
        n = 0
        xseg = equidistant_nodes_1D_range(0., 1., p+1)
        for j in range(p+1):
            xn[n:n+p+1-j,0] = xseg[:p+1-j]
            xn[n:n+p+1-j,1] = xseg[j]
            n += p+1-j
        # for j in range(p):
        #     for i in range(p-j):
        #         xn[k][0] = float(i)/float(p)
        #         xn[k][1] = float(j)/float(p)

    return xn, nb
  

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


def get_elem_mass_matrix(mesh, basis, order, elem=-1, PhysicalSpace=False, StaticData=None):
    '''
    Method: get_elem_mass_matrix
    --------------------------
    Calculate the mass matrix

    INPUTS:
        mesh: mesh object
        basis: type of basis function
        order: solution order
        PhysicalSpace: Flag to calc matrix in physical or reference space (default: False {reference space})
        elem: element index

    OUTPUTS: 
        MM: mass matrix  
    '''
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
        QuadOrder,QuadChanged = get_gaussian_quadrature_elem(mesh, mesh.QBasis, order*2, quadData=quadData)
    else:
        QuadOrder = order*2
        QuadChanged = True

    if QuadChanged:
        quadData = QuadData(mesh, basis, EntityType.Element, QuadOrder)

    quad_pts = quadData.quad_pts
    quad_wts = quadData.quad_wts
    nq = quad_pts.shape[0]

    if QuadChanged:
        PhiData = BasisData(basis,order,mesh)
        PhiData.eval_basis(quad_pts, Get_Phi=True)

    if PhysicalSpace:
        JData.element_jacobian(mesh,elem,quad_pts,get_djac=True)
        if JData.nq == 1:
            djac = np.full(nq, JData.djac[0])
        else:
            djac = JData.djac
    else:
        djac = np.full(nq, 1.)

    nb = PhiData.Phi.shape[1]
    #nb = PhiData.nn
    phi = PhiData.Phi

    MM = np.zeros([nb,nb])
    # for i in range(nn):
    #     for j in range(nn):
    #         t = 0.
    #         for iq in range(nq):
    #             t += phi[iq,i]*phi[iq,j]*wq[iq]*djac[iq] # JData.djac[iq*(JData.nq != 1)]
    #         MM[i,j] = t

    MM[:] = np.matmul(phi.transpose(), phi*quad_wts*djac)

    StaticData.pnq = nq
    StaticData.quadData = quadData
    StaticData.PhiData = PhiData
    StaticData.JData = JData

    return MM, StaticData

def get_elem_inv_mass_matrix(mesh, basis, order, elem=-1, PhysicalSpace=False, StaticData=None):
    '''
    Method: get_elem_inv_mass_matrix
    ---------------------------------
    Calculate the inverse mass matrix

    INPUTS:
        mesh: mesh object
        basis: type of basis function
        order: solution order
        elem: element index
        PhysicalSpace: Flag to calc matrix in physical or reference space (default: False {reference space})

    OUTPUTS: 
        iMM: inverse mass matrix  
    '''
    MM, StaticData = get_elem_mass_matrix(mesh, basis, order, elem, PhysicalSpace, StaticData)
    
    iMM = np.linalg.inv(MM) 

    return iMM, StaticData

def get_elem_inv_mass_matrix_ader(mesh, basis, order, elem=-1, PhysicalSpace=False, StaticData=None):
    '''
    Method: get_elem_inv_mass_matrix_ader
    --------------------------------------
    Calculate the inverse mass matrix for ADER-DG prediction step

    INPUTS:
        mesh: mesh object
        basis: type of basis function
        order: solution order
        PhysicalSpace: Flag to calc matrix in physical or reference space (default: False {reference space})
        elem: element index

    OUTPUTS: 
        iMM: inverse mass matrix for ADER-DG predictor step
    '''
    MM, StaticData = get_elem_mass_matrix_ader(mesh, basis, order, elem, PhysicalSpace, StaticData)

    iMM = np.linalg.inv(MM)

    return iMM, StaticData

def get_stiffness_matrix(mesh, basis, order, elem, StaticData=None):
    '''
    Method: get_stiffness_matrix
    --------------------------------------
    Calculate the stiffness_matrix

    INPUTS:
        mesh: mesh object
        basis: type of basis function
        order: solution order
        elem: element index

    OUTPUTS: 
        SM: stiffness matrix
    '''
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

    QuadOrder,QuadChanged = get_gaussian_quadrature_elem(mesh, mesh.QBasis, order*2, quadData=quadData)
    if QuadChanged:
        quadData = QuadData(mesh, mesh.QBasis, EntityType.Element, QuadOrder)

    quad_pts = quadData.quad_pts
    quad_wts = quadData.quad_wts
    nq = quad_pts.shape[0]

    if QuadChanged:
        PhiData = BasisData(basis,order,mesh)
        PhiData.eval_basis(quad_pts, Get_Phi=True, Get_GPhi=True)

    JData.element_jacobian(mesh,elem,quad_pts,get_djac=True,get_ijac=True)
    PhiData.eval_basis(quad_points, Get_gPhi=True, JData=JData)

    nb = PhiData.Phi.shape[1]

    phi = PhiData.Phi
    gPhi = PhiData.gPhi
    SM = np.zeros([nb,nb])
    for i in range(nb):
        for j in range(nb):
            t = 0.
            for iq in range(nq):
                t += gPhi[iq,i,0]*phi[iq,j]*wq[iq]*JData.djac[iq*(JData.nq != 1)]
            SM[i,j] = t

    StaticData.pnq = nq
    StaticData.quadData = quadData
    StaticData.PhiData = PhiData
    StaticData.JData = JData

    return SM, StaticData


def get_stiffness_matrix_ader(mesh, basis, order, elem, gradDir, StaticData=None):
    '''
    Method: get_stiffness_matrix_ader
    --------------------------------------
    Calculate the stiffness matrix for ADER-DG prediction step

    INPUTS:
        mesh: mesh object
        basis: type of basis function
        order: solution order
        elem: element index
        gradDir: direction of gradient calc

    OUTPUTS: 
        SM: stiffness matrix for ADER-DG
    '''
    if StaticData is None:
        pnq = -1
        quadData = None
        PhiData = None
        StaticData = GenericData()
    else:
        nq = StaticData.pnq
        quadData = StaticData.quadData
        PhiData = StaticData.PhiData
        # JData = StaticData.JData

    QuadOrder,QuadChanged = get_gaussian_quadrature_elem(mesh, basis, order*2., quadData=quadData)
    #Add one to QuadOrder to adjust the mesh.Dim addition in get_gaussian_quadrature_elem.
    QuadOrder+=1
    if QuadChanged:
        quadData = QuadDataADER(mesh, basis, EntityType.Element, QuadOrder)

    quad_pts = quadData.quad_pts
    quad_wts = quadData.quad_wts
    nq = quad_pts.shape[0]

    if QuadChanged:
        PhiData = BasisData(basis,order,mesh)
        PhiData.eval_basis(quad_pts, Get_Phi=True, Get_GPhi=True)

    nb = PhiData.Phi.shape[1]

    phi = PhiData.Phi
    GPhi = PhiData.GPhi
    SM = np.zeros([nb,nb])
    # for i in range(nn):
    #     for j in range(nn):
    #         t = 0.
    #         for iq in range(nq):
    #             t += GPhi[iq,i,gradDir]*phi[iq,j]*wq[iq]
    #         SM[i,j] = t
    #code.interact(local=locals())
    SM[:] = np.matmul(GPhi[:,:,gradDir].transpose(),phi*quad_wts)
    StaticData.pnq = nq
    StaticData.quadData = quadData
    StaticData.PhiData = PhiData

    return SM, StaticData

def get_temporal_flux_ader(mesh, basis1, basis2, order, elem=-1, PhysicalSpace=False, StaticData=None):
    '''
    Method: get_temporal_flux_ader
    --------------------------------------
    Calculate the temporal flux matrix for ADER-DG prediction step

    INPUTS:
        mesh: mesh object
        basis1: type of basis function
        basis2: type of basis function
        order: solution order
        elem: element index
        PhysicalSpace: Flag to calc matrix in physical or reference space (default: False {reference space})

    OUTPUTS: 
        FT: flux matrix for ADER-DG

    NOTES:
        Can work at tau_n and tau_n+1 depending on basis combinations
    '''

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
        face = 2 
    else:
        face = 0
    QuadOrder,QuadChanged = get_gaussian_quadrature_elem(mesh, mesh.QBasis, order*2, quadData=quadData)
  
    if QuadChanged:
        quadData = QuadData(mesh, mesh.QBasis, EntityType.Element, QuadOrder)

    quad_pts = quadData.quad_pts
    quad_wts = quadData.quad_wts
    nq = quad_pts.shape[0]
    if QuadChanged:
        if basis1 == basis2:
            face = 2
            basis = basis1
            PhiData = BasisData(basis,order,mesh)
            PsiData = PhiData
            xelem = np.zeros([nq,mesh.Dim+1])
            PhiData.eval_basis_on_face_ader(mesh, basis, face, quad_pts, xelem, Get_Phi=True)
            PsiData.eval_basis_on_face_ader(mesh, basis, face, quad_pts, xelem, Get_Phi=True)
        else:
            face = 0
            PhiData = BasisData(basis1,order,mesh)
            PsiData = BasisData(basis2,order,mesh)
            xelemPhi = np.zeros([nq,mesh.Dim+1])
            xelemPsi = np.zeros([nq,mesh.Dim])
            PhiData.eval_basis_on_face_ader(mesh, basis1, face, quad_pts, xelemPhi, Get_Phi=True)
            #PsiData.eval_basis_on_face_ader(mesh, basis2, face, xq, xelemPsi, Get_Phi=True)
            PsiData.eval_basis(quad_pts, Get_Phi=True, Get_GPhi=False)


    nb_st = PhiData.Phi.shape[1] #PhiData.nn
    nb = PsiData.Phi.shape[1] #PsiData.nn

    FT = np.zeros([nb_st,nb])
    # for i in range(nn1):
    #     for j in range(nn2):
    #         t = 0.
    #         for iq in range(nq):
    #             t += phi[iq,i]*psi[iq,j]*wq[iq]
    #         MM[i,j] = t

    FT[:] = np.matmul(PhiData.Phi.transpose(),PsiData.Phi*quad_wts)
    StaticData.pnq = nq
    StaticData.quadData = quadData
    StaticData.PhiData = PhiData
 
    return FT, StaticData


def get_elem_mass_matrix_ader(mesh, basis, order, elem=-1, PhysicalSpace=False, StaticData=None):
    '''
    Method: get_elem_mass_matrix_ader
    --------------------------------------
    Calculate the mass matrix for ADER-DG prediction step

    INPUTS:
        mesh: mesh object
        basis: type of basis function
        order: solution order
        elem: element index
        PhysicalSpace: Flag to calc matrix in physical or reference space (default: False {reference space})

    OUTPUTS: 
        MM: mass matrix for ADER-DG
    '''
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
        QuadOrder,QuadChanged = get_gaussian_quadrature_elem(mesh, mesh.QBasis, order*2, quadData=quadData)
    else:
        QuadOrder = order*2 + 1 #Add one for ADER method
        QuadChanged = True

    if QuadChanged:
        quadData = QuadDataADER(mesh, basis, EntityType.Element, QuadOrder)


    quad_pts = quadData.quad_pts
    quad_wts = quadData.quad_wts
    nq = quad_pts.shape[0]
    if QuadChanged:

        PhiData = BasisData(basis,order,mesh)
        PhiData.eval_basis(quad_pts, Get_Phi=True)

    if PhysicalSpace:
        JData.element_jacobian(mesh,elem,quad_pts,get_djac=True)
        if JData.nq == 1:
            djac = np.full(nq, JData.djac[0])
        else:
            djac = JData.djac
    else:
        djac = np.full(nq, 1.)

    nb_st = PhiData.Phi.shape[1]
    MM = np.zeros([nb_st,nb_st])
    # for i in range(nn):
    #     for j in range(nn):
    #         t = 0.
    #         for iq in range(nq):
    #             t += phi[iq,i]*phi[iq,j]*wq[iq]*djac[iq]
    #         MM[i,j] = t

    MM[:] = np.matmul(PhiData.Phi.transpose(), PhiData.Phi*quad_wts*djac)

    StaticData.pnq = nq
    StaticData.quadData = quadData
    StaticData.PhiData = PhiData
    StaticData.JData = JData

    return MM, StaticData

def get_projection_matrix(mesh, basis, basis_old, order, order_old, iMM):
    '''
    Method: get_projection_matrix
    --------------------------------------
    Calculate the projection matrix to increase order

    INPUTS:
        mesh: mesh object
        basis: type of basis function
        basis_old: type of basis function from previous order
        order: solution order
        order_old: previous solution order
        iMM: inverse mass matrix

    OUTPUTS: 
        PM: projection matrix
    '''
    QuadOrder = np.amax([order_old+order, 2*order])
    quadData = QuadData(mesh, basis, EntityType.Element, QuadOrder)

    quad_pts = quadData.quad_pts
    quad_wts = quadData.quad_wts
    nq = quad_pts.shape[0]

    PhiData_old = BasisData(basis_old, order_old, mesh)
    PhiData_old.eval_basis(quad_pts, Get_Phi=True)

    phi_old = PhiData_old.Phi
    nb_old = phi_old.shape[1]

    PhiData = BasisData(basis, order, mesh)
    PhiData.eval_basis(quad_pts, Get_Phi=True)
    phi = PhiData.Phi
    nb = phi.shape[1]

    A = np.zeros([nb,nb_old])
    # for i in range(nn):
    #     for j in range(nn_old):
    #         t = 0.
    #         for iq in range(nq):
    #             t += phi[iq,i]*phi_old[iq,j]*wq[iq] # JData.djac[iq*(JData.nq != 1)]
    #         A[i,j] = t
    A = np.matmul(phi.transpose(), phi_old*quad_wts)

    PM = np.matmul(iMM,A)

    return PM


def get_inv_stiffness_matrix(mesh, basis, order, elem, StaticData=None):
    '''
    Method: get_inv_stiffness_matrix
    --------------------------------------
    Calculate the inverse stiffness matrix (Currently not used)

    INPUTS:
        mesh: mesh object
        basis: type of basis function
        order: solution order
        elem: element index

    OUTPUTS: 
        iSM: inverse stiffness matrix
    '''
    SM, StaticData = get_stiffness_matrix(mesh, basis, order, elem, StaticData)

    iSM = np.linalg.inv(SM) 

    return iSM, StaticData

def get_inv_mass_matrices(mesh, EqnSet, solver=None):
    '''
    Method: compute_inv_mass_matrices
    --------------------------------------
    Calculate the inverse mass matrices

    INPUTS:
        mesh: mesh object
        EqnSet: type of equation set (i.e. scalar, euler, etc...)
        solver: type of solver (i.e. DG, ADER-DG, etc...)

    OUTPUTS: 
        iMM_all: all inverse mass matrices
    '''
    basis = EqnSet.Basis
    order = EqnSet.Order
    nb = order_to_num_basis_coeff(basis, order)
    iMM_all = np.zeros([mesh.nElem, nb, nb])

    StaticData = None

    # Uniform mesh?
    ReCalcMM = True
    if solver is not None:
        ReCalcMM = not solver.Params["UniformMesh"]
    for elem in range(mesh.nElem):
        if elem == 0 or ReCalcMM:
            # Only recalculate if not using uniform mesh
            iMM,StaticData = get_elem_inv_mass_matrix(mesh, basis, order, elem, True, StaticData)
        iMM_all[elem] = iMM

    if solver is not None:
        solver.DataSet.MMinv_all = iMM_all

    return iMM_all


def get_shapes(basis, order, quad_pts, phi=None):
    '''
    Method: get_shapes
    --------------------
    Choose basis evaluation based on order and basis

    INPUTS:
        basis: type of basis function
        order: solution order
        quad_pts: coordinates of nodes in ref space

    OUTPUTS: 
        phi: evaluated basis
    '''
    nq = quad_pts.shape[0]
    nb = order_to_num_basis_coeff(basis, order)

    if phi is None or phi.shape != (nq,nb):
        phi = np.zeros([nq,nb])
    else:
        phi[:] = 0.

    if basis == BasisType.LagrangeSeg:
        # for iq in range(nq): 
        #     shape_tensor_lagrange(1, Order, xq[iq], phi[iq,:])
        # code.interact(local=locals())
        shape_tensor_lagrange(1, order, quad_pts, phi)
    elif basis == BasisType.LagrangeQuad:
        # for iq in range(nq): 
        #     shape_tensor_lagrange(2, Order, xq[iq], phi[iq,:])
        shape_tensor_lagrange(2, order, quad_pts, phi)
    elif basis == BasisType.LagrangeTri:
        shape_lagrange_tri(order, quad_pts, phi)
    elif basis == BasisType.LegendreSeg:
        # for iq in range(nq):
        #     shape_tensor_legendre(1, Order, xq[iq], phi[iq,:])
        shape_tensor_legendre(1, order, quad_pts, phi)
    elif basis == BasisType.LegendreQuad:
        # for iq in range(nq):
        #     shape_tensor_legendre(2, Order, xq[iq], phi[iq,:])
        shape_tensor_legendre(2, order, quad_pts, phi)
    else:
        raise Exception("Basis not supported")

    return phi


def get_grads(basis, order, dim, quad_pts, GPhi=None):
    '''
    Method: get_grads
    --------------------
    Choose basis gradient evaluation based on order and basis

    INPUTS:
        basis: type of basis function
        order: solution order
        dim: dimension of mesh
        quad_pts: coordinates of nodes in ref space

    OUTPUTS: 
        Gphi: evaluated gradient of basis
    '''
    nq = quad_pts.shape[0]
    nb = order_to_num_basis_coeff(basis, order)

    if GPhi is None or GPhi.shape != (nq,nb,dim):
        GPhi = np.zeros([nq,nb,dim])
    else: 
        GPhi[:] = 0.

    if basis == BasisType.LagrangeSeg:
        # for iq in range(nq): 
        #     grad_tensor_lagrange(1, Order, xq[iq], GPhi[iq,:,:])
        grad_tensor_lagrange(1, order, quad_pts, GPhi)
    elif basis == BasisType.LagrangeQuad:
        # for iq in range(nq): 
            # grad_tensor_lagrange(2, Order, xq[iq], GPhi[iq,:,:])
        grad_tensor_lagrange(2, order, quad_pts, GPhi)
    elif basis == BasisType.LagrangeTri:
        grad_lagrange_tri(order, quad_pts, GPhi)
    elif basis == BasisType.LegendreSeg:
        # for iq in range(nq):
        #     grad_tensor_legendre(1, Order, xq[iq], GPhi[iq,:,:])
        grad_tensor_legendre(1, order, quad_pts, GPhi)
    elif basis == BasisType.LegendreQuad:
        # for iq in range(nq):
        #     grad_tensor_legendre(2, Order, xq[iq], GPhi[iq,:,:])
        grad_tensor_legendre(2, order, quad_pts, GPhi)
    else:
        raise Exception("Basis not supported")

    return GPhi


def get_lagrange_basis_1D(x, xnode, nnode, phi, gphi):
    '''
    Method: get_lagrange_basis_1D
    ------------------------------
    Calculates the 1D Lagrange basis functions

    INPUTS:
        x: coordinate of current node
        xnode: coordinates of nodes in 1D ref space
        nnode: number of nodes in 1D ref space
        
    OUTPUTS: 
        phi: evaluated basis 
        gphi: evaluated physical gradient of basis
    '''
    # for j in range(nnode):
    #     if phi is not None:
    #         pj = 1.
    #         for i in range(j): pj *= (x-xnode[i])/(xnode[j]-xnode[i])
    #         for i in range(j+1,nnode): pj *= (x-xnode[i])/(xnode[j]-xnode[i])
    #         phi[j] = pj
        # if gphi is not None:
        #     gphi[j] = 0.0;
        #     for i in range(nnode):
        #         if i != j:
        #             g = 1./(xnode[j] - xnode[i])
        #             for k in range(nnode):
        #                 if k != i and k != j:
        #                     g *= (x - xnode[k])/(xnode[j] - xnode[k])
        #             gphi[j] += g

    nnode = xnode.shape[0]
    mask = np.ones(nnode, bool)

    if phi is not None:
        phi[:] = 1.
        for j in range(nnode):
            mask[j] = False
            phi[:,j] = np.prod((x - xnode[mask])/(xnode[j] - xnode[mask]),axis=1)
            mask[j] = True

    if gphi is not None:
        gphi[:] = 0.

        for j in range(nnode):
            mask[j] = False
            for i in range(nnode):
                if i == j:
                    continue

                mask[i] = False
                if nnode > 2: 
                    gphi[:,j,:] += np.prod((x - xnode[mask])/(xnode[j] - xnode[mask]),
                        axis=1).reshape(-1,1)/(xnode[j] - xnode[i])
                else:
                    gphi[:,j,:] += 1./(xnode[j] - xnode[i])

                mask[i] = True
            mask[j] = True

def get_lagrange_basis_2D(x, xnode, nnode, phi, gphi):
    '''
    Method: get_lagrange_basis_2D
    ------------------------------
    Calculates the 2D Lagrange basis functions

    INPUTS:
        x: coordinate of current node
        xnode: coordinates of nodes in 1D ref space
        nnode: number of nodes in 1D ref space
        
    OUTPUTS: 
        phi: evaluated basis 
        gphi: evaluated gradient of basis
    '''
    if gphi is not None:
        gphix = np.zeros((x.shape[0],xnode.shape[0],1)); gphiy = np.zeros_like(gphix)
    else:
        gphix = None; gphiy = None
    # Always need phi
    phix = np.zeros((x.shape[0],xnode.shape[0])); phiy = np.zeros_like(phix)

    get_lagrange_basis_1D(x[:,0].reshape(-1,1), xnode, nnode, phix, gphix)
    get_lagrange_basis_1D(x[:,1].reshape(-1,1), xnode, nnode, phiy, gphiy)

    if phi is not None:
        for i in range(x.shape[0]):
            phi[i,:] = np.reshape(np.outer(phix[i,:], phiy[i,:]), (-1,), 'F')
    if gphi is not None:
        for i in range(x.shape[0]):
            gphi[i,:,0] = np.reshape(np.outer(gphix[i,:,0], phiy[i,:]), (-1,), 'F')
            gphi[i,:,1] = np.reshape(np.outer(phix[i,:], gphiy[i,:,0]), (-1,), 'F')


def shape_tensor_lagrange(dim, p, x, phi):
    '''
    Method: shape_tensor_lagrange
    ------------------------------
    Calculates lagrange bais for 1D or 2D quads

    INPUTS:
        dim: dimension of mesh
        p: order of polynomial space
        x: coordinate of current node

    OUTPUTS: 
        phi: evaluated basis 
    '''
    if p == 0:
    	phi[:] = 1.
    	return

    # nnode = p + 1
    # xnode = np.zeros(nnode)
    # dx = 2./float(p)
    # for i in range(nnode): xnode[i] = -1. + float(i)*dx
    # xnode, nnode = equidistant_nodes(BasisType.LagrangeSeg, p)
    # xnode.shape = nnode

    nnode = p+1
    xnode = equidistant_nodes_1D_range(-1., 1., nnode)

    if dim == 1:
        get_lagrange_basis_1D(x, xnode, nnode, phi, None)
    elif dim == 2:
        get_lagrange_basis_2D(x, xnode, nnode, phi, None)


def shape_lagrange_tri(p, xi, phi):
    '''
    Method: shape_lagrange_tri
    ------------------------------
    Calculates lagrange bais for triangles

    INPUTS:
        p: order of polynomial space
        xi: coordinates of current node

    OUTPUTS: 
        phi: evaluated basis 
    '''
    x = xi[:,0]; y = xi[:,1]

    if p == 0:
        phi[:] = 1.
    elif p == 1:
        phi[:,0] = 1-x-y
        phi[:,1] = x  
        phi[:,2] = y
    elif p == 2:
        phi[:,0] = 1.0-3.0*x-3.0*y+2.0*x*x+4.0*x*y+2.0*y*y
        phi[:,2] = -x+2.0*x*x
        phi[:,5] = -y+2.0*y*y
        phi[:,4] = 4.0*x*y
        phi[:,3] = 4.0*y-4.0*x*y-4.0*y*y
        phi[:,1] = 4.0*x-4.0*x*x-4.0*x*y
    elif p == 3:
        phi[:,0] = 1.0-11.0/2.0*x-11.0/2.0*y+9.0*x*x+18.0*x*y+9.0*y*y-9.0/2.0*x*x*x-27.0/2.0*x*x*y-27.0/2.0*x*y*y-9.0/2.0*y*y*y
        phi[:,3] = x-9.0/2.0*x*x+9.0/2.0*x*x*x
        phi[:,9] = y-9.0/2.0*y*y+9.0/2.0*y*y*y
        phi[:,6] = -9.0/2.0*x*y+27.0/2.0*x*x*y
        phi[:,8] = -9.0/2.0*x*y+27.0/2.0*x*y*y
        phi[:,7] = -9.0/2.0*y+9.0/2.0*x*y+18.0*y*y-27.0/2.0*x*y*y-27.0/2.0*y*y*y
        phi[:,4] = 9.0*y-45.0/2.0*x*y-45.0/2.0*y*y+27.0/2.0*x*x*y+27.0*x*y*y+27.0/2.0*y*y*y
        phi[:,1] = 9.0*x-45.0/2.0*x*x-45.0/2.0*x*y+27.0/2.0*x*x*x+27.0*x*x*y+27.0/2.0*x*y*y
        phi[:,2] = -9.0/2.0*x+18.0*x*x+9.0/2.0*x*y-27.0/2.0*x*x*x-27.0/2.0*x*x*y
        phi[:,5] = 27.0*x*y-27.0*x*x*y-27.0*x*y*y
    elif p == 4:
        phi[:, 0] = 1.0-25.0/3.0*x-25.0/3.0*y+70.0/3.0*x*x+140.0/3.0*x*y+70.0/3.0*y*y-80/3.0*x*x*x-80.0*x*x*y-80.0*x*y*y-80.0/3.0*y*y*y \
        +32.0/3.0*x*x*x*x+128/3.0*x*x*x*y+64.0*x*x*y*y+128.0/3.0*x*y*y*y+32.0/3.0*y*y*y*y 
        phi[:, 4] = -x+22.0/3.0*x*x-16.0*x*x*x+32.0/3.0*x*x*x*x 
        phi[:,14] = -y+22.0/3.0*y*y-16.0*y*y*y+32.0/3.0*y*y*y*y 
        phi[:, 8] = 16.0/3.0*x*y-32.0*x*x*y+128.0/3.0*x*x*x*y 
        phi[:,11] = 4.0*x*y-16.0*x*x*y-16.0*x*y*y+64.0*x*x*y*y 
        phi[:,13] = 16.0/3.0*x*y-32.0*x*y*y+128.0/3.0*x*y*y*y 
        phi[:,12] = 16.0/3.0*y-16.0/3.0*x*y-112.0/3.0*y*y+32.0*x*y*y+224.0/3.0*y*y*y-128.0/3.0*x*y*y*y-128.0/3.0*y*y*y*y 
        phi[:, 9] = -12.0*y+28.0*x*y+76.0*y*y-16.0*x*x*y-144.0*x*y*y-128.0*y*y*y+64.0*x*x*y*y+128.0*x*y*y*y+64.0*y*y*y*y 
        phi[:, 5] = 16.0*y-208.0/3.0*x*y-208.0/3.0*y*y+96.0*x*x*y+192.0*x*y*y+96.0*y*y*y-128.0/3.0*x*x*x*y- 128*x*x*y*y-128.0*x*y*y*y-128.0/3.0*y*y*y*y 
        phi[:, 1] = 16.0*x-208.0/3.0*x*x-208.0/3.0*x*y+96.0*x*x*x+192.0*x*x*y+96.0*x*y*y-128.0/3.0*x*x*x*x- 128*x*x*x*y-128.0*x*x*y*y-128.0/3.0*x*y*y*y 
        phi[:, 2] = -12.0*x+76.0*x*x+28.0*x*y-128.0*x*x*x-144.0*x*x*y-16.0*x*y*y+64.0*x*x*x*x+128.0*x*x*x*y+64.0*x*x*y*y 
        phi[:, 3] = 16.0/3.0*x-112.0/3.0*x*x-16.0/3.0*x*y+224.0/3.0*x*x*x+32.0*x*x*y-128.0/3.0*x*x*x*x-128.0/3.0*x*x*x*y 
        phi[:, 6] = 96.0*x*y-224.0*x*x*y-224.0*x*y*y+128.0*x*x*x*y+256.0*x*x*y*y+128.0*x*y*y*y 
        phi[:, 7] = -32.0*x*y+160.0*x*x*y+32.0*x*y*y-128.0*x*x*x*y-128.0*x*x*y*y 
        phi[:,10] = -32.0*x*y+32.0*x*x*y+160.0*x*y*y-128.0*x*x*y*y-128.0*x*y*y*y
    elif p == 5:
        phi[:, 0]  = 1.0-137.0/12.0*x-137.0/12.0*y+375.0/8.0*x*x+375.0/4.0*x*y+375.0/8.0*y*y-2125.0/24.0*x*x*x-2125.0/8.0*x*x*y-2125.0/8.0*x*y*y \
        -2125.0/24.0*y*y*y+ 625.0/8.0*x*x*x*x+625.0/2.0*x*x*x*y+1875.0/4.0*x*x*y*y+625.0/2.0*x*y*y*y+625.0/8.0*y*y*y*y-625.0/24.0*x*x*x*x*x \
        -3125.0/24.0*x*x*x*x*y-3125.0/12.0*x*x*x*y*y-3125.0/12.0*x*x*y*y*y-3125.0/24.0*x*y*y*y*y-625.0/24.0*y*y*y*y*y
        phi[:, 5]  = x-125.0/12.0*x*x+875.0/24.0*x*x*x-625.0/12.0*x*x*x*x+625.0/24.0*x*x*x*x*x
        phi[:,20]  = y-125.0/12.0*y*y+875.0/24.0*y*y*y-625.0/12.0*y*y*y*y+625.0/24.0*y*y*y*y*y
        phi[:,10]  = -25.0/4.0*x*y+1375.0/24.0*x*x*y-625.0/4.0*x*x*x*y+3125.0/24.0*x*x*x*x*y
        phi[:,14]  = -25.0/6.0*x*y+125.0/4.0*x*x*y+125.0/6.0*x*y*y-625.0/12.0*x*x*x*y-625.0/4.0*x*x*y*y+3125.0/12.0*x*x*x*y*y
        phi[:,17]  = -25.0/6.0*x*y+125.0/6.0*x*x*y+125.0/4.0*x*y*y-625.0/4.0*x*x*y*y-625.0/12.0*x*y*y*y+3125.0/12.0*x*x*y*y*y
        phi[:,19]  = -25.0/4.0*x*y+1375.0/24.0*x*y*y-625.0/4.0*x*y*y*y+3125.0/24.0*x*y*y*y*y
        phi[:,18]  = -25.0/4.0*y+25.0/4.0*x*y+1525.0/24.0*y*y-1375.0/24.0*x*y*y-5125.0/24.0*y*y*y+625.0/4.0*x*y*y*y+6875.0/24.0*y*y*y*y \
        -3125.0/24.0*x*y*y*y*y-3125.0/24.0*y*y*y*y*y
        phi[:,15]  = 50.0/3.0*y-75.0/2.0*x*y-325.0/2.0*y*y+125.0/6.0*x*x*y+3875.0/12.0*x*y*y+6125.0/12.0*y*y*y-625.0/4.0*x*x*y*y-3125.0/4.0*x*y*y*y \
        -625.0*y*y*y*y+3125.0/12.0*x*x*y*y*y+3125.0/6.0*x*y*y*y*y+3125.0/12.0*y*y*y*y*y
        phi[:,11]  = -25.0*y+1175.0/12.0*x*y+2675.0/12.0*y*y-125.0*x*x*y-8875.0/12.0*x*y*y-7375.0/12.0*y*y*y+625.0/12.0*x*x*x*y+3125.0/4.0*x*x*y*y \
        +5625.0/4.0*x*y*y*y+8125.0/12.0*y*y*y*y-3125.0/12.0*x*x*x*y*y-3125.0/4.0*x*x*y*y*y-3125.0/4.0*x*y*y*y*y-3125.0/12.0*y*y*y*y*y 
        phi[:, 6] = 25.0*y-1925.0/12.0*x*y-1925.0/12.0*y*y+8875.0/24.0*x*x*y+8875.0/12.0*x*y*y+8875.0/24.0*y*y*y-4375.0/12.0*x*x*x*y \
        -4375.0/4.0*x*x*y*y-4375.0/4.0*x*y*y*y-4375.0/12.0*y*y*y*y+3125.0/24.0*x*x*x*x*y+ 3125.0/6.0*x*x*x*y*y+3125.0/4.0*x*x*y*y*y \
        +3125.0/6.0*x*y*y*y*y+3125.0/24.0*y*y*y*y*y
        phi[:, 1] = 25.0*x-1925.0/12.0*x*x-1925.0/12.0*x*y+8875.0/24.0*x*x*x+8875.0/12.0*x*x*y+8875.0/24.0*x*y*y-4375.0/12.0*x*x*x*x-4375.0/4.0*x*x*x*y \
        -4375.0/4.0*x*x*y*y-4375.0/12.0*x*y*y*y+3125.0/24.0*x*x*x*x*x+3125.0/6.0*x*x*x*x*y+3125.0/4.0*x*x*x*y*y+3125.0/6.0*x*x*y*y*y+3125.0/24.0*x*y*y*y*y
        phi[:, 2] = -25.0*x+2675.0/12.0*x*x+1175.0/12.0*x*y-7375.0/12.0*x*x*x-8875.0/12.0*x*x*y-125.0*x*y*y+8125.0/12.0*x*x*x*x \
        +5625.0/4.0*x*x*x*y+3125.0/4.0*x*x*y*y+625.0/12.0*x*y*y*y-3125.0/12.0*x*x*x*x*x- 3125.0/4.0*x*x*x*x*y-3125.0/4.0*x*x*x*y*y-3125.0/12.0*x*x*y*y*y
        phi[:, 3] = 50.0/3.0*x-325.0/2.0*x*x-75.0/2.0*x*y+6125.0/12.0*x*x*x+3875.0/12.0*x*x*y+125.0/6.0*x*y*y-625.0*x*x*x*x \
        -3125.0/4.0*x*x*x*y-625.0/4.0*x*x*y*y+3125.0/12.0*x*x*x*x*x+3125.0/6.0*x*x*x*x*y+3125.0/12.0*x*x*x*y*y
        phi[:, 4] = -25.0/4.0*x+1525.0/24.0*x*x+25.0/4.0*x*y-5125.0/24.0*x*x*x-1375.0/24.0*x*x*y+6875.0/24.0*x*x*x*x+625.0/4.0*x*x*x*y \
        -3125.0/24.0*x*x*x*x*x-3125.0/24.0*x*x*x*x*y
        phi[:, 7] = 250.0*x*y-5875.0/6.0*x*x*y-5875.0/6.0*x*y*y+1250.0*x*x*x*y+2500.0*x*x*y*y+1250.0*x*y*y*y-3125.0/6.0*x*x*x*x*y \
        -3125.0/2.0*x*x*x*y*y-3125.0/2.0*x*x*y*y*y-3125.0/6.0*x*y*y*y*y
        phi[:, 8] = -125.0*x*y+3625.0/4.0*x*x*y+1125.0/4.0*x*y*y-3125.0/2.0*x*x*x*y-6875.0/4.0*x*x*y*y-625.0/4.0*x*y*y*y+3125.0/4.0*x*x*x*x*y \
        +3125.0/2.0*x*x*x*y*y+3125.0/4.0*x*x*y*y*y
        phi[:, 9] = 125.0/3.0*x*y-2125.0/6.0*x*x*y-125.0/3.0*x*y*y+2500.0/3.0*x*x*x*y+625.0/2.0*x*x*y*y-3125.0/6.0*x*x*x*x*y-3125.0/6.0*x*x*x*y*y
        phi[:,12] = -125.0*x*y+1125.0/4.0*x*x*y+3625.0/4.0*x*y*y-625.0/4.0*x*x*x*y-6875.0/4.0*x*x*y*y-3125.0/2.0*x*y*y*y+3125.0/4.0*x*x*x*y*y \
        +3125.0/2.0*x*x*y*y*y+3125.0/4.0*x*y*y*y*y
        phi[:,13] = 125.0/4.0*x*y-375.0/2.0*x*x*y-375.0/2.0*x*y*y+625.0/4.0*x*x*x*y+4375.0/4.0*x*x*y*y+625.0/4.0*x*y*y*y-3125.0/4.0*x*x*x*y*y \
        -3125.0/4.0*x*x*y*y*y
        phi[:,16] = 125.0/3.0*x*y-125.0/3.0*x*x*y-2125.0/6.0*x*y*y+625.0/2.0*x*x*y*y+2500.0/3.0*x*y*y*y-3125.0/6.0*x*x*y*y*y-3125.0/6.0*x*y*y*y*y


def grad_tensor_lagrange(dim, p, x, gphi):
    '''
    Method: grad_tensor_lagrange
    ------------------------------
    Calculates the lagrange basis gradients

    INPUTS:
        dim: dimension of mesh
        p: order of polynomial space
        x: coordinate of current node
        
    OUTPUTS: 
        gphi: evaluated gradient of basis
    '''
    if p == 0:
    	gphi[:,:] = 0.
    	return 

    nnode = p+1
    xnode = equidistant_nodes_1D_range(-1., 1., nnode)

    if dim == 1:
        get_lagrange_basis_1D(x, xnode, nnode, None, gphi)
    if dim == 2:
        get_lagrange_basis_2D(x, xnode, nnode, None, gphi)


def grad_lagrange_tri(p, xi, gphi):
    '''
    Method: grad_lagrange_tri
    ------------------------------
    Calculates lagrange basis gradient for triangles

    INPUTS:
        p: order of polynomial space
        xi: coordinates of current node

    OUTPUTS: 
        gphi: evaluated gradient of basis 
    '''
    x = xi[:,0]; y = xi[:,1]

    if p == 0:
        gphi[:] = 0.
    elif p == 1:
        gphi[:,0,0] =  -1.0
        gphi[:,1,0] =  1.0
        gphi[:,2,0] =  0.0
        gphi[:,0,1] =  -1.0
        gphi[:,1,1] =  0.0
        gphi[:,2,1] =  1.0
    elif p == 2:
        gphi[:,0,0] =  -3.0+4.0*x+4.0*y
        gphi[:,2,0] =  -1.0+4.0*x
        gphi[:,5,0] =  0.0
        gphi[:,4,0] =  4.0*y
        gphi[:,3,0] =  -4.0*y
        gphi[:,1,0] =  4.0-8.0*x-4.0*y
        gphi[:,0,1] =  -3.0+4.0*x+4.0*y
        gphi[:,2,1] =  0.0
        gphi[:,5,1] =  -1.0+4.0*y
        gphi[:,4,1] =  4.0*x
        gphi[:,3,1] =  4.0-4.0*x-8.0*y
        gphi[:,1,1] =  -4.0*x
    elif p == 3:
        gphi[:,0,0] =  -11.0/2.0+18.0*x+18.0*y-27.0/2.0*x*x-27.0*x*y-27.0/2.0*y*y
        gphi[:,3,0] =  1.0-9.0*x+27.0/2.0*x*x
        gphi[:,9,0] =  0.0
        gphi[:,6,0] =  -9.0/2.0*y+27.0*x*y
        gphi[:,8,0] =  -9.0/2.0*y+27.0/2.0*y*y
        gphi[:,7,0] =  9.0/2.0*y-27.0/2.0*y*y
        gphi[:,4,0] =  -45.0/2.0*y+27.0*x*y+27.0*y*y
        gphi[:,1,0] =  9.0-45.0*x-45.0/2.0*y+81.0/2.0*x*x+54.0*x*y+27.0/2.0*y*y
        gphi[:,2,0] =  -9.0/2.0+36.0*x+9.0/2.0*y-81.0/2.0*x*x-27.0*x*y
        gphi[:,5,0] =  27.0*y-54.0*x*y-27.0*y*y
        gphi[:,0,1] =  -11.0/2.0+18.0*x+18.0*y-27.0/2.0*x*x-27.0*x*y-27.0/2.0*y*y
        gphi[:,3,1] =  0.0
        gphi[:,9,1] =  1.0-9.0*y+27.0/2.0*y*y
        gphi[:,6,1] =  -9.0/2.0*x+27.0/2.0*x*x
        gphi[:,8,1] =  -9.0/2.0*x+27.0*x*y
        gphi[:,7,1] =  -9.0/2.0+9.0/2.0*x+36.0*y-27.0*x*y-81.0/2.0*y*y
        gphi[:,4,1] =  9.0-45.0/2.0*x-45.0*y+27.0/2.0*x*x+54.0*x*y+81.0/2.0*y*y
        gphi[:,1,1] =  -45.0/2.0*x+27.0*x*x+27.0*x*y
        gphi[:,2,1] =  9.0/2.0*x-27.0/2.0*x*x
        gphi[:,5,1] =  27.0*x-27.0*x*x-54.0*x*y
    elif p == 4:
        gphi[:, 0,0] =  -25.0/3.0+140.0/3.0*x+140.0/3.0*y-80.0*x*x-160.0*x*y-80.0*y*y+128.0/3.0*x*x*x+128.0*x*x*y+128.0*x*y*y+128.0/3.0*y*y*y
        gphi[:, 4,0] =  -1.0+44.0/3.0*x-48.0*x*x+128.0/3.0*x*x*x
        gphi[:,14,0] =  0.0
        gphi[:, 8,0] =  16.0/3.0*y-64.0*x*y+128.0*x*x*y
        gphi[:,11,0] =  4.0*y-32.0*x*y-16.0*y*y+128.0*x*y*y
        gphi[:,13,0] =  16.0/3.0*y-32.0*y*y+128.0/3.0*y*y*y
        gphi[:,12,0] =  -16.0/3.0*y+32.0*y*y-128.0/3.0*y*y*y
        gphi[:, 9,0] =  28.0*y-32.0*x*y-144.0*y*y+128.0*x*y*y+128.0*y*y*y
        gphi[:, 5,0] =  -208.0/3.0*y+192.0*x*y+192.0*y*y-128.0*x*x*y-256.0*x*y*y-128.0*y*y*y
        gphi[:, 1,0] =  16.0-416.0/3.0*x-208.0/3.0*y+288.0*x*x+384.0*x*y+96.0*y*y-512.0/3.0*x*x*x-384.0*x*x*y-256.0*x*y*y-128.0/3.0*y*y*y
        gphi[:, 2,0] =  -12.0+152.0*x+28.0*y-384.0*x*x-288.0*x*y-16.0*y*y+256.0*x*x*x+384.0*x*x*y+128.0*x*y*y
        gphi[:, 3,0] =  16.0/3.0-224.0/3.0*x-16.0/3.0*y+224.0*x*x+64.0*x*y-512.0/3.0*x*x*x-128.0*x*x*y
        gphi[:, 6,0] =  96.0*y-448.0*x*y-224.0*y*y+384.0*x*x*y+512.0*x*y*y+128.0*y*y*y
        gphi[:, 7,0] =  -32.0*y+320.0*x*y+32.0*y*y-384.0*x*x*y-256.0*x*y*y
        gphi[:,10,0] =  -32.0*y+64.0*x*y+160.0*y*y-256.0*x*y*y-128.0*y*y*y
        gphi[:, 0,1] =  -25.0/3.0+140.0/3.0*x+140.0/3.0*y-80.0*x*x-160.0*x*y-80.0*y*y+128.0/3.0*x*x*x+128.0*x*x*y+128.0*x*y*y+128.0/3.0*y*y*y
        gphi[:, 4,1] =  0.0
        gphi[:,14,1] =  -1.0+44.0/3.0*y-48.0*y*y+128.0/3.0*y*y*y
        gphi[:, 8,1] =  16.0/3.0*x-32.0*x*x+128.0/3.0*x*x*x
        gphi[:,11,1] =  4.0*x-16.0*x*x-32.0*x*y+128.0*x*x*y
        gphi[:,13,1] =  16.0/3.0*x-64.0*x*y+128.0*x*y*y
        gphi[:,12,1] =  16.0/3.0-16.0/3.0*x-224.0/3.0*y+64.0*x*y+224.0*y*y-128.0*x*y*y-512.0/3.0*y*y*y
        gphi[:, 9,1] =  -12.0+28.0*x+152.0*y-16.0*x*x-288.0*x*y-384.0*y*y+128.0*x*x*y+384.0*x*y*y+256.0*y*y*y
        gphi[:, 5,1] =  16.0-208.0/3.0*x-416.0/3.0*y+96.0*x*x+384.0*x*y+288.0*y*y-128.0/3.0*x*x*x-256.0*x*x*y-384.0*x*y*y-512.0/3.0*y*y*y
        gphi[:, 1,1] =  -208.0/3.0*x+192.0*x*x+192.0*x*y-128.0*x*x*x-256.0*x*x*y-128.0*x*y*y
        gphi[:, 2,1] =  28.0*x-144.0*x*x-32.0*x*y+128.0*x*x*x+128.0*x*x*y
        gphi[:, 3,1] =  -16.0/3.0*x+32.0*x*x-128.0/3.0*x*x*x
        gphi[:, 6,1] =  96.0*x-224.0*x*x-448.0*x*y+128.0*x*x*x+512.0*x*x*y+384.0*x*y*y
        gphi[:, 7,1] =  -32.0*x+160.0*x*x+64.0*x*y-128.0*x*x*x-256.0*x*x*y
        gphi[:,10,1] =  -32.0*x+32.0*x*x+320.0*x*y-256.0*x*x*y-384.0*x*y*y
    elif p == 5:
        gphi[:, 0,0] =  -137.0/12.0+375.0/4.0*x+375.0/4.0*y-2125.0/8.0*x*x-2125.0/4.0*x*y-2125.0/8.0*y*y+625.0/2.0*x*x*x+1875.0/2.0*x*x*y \
        +1875.0/2.0*x*y*y+625.0/2.0*y*y*y-3125.0/24.0*x*x*x*x-3125.0/6.0*x*x*x*y-3125.0/4.0*x*x*y*y-3125.0/6.0*x*y*y*y-3125.0/24.0*y*y*y*y
        gphi[:, 5,0] =  1.0-125.0/6.0*x+875.0/8.0*x*x-625.0/3.0*x*x*x+3125.0/24.0*x*x*x*x
        gphi[:,20,0] =  0.0
        gphi[:,10,0] =  -25.0/4.0*y+1375.0/12.0*x*y-1875.0/4.0*x*x*y+3125.0/6.0*x*x*x*y
        gphi[:,14,0] =  -25.0/6.0*y+125.0/2.0*x*y+125.0/6.0*y*y-625.0/4.0*x*x*y-625.0/2.0*x*y*y+3125.0/4.0*x*x*y*y
        gphi[:,17,0] =  -25.0/6.0*y+125.0/3.0*x*y+125.0/4.0*y*y-625.0/2.0*x*y*y-625.0/12.0*y*y*y+3125.0/6.0*x*y*y*y
        gphi[:,19,0] =  -25.0/4.0*y+1375.0/24.0*y*y-625.0/4.0*y*y*y+3125.0/24.0*y*y*y*y
        gphi[:,18,0] =  25.0/4.0*y-1375.0/24.0*y*y+625.0/4.0*y*y*y-3125.0/24.0*y*y*y*y
        gphi[:,15,0] =  -75.0/2.0*y+125.0/3.0*x*y+3875.0/12.0*y*y-625.0/2.0*x*y*y-3125.0/4.0*y*y*y+3125.0/6.0*x*y*y*y+3125.0/6.0*y*y*y*y
        gphi[:,11,0] =  1175.0/12.0*y-250.0*x*y-8875.0/12.0*y*y+625.0/4.0*x*x*y+3125.0/2.0*x*y*y+5625.0/4.0*y*y*y-3125.0/4.0*x*x*y*y \
        -3125.0/2.0*x*y*y*y-3125.0/4.0*y*y*y*y
        gphi[:, 6,0] =  -1925.0/12.0*y+8875.0/12.0*x*y+8875.0/12.0*y*y-4375.0/4.0*x*x*y-4375.0/2.0*x*y*y-4375.0/4.0*y*y*y+3125.0/6.0*x*x*x*y \
        +3125.0/2.0*x*x*y*y+3125.0/2.0*x*y*y*y+3125.0/6.0*y*y*y*y
        gphi[:, 1,0] =  25.0-1925.0/6.0*x-1925.0/12.0*y+8875.0/8.0*x*x+8875.0/6.0*x*y+8875.0/24.0*y*y-4375.0/3.0*x*x*x-13125.0/4.0*x*x*y \
        -4375.0/2.0*x*y*y-4375.0/12.0*y*y*y+15625.0/24.0*x*x*x*x+6250.0/3.0*x*x*x*y+9375.0/4.0*x*x*y*y+3125.0/3.0*x*y*y*y+3125.0/24.0*y*y*y*y
        gphi[:, 2,0] =  -25.0+2675.0/6.0*x+1175.0/12.0*y-7375.0/4.0*x*x-8875.0/6.0*x*y-125.0*y*y+8125.0/3.0*x*x*x+16875.0/4.0*x*x*y \
        +3125.0/2.0*x*y*y+625.0/12.0*y*y*y-15625.0/12.0*x*x*x*x-3125.0*x*x*x*y-9375.0/4.0*x*x*y*y-3125.0/6.0*x*y*y*y
        gphi[:, 3,0] =  50.0/3.0-325.0*x-75.0/2.0*y+6125.0/4.0*x*x+3875.0/6.0*x*y+125.0/6.0*y*y-2500.0*x*x*x-9375.0/4.0*x*x*y \
        -625.0/2.0*x*y*y+15625.0/12.0*x*x*x*x+6250.0/3.0*x*x*x*y+3125.0/4.0*x*x*y*y
        gphi[:, 4,0] =  -25.0/4.0+1525.0/12.0*x+25.0/4.0*y-5125.0/8.0*x*x-1375.0/12.0*x*y+6875.0/6.0*x*x*x+1875.0/4.0*x*x*y \
        -15625.0/24.0*x*x*x*x-3125.0/6.0*x*x*x*y
        gphi[:, 7,0] =  250.0*y-5875.0/3.0*x*y-5875.0/6.0*y*y+3750.0*x*x*y+5000.0*x*y*y+1250.0*y*y*y-6250.0/3.0*x*x*x*y \
        -9375.0/2.0*x*x*y*y-3125.0*x*y*y*y-3125.0/6.0*y*y*y*y
        gphi[:, 8,0] =  -125.0*y+3625.0/2.0*x*y+1125.0/4.0*y*y-9375.0/2.0*x*x*y-6875.0/2.0*x*y*y-625.0/4.0*y*y*y+3125.0*x*x*x*y \
        +9375.0/2.0*x*x*y*y+3125.0/2.0*x*y*y*y
        gphi[:, 9,0] =  125.0/3.0*y-2125.0/3.0*x*y-125.0/3.0*y*y+2500.0*x*x*y+625.0*x*y*y-6250.0/3.0*x*x*x*y-3125.0/2.0*x*x*y*y
        gphi[:,12,0] =  -125.0*y+1125.0/2.0*x*y+3625.0/4.0*y*y-1875.0/4.0*x*x*y-6875.0/2.0*x*y*y-3125.0/2.0*y*y*y+9375.0/4.0*x*x*y*y \
        +3125.0*x*y*y*y+3125.0/4.0*y*y*y*y
        gphi[:,13,0] =  125.0/4.0*y-375.0*x*y-375.0/2.0*y*y+1875.0/4.0*x*x*y+4375.0/2.0*x*y*y+625.0/4.0*y*y*y-9375.0/4.0*x*x*y*y \
        -3125.0/2.0*x*y*y*y
        gphi[:,16,0] =  125.0/3.0*y-250.0/3.0*x*y-2125.0/6.0*y*y+625.0*x*y*y+2500.0/3.0*y*y*y-3125.0/3.0*x*y*y*y-3125.0/6.0*y*y*y*y
        gphi[:, 0,1] =  -137.0/12.0+375.0/4.0*x+375.0/4.0*y-2125.0/8.0*x*x-2125.0/4.0*x*y-2125.0/8.0*y*y+625.0/2.0*x*x*x \
        +1875.0/2.0*x*x*y+1875.0/2.0*x*y*y+625.0/2.0*y*y*y-3125.0/24.0*x*x*x*x-3125.0/6.0*x*x*x*y-3125.0/4.0*x*x*y*y-3125.0/6.0*x*y*y*y-3125.0/24.0*y*y*y*y 
        gphi[:, 5,1] =  0.0
        gphi[:,20,1] =  1.0-125.0/6.0*y+875.0/8.0*y*y-625.0/3.0*y*y*y+3125.0/24.0*y*y*y*y
        gphi[:,10,1] =  -25.0/4.0*x+1375.0/24.0*x*x-625.0/4.0*x*x*x+3125.0/24.0*x*x*x*x
        gphi[:,14,1] =  -25.0/6.0*x+125.0/4.0*x*x+125.0/3.0*x*y-625.0/12.0*x*x*x-625.0/2.0*x*x*y+3125.0/6.0*x*x*x*y
        gphi[:,17,1] =  -25.0/6.0*x+125.0/6.0*x*x+125.0/2.0*x*y-625.0/2.0*x*x*y-625.0/4.0*x*y*y+3125.0/4.0*x*x*y*y
        gphi[:,19,1] =  -25.0/4.0*x+1375.0/12.0*x*y-1875.0/4.0*x*y*y+3125.0/6.0*x*y*y*y
        gphi[:,18,1] =  -25.0/4.0+25.0/4.0*x+1525.0/12.0*y-1375.0/12.0*x*y-5125.0/8.0*y*y+1875.0/4.0*x*y*y+6875.0/6.0*y*y*y \
        -3125.0/6.0*x*y*y*y-15625.0/24.0*y*y*y*y
        gphi[:,15,1] =  50.0/3.0-75.0/2.0*x-325.0*y+125.0/6.0*x*x+3875.0/6.0*x*y+6125.0/4.0*y*y-625.0/2.0*x*x*y-9375.0/4.0*x*y*y \
        -2500.0*y*y*y+3125.0/4.0*x*x*y*y+6250.0/3.0*x*y*y*y+15625.0/12.0*y*y*y*y
        gphi[:,11,1] =  -25.0+1175.0/12.0*x+2675.0/6.0*y-125.0*x*x-8875.0/6.0*x*y-7375.0/4.0*y*y+625.0/12.0*x*x*x+3125.0/2.0*x*x*y \
        +16875.0/4.0*x*y*y+8125.0/3.0*y*y*y-3125.0/6.0*x*x*x*y-9375.0/4.0*x*x*y*y-3125.0*x*y*y*y-15625.0/12.0*y*y*y*y
        gphi[:, 6,1] =  25.0-1925.0/12.0*x-1925.0/6.0*y+8875.0/24.0*x*x+8875.0/6.0*x*y+8875.0/8.0*y*y-4375.0/12.0*x*x*x-4375.0/2.0*x*x*y \
        -13125.0/4.0*x*y*y-4375.0/3.0*y*y*y+3125.0/24.0*x*x*x*x+3125.0/3.0*x*x*x*y+9375.0/4.0*x*x*y*y+6250.0/3.0*x*y*y*y+15625.0/24.0*y*y*y*y
        gphi[:, 1,1] =  -1925.0/12.0*x+8875.0/12.0*x*x+8875.0/12.0*x*y-4375.0/4.0*x*x*x-4375.0/2.0*x*x*y-4375.0/4.0*x*y*y \
        +3125.0/6.0*x*x*x*x+3125.0/2.0*x*x*x*y+3125.0/2.0*x*x*y*y+3125.0/6.0*x*y*y*y
        gphi[:, 2,1] =  1175.0/12.0*x-8875.0/12.0*x*x-250.0*x*y+5625.0/4.0*x*x*x+3125.0/2.0*x*x*y+625.0/4.0*x*y*y-3125.0/4.0*x*x*x*x \
        -3125.0/2.0*x*x*x*y-3125.0/4.0*x*x*y*y
        gphi[:, 3,1] =  -75.0/2.0*x+3875.0/12.0*x*x+125.0/3.0*x*y-3125.0/4.0*x*x*x-625.0/2.0*x*x*y+3125.0/6.0*x*x*x*x+3125.0/6.0*x*x*x*y
        gphi[:, 4,1] =  25.0/4.0*x-1375.0/24.0*x*x+625.0/4.0*x*x*x-3125.0/24.0*x*x*x*x
        gphi[:, 7,1] =  250.0*x-5875.0/6.0*x*x-5875.0/3.0*x*y+1250.0*x*x*x+5000.0*x*x*y+3750.0*x*y*y-3125.0/6.0*x*x*x*x \
        -3125.0*x*x*x*y-9375.0/2.0*x*x*y*y-6250.0/3.0*x*y*y*y
        gphi[:, 8,1] =  -125.0*x+3625.0/4.0*x*x+1125.0/2.0*x*y-3125.0/2.0*x*x*x-6875.0/2.0*x*x*y-1875.0/4.0*x*y*y+3125.0/4.0*x*x*x*x \
        +3125.0*x*x*x*y+9375.0/4.0*x*x*y*y
        gphi[:, 9,1] =  125.0/3.0*x-2125.0/6.0*x*x-250.0/3.0*x*y+2500.0/3.0*x*x*x+625.0*x*x*y-3125.0/6.0*x*x*x*x-3125.0/3.0*x*x*x*y
        gphi[:,12,1] =  -125.0*x+1125.0/4.0*x*x+3625.0/2.0*x*y-625.0/4.0*x*x*x-6875.0/2.0*x*x*y-9375.0/2.0*x*y*y+3125.0/2.0*x*x*x*y \
        +9375.0/2.0*x*x*y*y+3125.0*x*y*y*y
        gphi[:,13,1] =  125.0/4.0*x-375.0/2.0*x*x-375.0*x*y+625.0/4.0*x*x*x+4375.0/2.0*x*x*y+1875.0/4.0*x*y*y-3125.0/2.0*x*x*x*y-9375.0/4.0*x*x*y*y
        gphi[:,16,1] =  125.0/3.0*x-125.0/3.0*x*x-2125.0/3.0*x*y+625.0*x*x*y+2500.0*x*y*y-3125.0/2.0*x*x*y*y-6250.0/3.0*x*y*y*y



def shape_tensor_legendre(dim, p, x, phi):
    '''
    Method: shape_tensor_legendre
    ------------------------------
    Calculates legendre bais for 1D or 2D quads

    INPUTS:
        dim: dimension of mesh
        p: order of polynomial space
        x: coordinate of current node

    OUTPUTS: 
        phi: evaluated basis 
    '''
    if p == 0:
        phi[:] = 1.
        return

    if dim == 1:
        get_legendre_basis_1D(x, p, phi, None)
    elif dim == 2:
        get_legendre_basis_2D(x, p, phi, None)

def grad_tensor_legendre(dim, p, x, gphi):
    '''
    Method: grad_tensor_legendre
    ------------------------------
    Calculates the legendre basis gradients

    INPUTS:
        dim: dimension of mesh
        p: order of polynomial space
        x: coordinate of current node
        
    OUTPUTS: 
        gphi: evaluated gradient of basis
    '''
    if p == 0:
        gphi[:,:] = 0.
        return 
    if dim == 1:
        get_legendre_basis_1D(x, p, None, gphi)
    if dim == 2:
        get_legendre_basis_2D(x, p, None, gphi)


def get_legendre_basis_1D(x, p, phi, gphi):
    '''
    Method: get_legendre_basis_1D
    ------------------------------
    Calculates the 1D Legendre basis functions

    INPUTS:
        x: coordinate of current node
        p: order of polynomial space
        
    OUTPUTS: 
        phi: evaluated basis 
        gphi: evaluated physical gradient of basis
    '''

    if phi is not None:
        x.shape = -1
        if p >= 0:
            phi[:,0]  = 1.
        if p>=1:
            phi[:,1]  = x
        if p>=2:
            phi[:,2]  = 0.5*(3.*x*x - 1.)
        if p>=3:
            phi[:,3]  = 0.5*(5.*x*x*x - 3.*x)
        if p>=4:
            phi[:,4]  = 0.125*(35.*x*x*x*x - 30.*x*x + 3.)
        if p>=5:
            phi[:,5]  = 0.125*(63.*x*x*x*x*x - 70.*x*x*x + 15.*x)
        if p>=6:
            phi[:,6]  = 0.0625*(231.*x*x*x*x*x*x - 315.*x*x*x*x + 105.*x*x -5.)
        if p==7:
            phi[:,7]  = 0.0625*(429.*x*x*x*x*x*x*x - 693.*x*x*x*x*x + 315.*x*x*x - 35.*x)
        if p>7:
            raise NotImplementedError("Legendre Polynomial > 7 not supported")

        x.shape = -1,1

    if gphi is not None:
        if p >= 0:
            gphi[:,0] = 0.
        if p>=1:
            gphi[:,1] = 1.
        if p>=2:
            gphi[:,2] = 3.*x
        if p>=3:
            gphi[:,3] = 0.5*(15.*x*x - 3.)
        if p>=4:
            gphi[:,4] = 0.125*(35.*4.*x*x*x - 60.*x)
        if p>=5:
            gphi[:,5] = 0.125*(63.*5.*x*x*x*x - 210.*x*x + 15.)
        if p>=6:
            gphi[:,6] = 0.0625*(231.*6.*x*x*x*x*x - 315.*4.*x*x*x + 210.*x)
        if p==7:
            gphi[:,7] = 0.0625*(429.*7.*x*x*x*x*x*x - 693.*5.*x*x*x*x + 315.*3.*x*x - 35.)
        if p>7:
            raise NotImplementedError("Legendre Polynomial > 7 not supported")


def get_legendre_basis_2D(x, p, phi, gphi):
    '''
    Method: get_legendre_basis_2D
    ------------------------------
    Calculates the 2D Legendre basis functions

    INPUTS:
        x: coordinate of current node
        p: order of polynomial space
        
    OUTPUTS: 
        phi: evaluated basis 
        gphi: evaluated physical gradient of basis
    '''

    # if gphi is not None:
    #     gphix = np.zeros(p+1); gphiy = np.zeros(p+1)
    # else:
    #     gphix = None; gphiy = None
    # # Always need phi
    # phix = np.zeros(p+1); phiy = np.zeros(p+1)

    # get_legendre_basis_1D(x[0], p, phix, gphix)
    # get_legendre_basis_1D(x[1], p, phiy, gphiy)
   
    # if phi is not None:
    #     phi[:] = np.reshape(np.outer(phix, phiy), (-1,), 'F')
    # if gphi is not None:
    #     gphi[:,0] = np.reshape(np.outer(gphix, phiy), (-1,), 'F')
    #     gphi[:,1] = np.reshape(np.outer(phix, gphiy), (-1,), 'F')

    if gphi is not None:
        gphix = np.zeros((x.shape[0],p+1,1)); gphiy = np.zeros_like(gphix)
    else:
        gphix = None; gphiy = None
    # Always need phi
    phix = np.zeros((x.shape[0],p+1)); phiy = np.zeros_like(phix)

    get_legendre_basis_1D(x[:,0], p, phix, gphix)
    get_legendre_basis_1D(x[:,1], p, phiy, gphiy)

    if phi is not None:
        for i in range(x.shape[0]):
            phi[i,:] = np.reshape(np.outer(phix[i,:], phiy[i,:]), (-1,), 'F')
    if gphi is not None:
        for i in range(x.shape[0]):
            gphi[i,:,0] = np.reshape(np.outer(gphix[i,:,0], phiy[i,:]), (-1,), 'F')
            gphi[i,:,1] = np.reshape(np.outer(phix[i,:], gphiy[i,:,0]), (-1,), 'F')


class BasisData(object):
    '''
    Class: BasisData
    -------------------
    This class contains information about the basis functions

    ATTRIBUTES:
        basis: type of basis function
        order: solution order
        nq: number of quadrature points
        dim: dimension of mesh
        Phi: evaluated basis
        GPhi: gradient of basis in reference space
        gPhi: gradient of basis in physical space
        face: index of face in reference space
    '''
    def __init__(self,basis,order,mesh=None):
        '''
        Method: __init__
        -------------------
        This method initializes the object

        INPUTS:
            basis: type of basis function
            order: solution order
            mesh: mesh object
        '''
        self.basis = basis
        self.order = order
        #self.nb = order_to_num_basis_coeff(self.basis, self.order)
        #self.nq = nq
        #self.nbnqmax = self.nb * self.nq
        self.dim = Shape2Dim[Basis2Shape[self.basis]]
        self.Phi = None
        self.GPhi = None
        self.gPhi = None
        self.face = -1

    def get_physical_grad(self, JData):
        '''
        Method: get_physical_grad
        --------------------------
        Calculate the physical gradient

        INPUTS:
            JData: jacobian data

        OUTPUTS:
            gPhi: gradient of basis in physical space
        '''
        
        nq = JData.ijac.shape[0]

        if nq != JData.nq and JData.nq != 1:
            raise Exception("Quadrature doesn't match")
        dim = JData.dim
        if dim != self.dim:
            raise Exception("Dimensions don't match")
        nb = order_to_num_basis_coeff(self.basis,self.order)
        GPhi = self.GPhi 
        if GPhi is None:
            raise Exception("GPhi is an empty list")

        if self.gPhi is None or self.gPhi.shape != (nq,nb,dim):
            self.gPhi = np.zeros([nq,nb,dim])
        else:
            self.gPhi *= 0.

        gPhi = self.gPhi

        if gPhi.shape != GPhi.shape:
            raise Exception("gPhi and GPhi are different sizes")

        # for iq in range(nq):
        #     G = GPhi[iq,:,:] # [nb,dim]
        #     g = gPhi[iq,:,:] # [nb,dim]
        #     ijac = JData.ijac[iq*(JData.nq != 1),:,:] # [dim,dim]
        #     g[:] = np.transpose(np.matmul(ijac.transpose(),G.transpose()))

        gPhi[:] = np.transpose(np.matmul(JData.ijac.transpose(0,2,1), GPhi.transpose(0,2,1)), (0,2,1))

        return gPhi

    def eval_basis(self, quad_pts, Get_Phi=True, Get_GPhi=False, Get_gPhi=False, JData=None):
        '''
        Method: eval_basis
        --------------------
        Evaluate the basis functions

        INPUTS:
            quad_pts: coordinates of quadrature points
            Get_Phi: flag to calculate basis functions (Default: True)
            Get_GPhi: flag to calculate gradient of basis functions in ref space (Default: False)
            Get_gPhi: flag to calculate gradient of basis functions in phys space (Default: False)
            JData: jacobian data (needed if calculating physical gradients)
        '''

        if Get_Phi:
            self.Phi = get_shapes(self.basis, self.order, quad_pts, self.Phi)
        if Get_GPhi:
            self.GPhi = get_grads(self.basis, self.order, self.dim, quad_pts, self.GPhi)
        if Get_gPhi:
            if not JData:
                raise Exception("Need jacobian data")
            self.gPhi = self.get_physical_grad(JData)

    def eval_basis_on_face(self, mesh, face, quad_pts, xelem=None, Get_Phi=True, Get_GPhi=False, Get_gPhi=False, JData=False):
        '''
        Method: eval_basis_on_face
        ----------------------------
        Evaluate the basis functions on faces

        INPUTS:
            mesh: mesh object
            face: index of face in reference space
            quad_pts: coordinates of quadrature points
            Get_Phi: flag to calculate basis functions (Default: True)
            Get_GPhi: flag to calculate gradient of basis functions in ref space (Default: False)
            Get_gPhi: flag to calculate gradient of basis functions in phys space (Default: False)
            JData: jacobian data (needed if calculating physical gradients)

        OUTPUTS:
            xelem: coordinate of face
        '''
        self.face = face
        nq = quad_pts.shape[0]
        basis = mesh.QBasis
        if xelem is None or xelem.shape != (nq, mesh.Dim):
            xelem = np.zeros([nq, mesh.Dim])
        xelem = Mesh.ref_face_to_elem(Basis2Shape[basis], face, nq, quad_pts, xelem)
        self.eval_basis(xelem, Get_Phi, Get_GPhi, Get_gPhi, JData)

        return xelem

    def eval_basis_on_face_ader(self, mesh, basis, face, quad_pts, xelem=None, Get_Phi=True, Get_GPhi=False, Get_gPhi=False, JData=False):
        '''
        Method: eval_basis_on_face_ader
        ----------------------------
        Evaluate the basis functions on faces for ADER-DG

        INPUTS:
            mesh: mesh object
            basis: type of basis function
            face: index of face in reference space
            quad_pts: coordinates of quadrature points
            Get_Phi: flag to calculate basis functions (Default: True)
            Get_GPhi: flag to calculate gradient of basis functions in ref space (Default: False)
            Get_gPhi: flag to calculate gradient of basis functions in phys space (Default: False)
            JData: jacobian data (needed if calculating physical gradients)

        OUTPUTS:
            xelem: coordinate of face
        '''
        self.face = face
        nq = quad_pts.shape[0]
        if Shape2Dim[Basis2Shape[basis]] == ShapeType.Quadrilateral:
            if xelem is None or xelem.shape != (nq, mesh.Dim+1):
                xelem = np.zeros([nq, mesh.Dim+1])
            xelem = Mesh.ref_face_to_elem(Basis2Shape[basis], face, nq, quad_pts, xelem)
        elif Shape2Dim[Basis2Shape[basis]] == ShapeType.Segment:
            if xelem is None or xelem.shape != (nq, mesh.Dim):
                xelem = np.zeros([nq, mesh.Dim])
            xelem = Mesh.ref_face_to_elem(Basis2Shape[basis], face, nq, quad_pts, xelem)
        self.eval_basis(xelem, Get_Phi, Get_GPhi, Get_gPhi, JData)

        return xelem
    	

class JacobianData(object):
    '''
    Class: JacobianData
    -------------------
    This class contains information about the jacobian

    ATTRIBUTES:
        dim: dimension of mesh
        djac: determinant of the jacobian
        jac: jacobian
        ijac: inverse of the jacobian
        A: placeholder for matrix sizing
        gPhi: gradient of basis in physical space
    '''
    def __init__(self,mesh):
        '''
        Method: __init__
        -------------------
        This method initializes the object

        INPUTS:
            mesh: mesh object
        '''
        self.dim = mesh.Dim
        self.djac = None
        self.jac = None
        self.ijac = None
        self.A = None
        self.gPhi = None

    def element_jacobian(self, mesh, elem, quad_pts, get_djac=False, get_jac=False, get_ijac=False):
        '''
        Method: element_jacobian
        ----------------------------
        Evaluate the geometric jacobian for specified element

        INPUTS:
            mesh: mesh object
            elem: element index
            quad_pts: coordinates of quadrature points
            get_djac: flag to calculate jacobian determinant (Default: False)
            get_jac: flag to calculate jacobian (Default: False)
            get_ijac: flag to calculate inverse of the jacobian (Default: False)
        '''
        basis = mesh.QBasis
        order = mesh.QOrder
        Shape = Basis2Shape[basis]

        nq = quad_pts.shape[0]
        nb = order_to_num_basis_coeff(basis, order)
        dim = Shape2Dim[Basis2Shape[basis]]

        ## Check if we need to resize or recalculate 
        #if self.dim != dim or self.nq != nq: Resize = True
        #else: Resize = False
        if self.dim != dim: Resize = True
        else: Resize = False

        self.gPhi = get_grads(basis, order, dim, quad_pts, self.gPhi) # [nq, nb, dim]
        gPhi = self.gPhi

        self.dim = dim
        if dim != mesh.Dim:
            raise Exception("Dimensions don't match")

        self.nq = nq

        # if get_jac and (Resize or self.jac is None): 
        #     self.jac = np.zeros([nq,dim,dim])

        if self.jac is None or self.jac.shape != (nq,dim,dim): 
            # always have jac allocated (at least for temporary storage)
            self.jac = np.zeros([nq,dim,dim])
        if get_djac and (Resize or self.djac is None): 
            self.djac = np.zeros([nq,1])
        if get_ijac and (Resize or self.ijac is None): 
            self.ijac = np.zeros([nq,dim,dim])

        if Resize or self.A is None:
            self.A = np.zeros([dim,dim])

        # A = self.A
        Elem2Nodes = mesh.Elem2Nodes[elem]
        # for iq in range(nq):
        #     G = basis_grad[iq,:,:]
        #     A[:] = 0.
        #     for i in range(nb):
        #         for j in range(dim):
        #             for k in range(dim):
        #                 A[j,k] += mesh.Coords[Elem2Nodes[i],j]*G[i,k]
        #     if get_jac:
        #         self.jac[iq,:] = A[:]
        #         # for i in range(dim):
        #         #     for j in range(dim):
        #         #         self.J[iq,i,j] = A[i,j]
        #     if get_djac: djac_ = self.djac[iq]
        #     else: djac_ = None
        #     if get_ijac: ijac_ = self.ijac[iq,:,:]
        #     else: ijac_ = None 
        #     MatDetInv(A, dim, djac_, ijac_)

        #     if djac_ is not None and djac_ <= 0.:
        #         raise Exception("Nonpositive Jacobian (elem = %d)" % (elem))

        self.jac = np.tensordot(gPhi, mesh.Coords[Elem2Nodes].transpose(), \
            axes=[[1],[1]]).transpose((0,2,1))
        ijac = None; djac = None
        for i in range(nq):
            if get_ijac:
                # self.ijac[i] = np.linalg.inv(self.jac[i])
                ijac = self.ijac[i]
            if get_djac:
                # self.djac[i] = np.linalg.det(self.jac[i])
                djac = self.djac[i]

            MatDetInv(self.jac[i], dim, djac, ijac)

        if get_djac and np.any(self.djac[i] <= 0.):
            raise Exception("Nonpositive Jacobian (elem = %d)" % (elem))



