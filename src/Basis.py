import numpy as np
from enum import IntEnum
from General import *
from Quadrature import *
from Math import *
import Mesh
import code
from Data import ArrayList, GenericData
from abc import ABC, abstractmethod



Basis2Shape = {
    BasisType.LagrangeSeg : ShapeType.Segment,
    BasisType.LagrangeQuad : ShapeType.Quadrilateral,
    BasisType.LagrangeTri : ShapeType.Triangle,
    BasisType.LegendreSeg : ShapeType.Segment,
    BasisType.LegendreQuad : ShapeType.Quadrilateral,
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
        StaticData = GenericData()
    else:
        nq = StaticData.pnq
        quadData = StaticData.quadData

    if PhysicalSpace:
        QuadOrder,QuadChanged = get_gaussian_quadrature_elem(mesh, mesh.gbasis, order*2, quadData=quadData)
    else:
        QuadOrder = order*2
        QuadChanged = True

    if QuadChanged:
        quadData = QuadData(mesh, basis, EntityType.Element, QuadOrder)

    quad_pts = quadData.quad_pts
    quad_wts = quadData.quad_wts
    nq = quad_pts.shape[0]

    if QuadChanged:
        basis.eval_basis(quad_pts, Get_Phi=True)

    if PhysicalSpace:
        djac,_,_ = element_jacobian(mesh,elem,quad_pts,get_djac=True)

        if len(djac) == 1:
            djac = np.full(nq, djac[0])
    else:
        djac = np.full(nq, 1.)

    nb = basis.get_num_basis_coeff(order)
    phi = basis.basis_val

    MM = np.zeros([nb,nb])

    MM[:] = np.matmul(phi.transpose(), phi*quad_wts*djac) # [nb, nb]

    StaticData.pnq = nq
    StaticData.quadData = quadData

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
        StaticData = GenericData()
    else:
        nq = StaticData.pnq
        quadData = StaticData.quadData

    QuadOrder,QuadChanged = get_gaussian_quadrature_elem(mesh, basis, order*2., quadData=quadData)
    #Add one to QuadOrder to adjust the mesh.Dim addition in get_gaussian_quadrature_elem.
    QuadOrder+=1

    if QuadChanged:
        quadData = QuadDataADER(mesh, basis, EntityType.Element, QuadOrder)

    quad_pts = quadData.quad_pts
    quad_wts = quadData.quad_wts
    nq = quad_pts.shape[0]

    if QuadChanged:
        basis.eval_basis(quad_pts, Get_Phi=True, Get_GPhi=True)

    nb = basis.basis_val.shape[1]

    phi = basis.basis_val
    GPhi = basis.basis_grad
    SM = np.zeros([nb,nb])

    # for i in range(nn):
    #     for j in range(nn):
    #         t = 0.
    #         for iq in range(nq):
    #             t += GPhi[iq,i,gradDir]*phi[iq,j]*wq[iq]
    #         SM[i,j] = t

    SM[:] = np.matmul(GPhi[:,:,gradDir].transpose(),phi*quad_wts) # [nb,nb]
    StaticData.pnq = nq
    StaticData.quadData = quadData

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
        StaticData = GenericData()
    else:
        nq = StaticData.pnq
        quadData = StaticData.quadData

    if basis1 == basis2:
        face = 2 
    else:
        face = 0

    QuadOrder,QuadChanged = get_gaussian_quadrature_elem(mesh, mesh.gbasis, order*2, quadData=quadData)
  
    if QuadChanged:
        quadData = QuadData(mesh, mesh.gbasis, EntityType.Element, QuadOrder)

    quad_pts = quadData.quad_pts
    quad_wts = quadData.quad_wts
    nq = quad_pts.shape[0]

    if QuadChanged:
        if basis1 == basis2:
            face = 2

            PhiData = basis1
            PsiData = basis1

            xelem = np.zeros([nq,mesh.Dim+1])
            PhiData.eval_basis_on_face_ader(mesh, basis1, face, quad_pts, xelem, Get_Phi=True)
            PsiData.eval_basis_on_face_ader(mesh, basis1, face, quad_pts, xelem, Get_Phi=True)
        else:
            face = 0
            
            PhiData = basis1
            PsiData = basis2

            xelemPhi = np.zeros([nq,mesh.Dim+1])
            xelemPsi = np.zeros([nq,mesh.Dim])
            PhiData.eval_basis_on_face_ader(mesh, basis1, face, quad_pts, xelemPhi, Get_Phi=True)
            PsiData.eval_basis(quad_pts, Get_Phi=True, Get_GPhi=False)


    nb_st = PhiData.basis_val.shape[1]
    nb = PsiData.basis_val.shape[1]

    FT = np.zeros([nb_st,nb])

    # for i in range(nn1):
    #     for j in range(nn2):
    #         t = 0.
    #         for iq in range(nq):
    #             t += phi[iq,i]*psi[iq,j]*wq[iq]
    #         MM[i,j] = t

    FT[:] = np.matmul(PhiData.basis_val.transpose(),PsiData.basis_val*quad_wts) # [nb_st, nb]
    StaticData.pnq = nq
    StaticData.quadData = quadData
 
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
        StaticData = GenericData()
    else:
        nq = StaticData.pnq
        quadData = StaticData.quadData

    if PhysicalSpace:
        QuadOrder,QuadChanged = get_gaussian_quadrature_elem(mesh, mesh.gbasis, order*2, quadData=quadData)
    else:
        QuadOrder = order*2 + 1 #Add one for ADER method
        QuadChanged = True

    if QuadChanged:
        quadData = QuadDataADER(mesh, basis, EntityType.Element, QuadOrder)


    quad_pts = quadData.quad_pts
    quad_wts = quadData.quad_wts
    nq = quad_pts.shape[0]

    if QuadChanged:
        basis.eval_basis(quad_pts, Get_Phi=True)

    if PhysicalSpace:
        djac,_,_=element_jacobian(mesh,elem,quad_pts,get_djac=True)
        
        if len(djac) == 1:
            djac = np.full(nq, djac[0])
    else:
        djac = np.full(nq, 1.)

    nb_st = basis.basis_val.shape[1]
    MM = np.zeros([nb_st,nb_st])

    # for i in range(nn):
    #     for j in range(nn):
    #         t = 0.
    #         for iq in range(nq):
    #             t += phi[iq,i]*phi[iq,j]*wq[iq]*djac[iq]
    #         MM[i,j] = t

    MM[:] = np.matmul(basis.basis_val.transpose(), basis.basis_val*quad_wts*djac) # [nb_st,nb_st]

    StaticData.pnq = nq
    StaticData.quadData = quadData

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

    basis_old.eval_basis(quad_pts, Get_Phi=True)

    phi_old = basis_old.basis_val
    nb_old = phi_old.shape[1]

    basis.eval_basis(quad_pts, Get_Phi=True)
    phi = basis.basis_val
    nb = phi.shape[1]

    A = np.zeros([nb,nb_old])

    # for i in range(nn):
    #     for j in range(nn_old):
    #         t = 0.
    #         for iq in range(nq):
    #             t += phi[iq,i]*phi_old[iq,j]*wq[iq] # JData.djac[iq*(JData.nq != 1)]
    #         A[i,j] = t

    A = np.matmul(phi.transpose(), phi_old*quad_wts) # [nb, nb_old]

    PM = np.matmul(iMM,A) # [nb, nb_old]

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

def get_inv_stiffness_matrix_ader(mesh, basis, order, elem, gradDir, StaticData=None):
    '''
    Method: get_inv_stiffness_matrix_ader
    --------------------------------------
    Calculate the inverse stiffness matrix (Currently not used)

    INPUTS:
        mesh: mesh object
        basis: type of basis function
        order: solution order
        gradDir: direction to take the gradient in
        elem: element index

    OUTPUTS: 
        iSM: inverse stiffness matrix
    '''
    SM, StaticData = get_stiffness_matrix_ader(mesh, basis, order, elem, gradDir, StaticData)

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
    basis = solver.basis
    order = EqnSet.Order
    nb = basis.nb

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

def element_jacobian(mesh, elem, quad_pts, get_djac=False, get_jac=False, get_ijac=False):
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

    basis = mesh.gbasis
    order = mesh.gorder
    shape = basis.__class__.__bases__[1].__name__
    nb = basis.nb
    dim = basis.dim

    nq = quad_pts.shape[0]


    ## Check if we need to resize or recalculate 
    #if self.dim != dim or self.nq != nq: Resize = True
    #else: Resize = False
    if dim != dim: Resize = True
    else: Resize = False

    basis_pgrad = basis.get_grads(quad_pts, basis.basis_pgrad) # [nq, nb, dim]
    
    if dim != mesh.Dim:
        raise Exception("Dimensions don't match")

    # if get_jac and (Resize or self.jac is None): 
    #     self.jac = np.zeros([nq,dim,dim])

    #if jac is None or jac.shape != (nq,dim,dim): 
        # always have jac allocated (at least for temporary storage)
    jac = np.zeros([nq,dim,dim])
    # if get_djac: 
    djac = np.zeros([nq,1])
    # if get_ijac: 
    ijac = np.zeros([nq,dim,dim])


    A = np.zeros([dim,dim])

    Elem2Nodes = mesh.Elem2Nodes[elem]

    jac = np.tensordot(basis_pgrad, mesh.Coords[Elem2Nodes].transpose(), \
        axes=[[1],[1]]).transpose((0,2,1))

    for i in range(nq):
        MatDetInv(jac[i], dim, djac[i], ijac[i])
 
    if get_djac and np.any(djac[i] <= 0.):
        raise Exception("Nonpositive Jacobian (elem = %d)" % (elem))

    return djac, jac, ijac

class ShapeBase(ABC):
    
    @abstractmethod
    def get_num_basis_coeff(self,p):
        pass

    @abstractmethod
    def equidistant_nodes(self, p, xn=None):
        pass

class PointShape(ShapeBase):

    faceshape = None
    nfaceperelem = 0
    dim = 0
    
    def get_num_basis_coeff(self,p):
        return 1
    def equidistant_nodes(self, p, xn=None):
        pass

class SegShape(PointShape):

    faceshape = PointShape()
    nfaceperelem = 2
    dim = 1

    def get_num_basis_coeff(self,p):
        return p + 1

    def equidistant_nodes(self, p, xn=None):
        '''
        Method: equidistant_nodes
        --------------------------
        Calculate the coordinates in ref space

        INPUTS:
            p: order of polynomial space
            
        OUTPUTS: 
            xn: coordinates of nodes in ref space
        '''
        nb = self.get_num_basis_coeff(p)

        dim = self.dim

        adim = nb,dim
        if xn is None or xn.shape != adim:
            xn = np.zeros(adim)

        if p == 0:
            xn[:] = 0.0 # 0.5
            return xn, nb

        xn[:,0] = equidistant_nodes_1D_range(-1., 1., nb)
        
        return xn, nb

    def ref_face_to_elem(self, face, nq, xface, xelem=None):
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
        if xelem is None: xelem = np.zeros([1,1])
        if face == 0: xelem[0] = -1.
        elif face == 1: xelem[0] = 1.
        else: raise ValueError

        return xelem

class QuadShape(SegShape):

    faceshape = SegShape()
    nfaceperelem = 4
    dim = 2

    def get_num_basis_coeff(self,p):
        return (p + 1)**2

    def equidistant_nodes(self, p, xn=None):
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
        nb = self.get_num_basis_coeff(p)
        dim = self.dim

        adim = nb,dim
        if xn is None or xn.shape != adim:
            xn = np.zeros(adim)

        if p == 0:
            xn[:] = 0.0 # 0.5
            return xn, nb

        xseg = equidistant_nodes_1D_range(-1., 1., p+1)

        xn[:,0] = np.tile(xseg, (p+1,1)).reshape(-1)
        xn[:,1] = np.repeat(xseg, p+1, axis=0).reshape(-1)

        return xn, nb

    def ref_face_to_elem(self, face, nq, xface, xelem=None):
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
        if xelem is None: xelem = np.zeros([nq,2])

        fnodes, nfnode = self.local_q1_face_nodes(1, face)

        x0 = RefQ1Coords[BasisType.LagrangeQuad][fnodes[0]]
        x1 = RefQ1Coords[BasisType.LagrangeQuad][fnodes[1]]

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

        return xelem

class TriShape(QuadShape):

    faceshape = SegShape()
    nfaceperelem = 3
    dim = 2

    def get_num_basis_coeff(self,p):
        return (p + 1)*(p + 2)//2

    def equidistant_nodes(self, p, xn=None):
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

        nb = self.get_num_basis_coeff(p)
        dim = self.dim
        

        adim = nb,dim
        if xn is None or xn.shape != adim:
            xn = np.zeros(adim)

        if p == 0:
            xn[:] = 0.0 # 0.5
            return xn, nb
        n = 0
        xseg = equidistant_nodes_1D_range(0., 1., p+1)
        for j in range(p+1):
            xn[n:n+p+1-j,0] = xseg[:p+1-j]
            xn[n:n+p+1-j,1] = xseg[j]
            n += p+1-j

        return xn, nb

    def ref_face_to_elem(self, face, nq, xface, xelem=None):
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
        if xelem is None: xelem = np.zeros([nq,2])
        xf = np.zeros(nq)
        xf = xf.reshape((nq,1))
        # local q = 1 nodes on face
        fnodes, nfnode = self.local_q1_face_nodes(1, face)
        # coordinates of local q = 1 nodes on face
        x0 = RefQ1Coords[BasisType.LagrangeTri][fnodes[0]]
        x1 = RefQ1Coords[BasisType.LagrangeTri][fnodes[1]]
        # for i in range(nq):
        #     xf[i] = (xface[i] + 1.)/2.
        #     xelem[i,:] = (1. - xf[i])*x0 + xf[i]*x1
        xf = (xface + 1.)/2.
        xelem[:] = (1. - xf)*x0 + xf*x1

        return xelem

class BasisBase(ABC): 
    @abstractmethod
    def __init__(self, order, mesh=None):

        self.order = order
        self.basis_val = None
        self.basis_grad = None
        self.basis_pgrad = None
        self.face = -1
        self.nb = 0

    def get_physical_grad(self, ijac):
        '''
        Method: get_physical_grad
        --------------------------
        Calculate the physical gradient

        INPUTS:
            JData: jacobian data

        OUTPUTS:
            gPhi: gradient of basis in physical space
        '''
        nq = ijac.shape[0]

        # if nq != JData.nq and JData.nq != 1:
            # raise Exception("Quadrature doesn't match")
        # dim = JData.dim
        dim = self.dim
        # if dim != self.dim:
            # raise Exception("Dimensions don't match")
        nb = self.nb
        basis_grad = self.basis_grad 
        if basis_grad is None:
            raise Exception("basis_grad is an empty list")

        if self.basis_pgrad is None or self.basis_pgrad.shape != (nq,nb,dim):
            self.basis_pgrad = np.zeros([nq,nb,dim])
        else:
            self.basis_pgrad *= 0.

        basis_pgrad = self.basis_pgrad

        if basis_pgrad.shape != basis_grad.shape:
            raise Exception("basis_pgrad and basis_grad are different sizes")

        basis_pgrad[:] = np.transpose(np.matmul(ijac.transpose(0,2,1), basis_grad.transpose(0,2,1)), (0,2,1))

        return basis_pgrad

    def eval_basis(self, quad_pts, Get_Phi=True, Get_GPhi=False, Get_gPhi=False, ijac=None):
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
            self.basis_val = self.get_values(quad_pts, self.basis_val)
        if Get_GPhi:
            self.basis_grad = self.get_grads(quad_pts, self.basis_grad)
        if Get_gPhi:
            if ijac is None:
                raise Exception("Need jacobian data")
            self.basis_pgrad = self.get_physical_grad(ijac)

    def eval_basis_on_face(self, mesh, face, quad_pts, xelem=None, Get_Phi=True, Get_GPhi=False, Get_gPhi=False, ijac=False):
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
        basis = mesh.gbasis
        if xelem is None or xelem.shape != (nq, mesh.Dim):
            xelem = np.zeros([nq, mesh.Dim])
        xelem = basis.ref_face_to_elem(face, nq, quad_pts, xelem)
        self.eval_basis(xelem, Get_Phi, Get_GPhi, Get_gPhi, ijac)

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
        shape = basis.__class__.__bases__[1].__name__

        self.face = face
        nq = quad_pts.shape[0]
        if shape == 'QuadShape':
            if xelem is None or xelem.shape != (nq, mesh.Dim+1):
                xelem = np.zeros([nq, mesh.Dim+1])
            xelem = basis.ref_face_to_elem(face, nq, quad_pts, xelem)
        elif shape == 'SegShape':
            if xelem is None or xelem.shape != (nq, mesh.Dim):
                xelem = np.zeros([nq, mesh.Dim])
            xelem = basis.ref_face_to_elem(face, nq, quad_pts, xelem)
        self.eval_basis(xelem, Get_Phi, Get_GPhi, Get_gPhi, JData)

        return xelem

class LagrangeSeg(BasisBase, SegShape):
    def __init__(self, order, mesh=None):
        super().__init__(order)
        self.nb = self.get_num_basis_coeff(order)

    def get_values(self, quad_pts, basis_val=None):
        '''
        Method: get_values
        ------------------------------
        Calculates lagrange basis

        INPUTS:
            x: coordinate of current node

        OUTPUTS: 
            phi: evaluated basis 
        '''
        p = self.order
        nb = self.nb
        nq = quad_pts.shape[0]

        if basis_val is None or basis_val.shape != (nq,nb):
            basis_val = np.zeros([nq,nb])
        else:
            basis_val[:] = 0.

        if p == 0:
            basis_val[:] = 1.
            return basis_val

        nnode = p+1
        xnode = equidistant_nodes_1D_range(-1., 1., nnode)

        self.get_lagrange_basis_1D(quad_pts, xnode, nnode, basis_val, None)

        return basis_val

    def get_grads(self, quad_pts, basis_grad=None):
        '''
        Method: get_grads
        ------------------------------
        Calculates the lagrange basis gradients

        INPUTS:
            x: coordinate of current node
            
        OUTPUTS: 
            gphi: evaluated gradient of basis
        '''
        dim = self.dim
        p = self.order
        nb = self.nb
        nq = quad_pts.shape[0]

        if basis_grad is None or basis_grad.shape != (nq,nb,dim):
            basis_grad = np.zeros([nq,nb,dim])
        else: 
            basis_grad[:] = 0.

        if p == 0:
            basis_grad[:,:] = 0.
            return basis_grad

        nnode = p+1
        xnode = equidistant_nodes_1D_range(-1., 1., nnode)

        self.get_lagrange_basis_1D(quad_pts, xnode, nnode, None, basis_grad)

        return basis_grad

    def get_lagrange_basis_1D(self, x, xnode, nnode, phi, gphi):
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

    def calculate_normals(self, mesh, elem, face, quad_pts):
        '''
        Method: calculate_normals
        -------------------
        Calculate the normals

        INPUTS:
            mesh: Mesh object
            elem: element index
            face: face index
            quad_pts: points in reference space at which to calculate normals
        '''

        gorder = mesh.gorder
        nq = quad_pts.shape[0]

        if gorder == 1:
            nq = 1

        # if self.nvec is None or self.nvec.shape != (nq,mesh.Dim):
        nvec = np.zeros([nq,mesh.Dim])
        
        #1D normals calculation
        if face == 0:
            nvec[0] = -1.
        elif face == 1:
            nvec[0] = 1.
        else:
            raise ValueError

        return nvec

    def local_q1_face_nodes(self, p, face, fnodes=None):
        '''
        Method: local_q1_face_nodes
        -------------------
        Constructs the map for face nodes in 1D

        INPUTS:
            p: order of polynomial space
            face: face value in ref space

        OUTPUTS: 
            fnodes: index of face nodes
            nfnode: number of face nodes
        '''
        nfnode = 1
        if fnodes is None: fnodes = np.zeros(nfnode, dtype=int)
        if face == 0:
            fnodes[0] = 0
        elif face == 1:
            fnodes[0] = p
        else:
            raise IndexError

        return fnodes, nfnode

class LagrangeQuad(LagrangeSeg, QuadShape):
    def __init__(self, order, mesh=None):
        super().__init__(order)
        self.nb = self.get_num_basis_coeff(order)

    def get_values(self, quad_pts, basis_val=None):
        '''
        Method: get_values
        ------------------------------
        Calculates Lagrange basis for 2D quads

        INPUTS:
            x: coordinate of current node

        OUTPUTS: 
            phi: evaluated basis 
        '''
        p = self.order
        nb = self.nb
        nq = quad_pts.shape[0]

        if basis_val is None or basis_val.shape != (nq,nb):
            basis_val = np.zeros([nq,nb])
        else:
            basis_val[:] = 0.

        if p == 0:
            basis_val[:] = 1.
            return basis_val

        nnode = p+1
        xnode = equidistant_nodes_1D_range(-1., 1., nnode)

        self.get_lagrange_basis_2D(quad_pts, xnode, nnode, basis_val, None)

        return basis_val

    def get_grads(self, quad_pts, basis_grad=None):
        '''
        Method: get_grads
        ------------------------------
        Calculates the lagrange basis gradients for 2D quads

        INPUTS:
            x: coordinate of current node
            
        OUTPUTS: 
            gphi: evaluated gradient of basis
        '''
        dim = self.dim
        p = self.order
        nb = self.nb
        nq = quad_pts.shape[0]

        if basis_grad is None or basis_grad.shape != (nq,nb,dim):
            basis_grad = np.zeros([nq,nb,dim])
        else: 
            basis_grad[:] = 0.

        if p == 0:
            basis_grad[:,:] = 0.
            return basis_grad

        nnode = p+1
        xnode = equidistant_nodes_1D_range(-1., 1., nnode)

        self.get_lagrange_basis_2D(quad_pts, xnode, nnode, None, basis_grad)

        return basis_grad

    def get_lagrange_basis_2D(self, x, xnode, nnode, phi, gphi):
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

        self.get_lagrange_basis_1D(x[:,0].reshape(-1,1), xnode, nnode, phix, gphix)
        self.get_lagrange_basis_1D(x[:,1].reshape(-1,1), xnode, nnode, phiy, gphiy)

        if phi is not None:
            for i in range(x.shape[0]):
                phi[i,:] = np.reshape(np.outer(phix[i,:], phiy[i,:]), (-1,), 'F')
        if gphi is not None:
            for i in range(x.shape[0]):
                gphi[i,:,0] = np.reshape(np.outer(gphix[i,:,0], phiy[i,:]), (-1,), 'F')
                gphi[i,:,1] = np.reshape(np.outer(phix[i,:], gphiy[i,:,0]), (-1,), 'F')


    def calculate_normals(self, mesh, elem, face, quad_pts):
        '''
        Method: calculate_normals
        -------------------
        Calculate the normals for 2D shapes

        INPUTS:
            mesh: Mesh object
            elem: element index
            face: face index
            quad_pts: points in reference space at which to calculate normals
        '''
        gbasis = mesh.gbasis
        gorder = mesh.gorder

        nq = quad_pts.shape[0]

        if gorder == 1:
            nq = 1

        nvec = np.zeros([nq,mesh.Dim])

        # Calculate 2D normals
        ElemNodes = mesh.Elem2Nodes[elem]
        if gorder == 1:
            fnodes, nfnode = self.local_q1_face_nodes(gorder, face, fnodes=None)
            x0 = mesh.Coords[ElemNodes[fnodes[0]]]
            x1 = mesh.Coords[ElemNodes[fnodes[1]]]

            nvec[0,0] =  (x1[1]-x0[1])/2.;
            nvec[0,1] = -(x1[0]-x0[0])/2.;
        
        # Calculate normals for curved meshes
        else:
            x_s = np.zeros_like(nvec)
            fnodes, nfnode = self.local_face_nodes(gorder, face, fnodes=None)

            basis_seg = LagrangeSeg(gorder)
            basis_grad = basis_seg.get_grads(quad_pts, basis_grad=None)
            Coords = mesh.Coords[ElemNodes[fnodes]]

            # Face Jacobian (gradient of (x,y) w.r.t reference coordinate)
            x_s[:] = np.matmul(Coords.transpose(), basis_grad).reshape(x_s.shape)
            nvec[:,0] = x_s[:,1]
            nvec[:,1] = -x_s[:,0]

        return nvec

    def local_q1_face_nodes(self, p, face, fnodes=None):
        '''
        Method: local_q1_face_nodes
        -------------------
        Constructs the map for face nodes on 2D quads
        (For q1 elements only)

        INPUTS:
            p: order of polynomial space
            face: face value in ref space

        OUTPUTS: 
            fnodes: index of face nodes
            nfnode: number of face nodes
        '''
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

        return fnodes, nfnode

    def local_face_nodes(self, p, face, fnodes=None):
        '''
        Method: local_face_nodes
        -------------------
        Constructs the map for face nodes on 2D quads
        (For q > 1 elements)

        INPUTS:
            p: order of polynomial space
            face: face value in ref space

        OUTPUTS: 
            fnodes: index of face nodes
            nfnode: number of face nodes
        '''
        if p < 1:
            raise ValueError

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

        return fnodes, nfnode

class LagrangeTri(LagrangeQuad, TriShape):
    def __init__(self, order, mesh=None):
        super().__init__(order)
        self.nb = self.get_num_basis_coeff(order)

    def get_values(self, quad_pts, basis_val=None):
        '''
        Method: get_values
        ------------------------------
        Calculates Lagrange basis for triangles

        INPUTS:
            x: coordinate of current node

        OUTPUTS: 
            phi: evaluated basis 
        '''
        p = self.order
        nb = self.nb
        nq = quad_pts.shape[0]

        if basis_val is None or basis_val.shape != (nq,nb):
            basis_val = np.zeros([nq,nb])
        else:
            basis_val[:] = 0.

        if p == 0:
            basis_val[:] = 1.
            return basis_val

        self.get_lagrange_basis_tri(p, quad_pts, basis_val)

        return basis_val

    def get_grads(self, quad_pts, basis_grad=None):
        '''
        Method: get_grads
        ------------------------------
        Calculates the lagrange basis gradients

        INPUTS:
            x: coordinate of current node
            
        OUTPUTS: 
            gphi: evaluated gradient of basis
        '''
        dim = self.dim
        p = self.order
        nb = self.nb
        nq = quad_pts.shape[0]

        if basis_grad is None or basis_grad.shape != (nq,nb,dim):
            basis_grad = np.zeros([nq,nb,dim])
        else: 
            basis_grad[:] = 0.

        if p == 0:
            basis_grad[:,:] = 0.
            return basis_grad

        self.get_lagrange_grad_tri(p, quad_pts, basis_grad)

        return basis_grad

    def get_lagrange_basis_tri(self, p, xi, phi):

        xn, nb = self.equidistant_nodes(p)

        alpha = np.round(p*xn)
        alpha = np.c_[(p*np.ones(nb)-np.sum(alpha, axis=1),alpha)]
        l = np.c_[(np.ones(xi.shape[0]) - np.sum(xi, axis=1)),xi]

        if p == 0:
            phi[:] = 1.
            return 

        for i in range(nb):
            phi[:,i] = self.get_tri_area_coordinates(p, alpha[i], l)

        return phi

    def get_tri_area_coordinates(self, p, alpha, l):

        N = np.ones(l.shape[0])

        N *= self.get_eta_function(p, alpha[0], l[:,0])
        N *= self.get_eta_function(p, alpha[1], l[:,1])
        N *= self.get_eta_function(p, alpha[2], l[:,2])

        return N

    def get_eta_function(self, p, alpha, l, skip = -1):
        index = np.concatenate((np.arange(0, skip), np.arange(skip + 1, alpha)))

        eta = np.ones(l.shape[0])

        for i in index:
            eta *= (p * l - i) / (i + 1.)
        return eta

    def get_grad_eta_function(self, p, alpha, l):
        geta = np.zeros_like(l)

        for i in range(int(alpha)):
            geta += ( p / (i + 1)) * self.get_eta_function(p, alpha, l, i)

        return geta

    def get_lagrange_grad_tri(self, p, xi, gphi):

        xn, nb = self.equidistant_nodes(p)
        gphi_dir = np.zeros((xi.shape[0],nb,3))

        alpha = np.round(p*xn)
        alpha = np.c_[(p*np.ones(nb)-np.sum(alpha, axis=1),alpha)]
        l = np.c_[(np.ones(xi.shape[0]) - np.sum(xi, axis=1)),xi]

        if p == 0:
            gphi[:] = 0.
            return 
        for i in range(nb):
            gphi_dir[:,i,:] = self.get_tri_grad_area_coordinates(p, alpha[i], l)

        gphi[:,:,0] = gphi_dir[:,:,1] - gphi_dir[:,:,0]
        gphi[:,:,1] = gphi_dir[:,:,2] - gphi_dir[:,:,0]

        return gphi

    def get_tri_grad_area_coordinates(self, p, alpha, l):

        dN = np.ones((l.shape[0], 3))

        N1 = self.get_eta_function(p, alpha[0], l[:,0])
        N2 = self.get_eta_function(p, alpha[1], l[:,1])
        N3 = self.get_eta_function(p, alpha[2], l[:,2])

        dN1 = self.get_grad_eta_function(p, alpha[0], l[:,0])
        dN2 = self.get_grad_eta_function(p, alpha[1], l[:,1])
        dN3 = self.get_grad_eta_function(p, alpha[2], l[:,2])

        dN[:,0] = dN1 * N2 * N3
        dN[:,1] = N1 * dN2 * N3
        dN[:,2] = N1 * N2 * dN3

        return dN

    # def get_lagrange_basis_tri(self,p, xi, phi):
    #     '''
    #     Method: shape_lagrange_tri
    #     ------------------------------
    #     Calculates lagrange bais for triangles

    #     INPUTS:
    #         p: order of polynomial space
    #         xi: coordinates of current node

    #     OUTPUTS: 
    #         phi: evaluated basis 
    #     '''
    #     x = xi[:,0]; y = xi[:,1]

    #     if p == 0:
    #         phi[:] = 1.
    #     elif p == 1:
    #         phi[:,0] = 1-x-y
    #         phi[:,1] = x  
    #         phi[:,2] = y
    #     elif p == 2:
    #         phi[:,0] = 1.0-3.0*x-3.0*y+2.0*x*x+4.0*x*y+2.0*y*y
    #         phi[:,2] = -x+2.0*x*x
    #         phi[:,5] = -y+2.0*y*y
    #         phi[:,4] = 4.0*x*y
    #         phi[:,3] = 4.0*y-4.0*x*y-4.0*y*y
    #         phi[:,1] = 4.0*x-4.0*x*x-4.0*x*y
    #     elif p == 3:
    #         phi[:,0] = 1.0-11.0/2.0*x-11.0/2.0*y+9.0*x*x+18.0*x*y+9.0*y*y-9.0/2.0*x*x*x-27.0/2.0*x*x*y-27.0/2.0*x*y*y-9.0/2.0*y*y*y
    #         phi[:,3] = x-9.0/2.0*x*x+9.0/2.0*x*x*x
    #         phi[:,9] = y-9.0/2.0*y*y+9.0/2.0*y*y*y
    #         phi[:,6] = -9.0/2.0*x*y+27.0/2.0*x*x*y
    #         phi[:,8] = -9.0/2.0*x*y+27.0/2.0*x*y*y
    #         phi[:,7] = -9.0/2.0*y+9.0/2.0*x*y+18.0*y*y-27.0/2.0*x*y*y-27.0/2.0*y*y*y
    #         phi[:,4] = 9.0*y-45.0/2.0*x*y-45.0/2.0*y*y+27.0/2.0*x*x*y+27.0*x*y*y+27.0/2.0*y*y*y
    #         phi[:,1] = 9.0*x-45.0/2.0*x*x-45.0/2.0*x*y+27.0/2.0*x*x*x+27.0*x*x*y+27.0/2.0*x*y*y
    #         phi[:,2] = -9.0/2.0*x+18.0*x*x+9.0/2.0*x*y-27.0/2.0*x*x*x-27.0/2.0*x*x*y
    #         phi[:,5] = 27.0*x*y-27.0*x*x*y-27.0*x*y*y
    #     elif p == 4:
    #         phi[:, 0] = 1.0-25.0/3.0*x-25.0/3.0*y+70.0/3.0*x*x+140.0/3.0*x*y+70.0/3.0*y*y-80/3.0*x*x*x-80.0*x*x*y-80.0*x*y*y-80.0/3.0*y*y*y \
    #         +32.0/3.0*x*x*x*x+128/3.0*x*x*x*y+64.0*x*x*y*y+128.0/3.0*x*y*y*y+32.0/3.0*y*y*y*y 
    #         phi[:, 4] = -x+22.0/3.0*x*x-16.0*x*x*x+32.0/3.0*x*x*x*x 
    #         phi[:,14] = -y+22.0/3.0*y*y-16.0*y*y*y+32.0/3.0*y*y*y*y 
    #         phi[:, 8] = 16.0/3.0*x*y-32.0*x*x*y+128.0/3.0*x*x*x*y 
    #         phi[:,11] = 4.0*x*y-16.0*x*x*y-16.0*x*y*y+64.0*x*x*y*y 
    #         phi[:,13] = 16.0/3.0*x*y-32.0*x*y*y+128.0/3.0*x*y*y*y 
    #         phi[:,12] = 16.0/3.0*y-16.0/3.0*x*y-112.0/3.0*y*y+32.0*x*y*y+224.0/3.0*y*y*y-128.0/3.0*x*y*y*y-128.0/3.0*y*y*y*y 
    #         phi[:, 9] = -12.0*y+28.0*x*y+76.0*y*y-16.0*x*x*y-144.0*x*y*y-128.0*y*y*y+64.0*x*x*y*y+128.0*x*y*y*y+64.0*y*y*y*y 
    #         phi[:, 5] = 16.0*y-208.0/3.0*x*y-208.0/3.0*y*y+96.0*x*x*y+192.0*x*y*y+96.0*y*y*y-128.0/3.0*x*x*x*y- 128*x*x*y*y-128.0*x*y*y*y-128.0/3.0*y*y*y*y 
    #         phi[:, 1] = 16.0*x-208.0/3.0*x*x-208.0/3.0*x*y+96.0*x*x*x+192.0*x*x*y+96.0*x*y*y-128.0/3.0*x*x*x*x- 128*x*x*x*y-128.0*x*x*y*y-128.0/3.0*x*y*y*y 
    #         phi[:, 2] = -12.0*x+76.0*x*x+28.0*x*y-128.0*x*x*x-144.0*x*x*y-16.0*x*y*y+64.0*x*x*x*x+128.0*x*x*x*y+64.0*x*x*y*y 
    #         phi[:, 3] = 16.0/3.0*x-112.0/3.0*x*x-16.0/3.0*x*y+224.0/3.0*x*x*x+32.0*x*x*y-128.0/3.0*x*x*x*x-128.0/3.0*x*x*x*y 
    #         phi[:, 6] = 96.0*x*y-224.0*x*x*y-224.0*x*y*y+128.0*x*x*x*y+256.0*x*x*y*y+128.0*x*y*y*y 
    #         phi[:, 7] = -32.0*x*y+160.0*x*x*y+32.0*x*y*y-128.0*x*x*x*y-128.0*x*x*y*y 
    #         phi[:,10] = -32.0*x*y+32.0*x*x*y+160.0*x*y*y-128.0*x*x*y*y-128.0*x*y*y*y
    #     elif p == 5:
    #         phi[:, 0]  = 1.0-137.0/12.0*x-137.0/12.0*y+375.0/8.0*x*x+375.0/4.0*x*y+375.0/8.0*y*y-2125.0/24.0*x*x*x-2125.0/8.0*x*x*y-2125.0/8.0*x*y*y \
    #         -2125.0/24.0*y*y*y+ 625.0/8.0*x*x*x*x+625.0/2.0*x*x*x*y+1875.0/4.0*x*x*y*y+625.0/2.0*x*y*y*y+625.0/8.0*y*y*y*y-625.0/24.0*x*x*x*x*x \
    #         -3125.0/24.0*x*x*x*x*y-3125.0/12.0*x*x*x*y*y-3125.0/12.0*x*x*y*y*y-3125.0/24.0*x*y*y*y*y-625.0/24.0*y*y*y*y*y
    #         phi[:, 5]  = x-125.0/12.0*x*x+875.0/24.0*x*x*x-625.0/12.0*x*x*x*x+625.0/24.0*x*x*x*x*x
    #         phi[:,20]  = y-125.0/12.0*y*y+875.0/24.0*y*y*y-625.0/12.0*y*y*y*y+625.0/24.0*y*y*y*y*y
    #         phi[:,10]  = -25.0/4.0*x*y+1375.0/24.0*x*x*y-625.0/4.0*x*x*x*y+3125.0/24.0*x*x*x*x*y
    #         phi[:,14]  = -25.0/6.0*x*y+125.0/4.0*x*x*y+125.0/6.0*x*y*y-625.0/12.0*x*x*x*y-625.0/4.0*x*x*y*y+3125.0/12.0*x*x*x*y*y
    #         phi[:,17]  = -25.0/6.0*x*y+125.0/6.0*x*x*y+125.0/4.0*x*y*y-625.0/4.0*x*x*y*y-625.0/12.0*x*y*y*y+3125.0/12.0*x*x*y*y*y
    #         phi[:,19]  = -25.0/4.0*x*y+1375.0/24.0*x*y*y-625.0/4.0*x*y*y*y+3125.0/24.0*x*y*y*y*y
    #         phi[:,18]  = -25.0/4.0*y+25.0/4.0*x*y+1525.0/24.0*y*y-1375.0/24.0*x*y*y-5125.0/24.0*y*y*y+625.0/4.0*x*y*y*y+6875.0/24.0*y*y*y*y \
    #         -3125.0/24.0*x*y*y*y*y-3125.0/24.0*y*y*y*y*y
    #         phi[:,15]  = 50.0/3.0*y-75.0/2.0*x*y-325.0/2.0*y*y+125.0/6.0*x*x*y+3875.0/12.0*x*y*y+6125.0/12.0*y*y*y-625.0/4.0*x*x*y*y-3125.0/4.0*x*y*y*y \
    #         -625.0*y*y*y*y+3125.0/12.0*x*x*y*y*y+3125.0/6.0*x*y*y*y*y+3125.0/12.0*y*y*y*y*y
    #         phi[:,11]  = -25.0*y+1175.0/12.0*x*y+2675.0/12.0*y*y-125.0*x*x*y-8875.0/12.0*x*y*y-7375.0/12.0*y*y*y+625.0/12.0*x*x*x*y+3125.0/4.0*x*x*y*y \
    #         +5625.0/4.0*x*y*y*y+8125.0/12.0*y*y*y*y-3125.0/12.0*x*x*x*y*y-3125.0/4.0*x*x*y*y*y-3125.0/4.0*x*y*y*y*y-3125.0/12.0*y*y*y*y*y 
    #         phi[:, 6] = 25.0*y-1925.0/12.0*x*y-1925.0/12.0*y*y+8875.0/24.0*x*x*y+8875.0/12.0*x*y*y+8875.0/24.0*y*y*y-4375.0/12.0*x*x*x*y \
    #         -4375.0/4.0*x*x*y*y-4375.0/4.0*x*y*y*y-4375.0/12.0*y*y*y*y+3125.0/24.0*x*x*x*x*y+ 3125.0/6.0*x*x*x*y*y+3125.0/4.0*x*x*y*y*y \
    #         +3125.0/6.0*x*y*y*y*y+3125.0/24.0*y*y*y*y*y
    #         phi[:, 1] = 25.0*x-1925.0/12.0*x*x-1925.0/12.0*x*y+8875.0/24.0*x*x*x+8875.0/12.0*x*x*y+8875.0/24.0*x*y*y-4375.0/12.0*x*x*x*x-4375.0/4.0*x*x*x*y \
    #         -4375.0/4.0*x*x*y*y-4375.0/12.0*x*y*y*y+3125.0/24.0*x*x*x*x*x+3125.0/6.0*x*x*x*x*y+3125.0/4.0*x*x*x*y*y+3125.0/6.0*x*x*y*y*y+3125.0/24.0*x*y*y*y*y
    #         phi[:, 2] = -25.0*x+2675.0/12.0*x*x+1175.0/12.0*x*y-7375.0/12.0*x*x*x-8875.0/12.0*x*x*y-125.0*x*y*y+8125.0/12.0*x*x*x*x \
    #         +5625.0/4.0*x*x*x*y+3125.0/4.0*x*x*y*y+625.0/12.0*x*y*y*y-3125.0/12.0*x*x*x*x*x- 3125.0/4.0*x*x*x*x*y-3125.0/4.0*x*x*x*y*y-3125.0/12.0*x*x*y*y*y
    #         phi[:, 3] = 50.0/3.0*x-325.0/2.0*x*x-75.0/2.0*x*y+6125.0/12.0*x*x*x+3875.0/12.0*x*x*y+125.0/6.0*x*y*y-625.0*x*x*x*x \
    #         -3125.0/4.0*x*x*x*y-625.0/4.0*x*x*y*y+3125.0/12.0*x*x*x*x*x+3125.0/6.0*x*x*x*x*y+3125.0/12.0*x*x*x*y*y
    #         phi[:, 4] = -25.0/4.0*x+1525.0/24.0*x*x+25.0/4.0*x*y-5125.0/24.0*x*x*x-1375.0/24.0*x*x*y+6875.0/24.0*x*x*x*x+625.0/4.0*x*x*x*y \
    #         -3125.0/24.0*x*x*x*x*x-3125.0/24.0*x*x*x*x*y
    #         phi[:, 7] = 250.0*x*y-5875.0/6.0*x*x*y-5875.0/6.0*x*y*y+1250.0*x*x*x*y+2500.0*x*x*y*y+1250.0*x*y*y*y-3125.0/6.0*x*x*x*x*y \
    #         -3125.0/2.0*x*x*x*y*y-3125.0/2.0*x*x*y*y*y-3125.0/6.0*x*y*y*y*y
    #         phi[:, 8] = -125.0*x*y+3625.0/4.0*x*x*y+1125.0/4.0*x*y*y-3125.0/2.0*x*x*x*y-6875.0/4.0*x*x*y*y-625.0/4.0*x*y*y*y+3125.0/4.0*x*x*x*x*y \
    #         +3125.0/2.0*x*x*x*y*y+3125.0/4.0*x*x*y*y*y
    #         phi[:, 9] = 125.0/3.0*x*y-2125.0/6.0*x*x*y-125.0/3.0*x*y*y+2500.0/3.0*x*x*x*y+625.0/2.0*x*x*y*y-3125.0/6.0*x*x*x*x*y-3125.0/6.0*x*x*x*y*y
    #         phi[:,12] = -125.0*x*y+1125.0/4.0*x*x*y+3625.0/4.0*x*y*y-625.0/4.0*x*x*x*y-6875.0/4.0*x*x*y*y-3125.0/2.0*x*y*y*y+3125.0/4.0*x*x*x*y*y \
    #         +3125.0/2.0*x*x*y*y*y+3125.0/4.0*x*y*y*y*y
    #         phi[:,13] = 125.0/4.0*x*y-375.0/2.0*x*x*y-375.0/2.0*x*y*y+625.0/4.0*x*x*x*y+4375.0/4.0*x*x*y*y+625.0/4.0*x*y*y*y-3125.0/4.0*x*x*x*y*y \
    #         -3125.0/4.0*x*x*y*y*y
    #         phi[:,16] = 125.0/3.0*x*y-125.0/3.0*x*x*y-2125.0/6.0*x*y*y+625.0/2.0*x*x*y*y+2500.0/3.0*x*y*y*y-3125.0/6.0*x*x*y*y*y-3125.0/6.0*x*y*y*y*y

    # def get_lagrange_grad_tri(self, p, xi, gphi):
    #     '''
    #     Method: grad_lagrange_tri
    #     ------------------------------
    #     Calculates lagrange basis gradient for triangles

    #     INPUTS:
    #         p: order of polynomial space
    #         xi: coordinates of current node

    #     OUTPUTS: 
    #         gphi: evaluated gradient of basis 
    #     '''
    #     x = xi[:,0]; y = xi[:,1]

    #     if p == 0:
    #         gphi[:] = 0.
    #     elif p == 1:
    #         gphi[:,0,0] =  -1.0
    #         gphi[:,1,0] =  1.0
    #         gphi[:,2,0] =  0.0
    #         gphi[:,0,1] =  -1.0
    #         gphi[:,1,1] =  0.0
    #         gphi[:,2,1] =  1.0
    #     elif p == 2:
    #         gphi[:,0,0] =  -3.0+4.0*x+4.0*y
    #         gphi[:,2,0] =  -1.0+4.0*x
    #         gphi[:,5,0] =  0.0
    #         gphi[:,4,0] =  4.0*y
    #         gphi[:,3,0] =  -4.0*y
    #         gphi[:,1,0] =  4.0-8.0*x-4.0*y
    #         gphi[:,0,1] =  -3.0+4.0*x+4.0*y
    #         gphi[:,2,1] =  0.0
    #         gphi[:,5,1] =  -1.0+4.0*y
    #         gphi[:,4,1] =  4.0*x
    #         gphi[:,3,1] =  4.0-4.0*x-8.0*y
    #         gphi[:,1,1] =  -4.0*x
    #     elif p == 3:
    #         gphi[:,0,0] =  -11.0/2.0+18.0*x+18.0*y-27.0/2.0*x*x-27.0*x*y-27.0/2.0*y*y
    #         gphi[:,3,0] =  1.0-9.0*x+27.0/2.0*x*x
    #         gphi[:,9,0] =  0.0
    #         gphi[:,6,0] =  -9.0/2.0*y+27.0*x*y
    #         gphi[:,8,0] =  -9.0/2.0*y+27.0/2.0*y*y
    #         gphi[:,7,0] =  9.0/2.0*y-27.0/2.0*y*y
    #         gphi[:,4,0] =  -45.0/2.0*y+27.0*x*y+27.0*y*y
    #         gphi[:,1,0] =  9.0-45.0*x-45.0/2.0*y+81.0/2.0*x*x+54.0*x*y+27.0/2.0*y*y
    #         gphi[:,2,0] =  -9.0/2.0+36.0*x+9.0/2.0*y-81.0/2.0*x*x-27.0*x*y
    #         gphi[:,5,0] =  27.0*y-54.0*x*y-27.0*y*y
    #         gphi[:,0,1] =  -11.0/2.0+18.0*x+18.0*y-27.0/2.0*x*x-27.0*x*y-27.0/2.0*y*y
    #         gphi[:,3,1] =  0.0
    #         gphi[:,9,1] =  1.0-9.0*y+27.0/2.0*y*y
    #         gphi[:,6,1] =  -9.0/2.0*x+27.0/2.0*x*x
    #         gphi[:,8,1] =  -9.0/2.0*x+27.0*x*y
    #         gphi[:,7,1] =  -9.0/2.0+9.0/2.0*x+36.0*y-27.0*x*y-81.0/2.0*y*y
    #         gphi[:,4,1] =  9.0-45.0/2.0*x-45.0*y+27.0/2.0*x*x+54.0*x*y+81.0/2.0*y*y
    #         gphi[:,1,1] =  -45.0/2.0*x+27.0*x*x+27.0*x*y
    #         gphi[:,2,1] =  9.0/2.0*x-27.0/2.0*x*x
    #         gphi[:,5,1] =  27.0*x-27.0*x*x-54.0*x*y
    #     elif p == 4:
    #         gphi[:, 0,0] =  -25.0/3.0+140.0/3.0*x+140.0/3.0*y-80.0*x*x-160.0*x*y-80.0*y*y+128.0/3.0*x*x*x+128.0*x*x*y+128.0*x*y*y+128.0/3.0*y*y*y
    #         gphi[:, 4,0] =  -1.0+44.0/3.0*x-48.0*x*x+128.0/3.0*x*x*x
    #         gphi[:,14,0] =  0.0
    #         gphi[:, 8,0] =  16.0/3.0*y-64.0*x*y+128.0*x*x*y
    #         gphi[:,11,0] =  4.0*y-32.0*x*y-16.0*y*y+128.0*x*y*y
    #         gphi[:,13,0] =  16.0/3.0*y-32.0*y*y+128.0/3.0*y*y*y
    #         gphi[:,12,0] =  -16.0/3.0*y+32.0*y*y-128.0/3.0*y*y*y
    #         gphi[:, 9,0] =  28.0*y-32.0*x*y-144.0*y*y+128.0*x*y*y+128.0*y*y*y
    #         gphi[:, 5,0] =  -208.0/3.0*y+192.0*x*y+192.0*y*y-128.0*x*x*y-256.0*x*y*y-128.0*y*y*y
    #         gphi[:, 1,0] =  16.0-416.0/3.0*x-208.0/3.0*y+288.0*x*x+384.0*x*y+96.0*y*y-512.0/3.0*x*x*x-384.0*x*x*y-256.0*x*y*y-128.0/3.0*y*y*y
    #         gphi[:, 2,0] =  -12.0+152.0*x+28.0*y-384.0*x*x-288.0*x*y-16.0*y*y+256.0*x*x*x+384.0*x*x*y+128.0*x*y*y
    #         gphi[:, 3,0] =  16.0/3.0-224.0/3.0*x-16.0/3.0*y+224.0*x*x+64.0*x*y-512.0/3.0*x*x*x-128.0*x*x*y
    #         gphi[:, 6,0] =  96.0*y-448.0*x*y-224.0*y*y+384.0*x*x*y+512.0*x*y*y+128.0*y*y*y
    #         gphi[:, 7,0] =  -32.0*y+320.0*x*y+32.0*y*y-384.0*x*x*y-256.0*x*y*y
    #         gphi[:,10,0] =  -32.0*y+64.0*x*y+160.0*y*y-256.0*x*y*y-128.0*y*y*y
    #         gphi[:, 0,1] =  -25.0/3.0+140.0/3.0*x+140.0/3.0*y-80.0*x*x-160.0*x*y-80.0*y*y+128.0/3.0*x*x*x+128.0*x*x*y+128.0*x*y*y+128.0/3.0*y*y*y
    #         gphi[:, 4,1] =  0.0
    #         gphi[:,14,1] =  -1.0+44.0/3.0*y-48.0*y*y+128.0/3.0*y*y*y
    #         gphi[:, 8,1] =  16.0/3.0*x-32.0*x*x+128.0/3.0*x*x*x
    #         gphi[:,11,1] =  4.0*x-16.0*x*x-32.0*x*y+128.0*x*x*y
    #         gphi[:,13,1] =  16.0/3.0*x-64.0*x*y+128.0*x*y*y
    #         gphi[:,12,1] =  16.0/3.0-16.0/3.0*x-224.0/3.0*y+64.0*x*y+224.0*y*y-128.0*x*y*y-512.0/3.0*y*y*y
    #         gphi[:, 9,1] =  -12.0+28.0*x+152.0*y-16.0*x*x-288.0*x*y-384.0*y*y+128.0*x*x*y+384.0*x*y*y+256.0*y*y*y
    #         gphi[:, 5,1] =  16.0-208.0/3.0*x-416.0/3.0*y+96.0*x*x+384.0*x*y+288.0*y*y-128.0/3.0*x*x*x-256.0*x*x*y-384.0*x*y*y-512.0/3.0*y*y*y
    #         gphi[:, 1,1] =  -208.0/3.0*x+192.0*x*x+192.0*x*y-128.0*x*x*x-256.0*x*x*y-128.0*x*y*y
    #         gphi[:, 2,1] =  28.0*x-144.0*x*x-32.0*x*y+128.0*x*x*x+128.0*x*x*y
    #         gphi[:, 3,1] =  -16.0/3.0*x+32.0*x*x-128.0/3.0*x*x*x
    #         gphi[:, 6,1] =  96.0*x-224.0*x*x-448.0*x*y+128.0*x*x*x+512.0*x*x*y+384.0*x*y*y
    #         gphi[:, 7,1] =  -32.0*x+160.0*x*x+64.0*x*y-128.0*x*x*x-256.0*x*x*y
    #         gphi[:,10,1] =  -32.0*x+32.0*x*x+320.0*x*y-256.0*x*x*y-384.0*x*y*y
    #     elif p == 5:
    #         gphi[:, 0,0] =  -137.0/12.0+375.0/4.0*x+375.0/4.0*y-2125.0/8.0*x*x-2125.0/4.0*x*y-2125.0/8.0*y*y+625.0/2.0*x*x*x+1875.0/2.0*x*x*y \
    #         +1875.0/2.0*x*y*y+625.0/2.0*y*y*y-3125.0/24.0*x*x*x*x-3125.0/6.0*x*x*x*y-3125.0/4.0*x*x*y*y-3125.0/6.0*x*y*y*y-3125.0/24.0*y*y*y*y
    #         gphi[:, 5,0] =  1.0-125.0/6.0*x+875.0/8.0*x*x-625.0/3.0*x*x*x+3125.0/24.0*x*x*x*x
    #         gphi[:,20,0] =  0.0
    #         gphi[:,10,0] =  -25.0/4.0*y+1375.0/12.0*x*y-1875.0/4.0*x*x*y+3125.0/6.0*x*x*x*y
    #         gphi[:,14,0] =  -25.0/6.0*y+125.0/2.0*x*y+125.0/6.0*y*y-625.0/4.0*x*x*y-625.0/2.0*x*y*y+3125.0/4.0*x*x*y*y
    #         gphi[:,17,0] =  -25.0/6.0*y+125.0/3.0*x*y+125.0/4.0*y*y-625.0/2.0*x*y*y-625.0/12.0*y*y*y+3125.0/6.0*x*y*y*y
    #         gphi[:,19,0] =  -25.0/4.0*y+1375.0/24.0*y*y-625.0/4.0*y*y*y+3125.0/24.0*y*y*y*y
    #         gphi[:,18,0] =  25.0/4.0*y-1375.0/24.0*y*y+625.0/4.0*y*y*y-3125.0/24.0*y*y*y*y
    #         gphi[:,15,0] =  -75.0/2.0*y+125.0/3.0*x*y+3875.0/12.0*y*y-625.0/2.0*x*y*y-3125.0/4.0*y*y*y+3125.0/6.0*x*y*y*y+3125.0/6.0*y*y*y*y
    #         gphi[:,11,0] =  1175.0/12.0*y-250.0*x*y-8875.0/12.0*y*y+625.0/4.0*x*x*y+3125.0/2.0*x*y*y+5625.0/4.0*y*y*y-3125.0/4.0*x*x*y*y \
    #         -3125.0/2.0*x*y*y*y-3125.0/4.0*y*y*y*y
    #         gphi[:, 6,0] =  -1925.0/12.0*y+8875.0/12.0*x*y+8875.0/12.0*y*y-4375.0/4.0*x*x*y-4375.0/2.0*x*y*y-4375.0/4.0*y*y*y+3125.0/6.0*x*x*x*y \
    #         +3125.0/2.0*x*x*y*y+3125.0/2.0*x*y*y*y+3125.0/6.0*y*y*y*y
    #         gphi[:, 1,0] =  25.0-1925.0/6.0*x-1925.0/12.0*y+8875.0/8.0*x*x+8875.0/6.0*x*y+8875.0/24.0*y*y-4375.0/3.0*x*x*x-13125.0/4.0*x*x*y \
    #         -4375.0/2.0*x*y*y-4375.0/12.0*y*y*y+15625.0/24.0*x*x*x*x+6250.0/3.0*x*x*x*y+9375.0/4.0*x*x*y*y+3125.0/3.0*x*y*y*y+3125.0/24.0*y*y*y*y
    #         gphi[:, 2,0] =  -25.0+2675.0/6.0*x+1175.0/12.0*y-7375.0/4.0*x*x-8875.0/6.0*x*y-125.0*y*y+8125.0/3.0*x*x*x+16875.0/4.0*x*x*y \
    #         +3125.0/2.0*x*y*y+625.0/12.0*y*y*y-15625.0/12.0*x*x*x*x-3125.0*x*x*x*y-9375.0/4.0*x*x*y*y-3125.0/6.0*x*y*y*y
    #         gphi[:, 3,0] =  50.0/3.0-325.0*x-75.0/2.0*y+6125.0/4.0*x*x+3875.0/6.0*x*y+125.0/6.0*y*y-2500.0*x*x*x-9375.0/4.0*x*x*y \
    #         -625.0/2.0*x*y*y+15625.0/12.0*x*x*x*x+6250.0/3.0*x*x*x*y+3125.0/4.0*x*x*y*y
    #         gphi[:, 4,0] =  -25.0/4.0+1525.0/12.0*x+25.0/4.0*y-5125.0/8.0*x*x-1375.0/12.0*x*y+6875.0/6.0*x*x*x+1875.0/4.0*x*x*y \
    #         -15625.0/24.0*x*x*x*x-3125.0/6.0*x*x*x*y
    #         gphi[:, 7,0] =  250.0*y-5875.0/3.0*x*y-5875.0/6.0*y*y+3750.0*x*x*y+5000.0*x*y*y+1250.0*y*y*y-6250.0/3.0*x*x*x*y \
    #         -9375.0/2.0*x*x*y*y-3125.0*x*y*y*y-3125.0/6.0*y*y*y*y
    #         gphi[:, 8,0] =  -125.0*y+3625.0/2.0*x*y+1125.0/4.0*y*y-9375.0/2.0*x*x*y-6875.0/2.0*x*y*y-625.0/4.0*y*y*y+3125.0*x*x*x*y \
    #         +9375.0/2.0*x*x*y*y+3125.0/2.0*x*y*y*y
    #         gphi[:, 9,0] =  125.0/3.0*y-2125.0/3.0*x*y-125.0/3.0*y*y+2500.0*x*x*y+625.0*x*y*y-6250.0/3.0*x*x*x*y-3125.0/2.0*x*x*y*y
    #         gphi[:,12,0] =  -125.0*y+1125.0/2.0*x*y+3625.0/4.0*y*y-1875.0/4.0*x*x*y-6875.0/2.0*x*y*y-3125.0/2.0*y*y*y+9375.0/4.0*x*x*y*y \
    #         +3125.0*x*y*y*y+3125.0/4.0*y*y*y*y
    #         gphi[:,13,0] =  125.0/4.0*y-375.0*x*y-375.0/2.0*y*y+1875.0/4.0*x*x*y+4375.0/2.0*x*y*y+625.0/4.0*y*y*y-9375.0/4.0*x*x*y*y \
    #         -3125.0/2.0*x*y*y*y
    #         gphi[:,16,0] =  125.0/3.0*y-250.0/3.0*x*y-2125.0/6.0*y*y+625.0*x*y*y+2500.0/3.0*y*y*y-3125.0/3.0*x*y*y*y-3125.0/6.0*y*y*y*y
    #         gphi[:, 0,1] =  -137.0/12.0+375.0/4.0*x+375.0/4.0*y-2125.0/8.0*x*x-2125.0/4.0*x*y-2125.0/8.0*y*y+625.0/2.0*x*x*x \
    #         +1875.0/2.0*x*x*y+1875.0/2.0*x*y*y+625.0/2.0*y*y*y-3125.0/24.0*x*x*x*x-3125.0/6.0*x*x*x*y-3125.0/4.0*x*x*y*y-3125.0/6.0*x*y*y*y-3125.0/24.0*y*y*y*y 
    #         gphi[:, 5,1] =  0.0
    #         gphi[:,20,1] =  1.0-125.0/6.0*y+875.0/8.0*y*y-625.0/3.0*y*y*y+3125.0/24.0*y*y*y*y
    #         gphi[:,10,1] =  -25.0/4.0*x+1375.0/24.0*x*x-625.0/4.0*x*x*x+3125.0/24.0*x*x*x*x
    #         gphi[:,14,1] =  -25.0/6.0*x+125.0/4.0*x*x+125.0/3.0*x*y-625.0/12.0*x*x*x-625.0/2.0*x*x*y+3125.0/6.0*x*x*x*y
    #         gphi[:,17,1] =  -25.0/6.0*x+125.0/6.0*x*x+125.0/2.0*x*y-625.0/2.0*x*x*y-625.0/4.0*x*y*y+3125.0/4.0*x*x*y*y
    #         gphi[:,19,1] =  -25.0/4.0*x+1375.0/12.0*x*y-1875.0/4.0*x*y*y+3125.0/6.0*x*y*y*y
    #         gphi[:,18,1] =  -25.0/4.0+25.0/4.0*x+1525.0/12.0*y-1375.0/12.0*x*y-5125.0/8.0*y*y+1875.0/4.0*x*y*y+6875.0/6.0*y*y*y \
    #         -3125.0/6.0*x*y*y*y-15625.0/24.0*y*y*y*y
    #         gphi[:,15,1] =  50.0/3.0-75.0/2.0*x-325.0*y+125.0/6.0*x*x+3875.0/6.0*x*y+6125.0/4.0*y*y-625.0/2.0*x*x*y-9375.0/4.0*x*y*y \
    #         -2500.0*y*y*y+3125.0/4.0*x*x*y*y+6250.0/3.0*x*y*y*y+15625.0/12.0*y*y*y*y
    #         gphi[:,11,1] =  -25.0+1175.0/12.0*x+2675.0/6.0*y-125.0*x*x-8875.0/6.0*x*y-7375.0/4.0*y*y+625.0/12.0*x*x*x+3125.0/2.0*x*x*y \
    #         +16875.0/4.0*x*y*y+8125.0/3.0*y*y*y-3125.0/6.0*x*x*x*y-9375.0/4.0*x*x*y*y-3125.0*x*y*y*y-15625.0/12.0*y*y*y*y
    #         gphi[:, 6,1] =  25.0-1925.0/12.0*x-1925.0/6.0*y+8875.0/24.0*x*x+8875.0/6.0*x*y+8875.0/8.0*y*y-4375.0/12.0*x*x*x-4375.0/2.0*x*x*y \
    #         -13125.0/4.0*x*y*y-4375.0/3.0*y*y*y+3125.0/24.0*x*x*x*x+3125.0/3.0*x*x*x*y+9375.0/4.0*x*x*y*y+6250.0/3.0*x*y*y*y+15625.0/24.0*y*y*y*y
    #         gphi[:, 1,1] =  -1925.0/12.0*x+8875.0/12.0*x*x+8875.0/12.0*x*y-4375.0/4.0*x*x*x-4375.0/2.0*x*x*y-4375.0/4.0*x*y*y \
    #         +3125.0/6.0*x*x*x*x+3125.0/2.0*x*x*x*y+3125.0/2.0*x*x*y*y+3125.0/6.0*x*y*y*y
    #         gphi[:, 2,1] =  1175.0/12.0*x-8875.0/12.0*x*x-250.0*x*y+5625.0/4.0*x*x*x+3125.0/2.0*x*x*y+625.0/4.0*x*y*y-3125.0/4.0*x*x*x*x \
    #         -3125.0/2.0*x*x*x*y-3125.0/4.0*x*x*y*y
    #         gphi[:, 3,1] =  -75.0/2.0*x+3875.0/12.0*x*x+125.0/3.0*x*y-3125.0/4.0*x*x*x-625.0/2.0*x*x*y+3125.0/6.0*x*x*x*x+3125.0/6.0*x*x*x*y
    #         gphi[:, 4,1] =  25.0/4.0*x-1375.0/24.0*x*x+625.0/4.0*x*x*x-3125.0/24.0*x*x*x*x
    #         gphi[:, 7,1] =  250.0*x-5875.0/6.0*x*x-5875.0/3.0*x*y+1250.0*x*x*x+5000.0*x*x*y+3750.0*x*y*y-3125.0/6.0*x*x*x*x \
    #         -3125.0*x*x*x*y-9375.0/2.0*x*x*y*y-6250.0/3.0*x*y*y*y
    #         gphi[:, 8,1] =  -125.0*x+3625.0/4.0*x*x+1125.0/2.0*x*y-3125.0/2.0*x*x*x-6875.0/2.0*x*x*y-1875.0/4.0*x*y*y+3125.0/4.0*x*x*x*x \
    #         +3125.0*x*x*x*y+9375.0/4.0*x*x*y*y
    #         gphi[:, 9,1] =  125.0/3.0*x-2125.0/6.0*x*x-250.0/3.0*x*y+2500.0/3.0*x*x*x+625.0*x*x*y-3125.0/6.0*x*x*x*x-3125.0/3.0*x*x*x*y
    #         gphi[:,12,1] =  -125.0*x+1125.0/4.0*x*x+3625.0/2.0*x*y-625.0/4.0*x*x*x-6875.0/2.0*x*x*y-9375.0/2.0*x*y*y+3125.0/2.0*x*x*x*y \
    #         +9375.0/2.0*x*x*y*y+3125.0*x*y*y*y
    #         gphi[:,13,1] =  125.0/4.0*x-375.0/2.0*x*x-375.0*x*y+625.0/4.0*x*x*x+4375.0/2.0*x*x*y+1875.0/4.0*x*y*y-3125.0/2.0*x*x*x*y-9375.0/4.0*x*x*y*y
    #         gphi[:,16,1] =  125.0/3.0*x-125.0/3.0*x*x-2125.0/3.0*x*y+625.0*x*x*y+2500.0*x*y*y-3125.0/2.0*x*x*y*y-6250.0/3.0*x*y*y*y


    def local_q1_face_nodes(self, p, face, fnodes=None):
        '''
        Method: local_q1_face_nodes
        -------------------
        Constructs the map for face nodes on triangles
        (For q1 elements only)

        INPUTS:
            p: order of polynomial space
            face: face value in ref space

        OUTPUTS: 
            fnodes: index of face nodes
            nfnode: number of face nodes
        '''
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

        return fnodes, nfnode

    def local_face_nodes(self, p, face, fnodes=None):
        '''
        Method: local_face_nodes
        -------------------
        Constructs the map for face nodes on triangles
        (For q > 1 elements only)

        INPUTS:
            p: order of polynomial space
            face: face value in ref space

        OUTPUTS: 
            fnodes: index of face nodes
            nfnode: number of face nodes
        '''
        if p < 1:
            raise ValueError

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

        return fnodes, nfnode

class LegendreSeg(BasisBase, SegShape):
    def __init__(self, order, mesh=None):
        super().__init__(order)
        self.nb = self.get_num_basis_coeff(order)

    def get_values(self, quad_pts, basis_val=None):
        '''
        Method: get_values
        ------------------------------
        Calculates Legendre basis for segments

        INPUTS:
            x: coordinate of current node

        OUTPUTS: 
            phi: evaluated basis 
        '''
        p = self.order
        nb = self.nb
        nq = quad_pts.shape[0]

        if basis_val is None or basis_val.shape != (nq,nb):
            basis_val = np.zeros([nq,nb])
        else:
            basis_val[:] = 0.

        if p == 0:
            basis_val[:] = 1.
            return basis_val

        self.get_legendre_basis_1D(quad_pts, p, basis_val, None)

        return basis_val

    def get_grads(self, quad_pts, basis_grad=None):
        '''
        Method: grad_tensor_lagrange
        ------------------------------
        Calculates the Legendre basis gradients

        INPUTS:
            x: coordinate of current node
            
        OUTPUTS: 
            gphi: evaluated gradient of basis
        '''
        dim = self.dim
        p = self.order
        nb = self.nb
        nq = quad_pts.shape[0]

        if basis_grad is None or basis_grad.shape != (nq,nb,dim):
            basis_grad = np.zeros([nq,nb,dim])
        else: 
            basis_grad[:] = 0.

        if p == 0:
            basis_grad[:,:] = 0.
            return basis_grad

        self.get_legendre_basis_1D(quad_pts, p, None, basis_grad)

        return basis_grad
    
    def get_legendre_basis_1D(self, x, p, phi, gphi):
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
        leg_poly = np.polynomial.legendre.Legendre

        if phi is not None:
            phi[:,:] = 0.            
            x.shape = -1
            
            for it in range(p+1):
                phi[:,it] = leg_poly.basis(it)(x)

            # phi[:,:] = leg
            # phi[:,:] = np.polynomial.legendre.legval(x,c)
            # code.interact(local=locals())
            # if p >= 0:
            #     phi[:,0]  = 1.
            # if p>=1:
            #     phi[:,1]  = x
            # if p>=2:
            #     phi[:,2]  = 0.5*(3.*x*x - 1.)
            # if p>=3:
            #     phi[:,3]  = 0.5*(5.*x*x*x - 3.*x)
            # if p>=4:
            #     phi[:,4]  = 0.125*(35.*x*x*x*x - 30.*x*x + 3.)
            # if p>=5:
            #     phi[:,5]  = 0.125*(63.*x*x*x*x*x - 70.*x*x*x + 15.*x)
            # if p>=6:
            #     phi[:,6]  = 0.0625*(231.*x*x*x*x*x*x - 315.*x*x*x*x + 105.*x*x -5.)
            # if p==7:
            #     phi[:,7]  = 0.0625*(429.*x*x*x*x*x*x*x - 693.*x*x*x*x*x + 315.*x*x*x - 35.*x)
            # if p>7:
            #     raise NotImplementedError("Legendre Polynomial > 7 not supported")

            x.shape = -1,1

        if gphi is not None:
            gphi[:,:] = 0.

            for it in range(p+1):
                dleg = leg_poly.basis(it).deriv(1)
                gphi[:,it] = dleg(x)

            # if p >= 0:
            #     gphi[:,0] = 0.
            # if p>=1:
            #     gphi[:,1] = 1.
            # if p>=2:
            #     gphi[:,2] = 3.*x
            # if p>=3:
            #     gphi[:,3] = 0.5*(15.*x*x - 3.)
            # if p>=4:
            #     gphi[:,4] = 0.125*(35.*4.*x*x*x - 60.*x)
            # if p>=5:
            #     gphi[:,5] = 0.125*(63.*5.*x*x*x*x - 210.*x*x + 15.)
            # if p>=6:
            #     gphi[:,6] = 0.0625*(231.*6.*x*x*x*x*x - 315.*4.*x*x*x + 210.*x)
            # if p==7:
            #     gphi[:,7] = 0.0625*(429.*7.*x*x*x*x*x*x - 693.*5.*x*x*x*x + 315.*3.*x*x - 35.)
            # if p>7:
            #     raise NotImplementedError("Legendre Polynomial > 7 not supported")

class LegendreQuad(LegendreSeg, QuadShape):
    def __init__(self, order, mesh=None):
        super().__init__(order)
        self.nb = self.get_num_basis_coeff(order)

    def get_values(self, quad_pts, basis_val=None):
        '''
        Method: get_values
        ------------------------------
        Calculates Legendre basis for 2D quads

        INPUTS:
            x: coordinate of current node

        OUTPUTS: 
            phi: evaluated basis 
        '''
        p = self.order
        nb = self.nb
        nq = quad_pts.shape[0]

        if basis_val is None or basis_val.shape != (nq,nb):
            basis_val = np.zeros([nq,nb])
        else:
            basis_val[:] = 0.

        if p == 0:
            basis_val[:] = 1.
            return basis_val

        self.get_legendre_basis_2D(quad_pts, p, basis_val, None)

        return basis_val

    def get_grads(self, quad_pts, basis_grad=None):
        '''
        Method: get_grads
        ------------------------------
        Calculates the Legendre basis gradients

        INPUTS:
            dim: dimension of mesh
            p: order of polynomial space
            x: coordinate of current node
            
        OUTPUTS: 
            gphi: evaluated gradient of basis
        '''

        dim = self.dim
        p = self.order
        nb = self.nb
        nq = quad_pts.shape[0]

        if basis_grad is None or basis_grad.shape != (nq,nb,dim):
            basis_grad = np.zeros([nq,nb,dim])
        else: 
            basis_grad[:] = 0.

        if p == 0:
            basis_grad[:,:] = 0.
            return basis_grad

        self.get_legendre_basis_2D(quad_pts, p, None, basis_grad)

        return basis_grad

    def get_legendre_basis_2D(self, x, p, phi, gphi):
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

        if gphi is not None:
            gphix = np.zeros((x.shape[0],p+1,1)); gphiy = np.zeros_like(gphix)
        else:
            gphix = None; gphiy = None
        # Always need phi
        phix = np.zeros((x.shape[0],p+1)); phiy = np.zeros_like(phix)

        self.get_legendre_basis_1D(x[:,0], p, phix, gphix)
        self.get_legendre_basis_1D(x[:,1], p, phiy, gphiy)

        if phi is not None:
            for i in range(x.shape[0]):
                phi[i,:] = np.reshape(np.outer(phix[i,:], phiy[i,:]), (-1,), 'F')
        if gphi is not None:
            for i in range(x.shape[0]):
                gphi[i,:,0] = np.reshape(np.outer(gphix[i,:,0], phiy[i,:]), (-1,), 'F')
                gphi[i,:,1] = np.reshape(np.outer(phix[i,:], gphiy[i,:,0]), (-1,), 'F')


