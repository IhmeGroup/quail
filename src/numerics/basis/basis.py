from abc import ABC, abstractmethod
import code
import numpy as np

from data import ArrayList, GenericData
from general import SetSolverParams, BasisType, ShapeType, EntityType

import meshing.gmsh as mesh_gmsh

from numerics.basis import math
import numerics.quadrature.quadrature as quadrature


Basis2Shape = {
    BasisType.LagrangeEqSeg : ShapeType.Segment,
    BasisType.LagrangeEqQuad : ShapeType.Quadrilateral,
    BasisType.LagrangeEqTri : ShapeType.Triangle,
    BasisType.LegendreSeg : ShapeType.Segment,
    BasisType.LegendreQuad : ShapeType.Quadrilateral,
    BasisType.HierarchicH1Tri : ShapeType.Triangle
}

RefQ1Coords = {
    BasisType.LagrangeEqSeg : np.array([[-1.],[1.]]),
    BasisType.LagrangeEqQuad : np.array([[-1.,-1.],[1.,-1.],
                                [-1.,1.],[1.,1.]]),
    BasisType.LagrangeEqTri : np.array([[0.,0.],[1.,0.],
                                [0.,1.]]),
    BasisType.LegendreSeg : np.array([[-1.],[1.]]),
    BasisType.LegendreQuad : np.array([[-1.,-1.],[1.,-1.],
                                [-1.,1.],[1.,1.]]),
    BasisType.HierarchicH1Tri : np.array([[0.,0.],[1.,0.],
                                [0.,1.]])
}


def set_basis(mesh, order, BasisFunction):

    if BasisType[BasisFunction] == BasisType.LagrangeEqSeg:
        basis = LagrangeEqSeg(order, mesh)
    elif BasisType[BasisFunction] == BasisType.LegendreSeg:
        basis = LegendreSeg(order, mesh)
    elif BasisType[BasisFunction] == BasisType.LagrangeEqQuad:
        basis = LagrangeEqQuad(order, mesh)
    elif BasisType[BasisFunction] == BasisType.LegendreQuad:
        basis = LegendreQuad(order, mesh)
    elif BasisType[BasisFunction] == BasisType.LagrangeEqTri:
        basis = LagrangeEqTri(order, mesh)
    elif BasisType[BasisFunction] == BasisType.HierarchicH1Tri:
        basis = HierarchicH1Tri(order, mesh)
    else:
        raise NotImplementedError
    return basis

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

def get_elem_mass_matrix(mesh, basis, order, elem=-1, PhysicalSpace=False):
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

    if PhysicalSpace:
        QuadOrder,QuadChanged = quadrature.get_gaussian_quadrature_elem(mesh, mesh.gbasis, order*2)
    else:
        QuadOrder = order*2
        QuadChanged = True

    if QuadChanged:
        quadData = quadrature.QuadData(mesh, basis, EntityType.Element, QuadOrder)

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

    return MM

def get_elem_inv_mass_matrix(mesh, basis, order, elem=-1, PhysicalSpace=False):
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
    MM = get_elem_mass_matrix(mesh, basis, order, elem, PhysicalSpace)
    
    iMM = np.linalg.inv(MM) 

    return iMM

def get_elem_inv_mass_matrix_ader(mesh, basis, order, elem=-1, PhysicalSpace=False):
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
    MM = get_elem_mass_matrix_ader(mesh, basis, order, elem, PhysicalSpace)

    iMM = np.linalg.inv(MM)

    return iMM

def get_stiffness_matrix(mesh, basis, order, elem):
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
    QuadOrder,QuadChanged = quadrature.get_gaussian_quadrature_elem(mesh, mesh.QBasis, order*2)
    if QuadChanged:
        quadData = quadrature.QuadData(mesh, mesh.QBasis, EntityType.Element, QuadOrder)

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

    return SM

def get_stiffness_matrix_ader(mesh, basis, basis_st, order, dt, elem, gradDir, PhysicalSpace=False):
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
    
    dim = mesh.Dim

    QuadOrder_st,QuadChanged_st = quadrature.get_gaussian_quadrature_elem(mesh, basis_st, order*2)

    QuadOrder = QuadOrder_st
    QuadChanged = True
    if QuadChanged_st:
        quadData_st = quadrature.QuadData(mesh, basis_st, EntityType.Element, QuadOrder_st)
    if QuadChanged:
        quadData = quadrature.QuadData(mesh, basis, EntityType.Element, QuadOrder)

    quad_pts_st = quadData_st.quad_pts
    quad_wts_st = quadData_st.quad_wts
    nq_st = quad_pts_st.shape[0]

    quad_pts = quadData.quad_pts
    quad_wts = quadData.quad_wts
    nq = quad_pts.shape[0]
    
    if PhysicalSpace:
        djac,_,ijac=element_jacobian(mesh,elem,quad_pts_st,get_djac=True, get_ijac=True)
        if len(djac) == 1:
            djac = np.full(nq, djac[0])
        ijac_st = np.zeros([nq_st,dim+1,dim+1])
        ijac_st[:,0:dim,0:dim] = ijac
        ijac_st[:,dim,dim] = 2./dt
    else:
        djac = np.full(nq, 1.)

    basis_st.eval_basis(quad_pts_st, Get_Phi=True, Get_GPhi=True)
    nb_st = basis_st.basis_val.shape[1]
    phi = basis_st.basis_val

    if PhysicalSpace:
        basis_grad = basis_st.basis_grad
        GPhi = np.transpose(np.matmul(ijac_st.transpose(0,2,1), basis_grad.transpose(0,2,1)), (0,2,1))
    else:
        GPhi = basis_st.basis_grad

    SM = np.zeros([nb_st,nb_st])
    # code.interact(local=locals())
    # for i in range(nn):
    #     for j in range(nn):
    #         t = 0.
    #         for iq in range(nq):
    #             t += GPhi[iq,i,gradDir]*phi[iq,j]*wq[iq]
    #         SM[i,j] = t
    SM[:] = np.matmul(GPhi[:,:,gradDir].transpose(),phi*quad_wts_st) # [nb,nb]

    return SM

def get_temporal_flux_ader(mesh, basis1, basis2, order, elem=-1, PhysicalSpace=False):
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
    if basis1 == basis2:
        face = 2 
    else:
        face = 0

    QuadOrder,QuadChanged = quadrature.get_gaussian_quadrature_elem(mesh, mesh.gbasis, order*2)
  
    if QuadChanged:
        quadData = quadrature.QuadData(mesh, mesh.gbasis, EntityType.Element, QuadOrder)

    quad_pts = quadData.quad_pts
    quad_wts = quadData.quad_wts
    nq = quad_pts.shape[0]

    if QuadChanged:
        if basis1 == basis2:
            face = 2

            PhiData = basis1
            PsiData = basis1

            xelem = np.zeros([nq,mesh.Dim+1])
            PhiData.eval_basis_on_face(mesh, face, quad_pts, xelem, basis1, Get_Phi=True)
            PsiData.eval_basis_on_face(mesh, face, quad_pts, xelem, basis1, Get_Phi=True)
        else:
            face = 0
            
            PhiData = basis1
            PsiData = basis2

            xelemPhi = np.zeros([nq,mesh.Dim+1])
            xelemPsi = np.zeros([nq,mesh.Dim])
            PhiData.eval_basis_on_face(mesh, face, quad_pts, xelemPhi, basis1, Get_Phi=True)
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

    return FT


def get_elem_mass_matrix_ader(mesh, basis, order, elem=-1, PhysicalSpace=False):
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
    if PhysicalSpace:
        QuadOrder,QuadChanged = quadrature.get_gaussian_quadrature_elem(mesh, mesh.gbasis, order*2)
    else:
        QuadOrder = order*2 + 1 #Add one for ADER method
        QuadChanged = True

    if QuadChanged:
        quadData = quadrature.QuadData(mesh, basis, EntityType.Element, QuadOrder)


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

    return MM

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
    quadData = quadrature.QuadData(mesh, basis, EntityType.Element, QuadOrder)

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


def get_inv_stiffness_matrix(mesh, basis, order, elem):
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
    SM = get_stiffness_matrix(mesh, basis, order, elem)

    iSM = np.linalg.inv(SM) 

    return iSM

def get_inv_stiffness_matrix_ader(mesh, basis, order, elem, gradDir):
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
    SM = get_stiffness_matrix_ader(mesh, basis, order, elem, gradDir)

    iSM = np.linalg.inv(SM) 

    return iSM

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
    order = EqnSet.order
    nb = basis.nb

    iMM_all = np.zeros([mesh.nElem, nb, nb])

    # Uniform mesh?
    ReCalcMM = True
    # if solver is not None:
    #     ReCalcMM = not solver.Params["UniformMesh"]
    for elem in range(mesh.nElem):
        if elem == 0 or ReCalcMM:
            # Only recalculate if not using uniform mesh
            iMM = get_elem_inv_mass_matrix(mesh, basis, order, elem, True)
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
        math.MatDetInv(jac[i], dim, djac[i], ijac[i])
 
    if get_djac and np.any(djac[i] <= 0.):
        raise Exception("Nonpositive Jacobian (elem = %d)" % (elem))

    return djac, jac, ijac


class ShapeBase(ABC):
    @property
    @abstractmethod
    def faceshape(self):
        pass

    @property
    @abstractmethod
    def nfaceperelem(self):
        pass

    @property
    @abstractmethod
    def dim(self):
        pass

    @property
    @abstractmethod
    def centroid(self):
        pass
    
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
    centroid = np.array([[0.]])
    
    def get_num_basis_coeff(self,p):
        return 1
    def equidistant_nodes(self, p, xn=None):
        pass


class SegShape(PointShape):

    faceshape = PointShape()
    nfaceperelem = 2
    dim = 1
    centroid = np.array([[0.]])

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
    centroid = np.array([[0., 0.]])

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

        x0 = RefQ1Coords[BasisType.LagrangeEqQuad][fnodes[0]]
        x1 = RefQ1Coords[BasisType.LagrangeEqQuad][fnodes[1]]

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
    centroid = np.array([[1./3., 1./3.]])

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
        x0 = RefQ1Coords[BasisType.LagrangeEqTri][fnodes[0]]
        x1 = RefQ1Coords[BasisType.LagrangeEqTri][fnodes[1]]
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

    def eval_basis_on_face(self, mesh, face, quad_pts, xelem=None, basis = None, Get_Phi=True, Get_GPhi=False, Get_gPhi=False, ijac=False):
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
        if basis is None:
            basis = mesh.gbasis
        if xelem is None or xelem.shape != (nq, self.dim):
            xelem = np.zeros([nq, self.dim])
        xelem = basis.ref_face_to_elem(face, nq, quad_pts, xelem)
        self.eval_basis(xelem, Get_Phi, Get_GPhi, Get_gPhi, ijac)

        return xelem

class LagrangeEqSeg(BasisBase, SegShape):
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

    def local_face_nodes(self, p, face, fnodes=None):
        fnodes, nfnode = self.local_q1_face_nodes(p, face, fnodes=None)

        return fnodes, nfnode


class LagrangeEqQuad(LagrangeEqSeg, QuadShape):
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

            basis_seg = LagrangeEqSeg(gorder)
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


class LagrangeEqTri(LagrangeEqQuad, TriShape):
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

            x.shape = -1,1

        if gphi is not None:
            gphi[:,:] = 0.

            for it in range(p+1):
                dleg = leg_poly.basis(it).deriv(1)
                gphi[:,it] = dleg(x)

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

class HierarchicH1Tri(LegendreQuad, TriShape):
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

        self.get_modal_basis_tri(p, quad_pts, basis_val)

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

        self.get_modal_grad_tri(p, quad_pts, basis_grad)

        basis_grad = 2.*basis_grad

        return basis_grad

    def get_modal_basis_tri(self, p, xi, phi):

        xn, nb = self.equidistant_nodes(p)

        phi_reorder = np.zeros_like(phi)

        # Transform to the modal basis reference element
        # [-1,-1],[1,-1],[-1,1]
        xn = 2.*xn - 1.
        xi = 2.*xi - 1.

        # Define the affine coordinates
        l = np.zeros([xi.shape[0],3])

        l[:,0] = (xi[:,1]+1.)/2.
        l[:,1] = -1.*((xi[:,1]+xi[:,0])/2.)
        l[:,2] = (xi[:,0]+1.)/2.

        if p == 0:
            phi[:] = 1.
            return 

        phi_reorder[:,[0,1,2]] = l[:,[1,2,0]]

        e1 = np.arange(3,p-1+3,1)
        e2 = np.arange(p-1+3,2*p-2+3,1)
        e3 = np.arange(2*p-2+3,3*p-3+3,1)

        phi_reorder[:,e1] = self.get_edge_basis(p, l[:,2], l[:,1])
        phi_reorder[:,e2] = self.get_edge_basis(p, l[:,0], l[:,2])
        phi_reorder[:,e3] = self.get_edge_basis(p, l[:,1], l[:,0])

        internal = np.arange(3*p-3+3,nb,1)

        phi_reorder[:,internal] = self.get_internal_basis(p, internal, l)

        index = mesh_gmsh.gmsh_node_order_tri(p)

        phi[:,:] = phi_reorder[:,index]

        return phi

    def get_edge_basis(self, p, ll, lr):

        phi_e = np.zeros([ll.shape[0],p-1])
        for k in range(p-1):
            kernel = self.get_kernel_function(k,ll-lr)
            phi_e[:,k] = ll*lr*kernel

        return phi_e

    def get_internal_basis(self, p, index, l):

        phi_i = np.zeros([l.shape[0],len(index)])

        c=0
        for i in range(3,p+1):
            c += i-2
        
        n = np.zeros([c,2])
        n1 = np.arange(1,p-1,1)
        n2 = np.arange(1,p-1,1)
        k = 0
        for i in range(len(n1)):
            for j in range(len(n2)):
                if n1[i] + n2[j] <= p-1:
                    n[k,0] = n1[i]
                    n[k,1] = n2[j]
                    k+=1

        for m in range(c):
            phi_i[:,m] = l[:,0]*l[:,1]**n[m,0]*l[:,2]**n[m,1]

        return phi_i

    def get_kernel_function(self, p, x):

        p+=2
        # Initialize the legendre polynomial object
        leg_poly = np.polynomial.legendre.Legendre
        x.shape = -1

        # Construct the kernel's denominator (series of Lobatto fnc's)                    
        
        # First two lobatto shape functions 
        l0 =  (1.-x)/2.
        l1 =  (1.+x)/2.

        den = l0*l1

        leg_int = leg_poly.basis(p-1).integ(m=1,lbnd=-1)
        num = np.sqrt((2.*p-1.)/2.)*leg_int(x)

        kernel = num / (1e-12 + den)

        x.shape = -1,1

        return kernel


    def get_modal_grad_tri(self, p, xi, gphi):
     
        xn, nb = self.equidistant_nodes(p)

        gphi_reorder = np.zeros_like(gphi)
        # Transform to the modal basis reference element
        # [-1,-1],[1,-1],[-1,1]

        xn = 2.*xn - 1.
        xi = 2.*xi - 1.

        gl = np.zeros([xi.shape[0],3,2])
        l = np.zeros([xi.shape[0],3])

        # Calculate the affine coordinates
        l[:,0] = (xi[:,1]+1.)/2.
        l[:,1] = -1.*((xi[:,1]+xi[:,0])/2.)
        l[:,2] = (xi[:,0]+1.)/2.

        # Calculate vertex gradients
        gl[:,0,0] = 0.
        gl[:,0,1] = 0.5 
        
        gl[:,1,0] = -0.5
        gl[:,1,1] = -0.5

        gl[:,2,0] = 0.5
        gl[:,2,1] = 0.

        if p == 0:
            phi[:] = 1.
            return 

        gphi_reorder[:,[0,1,2],:] = gl[:,[1,2,0],:]

        # Calculate edge gradients
        e1 = np.arange(3,p-1+3,1)
        e2 = np.arange(p-1+3,2*p-2+3,1)
        e3 = np.arange(2*p-2+3,3*p-3+3,1)

        dxdxi = np.zeros([3,2])
        dxdxi[0,0] = 1.   ; dxdxi[0,1] = 0.5
        dxdxi[1,0] = -0.5 ; dxdxi[1,1] = 0.5
        dxdxi[2,0] = -0.5 ; dxdxi[2,1] = -1.

        gphi_reorder[:,e1,0] = self.get_edge_grad(p, dxdxi[0,0], gl[:,2,0], gl[:,1,0], l[:,2], l[:,1])
        gphi_reorder[:,e1,1] = self.get_edge_grad(p, dxdxi[0,1], gl[:,2,1], gl[:,1,1], l[:,2], l[:,1])
        gphi_reorder[:,e2,0] = self.get_edge_grad(p, dxdxi[1,0], gl[:,0,0], gl[:,2,0], l[:,0], l[:,2])
        gphi_reorder[:,e2,1] = self.get_edge_grad(p, dxdxi[1,1], gl[:,0,1], gl[:,2,1], l[:,0], l[:,2])
        gphi_reorder[:,e3,0] = self.get_edge_grad(p, dxdxi[2,0], gl[:,1,0], gl[:,0,0], l[:,1], l[:,0])
        gphi_reorder[:,e3,1] = self.get_edge_grad(p, dxdxi[2,1], gl[:,1,1], gl[:,0,1], l[:,1], l[:,0])

        internal = np.arange(3*p-3+3,nb,1)

        gphi_reorder[:,internal,0] = self.get_internal_grad(p, internal, gl[:,:,0], l)
        gphi_reorder[:,internal,1] = self.get_internal_grad(p, internal, gl[:,:,1], l)

        index = mesh_gmsh.gmsh_node_order_tri(p)

        gphi[:,:,:] = gphi_reorder[:,index,:]

        return gphi


    def get_edge_grad(self, p, dxdxi, gl, gr, ll, lr):

        gphi_e = np.zeros([ll.shape[0],p-1])
        for k in range(p-1):
            gkernel = self.get_kernel_grad(k, dxdxi, ll-lr)
            kernel = self.get_kernel_function(k,ll-lr)
            gphi_e[:,k] = (ll*gr+lr*gl)*kernel + ll*lr*gkernel

        return gphi_e

    def get_kernel_grad(self, p, dxdxi, x):

        p+=2
        leg_poly = np.polynomial.legendre.Legendre
        x.shape = -1

        # First two lobatto shape functions 
        l0 =  (1.-x)/2.
        l1 =  (1.+x)/2.
        dl0 = -0.5
        dl1 = 0.5

        leg_int = leg_poly.basis(p-1).integ(m=1,lbnd=-1)
        lk = np.sqrt((2.*p-1.)/2.)*leg_int(x)

        leg = leg_poly.basis(p-1)
        dl = np.sqrt((2.*p-1.)/2.)*dxdxi*leg(x)

        num = l0*l1*dl - lk*(l1*dl0*dxdxi+l0*dl1*dxdxi)
        den = (l0*l1)**2

        kernel = num / (1.e-12+den)

        return kernel


    def get_internal_grad(self, p, index, gl,l):

        gphi_i = np.zeros([l.shape[0],len(index)])

        c=0
        for i in range(3,p+1):
            c += i-2
        
        n = np.zeros([c,2])
        n1 = np.arange(1,p-1,1)
        n2 = np.arange(1,p-1,1)
        k = 0
        for i in range(len(n1)):
            for j in range(len(n2)):
                if n1[i] + n2[j] <= p-1:
                    n[k,0] = n1[i]
                    n[k,1] = n2[j]
                    k+=1

        for m in range(c):
            dl2l3_1 = n[m,0]*l[:,1]**(n[m,0]-1)*l[:,2]**n[m,1]*gl[:,1]
            dl2l3_2 = n[m,1]*l[:,2]**(n[m,1]-1)*l[:,1]**n[m,0]*gl[:,2]
            gphi_i[:,m] = gl[:,0]*l[:,1]**n[m,0]*l[:,2]**n[m,1]+l[:,0]*(dl2l3_1+dl2l3_2)

        return gphi_i



