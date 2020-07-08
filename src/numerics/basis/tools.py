import code
import numpy as np

from data import ArrayList, GenericData
from general import SetSolverParams, BasisType, ShapeType, NodeType

import meshing.gmsh as mesh_gmsh


import numerics.basis.basis as basis_defs

import numerics.quadrature.segment as segment

Basis2Shape = {
    BasisType.LagrangeEqSeg : ShapeType.Segment,
    BasisType.LagrangeEqQuad : ShapeType.Quadrilateral,
    BasisType.LagrangeEqTri : ShapeType.Triangle,
    BasisType.LegendreSeg : ShapeType.Segment,
    BasisType.LegendreQuad : ShapeType.Quadrilateral,
    BasisType.HierarchicH1Tri : ShapeType.Triangle
}


def set_basis(order, basis_name):

    if BasisType[basis_name] == BasisType.LagrangeEqSeg:
        basis = basis_defs.LagrangeEqSeg(order)
    elif BasisType[basis_name] == BasisType.LegendreSeg:
        basis = basis_defs.LegendreSeg(order)
    elif BasisType[basis_name] == BasisType.LagrangeEqQuad:
        basis = basis_defs.LagrangeEqQuad(order)
    elif BasisType[basis_name] == BasisType.LegendreQuad:
        basis = basis_defs.LegendreQuad(order)
    elif BasisType[basis_name] == BasisType.LagrangeEqTri:
        basis = basis_defs.LagrangeEqTri(order)
    elif BasisType[basis_name] == BasisType.HierarchicH1Tri:
        basis = basis_defs.HierarchicH1Tri(order)
    else:
        raise NotImplementedError
    return basis

def set_node_type(node_type):
    if NodeType[node_type] == NodeType.GaussLegendre:
        fcn = equidistant_nodes_1D_range
    elif NodeType[node_type] == NodeType.GaussLobatto:
        fcn = gauss_lobatto_nodes_1D_range
    else:
        raise NotImplementedError

    return fcn


def equidistant_nodes_1D_range(start, stop, nnode):
    '''
    This function calculates the 1D coordinates in reference space.

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

def gauss_lobatto_nodes_1D_range(start, stop, nnode):

    if nnode <= 1: 
        raise ValueError
    if stop <= start:
        raise ValueError
    
    xnode,_ = segment.gauss_lobatto(nnode) 

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
        PhysicalSpace: Flag to calc matrix in physical or reference 
                       space (default: False {reference space}) 
        elem: element index

    OUTPUTS: 
        MM: mass matrix  
    '''
    gbasis = mesh.gbasis
    if PhysicalSpace:
        quad_order = gbasis.get_quadrature(mesh, order*2)
    else:
        quad_order = order*2

    quad_pts, quad_wts = basis.get_quad_data(quad_order)
    # quad_pts = basis.quad_pts
    # quad_wts = basis.quad_wts

    # quad_pts = quadData.quad_pts
    # quad_wts = quadData.quad_wts
    nq = quad_pts.shape[0]

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

def get_elem_inv_mass_matrix(mesh, basis, order, elem=-1, 
        PhysicalSpace=False):
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
    # QuadOrder,QuadChanged = quadrature.get_gaussian_quadrature_elem(mesh, mesh.QBasis, order*2)
    qbasis = mesh.QBasis 
    quad_order = qbasis.get_quadrature(mesh,order*2)
    quad_pts, quad_wts = qbasis.get_quad_data(quad_order)

    # quad_pts = qbasis.quad_pts
    # quad_wts = qbasis.quad_wts
    nq = quad_pts.shape[0]

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
                t += gPhi[iq,i,0]*phi[iq,j]*wq[iq]* \
                    JData.djac[iq*(JData.nq != 1)]
            SM[i,j] = t

    return SM


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
    quad_order = np.amax([order_old+order, 2*order])
    quad_pts, quad_wts = basis.get_quad_data(quad_order)

    # quad_pts = basis.quad_pts
    # quad_wts = basis.quad_wts
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



def get_inv_mass_matrices(mesh, EqnSet, basis):
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

    # if solver is not None:
    #     solver.DataSet.MMinv_all = iMM_all

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

    basis_pgrad = basis.get_grads(quad_pts) #, basis.basis_pgrad) # [nq, nb, dim]
    
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

def calculate_1D_normals(self, mesh, elem, face, quad_pts):
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
    
def calculate_2D_normals(basis, mesh, elem, face, quad_pts):
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
        fnodes, nfnode = basis.local_q1_face_nodes(gorder, face, fnodes=None)
        x0 = mesh.Coords[ElemNodes[fnodes[0]]]
        x1 = mesh.Coords[ElemNodes[fnodes[1]]]

        nvec[0,0] =  (x1[1]-x0[1])/2.;
        nvec[0,1] = -(x1[0]-x0[0])/2.;
    
    # Calculate normals for curved meshes
    else:
        x_s = np.zeros_like(nvec)
        fnodes, nfnode = basis.local_face_nodes(gorder, face, fnodes=None)

        basis_seg = basis_defs.LagrangeEqSeg(gorder)
        basis_grad = basis_seg.get_grads(quad_pts)
        Coords = mesh.Coords[ElemNodes[fnodes]]

        # Face Jacobian (gradient of (x,y) w.r.t reference coordinate)
        x_s[:] = np.matmul(Coords.transpose(), basis_grad).reshape(x_s.shape)
        nvec[:,0] = x_s[:,1]
        nvec[:,1] = -x_s[:,0]

    return nvec

def MatDetInv(A, d, detA, iA):
    if d == 1:
        det = A[0]
        if detA is not None: detA[0] = det;
        if iA is not None:
            if det == 0.:
                raise Exception("Singular matrix")
            iA[0] = 1./det
    elif d == 2:
        det = A[0,0]*A[1,1] - A[0,1]*A[1,0]
        if detA is not None: detA[0] = det;
        if iA is not None:
            if det == 0.:
                raise Exception("Singular matrix")
            iA[0,0] =  A[1,1]/det
            iA[0,1] = -A[0,1]/det
            iA[1,0] = -A[1,0]/det
            iA[1,1] =  A[0,0]/det;
    else:
        raise Exception("Can only deal with 2x2 matrices or smaller")