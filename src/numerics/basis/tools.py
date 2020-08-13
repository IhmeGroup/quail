# ------------------------------------------------------------------------ #
#
#       File : src/numerics/basis/tools.py
#
#       Contains helper definitions for the shape and basis classes.
#
#       Authors: Eric Ching and Brett Bornhoft
#
#       Created: January, 2020
#      
# ------------------------------------------------------------------------ #
import code
import numpy as np

from data import ArrayList, GenericData
from general import SetSolverParams, BasisType, ShapeType, NodeType

import meshing.gmsh as mesh_gmsh

import numerics.basis.basis as basis_defs

import numerics.quadrature.segment as segment

# mapping for ShapeType enum to appropriate BasisType enums
Basis2Shape = {
    BasisType.LagrangeSeg : ShapeType.Segment,
    BasisType.LagrangeQuad : ShapeType.Quadrilateral,
    BasisType.LagrangeTri : ShapeType.Triangle,
    BasisType.LegendreSeg : ShapeType.Segment,
    BasisType.LegendreQuad : ShapeType.Quadrilateral,
    BasisType.HierarchicH1Tri : ShapeType.Triangle
}


def set_basis(order, basis_name):
    '''
    Sets the basis class given the basis_name string argument

    INPUTS: 
        order: solution order
        basis_name: name of the basis function we wish to instantiate 
            as a class

    OUTPUTS:
        basis: instantiated basis class

    RAISE:
        If the basis class is not defined returns a NotImplementedError
    '''
    if BasisType[basis_name] == BasisType.LagrangeSeg:
        basis = basis_defs.LagrangeSeg(order)
    elif BasisType[basis_name] == BasisType.LegendreSeg:
        basis = basis_defs.LegendreSeg(order)
    elif BasisType[basis_name] == BasisType.LagrangeQuad:
        basis = basis_defs.LagrangeQuad(order)
    elif BasisType[basis_name] == BasisType.LegendreQuad:
        basis = basis_defs.LegendreQuad(order)
    elif BasisType[basis_name] == BasisType.LagrangeTri:
        basis = basis_defs.LagrangeTri(order)
    elif BasisType[basis_name] == BasisType.HierarchicH1Tri:
        basis = basis_defs.HierarchicH1Tri(order)
    else:
        raise NotImplementedError
    return basis


def set_1D_node_calc(node_type):
    '''
    Sets the get_1d_nodes attribute from BasisBase in basis.py

    INPUTS: 
        node_type: name of the node type (available NodeType listed in 
            src/general.py)

    OUTPUTS:
        fcn: method to calculate 1D nodes
    '''
    if NodeType[node_type] == NodeType.Equidistant:
        fcn = equidistant_nodes_1D_range
    elif NodeType[node_type] == NodeType.GaussLobatto:
        fcn = gauss_lobatto_nodes_1D_range
    else:
        raise NotImplementedError

    return fcn


def equidistant_nodes_1D_range(start, stop, nnode):
    '''
    This function calculates the 1D coordinates in reference space 
    equidistantly

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
    dx = (stop-start) / float(nnode-1)
    for i in range(nnode): xnode[i] = start + float(i)*dx

    return xnode # [nnode, 1]


def gauss_lobatto_nodes_1D_range(start, stop, nnode):
    '''
    This function calculates the 1D coordinates in reference space 
    using Gauss Lobatto nodes

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
        
    order = 2*nnode - 3
    xnode,_ = segment.gauss_lobatto(order) 

    return xnode # [nnode, 1]


def get_inv_mass_matrices(mesh, physics, basis):
    '''
    Calculate the inverse mass matrices

    INPUTS:
        mesh: mesh object
        physics: type of physics set (i.e. scalar, euler, etc...)
        basis: instantiation of the basis

    OUTPUTS: 
        iMM_all: all inverse mass matrices
    '''
    order = physics.order
    nb = basis.nb

    iMM_all = np.zeros([mesh.num_elems, nb, nb])

    # TODO: Set logic to recalculate only if using non-uniform mesh
    recalc_mm = True

    for elem in range(mesh.num_elems):
        if elem == 0 or recalc_mm:
            # calculate the inv mass matrix in physical space
            iMM = get_elem_inv_mass_matrix(mesh, basis, order, elem, True)
        iMM_all[elem] = iMM

    return iMM_all # [mesh.num_elems, nb, nb]


def get_elem_inv_mass_matrix(mesh, basis, order, elem=-1, 
        PhysicalSpace=False):
    '''
    Calculate the inverse mass matrix for a given element

    INPUTS:
        mesh: mesh object
        basis: basis function object
        order: solution order
        elem: [OPTIONAL] element index
        PhysicalSpace: [OPTIONAL] Flag to calc matrix in physical or 
            reference space (default: False {reference space})

    OUTPUTS: 
        iMM: inverse mass matrix [nb, nb]
    '''
    MM = get_elem_mass_matrix(mesh, basis, order, elem, PhysicalSpace)
    
    iMM = np.linalg.inv(MM) 

    return iMM # [nb, nb]


def get_elem_mass_matrix(mesh, basis, order, elem=-1, PhysicalSpace=False):
    '''
    Calculate the mass matrix for a given element

    INPUTS:
        mesh: mesh object
        basis: basis function object
        order: solution order [int]
        elem: [OPTIONAL] element index [int]
        PhysicalSpace: [OPTIONAL] Flag to calc matrix in physical or 
            reference space (default: False {reference space}) 

    OUTPUTS: 
        MM: mass matrix [nb, nb]
    '''
    gbasis = mesh.gbasis

    if PhysicalSpace:
        quad_order = gbasis.get_quadrature_order(mesh, order*2)
    else:
        quad_order = order*2

    quad_pts, quad_wts = basis.get_quadrature_data(quad_order)

    nq = quad_pts.shape[0]

    basis.get_basis_val_grads(quad_pts, get_val=True)

    if PhysicalSpace:
        djac,_,_ = element_jacobian(mesh,elem,quad_pts,get_djac=True)

        if len(djac) == 1:
            djac = np.full(nq, djac[0])
    else:
        djac = np.full(nq, 1.)

    nb = basis.get_num_basis_coeff(order)
    phi = basis.basis_val

    MM = np.matmul(phi.transpose(), phi*quad_wts*djac) # [nb, nb]

    return MM # [nb, nb]


def get_stiffness_matrix(solver, mesh, order, elem):
    '''
    Calculate the stiffness_matrix

    INPUTS:
        solver: instance of solver object
        mesh: instance of mesh object
        order: solution order
        elem: element index

    OUTPUTS: 
        SM: stiffness matrix # [nb, nb]
    '''
    quad_pts = solver.elem_operators.quad_pts
    quad_wts = solver.elem_operators.quad_wts
    djac = solver.elem_operators.djac_elems[elem]
    phi = solver.elem_operators.basis_val
    gPhi = solver.elem_operators.basis_phys_grad_elems[elem]

    nq = quad_pts.shape[0]
    nb = phi.shape[1]

    # ------------------------------------------------------------------- #
    # Example of Stiffness Matrix calculation using for-loops
    # ------------------------------------------------------------------- #
    #
    # SM = np.zeros([nb, nb])
    # for i in range(nb):
    #     for j in range(nb):
    #         t = 0.
    #         for iq in range(nq):
    #             t += gPhi[iq,i,0]*phi[iq,j]*quad_wts[iq]* \
    #                 djac[iq]
    #         SM[i,j] = t
    #
    # ------------------------------------------------------------------- #
    SM = np.matmul(gPhi[:,:,0].transpose(), phi*quad_wts*djac) # [nb, nb]

    return SM # [nb, nb]


def get_projection_matrix(mesh, basis, basis_old, order, order_old, iMM):
    '''
    Calculate the projection matrix to increase order

    INPUTS:
        mesh: mesh object
        basis: basis function object
        basis_old: basis function from previous order
        order: solution order
        order_old: previous solution order
        iMM: inverse mass matrix [nb, nb]

    OUTPUTS: 
        PM: projection matrix # [nb, nb_old]
    '''
    quad_order = np.amax([order_old+order, 2*order])
    quad_pts, quad_wts = basis.get_quadrature_data(quad_order)

    nq = quad_pts.shape[0]

    basis_old.get_basis_val_grads(quad_pts, get_val=True)

    phi_old = basis_old.basis_val
    nb_old = phi_old.shape[1]

    basis.get_basis_val_grads(quad_pts, get_val=True)
    phi = basis.basis_val
    nb = phi.shape[1]

    A = np.zeros([nb,nb_old])

    # ------------------------------------------------------------------- #
    # Example of Projection Matrix calculation using for-loops
    # ------------------------------------------------------------------- #
    #
    # for i in range(nb):
    #     for j in range(nb_old):
    #         t = 0.
    #         for iq in range(nq):
    #             t += phi[iq,i]*phi_old[iq,j]*quad_wts[iq]
    #         A[i,j] = t
    #
    # ------------------------------------------------------------------- #

    A = np.matmul(phi.transpose(), phi_old*quad_wts) # [nb, nb_old]

    PM = np.matmul(iMM,A) # [nb, nb_old]

    return PM # [nb, nb_old]


def element_jacobian(mesh, elem, quad_pts, get_djac=False, get_jac=False, get_ijac=False):
    '''
    Evaluate the geometric jacobian for a specified element

    INPUTS:
        mesh: mesh object 
        elem: element index
        quad_pts: coordinates of quadrature points
        get_djac: [OPTIONAL]flag to calculate jacobian determinant 
            (Default: False)
        get_jac: [OPTIONAL] flag to calculate jacobian (Default: False)
        get_ijac: [OPTIONAL] flag to calculate inverse of the jacobian 
            (Default: False)

    OUTPUTS:
        djac: determinant of the jacobian [nq, 1]
        jac: jacobian [nq, dim, dim]
        ijac: inverse jacobian [nq, dim, dim]
    '''
    basis = mesh.gbasis
    order = mesh.gorder
    shape = basis.__class__.__bases__[1].__name__
    nb = basis.nb
    dim = basis.DIM

    nq = quad_pts.shape[0]

    if dim != dim: Resize = True
    else: Resize = False

    basis_phys_grad = basis.get_grads(quad_pts) # [nq, nb, dim]
    
    if dim != mesh.dim:
        raise Exception("Dimensions don't match")

    jac = np.zeros([nq,dim,dim])
    djac = np.zeros([nq,1])
    ijac = np.zeros([nq,dim,dim])

    A = np.zeros([dim,dim])

    elem_coords = mesh.elements[elem].node_coords

    jac = np.tensordot(basis_phys_grad, elem_coords.transpose(), \
        axes=[[1],[1]]).transpose((0,2,1))

    for i in range(nq):
        MatDetInv(jac[i], dim, djac[i], ijac[i])
 
    if get_djac and np.any(djac[i] <= 0.):
        raise Exception("Nonpositive Jacobian (elem = %d)" % (elem))

    return djac, jac, ijac # [nq, 1], [nq, dim, dim], and [nq, dim, dim]


def calculate_1D_normals(mesh, elem, face, quad_pts):

    '''
    Calculate the normals for a 1D face

    INPUTS:
        mesh: mesh object
        elem: element index
        face: face index
        quad_pts: points in reference space at which to calculate normals

    OUTPUTS:
        nvec: normal vector [nq, dim]
    '''
    gorder = mesh.gorder
    nq = quad_pts.shape[0]

    if gorder == 1:
        nq = 1

    nvec = np.zeros([nq,mesh.dim])
    
    #1D normals calculation
    if face == 0:
        nvec[0] = -1.
    elif face == 1:
        nvec[0] = 1.
    else:
        raise ValueError

    return nvec # [nq, dim]

    
def calculate_2D_normals(mesh, elem, face, quad_pts):
    '''
    Calculate the normals for 2D shapes

    INPUTS:
        mesh: mesh object
        elem: element index 
        face: face index
        quad_pts: points in reference space at which to calculate normals
    '''
    gbasis = mesh.gbasis
    gorder = mesh.gorder

    # Calculate 2D normals
    ElemNodes = mesh.elem_to_node_ids[elem]
    elem_coords = mesh.elements[elem].node_coords

    fnodes = gbasis.get_local_face_node_nums(gorder, face)

    basis_seg = basis_defs.LagrangeSeg(gorder)
    basis_ref_grad = basis_seg.get_grads(quad_pts)
    face_coords = elem_coords[fnodes]

    x_s = np.matmul(face_coords.transpose(), basis_ref_grad)[:, :, 0]
    nvec = x_s[:,::-1]
    nvec[:,1] *= -1.

    return nvec # [nq, dim]


def get_lagrange_basis_1D(x, xnodes, phi=None, gphi=None):
    '''
    Calculates the 1D Lagrange basis functions

    INPUTS:
        x: coordinate of current node [nq, 1]
        xnodes: coordinates of nodes in 1D ref space [nb, 1] 
        
    OUTPUTS: 
        phi: evaluated basis [nq, nb]
        gphi: evaluated physical gradient of basis [nq, nb, dim]
    '''
    nnodes = xnodes.shape[0]
    mask = np.ones(nnodes, bool)

    if phi is not None:
        phi[:] = 1.
        for j in range(nnodes):
            mask[j] = False
            phi[:,j] = np.prod((x - xnodes[mask])/(xnodes[j] - xnodes[mask]),
                axis=1)
            mask[j] = True

    if gphi is not None:
        gphi[:] = 0.

        for j in range(nnodes):
            mask[j] = False
            for i in range(nnodes):
                if i == j:
                    continue

                mask[i] = False
                if nnodes > 2: 
                    gphi[:,j,:] += np.prod((x - xnodes[mask])/(xnodes[j] - 
                        xnodes[mask]), axis=1).reshape(-1,1)/(xnodes[j] -
                        xnodes[i])
                else:
                    gphi[:,j,:] += 1./(xnodes[j] - xnodes[i])

                mask[i] = True
            mask[j] = True


def get_lagrange_basis_2D(x, xnodes, phi=None, gphi=None):
    '''
    Calculates the 2D Lagrange basis functions

    INPUTS:
        x: coordinate of current node [nq, dim]
        xnodes: coordinates of nodes in 1D ref space [nb, dim]
        
    OUTPUTS: 
        phi: evaluated basis [nq, nb]
        gphi: evaluated gradient of basis [nq, nb, dim]
    '''
    if gphi is not None:
        gphix = np.zeros((x.shape[0], xnodes.shape[0], 1))
        gphiy = np.zeros_like(gphix)
    else:
        gphix = None; gphiy = None
    # Always need phi
    phix = np.zeros((x.shape[0], xnodes.shape[0]))
    phiy = np.zeros_like(phix)

    nnodes_1D = xnodes.shape[0]
    lagrange_eq_seg = basis_defs.LagrangeSeg(nnodes_1D-1)
    get_lagrange_basis_1D(x[:, 0].reshape(-1, 1), xnodes, phix, gphix)
    get_lagrange_basis_1D(x[:, 1].reshape(-1, 1), xnodes, phiy, gphiy)

    if phi is not None:
        for i in range(x.shape[0]):
            phi[i, :] = np.reshape(np.outer(phix[i, :], \
                phiy[i, :]), (-1, ), 'F')
    if gphi is not None:
        for i in range(x.shape[0]):
            gphi[i, :, 0] = np.reshape(np.outer(gphix[i, :, 0], \
                phiy[i, :]), (-1, ), 'F')
            gphi[i, :, 1] = np.reshape(np.outer(phix[i, :], \
                gphiy[i, :, 0]), (-1, ), 'F')


def get_lagrange_basis_tri(x, p, xn, phi):
    '''
    Calculates the value for Lagrange triangle basis function

    INPUTS:
        x: coordinate of quadrature points [nq, dim]
        p: polynomial solution order
        xn: coordinates of nodes in 1D equidistant ref space 
            [nb, dim]
        
    OUTPUTS: 
        phi: evaluated basis [nq, nb]
    '''
    nb = xn.shape[0]

    alpha = np.round(p*xn)
    alpha = np.c_[(p*np.ones(nb) - np.sum(alpha, axis=1), alpha)]
    l = np.c_[(np.ones(x.shape[0]) - np.sum(x, axis=1)), x]

    if p == 0:
        phi[:] = 1.
        return 

    for i in range(nb):
        phi[:,i] = get_tri_area_coordinates(p, alpha[i], l)

    return phi # [nq, nb]


def get_tri_area_coordinates(p, alpha, l):
    '''
    Helper function for Lagrange triangular basis
    '''
    N = np.ones(l.shape[0])

    N *= get_eta_function(p, alpha[0], l[:,0])
    N *= get_eta_function(p, alpha[1], l[:,1])
    N *= get_eta_function(p, alpha[2], l[:,2])

    return N


def get_eta_function(p, alpha, l, skip = -1):
    '''
    Helper function for Lagrange triangular basis
    '''
    index = np.concatenate((np.arange(0, skip), np.arange(skip + 1, alpha)))

    eta = np.ones(l.shape[0])

    for i in index:
        eta *= (p * l - i) / (i + 1.)
    return eta


def get_grad_eta_function(p, alpha, l):
    '''
    Helper function for Lagrange triangular basis
    '''
    get_a = np.zeros_like(l)
    for i in range(int(alpha)):
        get_a += ( p / (i + 1)) * get_eta_function(p, alpha, l, i)

    return get_a


def get_lagrange_grad_tri(x, p, xn, gphi):
    '''
    Calculates the gradient of the triangular basis functions 

    INPUTS:
        x: coordinate of quadrature points [nq, dim]
        p: polynomial solution order
        xn: coordinates of nodes in 1D equidistant ref space 
            [nb, dim]
        
    OUTPUTS: 
        gphi: evaluated gradient of basis function [nq, nb, dim]
    '''
    nb = xn.shape[0]
    gphi_dir = np.zeros((x.shape[0], nb, 3))

    alpha = np.round(p*xn)
    alpha = np.c_[(p*np.ones(nb) - np.sum(alpha, axis=1), alpha)]
    l = np.c_[(np.ones(x.shape[0]) - np.sum(x, axis=1)), x]

    if p == 0:
        gphi[:] = 0.
        return 
    for i in range(nb):
        gphi_dir[:,i,:] = get_tri_grad_area_coordinates(p, alpha[i], l)

    gphi[:,:,0] = gphi_dir[:,:,1] - gphi_dir[:,:,0]
    gphi[:,:,1] = gphi_dir[:,:,2] - gphi_dir[:,:,0]

    return gphi # [nq, nb, dim]

def get_tri_grad_area_coordinates(p, alpha, l):
    '''
    Helper function for Lagrange triangular basis
    '''
    dN = np.ones((l.shape[0], 3))

    N1 = get_eta_function(p, alpha[0], l[:,0])
    N2 = get_eta_function(p, alpha[1], l[:,1])
    N3 = get_eta_function(p, alpha[2], l[:,2])

    dN1 = get_grad_eta_function(p, alpha[0], l[:,0])
    dN2 = get_grad_eta_function(p, alpha[1], l[:,1])
    dN3 = get_grad_eta_function(p, alpha[2], l[:,2])

    dN[:,0] = dN1 * N2 * N3
    dN[:,1] = N1 * dN2 * N3
    dN[:,2] = N1 * N2 * dN3

    return dN


def get_legendre_basis_1D(x, p, phi=None, gphi=None):
    '''
    Calculates the 1D Legendre basis functions

    INPUTS:
        x: coordinate of current node [nq, dim]
        p: order of polynomial space
        
    OUTPUTS: 
        phi: evaluated basis [nq, nb]
        gphi: evaluated physical gradient of basis [nq, nb, dim]
    '''

    # use numpy legendre polynomials
    leg_poly = np.polynomial.legendre.Legendre

    if phi is not None:
        phi[:, :] = 0.            
        x.shape = -1
        
        for it in range(p+1):
            phi[:, it] = leg_poly.basis(it)(x)
        x.shape = -1, 1

    if gphi is not None:
        gphi[:,:] = 0.

        for it in range(p+1):
            dleg = leg_poly.basis(it).deriv(1)
            gphi[:,it] = dleg(x)


def get_legendre_basis_2D(x, p, phi=None, gphi=None):
    '''
    Calculates the 2D Legendre basis functions

    INPUTS:
        x: coordinate of current node [nq, dim]
        p: order of polynomial space
        
    OUTPUTS: 
        phi: evaluated basis [nq, nb]
        gphi: evaluated physical gradient of basis [nq, nb, dim]
    '''
    nq = x.shape[0]
    if gphi is not None:
        gphix = np.zeros((nq, p+1, 1)); gphiy = np.zeros_like(gphix)
    else:
        gphix = None; gphiy = None
    # Always need phi
    phix = np.zeros((nq, p+1)); phiy = np.zeros_like(phix)

    legendre_seg = basis_defs.LegendreSeg(p)
    get_legendre_basis_1D(x[:, 0], p, phix, gphix)
    get_legendre_basis_1D(x[:, 1], p, phiy, gphiy)

    if phi is not None:
        for i in range(nq):
            phi[i, :] = np.reshape(np.outer(phix[i, :], \
                phiy[i, :]), (-1, ), 'F')
    if gphi is not None:
        for i in range(nq):
            gphi[i, :, 0] = np.reshape(np.outer(gphix[i, :, 0], \
                phiy[i, :]), (-1, ), 'F')
            gphi[i, :, 1] = np.reshape(np.outer(phix[i, :], \
                gphiy[i, :, 0]), (-1, ), 'F')


def get_modal_basis_tri(xi, p, xn, phi):
    '''
    Calculates the value for Hierarchical triangle basis function

    INPUTS:
        xi: coordinate of quadrature points [nq, dim]
        p: polynomial solution order
        xn: coordinates of nodes in 1D equidistant ref space 
            [nb, dim]
        
    OUTPUTS: 
        phi: evaluated basis [nq, nb]
    '''
    nb = xn.shape[0]

    phi_reorder = np.zeros_like(phi)

    # Transform to the modal basis reference element
    # [-1,-1],[1,-1],[-1,1]
    xn = 2.*xn - 1.
    xi = 2.*xi - 1.

    # Define the affine coordinates
    l = np.zeros([xi.shape[0], 3])

    l[:,0] = (xi[:,1] + 1.) / 2.
    l[:,1] = -1.*((xi[:,1] + xi[:,0]) / 2.)
    l[:,2] = (xi[:,0] + 1.) / 2.

    if p == 0:
        phi[:] = 1.
        return 

    phi_reorder[:,[0, 1, 2]] = l[:,[1, 2, 0]]

    e1 = np.arange(3, p-1+3, 1)
    e2 = np.arange(p-1+3, 2*p-2+3, 1)
    e3 = np.arange(2*p-2+3, 3*p-3+3, 1)

    phi_reorder[:,e1] = get_edge_basis(p, l[:,2], l[:,1])
    phi_reorder[:,e2] = get_edge_basis(p, l[:,0], l[:,2])
    phi_reorder[:,e3] = get_edge_basis(p, l[:,1], l[:,0])

    internal = np.arange(3*p-3+3, nb, 1)

    phi_reorder[:,internal] = get_internal_basis(p, internal, l)

    index = mesh_gmsh.gmsh_node_order_tri(p)

    phi[:,:] = phi_reorder[:,index]

    return phi # [nb, nb]


def get_edge_basis(p, ll, lr):
    '''
    Helper function for Hierarchical triangular basis
    Note: Eq. 2.21 from Sorin et al.
    '''
    phi_e = np.zeros([ll.shape[0], p-1])
    for k in range(p-1):
        kernel = get_kernel_function(k, ll-lr)
        phi_e[:,k] = ll*lr*kernel

    return phi_e

def get_internal_basis(p, index, l):
    '''
    Helper function for Hierarchical triangular basis
    Note: Eq. 2.22 from Sorin et al.
    '''
    phi_i = np.zeros([l.shape[0], len(index)])

    c = 0
    for i in range(3,p+1):
        c += i-2
    
    n = np.zeros([c, 2])
    n1 = np.arange(1, p-1, 1)
    n2 = np.arange(1, p-1, 1)
    k = 0
    for i in range(len(n1)):
        for j in range(len(n2)):
            if n1[i] + n2[j] <= p-1:
                n[k,0] = n1[i]
                n[k,1] = n2[j]
                k += 1

    for m in range(c):
        phi_i[:,m] = l[:,0]*l[:,1]**n[m,0]*l[:,2]**n[m,1]

    return phi_i

def get_kernel_function(p, x):
    '''
    Helper function for Hierarchical triangular basis
    Note: pp. 27 of Sorin et al.
    '''
    p+=2
    # Initialize the legendre polynomial object
    leg_poly = np.polynomial.legendre.Legendre
    x.shape = -1

    # Construct the kernel's denominator 
    # (series of Lobatto fnc's)                    
    
    # First two lobatto shape functions 
    l0 =  (1.-x)/2.
    l1 =  (1.+x)/2.

    den = l0*l1

    leg_int = leg_poly.basis(p-1).integ(m=1,lbnd=-1)
    num = np.sqrt((2.*p-1.)/2.)*leg_int(x)

    kernel = num / (1e-12 + den)

    x.shape = -1,1

    return kernel


def get_modal_grad_tri(xi, p, xn, gphi):
    '''
    Calculates the gradient of the triangular basis functions 

    INPUTS:
        xi: coordinate of quadrature points [nq, dim]
        p: polynomial solution order
        xn: coordinates of nodes in 1D equidistant ref space 
            [nb, dim]
        
    OUTPUTS: 
        gphi: evaluated gradient of basis function [nq, nb, dim]
    '''
    nb = xn.shape[0]

    gphi_reorder = np.zeros_like(gphi)

    xn = 2.*xn - 1.
    xi = 2.*xi - 1.

    gl = np.zeros([xi.shape[0],3,2])
    l = np.zeros([xi.shape[0],3])

    # Calculate the affine coordinates
    l[:,0] = (xi[:,1] + 1.) / 2.
    l[:,1] = -1.*((xi[:,1] + xi[:,0]) / 2.)
    l[:,2] = (xi[:,0] + 1.) / 2.

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
    e1 = np.arange(3, p-1+3, 1)
    e2 = np.arange(p-1+3, 2*p-2+3, 1)
    e3 = np.arange(2*p-2+3, 3*p-3+3, 1)

    dxdxi = np.zeros([3,2])
    dxdxi[0,0] = 1.   ; dxdxi[0,1] = 0.5
    dxdxi[1,0] = -0.5 ; dxdxi[1,1] = 0.5
    dxdxi[2,0] = -0.5 ; dxdxi[2,1] = -1.

    gphi_reorder[:,e1,0] = get_edge_grad(p, dxdxi[0,0], gl[:,2,0], \
        gl[:,1,0], l[:,2], l[:,1])
    gphi_reorder[:,e1,1] = get_edge_grad(p, dxdxi[0,1], gl[:,2,1], \
        gl[:,1,1], l[:,2], l[:,1])
    gphi_reorder[:,e2,0] = get_edge_grad(p, dxdxi[1,0], gl[:,0,0], \
        gl[:,2,0], l[:,0], l[:,2])
    gphi_reorder[:,e2,1] = get_edge_grad(p, dxdxi[1,1], gl[:,0,1], \
        gl[:,2,1], l[:,0], l[:,2])
    gphi_reorder[:,e3,0] = get_edge_grad(p, dxdxi[2,0], gl[:,1,0], \
        gl[:,0,0], l[:,1], l[:,0])
    gphi_reorder[:,e3,1] = get_edge_grad(p, dxdxi[2,1], gl[:,1,1], \
        gl[:,0,1], l[:,1], l[:,0])

    internal = np.arange(3*p-3+3,nb,1)

    gphi_reorder[:,internal,0] = get_internal_grad(p, internal, gl[:,:,0], l)
    gphi_reorder[:,internal,1] = get_internal_grad(p, internal, gl[:,:,1], l)

    index = mesh_gmsh.gmsh_node_order_tri(p)

    gphi[:,:,:] = gphi_reorder[:,index,:]

    return gphi # [nq, nb, dim]


def get_edge_grad(p, dxdxi, gl, gr, ll, lr):
    '''
    Helper function for Hierarchical triangular basis
    '''
    gphi_e = np.zeros([ll.shape[0],p-1])
    for k in range(p-1):
        gkernel = get_kernel_grad(k, dxdxi, ll-lr)
        kernel = get_kernel_function(k,ll-lr)
        gphi_e[:,k] = (ll*gr+lr*gl)*kernel + ll*lr*gkernel

    return gphi_e


def get_kernel_grad(p, dxdxi, x):
    '''
    Helper function for Hierarchical triangular basis
    '''
    p += 2
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


def get_internal_grad(p, index, gl,l):
    '''
    Helper function for Hierarchical triangular basis
    '''
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
        gphi_i[:,m] = gl[:,0]*l[:,1]**n[m,0]*l[:,2]**n[m,1]+l[:,0] \
            * (dl2l3_1+dl2l3_2)

    return gphi_i


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