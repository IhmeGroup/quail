# ------------------------------------------------------------------------ #
#
#       File : src/numerics/basis/tools.py
#
#       Contains helper definitions for the shape and basis classes.
#      
# ------------------------------------------------------------------------ #
import numpy as np

from general import BasisType, ShapeType, NodeType

import meshing.gmsh as mesh_gmsh

import numerics.basis.basis as basis_defs
import numerics.quadrature.segment as segment


def set_basis(order, basis_name):
    '''
    Sets the basis class given the basis_name string argument

    Inputs:
    ------- 
        order: solution order
        basis_name: name of the basis function we wish to instantiate 
            as a class

    Outputs:
    --------
        basis: instantiated basis class

    Raise:
    ------
        If the basis class is not defined, returns a NotImplementedError
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

    Inputs:
    ------- 
        node_type: name of the node type (available NodeType listed in 
            src/general.py)

    Outputs:
    --------
        fcn: method to calculate 1D nodes
    '''
    if NodeType[node_type] == NodeType.Equidistant:
        fcn = equidistant_nodes_1D_range
    elif NodeType[node_type] == NodeType.GaussLobatto:
        fcn = gauss_lobatto_nodes_1D_range
    else:
        raise NotImplementedError

    return fcn


def equidistant_nodes_1D_range(start, stop, nnodes):
    '''
    This function gets the 1D equidistant coordinates in reference space 

    Inputs:
    -------
        start: start of ref space (typically -1)
        stop:  end of ref space (typically 1)
        nnodes: num of nodes in 1D ref space

    Outputs:
    -------- 
        xnodes: coordinates of nodes in 1D ref space
    '''
    if nnodes <= 1:
        raise ValueError
    if stop <= start:
        raise ValueError

    # Note: this is faster than linspace unless p is large
    xnodes = np.zeros(nnodes)
    dx = (stop-start) / float(nnodes-1)
    for i in range(nnodes): xnodes[i] = start + float(i)*dx

    return xnodes # [nnodes, 1]


def gauss_lobatto_nodes_1D_range(start, stop, nnodes):
    '''
    This function calculates the 1D coordinates in reference space 
    using Gauss Lobatto nodes

    Inputs:
    -------
        start: start of ref space (typically -1)
        stop:  end of ref space (typically 1)
        nnodes: num of nodes in 1D ref space

    Outputs:
    -------- 
        xnodes: coordinates of nodes in 1D ref space
    '''
    if nnodes <= 1: 
        raise ValueError
    if stop <= start:
        raise ValueError
        
    order = 2*nnodes - 3
    xnodes, _ = segment.gauss_lobatto(order) 

    return xnodes # [nnodes, 1]


def get_inv_mass_matrices(mesh, physics, basis):
    '''
    Calculate the inverse mass matrices for all elements

    Inputs:
    -------
        mesh: mesh object
        physics: physics object (e.g., scalar, euler, etc...)
        basis: instantiation of the basis

    Outputs:
    -------- 
        iMM_all: all inverse mass matrices
    '''
    order = physics.order
    nb = basis.nb

    iMM_all = np.zeros([mesh.num_elems, nb, nb])

    for elem_ID in range(mesh.num_elems):
        # Calculate the inv mass matrix in physical space
        iMM = get_elem_inv_mass_matrix(mesh, basis, order, elem_ID, True)
        # Store
        iMM_all[elem_ID] = iMM

    return iMM_all # [mesh.num_elems, nb, nb]


def get_elem_inv_mass_matrix(mesh, basis, order, elem_ID=-1, 
        physical_space=False):
    '''
    Calculate the inverse mass matrix for a given element

    Inputs:
    -------
        mesh: mesh object
        basis: basis function object
        order: solution order
        elem_ID: [OPTIONAL] element index
        physical_space: [OPTIONAL] Flag to calc matrix in physical or 
            reference space (default: False {reference space})

    Outputs:
    -------- 
        iMM: inverse mass matrix [nb, nb]
    '''
    MM = get_elem_mass_matrix(mesh, basis, order, elem_ID, physical_space)
    
    iMM = np.linalg.inv(MM) 

    return iMM # [nb, nb]


def get_elem_mass_matrix(mesh, basis, order, elem_ID=-1, 
        physical_space=False):
    '''
    Calculate the mass matrix for a given element

    Inputs:
    -------
        mesh: mesh object
        basis: basis function object
        order: solution order [int]
        elem_ID: [OPTIONAL] element index [int]
        physical_space: [OPTIONAL] Flag to calc matrix in physical or 
            reference space (default: False {reference space}) 

    Outputs:
    -------- 
        MM: mass matrix [nb, nb]
    '''
    gbasis = mesh.gbasis

    if physical_space:
        quad_order = gbasis.get_quadrature_order(mesh, order*2)
    else:
        quad_order = order*2

    quad_pts, quad_wts = basis.get_quadrature_data(quad_order)

    nq = quad_pts.shape[0]

    # Compute basis values
    basis.get_basis_val_grads(quad_pts, get_val=True)

    if physical_space:
        djac, _, _ = element_jacobian(mesh, elem_ID, quad_pts, get_djac=True)

        if len(djac) == 1:
            djac = np.full(nq, djac[0])
    else:
        djac = np.full(nq, 1.)

    basis_val = basis.basis_val

    MM = np.matmul(basis_val.transpose(), basis_val*quad_wts*djac) # [nb, nb]

    return MM # [nb, nb]


def get_stiffness_matrix(solver, mesh, order, elem_ID):
    '''
    Calculate the stiffness matrix

    Inputs:
    -------
        solver: instance of solver object
        mesh: instance of mesh object
        order: solution order
        elem_ID: element index

    Outputs:
    -------- 
        SM: stiffness matrix # [nb, nb]
    '''
    # Unpack
    quad_pts = solver.elem_operators.quad_pts
    quad_wts = solver.elem_operators.quad_wts
    djac = solver.elem_operators.djac_elems[elem_ID]
    basis_val = solver.elem_operators.basis_val
    basis_grad = solver.elem_operators.basis_phys_grad_elems[elem_ID]

    # ------------------------------------------------------------------- #
    # Example of Stiffness Matrix calculation using for-loops
    # ------------------------------------------------------------------- #
    #
    # nq = quad_pts.shape[0]
    # nb = basis_val.shape[1]
    #
    # SM = np.zeros([nb, nb])
    # for i in range(nb):
    #     for j in range(nb):
    #         a = 0.
    #         for iq in range(nq):
    #             a += basis_grad[iq, i, 0]*basis_val[iq, j]*quad_wts[iq]* \
    #                 djac[iq]
    #         SM[i,j] = a
    #
    # ------------------------------------------------------------------- #
    SM = np.matmul(basis_grad[:, :, 0].transpose(), basis_val*quad_wts*djac) 
        # [nb, nb]

    return SM # [nb, nb]


def element_jacobian(mesh, elem_ID, quad_pts, get_djac=False, get_jac=False, 
        get_ijac=False):
    '''
    Evaluate the geometric Jacobian for a specified element

    Inputs:
    -------
        mesh: mesh object 
        elem_ID: element index
        quad_pts: coordinates of quadrature points
        get_djac: [OPTIONAL] flag to calculate Jacobian determinant 
            (Default: False)
        get_jac: [OPTIONAL] flag to calculate Jacobian (Default: False)
        get_ijac: [OPTIONAL] flag to calculate inverse of the Jacobian 
            (Default: False)

    Outputs:
    --------
        djac: determinant of the Jacobian [nq, 1]
        jac: Jacobian [nq, dim, dim]
        ijac: inverse Jacobian [nq, dim, dim]
    '''
    gbasis = mesh.gbasis
    dim = gbasis.DIM

    nq = quad_pts.shape[0]

    # Gradients in reference space
    basis_ref_grad = gbasis.get_grads(quad_pts) # [nq, nb, dim]
    
    if dim != mesh.dim:
        raise Exception("Dimensions don't match")

    elem_coords = mesh.elements[elem_ID].node_coords

    # Compute Jacobian
    jac = np.tensordot(basis_ref_grad, elem_coords.transpose(),
            axes=[[1], [1]]).transpose((0, 2, 1))

    # Get inverse and determinant
    ijac = np.linalg.inv(jac)
    djac = np.linalg.det(jac).reshape(-1, 1)
 
    # Check for nonpositive Jacobian
    if get_djac and np.any(djac <= 0.):
        raise Exception("Nonpositive Jacobian (elem_ID = %d)" % (elem_ID))

    return djac, jac, ijac # [nq, 1], [nq, dim, dim], and [nq, dim, dim]


def calculate_1D_normals(mesh, elem_ID, face_ID, quad_pts):

    '''
    Calculate the normals for a 1D face

    Inputs:
    -------
        mesh: mesh object
        elem_ID: element index
        face_ID: face index
        quad_pts: points in reference space at which to calculate normals

    Outputs:
    --------
        normals: normal vector [nq, dim]
    '''
    nq = quad_pts.shape[0]

    # nq should be 1
    # if nq != 1:
    #     raise ValueError

    normals = np.zeros([nq, mesh.dim])
    
    # 1D normals calculation
    if face_ID == 0:
        normals[0] = -1.
    elif face_ID == 1:
        normals[0] = 1.
    else:
        raise ValueError

    return normals # [nq, dim]

    
def calculate_2D_normals(mesh, elem_ID, face_ID, quad_pts):
    '''
    Calculate the normals for 2D shapes (triangles and quadrilaterals).

    Inputs:
    -------
        mesh: mesh object
        elem_ID: element index 
        face_ID: face index
        quad_pts: points in reference space at which to calculate normals

    Outputs:
    --------
        normals: normal vector [nq, dim]
    '''
    gbasis = mesh.gbasis
    gorder = mesh.gorder
    elem_coords = mesh.elements[elem_ID].node_coords

    ''' Get face coordinates '''
    # Get local IDs of face nodes
    fnodes = gbasis.get_local_face_node_nums(gorder, face_ID)
    # Instantiate segment basis
    basis_seg = basis_defs.LagrangeSeg(gorder)
    # Compute basis values
    basis_ref_grad = basis_seg.get_grads(quad_pts)
    # Extract coordinates of face nodes
    face_coords = elem_coords[fnodes]

    ''' Calculate 2D normals '''
    xphys_grad = np.matmul(face_coords.transpose(), basis_ref_grad)[:, 
            :, 0] # gradient of physical space w.r.t ref space
    normals = xphys_grad[:, ::-1]
    normals[:, 1] *= -1.

    return normals # [nq, dim]


def get_lagrange_basis_1D(xq, xnodes, basis_val=None, basis_ref_grad=None):
    '''
    Calculates the 1D Lagrange basis functions

    Inputs:
    -------
        xq: coordinates of quadrature points [nq, 1]
        xnodes: coordinates of nodes in 1D ref space [nb, 1] 
        
    Outputs:
    -------- 
        basis_val: evaluated basis [nq, nb]
        basis_ref_grad: evaluated gradient of basis [nq, nb, dim]
    '''
    nnodes = xnodes.shape[0]
    mask = np.ones(nnodes, bool)

    if basis_val is not None:
        basis_val[:] = 1.
        for j in range(nnodes):
            mask[j] = False
            basis_val[:,j] = np.prod((xq - xnodes[mask])/(xnodes[j] - xnodes[
                    mask]), axis=1)
            mask[j] = True

    if basis_ref_grad is not None:
        basis_ref_grad[:] = 0.

        for j in range(nnodes):
            mask[j] = False
            for i in range(nnodes):
                if i == j:
                    continue

                mask[i] = False
                if nnodes > 2: 
                    basis_ref_grad[:,j,:] += np.prod((xq - xnodes[mask])/(
                            xnodes[j] - xnodes[mask]), axis=1).reshape(-1, 
                            1)/(xnodes[j] - xnodes[i])
                else:
                    basis_ref_grad[:,j,:] += 1./(xnodes[j] - xnodes[i])

                mask[i] = True
            mask[j] = True


def get_lagrange_basis_2D(xq, xnodes, basis_val=None, basis_ref_grad=None):
    '''
    Calculates the 2D Lagrange basis functions

    Inputs:
    -------
        xq: coordinates of quadrature points [nq, dim]
        xnodes: coordinates of nodes in 1D ref space [nb, dim]
        
    Outputs:
    -------- 
        basis_val: evaluated basis [nq, nb]
        basis_ref_grad: evaluated gradient of basis [nq, nb, dim]
    '''
    if basis_ref_grad is not None:
        gradx = np.zeros((xq.shape[0], xnodes.shape[0], 1))
        grady = np.zeros_like(gradx)
    else:
        gradx = None; grady = None
    # Always need basis_val
    valx = np.zeros((xq.shape[0], xnodes.shape[0]))
    valy = np.zeros_like(valx)

    # Get 1D basis values first
    nnodes_1D = xnodes.shape[0]
    lagrange_eq_seg = basis_defs.LagrangeSeg(nnodes_1D-1)
    get_lagrange_basis_1D(xq[:, 0].reshape(-1, 1), xnodes, valx, gradx)
    get_lagrange_basis_1D(xq[:, 1].reshape(-1, 1), xnodes, valy, grady)

    # Tensor products to get 2D basis values
    if basis_val is not None:
        for i in range(xq.shape[0]):
            basis_val[i, :] = np.reshape(np.outer(valx[i, :],
                    valy[i, :]), (-1, ), 'F')
    if basis_ref_grad is not None:
        for i in range(xq.shape[0]):
            basis_ref_grad[i, :, 0] = np.reshape(np.outer(gradx[i, :, 0],
                    valy[i, :]), (-1, ), 'F')
            basis_ref_grad[i, :, 1] = np.reshape(np.outer(valx[i, :],
                    grady[i, :, 0]), (-1, ), 'F')


def get_lagrange_basis_tri(xq, p, xnodes, basis_val):
    '''
    Calculates the value for Lagrange triangle basis function

    Inputs:
    -------
        xq: coordinate of quadrature points [nq, dim]
        p: polynomial solution order
        xnodes: coordinates of nodes in 1D equidistant ref space 
            [nb, dim]
        
    Outputs:
    -------- 
        basis_val: evaluated basis [nq, nb]
    '''
    nb = xnodes.shape[0]

    alpha = np.round(p*xnodes)
    alpha = np.c_[(p*np.ones(nb) - np.sum(alpha, axis=1), alpha)]
    l = np.c_[(np.ones(xq.shape[0]) - np.sum(xq, axis=1)), xq]

    if p == 0:
        basis_val[:] = 1.
        return 

    for i in range(nb):
        basis_val[:, i] = get_tri_area_coordinates(p, alpha[i], l)

    return basis_val # [nq, nb]


def get_tri_area_coordinates(p, alpha, l):
    '''
    Helper function for Lagrange triangular basis
    '''
    N = np.ones(l.shape[0])

    N *= get_eta_function(p, alpha[0], l[:,0])
    N *= get_eta_function(p, alpha[1], l[:,1])
    N *= get_eta_function(p, alpha[2], l[:,2])

    return N


def get_eta_function(p, alpha, l, skip=-1):
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
        get_a += (p / (i + 1)) * get_eta_function(p, alpha, l, i)

    return get_a


def get_lagrange_grad_tri(xq, p, xnodes, basis_ref_grad):
    '''
    Calculates the gradient of the Lagrange triangular basis functions 

    Inputs:
    -------
        xq: coordinate of quadrature points [nq, dim]
        p: polynomial solution order
        xnodes: coordinates of nodes in 1D equidistant ref space 
            [nb, dim]
        
    Outputs:
    -------- 
        basis_ref_grad: evaluated gradient of basis function [nq, nb, dim]
    '''
    nb = xnodes.shape[0]
    grad_dir = np.zeros((xq.shape[0], nb, 3))

    alpha = np.round(p*xnodes)
    alpha = np.c_[(p*np.ones(nb) - np.sum(alpha, axis=1), alpha)]
    l = np.c_[(np.ones(xq.shape[0]) - np.sum(xq, axis=1)), xq]

    if p == 0:
        basis_ref_grad[:] = 0.
        return 
    for i in range(nb):
        grad_dir[:, i, :] = get_tri_grad_area_coordinates(p, alpha[i], l)

    basis_ref_grad[:, :, 0] = grad_dir[:, :, 1] - grad_dir[:, :, 0]
    basis_ref_grad[:, :, 1] = grad_dir[:, :, 2] - grad_dir[:, :, 0]

    return basis_ref_grad # [nq, nb, dim]


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


def get_legendre_basis_1D(xq, p, basis_val=None, basis_ref_grad=None):
    '''
    Calculates the 1D Legendre basis functions

    Inputs:
    -------
        xq: coordinate of current node [nq, dim]
        p: order of polynomial space
        
    Outputs:
    -------- 
        basis_val: evaluated basis [nq, nb]
        basis_ref_grad: evaluated physical gradient of basis [nq, nb, dim]
    '''
    # Use numpy legendre polynomials
    leg_poly = np.polynomial.legendre.Legendre

    if basis_val is not None:
        basis_val[:, :] = 0.            
        xq.shape = -1
        
        for it in range(p+1):
            basis_val[:, it] = leg_poly.basis(it)(xq)
        xq.shape = -1, 1

    if basis_ref_grad is not None:
        basis_ref_grad[:,:] = 0.

        for it in range(p+1):
            dleg = leg_poly.basis(it).deriv(1)
            basis_ref_grad[:,it] = dleg(xq)


def get_legendre_basis_2D(xq, p, basis_val=None, basis_ref_grad=None):
    '''
    Calculates the 2D Legendre basis functions

    Inputs:
    -------
        xq: coordinate of current node [nq, dim]
        p: order of polynomial space
        
    Outputs:
    -------- 
        basis_val: evaluated basis [nq, nb]
        basis_ref_grad: evaluated physical gradient of basis [nq, nb, dim]
    '''
    nq = xq.shape[0]
    if basis_ref_grad is not None:
        gradx = np.zeros((nq, p+1, 1)); grady = np.zeros_like(gradx)
    else:
        gradx = None; grady = None
    # Always need basis_val
    valx = np.zeros((nq, p+1)); valy = np.zeros_like(valx)

    legendre_seg = basis_defs.LegendreSeg(p)
    get_legendre_basis_1D(xq[:, 0], p, valx, gradx)
    get_legendre_basis_1D(xq[:, 1], p, valy, grady)

    if basis_val is not None:
        for i in range(nq):
            basis_val[i, :] = np.reshape(np.outer(valx[i, :], \
                    valy[i, :]), (-1, ), 'F')
    if basis_ref_grad is not None:
        for i in range(nq):
            basis_ref_grad[i, :, 0] = np.reshape(np.outer(gradx[i, :, 0], \
                    valy[i, :]), (-1, ), 'F')
            basis_ref_grad[i, :, 1] = np.reshape(np.outer(valx[i, :], \
                    grady[i, :, 0]), (-1, ), 'F')


def get_modal_basis_tri(xi, p, xnodes, basis_val):
    '''
    Calculates the value for Hierarchical triangle basis function

    Inputs:
    -------
        xi: coordinate of quadrature points [nq, dim]
        p: polynomial solution order
        xnodes: coordinates of nodes in 1D equidistant ref space 
            [nb, dim]
        
    Outputs:
    -------- 
        basis_val: evaluated basis [nq, nb]
    '''
    nb = xnodes.shape[0]

    val_reorder = np.zeros_like(basis_val)

    # Transform to the modal basis reference element
    # [-1,-1], [1,-1], [-1,1]
    xnodes = 2.*xnodes - 1.
    xi = 2.*xi - 1.

    # Define the affine coordinates
    l = np.zeros([xi.shape[0], 3])

    l[:, 0] = (xi[:, 1] + 1.) / 2.
    l[:, 1] = -1.*((xi[:, 1] + xi[:, 0]) / 2.)
    l[:, 2] = (xi[:, 0] + 1.) / 2.

    if p == 0:
        basis_val[:] = 1.
        return 

    val_reorder[:,[0, 1, 2]] = l[:,[1, 2, 0]]

    e1 = np.arange(3, p-1+3, 1)
    e2 = np.arange(p-1+3, 2*p-2+3, 1)
    e3 = np.arange(2*p-2+3, 3*p-3+3, 1)

    val_reorder[:, e1] = get_edge_basis(p, l[:, 2], l[:, 1])
    val_reorder[:, e2] = get_edge_basis(p, l[:, 0], l[:, 2])
    val_reorder[:, e3] = get_edge_basis(p, l[:, 1], l[:, 0])

    internal = np.arange(3*p-3+3, nb, 1)

    val_reorder[:,internal] = get_internal_basis(p, internal, l)

    index = mesh_gmsh.gmsh_node_order_tri(p)

    basis_val[:, :] = val_reorder[:, index]

    return basis_val # [nb, nb]


def get_edge_basis(p, ll, lr):
    '''
    Helper function for Hierarchical triangular basis
    Note: Eq. 2.21 from Sorin et al.
    '''
    phi_e = np.zeros([ll.shape[0], p-1])
    for k in range(p-1):
        kernel = get_kernel_function(k, ll-lr)
        phi_e[:, k] = ll*lr*kernel

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
                n[k, 0] = n1[i]
                n[k, 1] = n2[j]
                k += 1

    for m in range(c):
        phi_i[:, m] = l[:, 0]*l[:, 1]**n[m, 0]*l[:, 2]**n[m, 1]

    return phi_i


def get_kernel_function(p, x):
    '''
    Helper function for Hierarchical triangular basis
    Note: pp. 27 of Sorin et al.
    '''
    p += 2
    # Initialize the legendre polynomial object
    leg_poly = np.polynomial.legendre.Legendre
    x.shape = -1

    # Construct the kernel's denominator 
    # (series of Lobatto fcns)                    
    
    # First two lobatto shape functions 
    l0 = (1. - x)/2.
    l1 = (1. + x)/2.

    den = l0*l1

    leg_int = leg_poly.basis(p-1).integ(m=1, lbnd=-1)
    num = np.sqrt((2.*p-1.)/2.)*leg_int(x)

    kernel = num / (1e-12 + den)

    x.shape = -1, 1

    return kernel


def get_modal_grad_tri(xi, p, xnodes, basis_ref_grad):
    '''
    Calculates the gradient of the triangular basis functions 

    Inputs:
    -------
        xi: coordinates of quadrature points [nq, dim]
        p: polynomial solution order
        xnodes: coordinates of nodes in 1D equidistant ref space 
            [nb, dim]
        
    Outputs:
    -------- 
        basis_ref_grad: evaluated gradient of basis function [nq, nb, dim]
    '''
    nb = xnodes.shape[0]

    grad_reorder = np.zeros_like(basis_ref_grad)

    xnodes = 2.*xnodes - 1.
    xi = 2.*xi - 1.

    gl = np.zeros([xi.shape[0], 3, 2])
    l = np.zeros([xi.shape[0], 3])

    # Calculate the affine coordinates
    l[:, 0] = (xi[:, 1] + 1.) / 2.
    l[:, 1] = -1.*((xi[:, 1] + xi[:, 0]) / 2.)
    l[:, 2] = (xi[:, 0] + 1.) / 2.

    # Calculate vertex gradients
    gl[:, 0, 0] = 0.
    gl[:, 0, 1] = 0.5 
    
    gl[:, 1, 0] = -0.5
    gl[:, 1, 1] = -0.5

    gl[:, 2, 0] = 0.5
    gl[:, 2, 1] = 0.

    if p == 0:
        basis_val[:] = 1.
        return 

    grad_reorder[:, [0, 1, 2], :] = gl[:, [1, 2, 0], :]

    # Calculate edge gradients
    e1 = np.arange(3, p-1+3, 1)
    e2 = np.arange(p-1+3, 2*p-2+3, 1)
    e3 = np.arange(2*p-2+3, 3*p-3+3, 1)

    dxdxi = np.zeros([3, 2])
    dxdxi[0, 0] = 1.   ; dxdxi[0, 1] = 0.5
    dxdxi[1, 0] = -0.5 ; dxdxi[1, 1] = 0.5
    dxdxi[2, 0] = -0.5 ; dxdxi[2, 1] = -1.

    grad_reorder[:, e1, 0] = get_edge_grad(p, dxdxi[0, 0], gl[:, 2, 0],
            gl[:, 1, 0], l[:, 2], l[:, 1])
    grad_reorder[:, e1, 1] = get_edge_grad(p, dxdxi[0, 1], gl[:, 2, 1],
            gl[:, 1, 1], l[:, 2], l[:, 1])
    grad_reorder[:, e2, 0] = get_edge_grad(p, dxdxi[1, 0], gl[:, 0, 0],
            gl[:, 2, 0], l[:, 0], l[:, 2])
    grad_reorder[:, e2, 1] = get_edge_grad(p, dxdxi[1, 1], gl[:, 0, 1],
            gl[:, 2, 1], l[:, 0], l[:, 2])
    grad_reorder[:, e3, 0] = get_edge_grad(p, dxdxi[2, 0], gl[:, 1, 0],
            gl[:, 0, 0], l[:, 1], l[:, 0])
    grad_reorder[:, e3, 1] = get_edge_grad(p, dxdxi[2, 1], gl[:, 1, 1],
            gl[:, 0, 1], l[:, 1], l[:, 0])

    internal = np.arange(3*p-3+3, nb, 1)

    grad_reorder[:, internal, 0] = get_internal_grad(p, internal, 
            gl[:, :, 0], l)
    grad_reorder[:, internal, 1] = get_internal_grad(p, internal, 
            gl[:, :, 1], l)

    index = mesh_gmsh.gmsh_node_order_tri(p)

    basis_ref_grad[:,:,:] = grad_reorder[:,index,:]

    return basis_ref_grad # [nq, nb, dim]


def get_edge_grad(p, dxdxi, gl, gr, ll, lr):
    '''
    Helper function for Hierarchical triangular basis
    '''
    grad_e = np.zeros([ll.shape[0], p-1])
    for k in range(p-1):
        gkernel = get_kernel_grad(k, dxdxi, ll-lr)
        kernel = get_kernel_function(k, ll-lr)
        grad_e[:,k] = (ll*gr+lr*gl)*kernel + ll*lr*gkernel

    return grad_e


def get_kernel_grad(p, dxdxi, x):
    '''
    Helper function for Hierarchical triangular basis
    '''
    p += 2
    leg_poly = np.polynomial.legendre.Legendre
    x.shape = -1

    # First two lobatto shape functions 
    l0 =  (1. - x)/2.
    l1 =  (1. + x)/2.
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
    grad_i = np.zeros([l.shape[0], len(index)])

    c = 0
    for i in range(3, p+1):
        c += i-2
    
    n = np.zeros([c, 2])
    n1 = np.arange(1, p-1, 1)
    n2 = np.arange(1, p-1, 1)
    k = 0
    for i in range(len(n1)):
        for j in range(len(n2)):
            if n1[i] + n2[j] <= p-1:
                n[k, 0] = n1[i]
                n[k, 1] = n2[j]
                k+=1

    for m in range(c):
        dl2l3_1 = n[m, 0]*l[:, 1]**(n[m, 0]-1)*l[:, 2]**n[m, 1]*gl[:, 1]
        dl2l3_2 = n[m, 1]*l[:, 2]**(n[m, 1]-1)*l[:, 1]**n[m, 0]*gl[:, 2]
        grad_i[:, m] = gl[:, 0]*l[:, 1]**n[m, 0]*l[:, 2]**n[m, 1]+l[:, 0] \
                * (dl2l3_1+dl2l3_2)

    return grad_i