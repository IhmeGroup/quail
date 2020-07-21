# ------------------------------------------------------------------------ #
#
#       File : numerics/basis/ader_tools.py
#
#       Contains helper definitions for the shape and basis classes with
#       specific application to the ADER-DG solver.
#
#       Authors: Brett Bornhoft and Eric Ching
#
#       Created: January, 2020
#      
# ------------------------------------------------------------------------ #
from abc import ABC, abstractmethod
import code
import numpy as np

from data import ArrayList, GenericData
from general import SetSolverParams, BasisType, ShapeType

import meshing.gmsh as mesh_gmsh

import numerics.basis.basis as basis_defs
import numerics.basis.tools as basis_tools

def set_basis_spacetime(mesh, order, basis_name):
    '''
    Method: set_basis_spacetime
    ----------------------------
    Sets the space-time basis class given the basis_name string argunemnt

    INPUTS: 
        order: solution order [int]
        basis_name: name of the spacial basis function used to determine
                    the space-time basis function we wish to instantiate
                    as a class [str]

    OUTPUTS:
        basis_st: instantiated space-time basis class [basis object]

    RAISE:
        If the basis class is not defined returns a NotImplementedError
    '''
    if BasisType[basis_name] == BasisType.LagrangeSeg:
        basis_st = basis_defs.LagrangeQuad(order)
    elif BasisType[basis_name] == BasisType.LegendreSeg:
        basis_st = basis_defs.LegendreQuad(order)
    else:
        raise NotImplementedError

    return basis_st

def get_elem_inv_mass_matrix_ader(mesh, basis, order, elem=-1, 
    PhysicalSpace=False):
    '''
    Method: get_elem_inv_mass_matrix_ader
    --------------------------------------
    Calculate the inverse mass matrix for ADER-DG prediction step

    INPUTS:
        mesh: mesh object
        basis: basis object
        order: solution order [int]
        elem: element index [int]
        PhysicalSpace: [OPTIONAL] Flag to calc matrix in physical or 
                       reference space (default: False {reference space})

    OUTPUTS: 
        iMM: inverse mass matrix for ADER-DG predictor step [nb, nb]
    '''
    MM = get_elem_mass_matrix_ader(mesh, basis, order, elem, PhysicalSpace)

    iMM = np.linalg.inv(MM)

    return iMM # [nb_st, nb_st]


def get_inv_stiffness_matrix_ader(mesh, basis, order, elem, gradDir):
    '''
    Method: get_inv_stiffness_matrix_ader
    --------------------------------------
    Calculate the inverse stiffness matrix (Currently not used)

    INPUTS:
        mesh: mesh object
        basis: basis object
        order: solution order [int]
        elem: element index [int]
        gradDir: direction to take the gradient in [int]

    OUTPUTS: 
        iSM: inverse stiffness matrix [nb_st, nb_st]
    '''
    SM = get_stiffness_matrix_ader(mesh, basis, order, elem, gradDir)

    iSM = np.linalg.inv(SM) 

    return iSM # [nb_st, nb_st]


def get_stiffness_matrix_ader(mesh, basis, basis_st, order, dt, elem, gradDir
    , PhysicalSpace=False):
    '''
    Method: get_stiffness_matrix_ader
    --------------------------------------
    Calculate the stiffness matrix for ADER-DG prediction step
    INPUTS:
        mesh: mesh object
        basis: basis object
        basis_st : space-time basis object
        order: solution order [int]
        dt: time step [float]
        elem: element index [int]
        gradDir: direction of gradient calc [int]
    OUTPUTS: 
        SM: stiffness matrix for ADER-DG # [nb_st, nb_st]
    '''
    dim = mesh.Dim

    quad_order_st = basis_st.get_quadrature_order(mesh, order*2)
    quad_order = quad_order_st
    
    quad_pts_st, quad_wts_st = basis_st.get_quadrature_data(quad_order_st)
    quad_pts, quad_wts = basis.get_quadrature_data(quad_order)

    nq_st = quad_pts_st.shape[0]
    nq = quad_pts.shape[0]
    
    if PhysicalSpace:
        djac, _, ijac = basis_tools.element_jacobian(mesh, elem, quad_pts_st,
            get_djac=True, get_ijac=True)
        
        if len(djac) == 1:
            djac = np.full(nq, djac[0])
        
        ijac_st = np.zeros([nq_st, dim+1, dim+1])
        ijac_st[:, 0:dim, 0:dim] = ijac
        
        # add the temporal jacobian in the dim+1 dimension
        ijac_st[:, dim, dim] = 2./dt 
    
    else:
        djac = np.full(nq, 1.)

    basis_st.get_basis_val_grads(quad_pts_st, get_val=True, 
        get_ref_grad=True)
    
    nb_st = basis_st.basis_val.shape[1]
    phi = basis_st.basis_val

    if PhysicalSpace:
        basis_ref_grad = basis_st.basis_ref_grad
        GPhi = np.transpose(np.matmul(ijac_st.transpose(0, 2, 1), \
            basis_ref_grad.transpose(0, 2, 1)), (0, 2, 1))
    else:
        GPhi = basis_st.basis_ref_grad

    SM = np.zeros([nb_st, nb_st])
    # ------------------------------------------------------------------- #
    # Example of ADER Stiffness Matrix calculation using for-loops
    # ------------------------------------------------------------------- # 
    #
    # for i in range(nb_st):
    #     for j in range(nb_st):
    #         t = 0.
    #         for iq in range(nq_st):
    #             t += GPhi[iq,i,gradDir]*phi[iq,j]*quad_wts_st[iq]
    #         SM[i,j] = t
    #
    # ------------------------------------------------------------------- #
    SM[:] = np.matmul(GPhi[:,:,gradDir].transpose(), phi * quad_wts_st)

    return SM # [nb_st, nb_st]


def get_temporal_flux_ader(mesh, basis1, basis2, order, elem=-1, 
    PhysicalSpace=False):
    '''
    Method: get_temporal_flux_ader
    --------------------------------------
    Calculate the temporal flux matrix for ADER-DG prediction step

    INPUTS:
        mesh: mesh object
        basis1: basis object
        basis2: basis object 
        order: solution order [int]
        elem: [OPTIONAL] element index [int]
        PhysicalSpace: [OPTIONAL] Flag to calc matrix in physical or 
                       reference space (default: False {reference space})

    OUTPUTS: 
        FT: flux matrix for ADER-DG # [nb_st, nb] or [nb_st, nb_st]
            (Shape depends on basis objects passed in)

    NOTES:
        Can work at tau_n and tau_n+1 depending on basis combinations
    '''
    if basis1 == basis2:
        # if both basis are space-time you are at tau_{n+1} in ref time
        face = 2 
    else:
        # if basis are different you are at tau_{n} in ref time
        face = 0

    gbasis = mesh.gbasis
    quad_order = gbasis.get_quadrature_order(mesh, order*2)
    quad_pts, quad_wts = gbasis.get_quadrature_data(quad_order)

    nq = quad_pts.shape[0]

    if basis1 == basis2:
        # evaluate basis functions at tau_{n+1}
        face = 2

        PhiData = basis1
        PsiData = basis1

        PhiData.get_basis_face_val_grads(mesh, face, quad_pts, basis1, 
            get_val=True)
        PsiData.get_basis_face_val_grads(mesh, face, quad_pts, basis1, 
            get_val=True)
    else:
        # evaluate basis at tau_{n}
        face = 0
        
        PhiData = basis1
        PsiData = basis2

        PhiData.get_basis_face_val_grads(mesh, face, quad_pts, basis1, 
            get_val=True)
        PsiData.get_basis_val_grads(quad_pts, get_val=True, 
            get_ref_grad=False)

    nb_st = PhiData.basis_val.shape[1]
    nb = PsiData.basis_val.shape[1]

    FT = np.zeros([nb_st, nb])
    # ------------------------------------------------------------------- #
    # Example of ADER Flux Matrix calculation using for-loops
    # ------------------------------------------------------------------- # 
    #
    # phi = PhiData.basis_val
    # psi = PsiData.basis_val
    #
    # for i in range(nb_st):
    #     for j in range(nb):
    #         t = 0.
    #         for iq in range(nq):
    #             t += phi[iq,i]*psi[iq,j]*quad_wts[iq]
    #         FT[i,j] = t
    #
    # ------------------------------------------------------------------- # 
    FT[:] = np.matmul(PhiData.basis_val.transpose(), PsiData.basis_val * \
        quad_wts) # [nb_st, nb] or [nb_st, nb_st]

    return FT # [nb_st, nb] or [nb_st, nb_st]


def get_elem_mass_matrix_ader(mesh, basis, order, elem=-1, 
    PhysicalSpace=False):
    '''
    Method: get_elem_mass_matrix_ader
    --------------------------------------
    Calculate the mass matrix for ADER-DG prediction step

    INPUTS:
        mesh: mesh object
        basis: basis object
        order: solution order [int]
        elem: [OPTIONAL] element index [int]
        PhysicalSpace: [OPTIONAL] Flag to calc matrix in physical or 
                       reference space (default: False {reference space})

    OUTPUTS: 
        MM: mass matrix for ADER-DG [nb_st, nb_st]
    '''
    if PhysicalSpace:
        gbasis = mesh.gbasis
        quad_order = gbasis.get_quadrature_order(mesh, order*2)
    else:
        quad_order = order*2 # TODO: Verify (+ 1 #Add one for ADER method)

    quad_pts, quad_wts = basis.get_quadrature_data(quad_order)

    nq = quad_pts.shape[0]

    basis.get_basis_val_grads(quad_pts, get_val=True)

    if PhysicalSpace:
        djac,_,_=basis_tools.element_jacobian(mesh, elem, quad_pts,
            get_djac=True)
        
        if len(djac) == 1:
            djac = np.full(nq, djac[0])
    else:
        djac = np.full(nq, 1.).reshape(nq, 1)

    nb_st = basis.basis_val.shape[1]
    MM = np.zeros([nb_st, nb_st])
    # ------------------------------------------------------------------- #
    # Example of ADER Flux Matrix calculation using for-loops
    # ------------------------------------------------------------------- # 
    #
    # phi = basis.basis_val
    #
    # for i in range(nb_st):
    #     for j in range(nb_st):
    #         t = 0.
    #         for iq in range(nq):
    #             t += phi[iq,i]*phi[iq,j]*quad_wts[iq]*djac[iq]
    #         MM[i,j] = t
    #
    # ------------------------------------------------------------------- # 
    MM[:] = np.matmul(basis.basis_val.transpose(), basis.basis_val * \
            quad_wts * djac) # [nb_st,nb_st]

    return MM # [nb_st, nb_st]