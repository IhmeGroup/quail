from abc import ABC, abstractmethod
import code
import numpy as np

from data import ArrayList, GenericData
from general import SetSolverParams, BasisType, ShapeType

import meshing.gmsh as mesh_gmsh

import numerics.basis.basis as basis_defs
import numerics.basis.tools as basis_tools

def set_basis_spacetime(mesh, order, BasisFunction):
    if BasisType[BasisFunction] == BasisType.LagrangeSeg:
        basis_st = basis_defs.LagrangeQuad(order)
    elif BasisType[BasisFunction] == BasisType.LegendreSeg:
        basis_st = basis_defs.LegendreQuad(order)
    else:
        raise NotImplementedError

    return basis_st

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

    quad_order_st = basis_st.get_quadrature_order(mesh, order*2)
    quad_order = quad_order_st
    
    quad_pts_st, quad_wts_st = basis_st.get_quadrature_data(quad_order_st)
    quad_pts, quad_wts = basis.get_quadrature_data(quad_order)

    # quad_pts_st = basis_st.quad_pts
    # quad_wts_st = basis_st.quad_wts
    nq_st = quad_pts_st.shape[0]

    # quad_pts = basis.quad_pts
    # quad_wts = basis.quad_wts
    nq = quad_pts.shape[0]
    
    if PhysicalSpace:
        djac,_,ijac=basis_tools.element_jacobian(mesh,elem,quad_pts_st,get_djac=True, get_ijac=True)
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

    gbasis = mesh.gbasis
    quad_order = gbasis.get_quadrature_order(mesh, order*2)
    quad_pts, quad_wts = gbasis.get_quadrature_data(quad_order)

    # quad_pts = gbasis.quad_pts
    # quad_wts = gbasis.quad_wts
    nq = quad_pts.shape[0]

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
        gbasis = mesh.gbasis
        quad_order = gbasis.get_quadrature_order(mesh, order*2)
    else:
        quad_order = order*2 + 1 #Add one for ADER method

    quad_pts, quad_wts = basis.get_quadrature_data(quad_order)

    # quad_pts = basis.quad_pts
    # quad_wts = basis.quad_wts
    nq = quad_pts.shape[0]

    basis.eval_basis(quad_pts, Get_Phi=True)

    if PhysicalSpace:
        djac,_,_=basis_tools.element_jacobian(mesh,elem,quad_pts,get_djac=True)
        
        if len(djac) == 1:
            djac = np.full(nq, djac[0])
    else:
        djac = np.full(nq, 1.).reshape(nq,1)

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