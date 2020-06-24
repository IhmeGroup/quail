import code
import copy
import numpy as np
from scipy.optimize import fsolve, root

import general

import meshing.tools as mesh_tools

import numerics.basis.basis as basis_defs


# Update default solver params.
general.SolverParams.update({
	"TimeScheme": "ADER",
	"SourceTreatment": "Explicit",
})

def set_source_treatment(ns, SourceTreatment):
	if SourceTreatment == "Explicit":
		fcn = predictor_elem_explicit
	elif SourceTreatment == "Implicit":
		if ns is 1:
			fcn = predictor_elem_implicit
		else:
			fcn = predictor_elem_sylvester
	else:
		raise NotImplementedError

	return fcn

def calculate_inviscid_flux_volume_integral(solver, elem_ops, elem_ops_st, elem, Fq):
	
	quad_wts = elem_ops.quad_wts
	quad_wts_st = elem_ops_st.quad_wts
	basis_val = elem_ops.basis_val 
	basis_pgrad_elems = elem_ops.basis_pgrad_elems
	djac_elems = elem_ops.djac_elems 

	basis_pgrad = basis_pgrad_elems[elem]
	djac = djac_elems[elem]

	nb = basis_val.shape[1]
	nq = quad_wts.shape[0]
	nq_st = quad_wts_st.shape[0]
	
	# for ir in range(sr):
	# 	for k in range(nn): # Loop over basis function in space
	# 		for i in range(nq): # Loop over time
	# 			for j in range(nq): # Loop over space
	# 				gPsi = PsiData.gPhi[j,k]
	# 				ER[k,ir] += wq[i]*wq[j]*JData.djac[j*(JData.nq!=1)]*F[i,j,ir]*gPsi

	# F = np.reshape(F,(nqST,sr,dim))
	
	ER = np.tensordot(np.tile(basis_pgrad,(nb,1,1)), Fq*(quad_wts_st.reshape(nq,nq)*djac).reshape(nq_st,1,1), axes=([0,2],[0,2])) # [nb, ns]

	return ER

def calculate_inviscid_flux_boundary_integral(basis_val, quad_wts_st, Fq):

	# F = np.reshape(F,(nq,nqST,sr))
	
	# for ir in range(sr):
	# 	#for k in range(nn): # Loop over basis function in space
	# 	for i in range(nqST): # Loop over time
	# 		for j in range(nq): # Loop over space
	# 			PsiL = PsiDataL.Phi[j,:]
	# 			PsiR = PsiDataR.Phi[j,:]
	# 			RL[:,ir] -= wqST[i]*wq[j]*F[j,i,ir]*PsiL
	# 			RR[:,ir] += wqST[i]*wq[j]*F[j,i,ir]*PsiR

	# F = np.reshape(F,(nqST,sr))
	
	nb = basis_val.shape[1]
	R = np.matmul(np.tile(basis_val,(nb,1)).transpose(), Fq*quad_wts_st) # [nb, ns]

	return R

def calculate_source_term_integral(elem_ops, elem_ops_st, elem, Sq):
	
	quad_wts = elem_ops.quad_wts
	quad_wts_st = elem_ops_st.quad_wts

	basis_val = elem_ops.basis_val 
	djac_elems = elem_ops.djac_elems 
	djac = djac_elems[elem]

	nb = basis_val.shape[1]
	nq = quad_wts.shape[0]
	nq_st = quad_wts_st.shape[0]

	# s = np.reshape(s,(nq,nq,sr))
	# #Calculate source term integral
	# for ir in range(sr):
	# 	for k in range(nn):
	# 		for i in range(nq): # Loop over time
	# 			for j in range(nq): # Loop over space
	# 				Psi = PsiData.Phi[j,k]
	# 				ER[k,ir] += wq[i]*wq[j]*s[i,j,ir]*JData.djac[j*(JData.nq!=1)]*Psi
	# s = np.reshape(s,(nqST,sr))

	ER = np.matmul(np.tile(basis_val,(nb,1)).transpose(),Sq*(quad_wts_st.reshape(nq,nq)*djac).reshape(nq_st,1)) # [nb, ns]

	return ER

def predictor_elem_explicit(solver, elem, dt, Wp, Up):
	'''
	Method: calculate_predictor_elem
	-------------------------------------------
	Calculates the predicted solution state for the ADER-DG method using a nonlinear solve of the
	weak form of the DG discretization in time.

	INPUTS:
		elem: element index
		dt: time step 
		Wp: previous time step solution in space only

	OUTPUTS:
		Up: predicted solution in space-time
	'''
	EqnSet = solver.EqnSet
	ns = EqnSet.StateRank
	mesh = solver.mesh

	basis = solver.basis
	basis_st = solver.basis_st

	order = EqnSet.order
	
	elem_ops = solver.elem_operators
	ader_ops = solver.ader_operators
	
	quad_wts = elem_ops.quad_wts
	basis_val = elem_ops.basis_val 
	djac_elems = elem_ops.djac_elems 
	djac = djac_elems[elem]

	FTR = ader_ops.FTR
	MM = ader_ops.MM
	SMS = ader_ops.SMS_elems[elem]
	iK = ader_ops.iK


	srcpoly = solver.source_coefficients(elem, dt, order, basis_st, Up)
	flux = solver.flux_coefficients(elem, dt, order, basis_st, Up)
	ntest = 100
	for i in range(ntest):

		Up_new = np.matmul(iK,(np.matmul(MM,srcpoly)-np.einsum('ijk,jlk->il',SMS,flux)+np.matmul(FTR,Wp)))
		err = Up_new - Up

		if np.amax(np.abs(err))<1e-10:
			Up = Up_new
			break

		Up = Up_new
		
		srcpoly = solver.source_coefficients(elem, dt, order, basis_st, Up)
		flux = solver.flux_coefficients(elem, dt, order, basis_st, Up)

	return Up


def predictor_elem_implicit(solver, elem, dt, Wp, Up):
	'''
	Method: calculate_predictor_elem
	-------------------------------------------
	Calculates the predicted solution state for the ADER-DG method using a nonlinear solve of the
	weak form of the DG discretization in time.

	INPUTS:
		elem: element index
		dt: time step 
		Wp: previous time step solution in space only

	OUTPUTS:
		Up: predicted solution in space-time
	'''
	EqnSet = solver.EqnSet
	Sources = EqnSet.Sources

	ns = EqnSet.StateRank
	mesh = solver.mesh

	basis = solver.basis
	basis_st = solver.basis_st

	order = EqnSet.order
	
	elem_ops = solver.elem_operators
	ader_ops = solver.ader_operators
	
	quad_wts = elem_ops.quad_wts
	basis_val = elem_ops.basis_val 
	djac_elems = elem_ops.djac_elems 
	
	djac = djac_elems[elem]
	_, ElemVols = mesh_tools.element_volumes(mesh, solver)

	FTR = ader_ops.FTR
	MM = ader_ops.MM
	SMS = ader_ops.SMS_elems[elem]
	K = ader_ops.K

	W_bar = np.zeros([1,ns])
	Wq = np.matmul(basis_val, Wp)
	vol = ElemVols[elem]

	W_bar[:] = np.matmul(Wq.transpose(),quad_wts*djac).T/vol

	# def F(u):
	# 	S = 0.
	# 	S = EqnSet.SourceState(1, 0., 0., u, S)
	# 	F = u - S - W_bar[0,0]
	# 	return F

	# U_bar = fsolve(F, W_bar)

	jac = 0.0
	for Source in Sources:
		jac += Source.get_jacobian(W_bar)

	Kp = K-MM*dt*jac 
	iK = np.linalg.inv(Kp)

	Up[:] = W_bar

	srcpoly = solver.source_coefficients(elem, dt, order, basis_st, Up)
	flux = solver.flux_coefficients(elem, dt, order, basis_st, Up)
	ntest = 100
	for i in range(ntest):

		Up_new = np.matmul(iK,(np.matmul(MM,srcpoly)-np.einsum('ijk,jlk->il',SMS,flux)+np.matmul(FTR,Wp)-np.matmul(MM,dt*jac*Up)))
		err = Up_new - Up

		if np.amax(np.abs(err))<1e-10:
			Up = Up_new
			break

		Up = Up_new
		

		srcpoly = solver.source_coefficients(elem, dt, order, basis_st, Up)
		flux = solver.flux_coefficients(elem, dt, order, basis_st, Up)

	return Up

def predictor_elem_sylvester(solver, elem, dt, Wp, Up):
	'''
	Method: calculate_predictor_elem
	-------------------------------------------
	Calculates the predicted solution state for the ADER-DG method using a nonlinear solve of the
	weak form of the DG discretization in time.
	INPUTS:
		elem: element index
		dt: time step 
		W: previous time step solution in space only
	OUTPUTS:
		Up: predicted solution in space-time
	'''
	EqnSet = solver.EqnSet
	Sources = EqnSet.Sources

	ns = EqnSet.StateRank
	mesh = solver.mesh

	basis = solver.basis
	basis_st = solver.basis_st

	order = EqnSet.order
	
	elem_ops = solver.elem_operators
	ader_ops = solver.ader_operators
	
	quad_wts = elem_ops.quad_wts
	basis_val = elem_ops.basis_val 
	djac_elems = elem_ops.djac_elems 
	
	djac = djac_elems[elem]
	_, ElemVols = mesh_tools.element_volumes(mesh, solver)

	FTR = ader_ops.FTR
	iMM = ader_ops.iMM_elems[elem]
	SMS = ader_ops.SMS_elems[elem]
	K = ader_ops.K

	W_bar = np.zeros([1,ns])
	Wq = np.matmul(basis_val, Wp)
	vol = ElemVols[elem]
	W_bar[:] = np.matmul(Wq.transpose(),quad_wts*djac).T/vol

	jac = np.zeros([ns,ns])
	for Source in Sources:
		jac += Source.get_jacobian(W_bar) 

	srcpoly = solver.source_coefficients(elem, dt, order, basis_st, Up)
	flux = solver.flux_coefficients(elem, dt, order, basis_st, Up)

	ntest = 100
	for i in range(ntest):

		A = np.matmul(iMM,K)/dt
		B = -1.0*jac.transpose()

		C = np.matmul(FTR,Wp) - np.einsum('ijk,jlk->il',SMS,flux)
		Q = srcpoly/dt - np.matmul(Up,jac.transpose()) + np.matmul(iMM,C)/dt

		Up_new = solve_sylvester(A,B,Q)

		err = Up_new - Up
		if np.amax(np.abs(err))<1e-10:
			Up = Up_new
			break

		Up = Up_new
		
		srcpoly = solver.source_coefficients(elem, dt, order, basis_st, Up)
		flux = solver.flux_coefficients(elem, dt, order, basis_st, Up)
		if i == ntest-1:
			print('Sub-iterations not converging',i)

	return Up

def ref_to_phys_time(mesh, elem, time, dt, gbasis, xref, tphys=None, PointsChanged=False):
    '''
    Function: ref_to_phys_time
    ------------------------------
    This function converts reference time coordinates to physical
    time coordinates

    INPUTS:
        mesh: Mesh object
        elem: element 
        PhiData: basis data
        npoint: number of coordinates to convert
        xref: coordinates in reference space
        tphys: pre-allocated storage for physical time coordinates (optional) 

    OUTPUTS:
        tphys: coordinates in temporal space
    '''
    gorder = 1
    gbasis = basis_defs.LagrangeEqQuad(gorder)

    npoint = xref.shape[0]

    gbasis.eval_basis(xref, Get_Phi=True)

    dim = mesh.Dim
    
    Phi = gbasis.basis_val

    if tphys is None:
        tphys = np.zeros([npoint,dim])
    else:
        tphys[:] = time
    for ipoint in range(npoint):
        #for n in range(nn):
            #nodeNum = ElemNodes[n]
            #val = Phi[ipoint][n]
            #for d in range(dim):
        tphys[ipoint] = (time/2.)*(1-xref[ipoint,dim])+((time+dt)/2.0)*(1+xref[ipoint,dim])

    return tphys, gbasis