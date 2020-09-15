# ------------------------------------------------------------------------ #
#
#       File : src/numerics/solver/ader_tools.py
#
#       Contains additional methods (tools) for the ADERDG solver class
#
#       Authors: Brett Bornhoft and Eric Ching
#
#       Created: January 2020
#      
# ------------------------------------------------------------------------ #
import code
import copy
import numpy as np
from scipy.optimize import fsolve, root
from scipy.linalg import solve_sylvester


import general

import meshing.tools as mesh_tools

import numerics.basis.basis as basis_defs
import numerics.helpers.helpers as helpers


def set_source_treatment(ns, SourceTreatmentADER):
	'''
	This method sets the appropriate predictor function for the ADER-DG
	scheme given the input deck parameters

	Inputs:
	-------
		ns: number of state variables
		SourceTreatmentADER: string from input deck to determine if the source
			term should be taken implicitly or explicitly

	Outputs:
	--------
		fcn: the name of the function chosen for the calculate_predictor_elem
	'''
	if SourceTreatmentADER == "Explicit":
		fcn = predictor_elem_explicit
	elif SourceTreatmentADER == "Implicit":
		if ns == 1:
			fcn = predictor_elem_implicit
		else:
			fcn = predictor_elem_sylvester
	else:
		raise NotImplementedError

	return fcn


def calculate_inviscid_flux_volume_integral(solver, elem_ops, elem_ops_st, 
		elem, Fq):
	'''
	Calculates the inviscid flux volume integral for the ADERDG scheme

	Inputs:
	-------
		solver: solver object
		elem_ops: helper operators defined in ElemHelpers
		elem_ops_st: space-time helper operators defined in ElemHelpers
		elem: element index
		Fq: flux array evaluated at the quadrature points [nq, ns, dim]

	Outputs:
	--------
		ER: calculated residual array (for volume integral of specified 
		element) [nb, ns]
	'''
	quad_wts = elem_ops.quad_wts
	quad_wts_st = elem_ops_st.quad_wts
	basis_val = elem_ops.basis_val 
	basis_phys_grad_elems = elem_ops.basis_phys_grad_elems
	djac_elems = elem_ops.djac_elems 

	basis_phys_grad = basis_phys_grad_elems[elem]
	djac = djac_elems[elem]

	nb = basis_val.shape[1]
	nq = quad_wts.shape[0]
	nq_st = quad_wts_st.shape[0]
	
	ER = np.tensordot(np.tile(basis_phys_grad,(nq,1,1)), 
			Fq*(quad_wts_st.reshape(nq,nq)*djac).reshape(nq_st,1,1), 
			axes=([0,2],[0,2])) # [nb, ns]

	return ER # [nb, ns]


def calculate_inviscid_flux_boundary_integral(basis_val, quad_wts_st, Fq):
	'''
	Calculates the inviscid flux boundary integral for the ADERDG scheme

	Inputs:
	-------
		basis_val: basis function for the interior element [nq, nb]
		quad_wts_st: space-time quadrature weights [nq, 1]
		Fq: flux array evaluated at the quadrature points [nq, ns, dim]

	Outputs:
	--------
		R: calculated residual array (from boundary face) [nb, ns]
	'''
	nb = basis_val.shape[1]
	nq = quad_wts_st.shape[0]

	R = np.matmul(np.tile(basis_val,(nq,1)).transpose(), Fq*quad_wts_st) 

	return R # [nb, ns]


def calculate_source_term_integral(elem_ops, elem_ops_st, elem, Sq):
	'''
	Calculates the source term volume integral for the ADERDG scheme

	Inputs:
	-------
		elem_ops: helper operators defined in ElemHelpers
		elem_ops_st: space-time helper operators defined in ElemHelpers
		elem: element index
		Sq: source term array evaluated at the quadrature points [nq, ns]

	Outputs:
	--------
		ER: calculated residual array (for volume integral of specified 
		element) [nb, ns]
	'''
	quad_wts = elem_ops.quad_wts
	quad_wts_st = elem_ops_st.quad_wts

	basis_val = elem_ops.basis_val 
	djac_elems = elem_ops.djac_elems 
	djac = djac_elems[elem]

	nb = basis_val.shape[1]
	nq = quad_wts.shape[0]
	nq_st = quad_wts_st.shape[0]

	ER = np.matmul(np.tile(basis_val,(nq,1)).transpose(), 
			Sq*(quad_wts_st.reshape(nq,nq)*djac).reshape(nq_st,1))

	return ER # [nb, ns]

def predictor_elem_explicit(solver, elem, dt, W, U_pred):
	'''
	Calculates the predicted solution state for the ADER-DG method using a 
	nonlinear solve of the weak form of the DG discretization in time.

	This function applies the source term explicitly. Appropriate for 
	non-stiff systems.

	Inputs:
	-------
		solver: solver object
		elem: element index
		dt: time step 
		W: previous time step solution in space only [nb, ns]

	Outputs:
	--------
		U_pred: predicted solution in space-time [nb_st, ns]
	'''
	physics = solver.physics
	ns = physics.NUM_STATE_VARS
	mesh = solver.mesh

	basis = solver.basis
	basis_st = solver.basis_st

	order = physics.order
	
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

	vol_elems = elem_ops.vol_elems
	W_bar = np.zeros([1,ns])

	Wq = helpers.evaluate_state(W, basis_val, skip_interp=basis.skip_interp)
	vol = vol_elems[elem]

	W_bar = helpers.get_element_mean(Wq, quad_wts, djac, vol)
	U_pred[:] = W_bar

	srcpoly = solver.source_coefficients(elem, dt, order, basis_st, U_pred)
	flux = solver.flux_coefficients(elem, dt, order, basis_st, U_pred)
	ntest = 100
	for i in range(ntest):

		U_pred_new = np.matmul(iK,(np.matmul(MM,srcpoly) - \
				np.einsum('ijk,jlk->il',SMS,flux)+np.matmul(FTR,W)))
		err = U_pred_new - U_pred

		if np.amax(np.abs(err))<1e-10:
			U_pred = U_pred_new
			break

		U_pred = U_pred_new
		
		srcpoly = solver.source_coefficients(elem, dt, order, 
				basis_st, U_pred)
		flux = solver.flux_coefficients(elem, dt, order, basis_st, U_pred)

	return U_pred # [nb_st, ns]


def predictor_elem_implicit(solver, elem, dt, W, U_pred):
	'''
	Calculates the predicted solution state for the ADER-DG method using a 
	nonlinear solve of the weak form of the DG discretization in time.

	This function applies the source term implicitly. Appropriate for 
	stiff scalar equations.

	Inputs:
	-------
		solver: solver object
		elem: element index
		dt: time step 
		W: previous time step solution in space only [nb, 1]

	Outputs:
	--------
		U_pred: predicted solution in space-time [nb_st, 1]
	'''
	physics = solver.physics
	source_terms = physics.source_terms

	ns = physics.NUM_STATE_VARS
	mesh = solver.mesh

	basis = solver.basis
	basis_st = solver.basis_st

	order = physics.order
	
	elem_ops = solver.elem_operators
	ader_ops = solver.ader_operators
	
	quad_wts = elem_ops.quad_wts
	basis_val = elem_ops.basis_val 
	djac_elems = elem_ops.djac_elems 
	djac = djac_elems[elem]
	x_elems = elem_ops.x_elems
	x = x_elems[elem]

	FTR = ader_ops.FTR
	MM = ader_ops.MM
	SMS = ader_ops.SMS_elems[elem]
	K = ader_ops.K

	vol_elems = elem_ops.vol_elems
	# W_bar = np.zeros([1,ns])
	# Wq = np.matmul(basis_val, W)
	Wq = helpers.evaluate_state(W, basis_val, skip_interp=basis.skip_interp)
	vol = vol_elems[elem]

	# W_bar[:] = np.matmul(Wq.transpose(),quad_wts*djac).T/vol
	W_bar = helpers.get_element_mean(Wq, quad_wts, djac, vol)

	# def F(u):
	# 	S = 0.
	# 	S = physics.SourceState(1, 0., 0., u, S)
	# 	F = u - S - W_bar[0,0]
	# 	return F

	# U_bar = fsolve(F, W_bar)

	jac_q = np.zeros([1,ns,ns])

	jac_q = physics.SourceJacobianState(1, x, solver.time, W_bar, jac_q) 
	jac = jac_q[0,:,:]
	
	Kp = K-MM*dt*jac 
	iK = np.linalg.inv(Kp)

	U_pred[:] = W_bar

	srcpoly = solver.source_coefficients(elem, dt, order, basis_st, U_pred)
	flux = solver.flux_coefficients(elem, dt, order, basis_st, U_pred)
	ntest = 100
	for i in range(ntest):

		U_pred_new = np.matmul(iK,(np.matmul(MM,srcpoly) - \
				np.einsum('ijk,jlk->il',SMS,flux)+np.matmul(FTR,W) - \
				np.matmul(MM,dt*jac*U_pred)))
		err = U_pred_new - U_pred

		if np.amax(np.abs(err))<1e-10:
			U_pred = U_pred_new
			break

		U_pred = U_pred_new

		srcpoly = solver.source_coefficients(elem, dt, order, 
				basis_st, U_pred)
		flux = solver.flux_coefficients(elem, dt, order, basis_st, U_pred)

	return U_pred # [nb_st, ns]

def predictor_elem_sylvester(solver, elem, dt, W, U_pred):
	'''
	Calculates the predicted solution state for the ADER-DG method using a 
	nonlinear solve of the weak form of the DG discretization in time.

	This function applies the source term implicitly. Appropriate for 
	stiff systems of equations. The implicit solve utilizes the Sylvester
	equation of the form:

		AX + XB = C 

	This is a built-in function via the scipy.linalg library.

	Inputs:
	-------
		solver: solver object
		elem: element index
		dt: time step 
		W: previous time step solution in space only [nb, ns]

	Outputs:
	--------
		U_pred: predicted solution in space-time [nb_st, ns]
	'''
	physics = solver.physics
	source_terms = physics.source_terms

	ns = physics.NUM_STATE_VARS
	mesh = solver.mesh

	basis = solver.basis
	basis_st = solver.basis_st

	order = physics.order
	
	elem_ops = solver.elem_operators
	ader_ops = solver.ader_operators
	
	quad_wts = elem_ops.quad_wts
	basis_val = elem_ops.basis_val 
	djac_elems = elem_ops.djac_elems 
	djac = djac_elems[elem]
	x_elems = elem_ops.x_elems
	x = x_elems[elem]

	FTR = ader_ops.FTR
	# iMM = ader_ops.iMM_elems[elem]
	iMM = ader_ops.iMM
	SMS = ader_ops.SMS_elems[elem]
	K = ader_ops.K
	vol_elems = elem_ops.vol_elems

	Wq = np.matmul(basis_val, W)

	vol = vol_elems[elem]
	W_bar = helpers.get_element_mean(Wq, quad_wts, djac, vol)

	jac_q = np.zeros([1,ns,ns])
	jac_q = physics.SourceJacobianState(1, x, solver.time, W_bar, jac_q)
	jac = jac_q[0,:,:]
	
	U_pred[:] = W_bar

	srcpoly = solver.source_coefficients(elem, dt, order, basis_st, U_pred)
	flux = solver.flux_coefficients(elem, dt, order, basis_st, U_pred)

	ntest = 100
	for i in range(ntest):

		A = np.matmul(iMM,K)/dt
		B = -1.0*jac.transpose()

		C = np.matmul(FTR,W) - np.einsum('ijk,jlk->il',SMS,flux)
		Q = srcpoly/dt - np.matmul(U_pred,jac.transpose()) + \
				np.matmul(iMM,C)/dt

		U_pred_new = solve_sylvester(A,B,Q)

		err = U_pred_new - U_pred
		if np.amax(np.abs(err))<1e-10:
			U_pred = U_pred_new
			break

		U_pred = U_pred_new
		
		srcpoly = solver.source_coefficients(elem, dt, order, 
				basis_st, U_pred)
		flux = solver.flux_coefficients(elem, dt, order, basis_st, U_pred)
		if i == ntest-1:
			print('Sub-iterations not converging',i)

	return U_pred

def L2_projection(mesh, iMM, basis, quad_pts, quad_wts, djac, f, U):
	'''
	Performs an L2 projection for the space-time solution state vector

	Inputs:
	-------
		mesh: mesh object
		iMM: space-time inverse mass matrix
		basis: basis object
		quad_pts: quadrature coordinates in reference space
		quad_wts: quadrature weights
		djac: determinant of the jacobian
		f: array of values to be projected from

	Outpust:
	--------
		U: array of values to be projected too
	'''
	if basis.basis_val.shape[0] != quad_wts.shape[0]:
		basis.get_basis_val_grads(quad_pts, get_val=True)

	rhs = np.matmul(basis.basis_val.transpose(), f*quad_wts*djac) # [nb, ns]
	U[:,:] = np.matmul(iMM, rhs)

def ref_to_phys_time(mesh, elem, time, dt, gbasis, xref, 
		tphys=None, PointsChanged=False):
    '''
    This function converts reference time coordinates to physical
    time coordinates

    Intputs:
    --------
        mesh: mesh object
        elem: element index 
        time: current solution time
        dt: solution time step
        gbasis: geometric basis object
        xref: coordinates in reference space
        PointsChanged: flag to determine in points have changed between
        		each time step 

	Outputs:
	--------
        tphys: coordinates in temporal space [nq, dim]
    '''
    gorder = 1
    gbasis = basis_defs.LagrangeQuad(gorder)

    npoint = xref.shape[0]

    gbasis.get_basis_val_grads(xref, get_val=True)

    dim = mesh.dim
    
    Phi = gbasis.basis_val

    if tphys is None:
        tphys = np.zeros([npoint,dim])
    else:
        tphys[:] = time
    for ipoint in range(npoint):
        tphys[ipoint] = (time/2.)*(1-xref[ipoint,dim])+((time+dt)/2.0) * \
        		(1+xref[ipoint,dim])

    return tphys, gbasis