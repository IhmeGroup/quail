# ------------------------------------------------------------------------ #
#
#       File : src/numerics/solver/ader_tools.py
#
#       Contains additional functions (tools) for the ADERDG solver class
#      
# ------------------------------------------------------------------------ #
import numpy as np
from scipy.optimize import fsolve, root
from scipy.linalg import solve_sylvester

import general

import meshing.tools as mesh_tools

import numerics.basis.basis as basis_defs
import numerics.helpers.helpers as helpers


def set_source_treatment(ns, source_treatment):
	'''
	This method sets the appropriate predictor function for the ADER-DG
	scheme given the input deck parameters

	Inputs:
	-------
		ns: number of state variables
		source_treatment: string from input deck to determine if the source
			term should be taken implicitly or explicitly

	Outputs:
	--------
		fcn: the name of the function chosen for the calculate_predictor_elem
	'''
	if source_treatment == "Explicit":
		fcn = predictor_elem_explicit
	elif source_treatment == "Implicit":
		if ns == 1:
			fcn = predictor_elem_implicit
		else:
			fcn = predictor_elem_sylvester
	else:
		raise NotImplementedError

	return fcn


def calculate_inviscid_flux_volume_integral(solver, elem_helpers, 
		elem_helpers_st, elem_ID, Fq):
	'''
	Calculates the inviscid flux volume integral for the ADERDG scheme

	Inputs:
	-------
		solver: solver object
		elem_helpers: helpers defined in ElemHelpers
		elem_helpers_st: space-time helpers defined in ElemHelpers
		elem_ID: element ID
		Fq: flux array evaluated at the quadrature points [nq, ns, dim]

	Outputs:
	--------
		R_elem: residual contribution (for volume integral of inviscid flux) 
			[nb, ns]
	'''
	quad_wts = elem_helpers.quad_wts
	quad_wts_st = elem_helpers_st.quad_wts
	basis_val = elem_helpers.basis_val 
	basis_phys_grad_elems = elem_helpers.basis_phys_grad_elems
	djac_elems = elem_helpers.djac_elems 

	basis_phys_grad = basis_phys_grad_elems[elem_ID]
	djac = djac_elems[elem_ID]

	nb = basis_val.shape[1]
	nq = quad_wts.shape[0]
	nq_st = quad_wts_st.shape[0]
	
	# Integrate
	R_elem = np.tensordot(np.tile(basis_phys_grad, (nq, 1, 1)), 
			Fq*(quad_wts_st.reshape(nq, nq)*djac).reshape(nq_st, 1, 1), 
			axes=([0, 2], [0, 2])) # [nb, ns]

	return R_elem # [nb, ns]


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
		R_B: residual contribution (from boundary face) [nb, ns]
	'''
	nb = basis_val.shape[1]
	nq = quad_wts_st.shape[0]

	# Integrate
	R_B = np.matmul(np.tile(basis_val,(nq,1)).transpose(), Fq*quad_wts_st) 

	return R_B # [nb, ns]


def calculate_source_term_integral(elem_helpers, elem_helpers_st, elem_ID, 
		Sq):
	'''
	Calculates the source term volume integral for the ADERDG scheme

	Inputs:
	-------
		elem_helpers: helpers defined in ElemHelpers
		elem_helpers_st: space-time helpers defined in ElemHelpers
		elem_ID: element ID
		Sq: source term array evaluated at the quadrature points [nq, ns]

	Outputs:
	--------
		R_elem: residual contribution (from volume integral of source term) 
			[nb, ns]
	'''
	quad_wts = elem_helpers.quad_wts
	quad_wts_st = elem_helpers_st.quad_wts

	basis_val = elem_helpers.basis_val 
	djac_elems = elem_helpers.djac_elems 
	djac = djac_elems[elem_ID]

	nb = basis_val.shape[1]
	nq = quad_wts.shape[0]
	nq_st = quad_wts_st.shape[0]

	# Integrate
	R_elem = np.matmul(np.tile(basis_val,(nq, 1)).transpose(), 
			Sq*(quad_wts_st.reshape(nq, nq)*djac).reshape(nq_st, 1))

	return R_elem # [nb, ns]


def predictor_elem_explicit(solver, elem_ID, dt, W, U_pred):
	'''
	Calculates the predicted solution state for the ADER-DG method using a 
	nonlinear solve of the weak form of the DG discretization in time.

	This function treats the source term explicitly. Appropriate for 
	non-stiff systems.

	Inputs:
	-------
		solver: solver object
		elem_ID: element ID
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

	order = solver.order
	
	elem_helpers = solver.elem_helpers
	ader_helpers = solver.ader_helpers
	
	quad_wts = elem_helpers.quad_wts
	basis_val = elem_helpers.basis_val 
	djac_elems = elem_helpers.djac_elems 
	djac = djac_elems[elem_ID]

	FTR = ader_helpers.FTR
	MM = ader_helpers.MM
	SMS = ader_helpers.SMS_elems[elem_ID]
	iK = ader_helpers.iK

	vol_elems = elem_helpers.vol_elems
	W_bar = np.zeros([1, ns])

	Wq = helpers.evaluate_state(W, basis_val, skip_interp=basis.skip_interp)
	vol = vol_elems[elem_ID]

	W_bar = helpers.get_element_mean(Wq, quad_wts, djac, vol)
	U_pred[:] = W_bar

	source_coeffs = solver.source_coefficients(elem_ID, dt, order, basis_st, 
			U_pred)
	flux_coeffs = solver.flux_coefficients(elem_ID, dt, order, basis_st, 
			U_pred)

	niter = 100
	for i in range(niter):

		U_pred_new = np.matmul(iK, (np.matmul(MM, source_coeffs) - np.einsum(
				'ijk,jlk->il', SMS, flux_coeffs)+np.matmul(FTR, W)))
		err = U_pred_new - U_pred

		if np.amax(np.abs(err)) < 1.e-10:
			U_pred = U_pred_new
			break

		U_pred = U_pred_new
		
		source_coeffs = solver.source_coefficients(elem_ID, dt, order, 
				basis_st, U_pred)
		flux_coeffs = solver.flux_coefficients(elem_ID, dt, order, basis_st, 
				U_pred)

		if i == niter - 1:
			print('Sub-iterations not converging')
			raise ValueError

	return U_pred # [nb_st, ns]


def predictor_elem_implicit(solver, elem_ID, dt, W, U_pred):
	'''
	Calculates the predicted solution state for the ADER-DG method using a 
	nonlinear solve of the weak form of the DG discretization in time.

	This function treats the source term implicitly. Appropriate for 
	stiff scalar equations.

	Inputs:
	-------
		solver: solver object
		elem_ID: element ID
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

	order = solver.order
	
	elem_helpers = solver.elem_helpers
	ader_helpers = solver.ader_helpers
	
	quad_wts = elem_helpers.quad_wts
	basis_val = elem_helpers.basis_val 
	djac_elems = elem_helpers.djac_elems 
	djac = djac_elems[elem_ID]
	x_elems = elem_helpers.x_elems
	x = x_elems[elem_ID]

	FTR = ader_helpers.FTR
	MM = ader_helpers.MM
	SMS = ader_helpers.SMS_elems[elem_ID]
	K = ader_helpers.K

	vol_elems = elem_helpers.vol_elems
	Wq = helpers.evaluate_state(W, basis_val, skip_interp=basis.skip_interp)
	vol = vol_elems[elem_ID]

	W_bar = helpers.get_element_mean(Wq, quad_wts, djac, vol)

	Sjac_q = np.zeros([1, ns, ns])
	Sjac_q = physics.eval_source_term_jacobians(W_bar, x, solver.time, 
			Sjac_q) 
	Sjac = Sjac_q[0, :, :]
	
	Kp = K - MM*dt*Sjac 
	iK = np.linalg.inv(Kp)

	U_pred[:] = W_bar

	source_coeffs = solver.source_coefficients(elem_ID, dt, order, basis_st, 
			U_pred)
	flux_coeffs = solver.flux_coefficients(elem_ID, dt, order, basis_st, 
			U_pred)

	niter = 100
	for i in range(niter):
		U_pred_new = np.matmul(iK, np.matmul(MM, source_coeffs) - np.einsum(
				'ijk,jlk->il', SMS, flux_coeffs)+np.matmul(FTR, W) - \
				np.matmul(MM, dt*Sjac*U_pred))
		err = U_pred_new - U_pred
		test4 = np.matmul(MM, dt*Sjac*U_pred)
		if np.amax(np.abs(err)) < 1.e-10:
			U_pred = U_pred_new
			break
		import code; code.interact(local=locals())
		U_pred = U_pred_new

		source_coeffs = solver.source_coefficients(elem_ID, dt, order, 
				basis_st, U_pred)
		flux_coeffs = solver.flux_coefficients(elem_ID, dt, order, basis_st, 
				U_pred)
		
		if i == niter - 1:
			print('Sub-iterations not converging')
			raise ValueError

	return U_pred # [nb_st, ns]

def predictor_elem_sylvester(solver, elem_ID, dt, W, U_pred):
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
		elem_ID: element ID
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

	order = solver.order
	
	elem_helpers = solver.elem_helpers
	ader_helpers = solver.ader_helpers
	
	quad_wts = elem_helpers.quad_wts
	basis_val = elem_helpers.basis_val 
	djac_elems = elem_helpers.djac_elems 
	djac = djac_elems[elem_ID]
	x_elems = elem_helpers.x_elems
	x = x_elems[elem_ID]

	FTR = ader_helpers.FTR
	iMM = ader_helpers.iMM
	SMS = ader_helpers.SMS_elems[elem_ID]
	K = ader_helpers.K
	vol_elems = elem_helpers.vol_elems

	Wq = np.matmul(basis_val, W)

	vol = vol_elems[elem_ID]
	W_bar = helpers.get_element_mean(Wq, quad_wts, djac, vol)

	Sjac_q = np.zeros([1, ns, ns])
	Sjac_q = physics.eval_source_term_jacobians(W_bar, x, solver.time, 
			Sjac_q)
	Sjac = Sjac_q[0, :, :]
	
	U_pred[:] = W_bar

	source_coeffs = solver.source_coefficients(elem_ID, dt, order, basis_st,
			U_pred)
	flux_coeffs = solver.flux_coefficients(elem_ID, dt, order, basis_st, 
			U_pred)

	niter = 100
	for i in range(niter):

		A = np.matmul(iMM,K)/dt
		B = -1.0*Sjac.transpose()

		C = np.matmul(FTR, W) - np.einsum('ijk,jlk->il', SMS, flux_coeffs)
		Q = source_coeffs/dt - np.matmul(U_pred, Sjac.transpose()) + \
				np.matmul(iMM, C)/dt

		U_pred_new = solve_sylvester(A, B, Q)

		err = U_pred_new - U_pred
		if np.amax(np.abs(err)) < 1.e-10:
			U_pred = U_pred_new
			break

		U_pred = U_pred_new
		
		source_coeffs = solver.source_coefficients(elem_ID, dt, order, 
				basis_st, U_pred)
		flux_coeffs = solver.flux_coefficients(elem_ID, dt, order, basis_st,
				U_pred)

		if i == niter - 1:
			print('Sub-iterations not converging')
			raise ValueError

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
		djac: determinant of the Jacobian
		f: array of values to be projected from

	Outpust:
	--------
		U: array of values to be projected to
	'''
	if basis.basis_val.shape[0] != quad_wts.shape[0]:
		basis.get_basis_val_grads(quad_pts, get_val=True)

	rhs = np.matmul(basis.basis_val.transpose(), f*quad_wts*djac) # [nb, ns]
	U[:, :] = np.matmul(iMM, rhs)


def ref_to_phys_time(mesh, elem_ID, time, dt, tref, basis=None):
    '''
    This function converts reference time coordinates to physical
    time coordinates

    Intputs:
    --------
        mesh: mesh object
        elem_ID: element ID 
        time: current solution time
        dt: solution time step
        tref: time in reference space [nq, 1]
        basis: basis object

	Outputs:
	--------
        tphys: coordinates in temporal space [nq, 1]
    '''
    gorder = 1
    if basis is None:
    	basis = basis_defs.LagrangeSeg(gorder)
    	basis.get_basis_val_grads(tref, get_val=True)

    tphys = (time/2.)*(1. - tref) + (time + dt)/2.*(1. + tref)

    return tphys, basis