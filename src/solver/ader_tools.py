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
		elem_helpers_st, Fq):
	'''
	Calculates the inviscid flux volume integral for the ADERDG scheme

	Inputs:
	-------
		solver: solver object
		elem_helpers: helpers defined in ElemHelpers
		elem_helpers_st: space-time helpers defined in ElemHelpers
		Fq: flux array evaluated at the quadrature points [ne, nq, ns, dim]

	Outputs:
	--------
		res_elem: residual contribution (for volume integral of inviscid flux) 
			[ne, nb, ns]
	'''
	quad_wts_st = elem_helpers_st.quad_wts
	basis_phys_grad_elems = elem_helpers.basis_phys_grad_elems
	djac_elems = elem_helpers.djac_elems 

	nb = elem_helpers.basis_val.shape[1]
	nq = elem_helpers.quad_wts.shape[0]
	nq_st = quad_wts_st.shape[0]

	tile_basis_phys_grads = np.tile(basis_phys_grad_elems, (1, nq, 1, 1))
	quad_wts_st_djac = (quad_wts_st.reshape(nq, nq)*
			djac_elems).reshape(Fq.shape[0], nq_st, 1, 1)

	# integrate
	res_elem = np.einsum('ijkl, ijml -> ikm', tile_basis_phys_grads,
			Fq * quad_wts_st_djac) # [ne, nb, ns]

	return res_elem # [ne, nb, ns]


def calculate_inviscid_flux_boundary_integral(basis_val, quad_wts_st, Fq):
	'''
	Calculates the inviscid flux boundary integral for the ADERDG scheme

	Inputs:
	-------
		basis_val: basis function for the interior element [nf, nq, nb]
		quad_wts_st: space-time quadrature weights [nq, 1]
		Fq: flux array evaluated at the quadrature points [nf, nq, ns, dim]

	Outputs:
	--------
		R_B: residual contribution (from boundary face) [nf, nb, ns]
	'''
	nb = basis_val.shape[1]
	nq = quad_wts_st.shape[0]

	# Calculate the flux quadrature
	Fq_quad = np.einsum('ijk, jm -> ijk', Fq, quad_wts_st)
	# Calculate residual
	R_B = np.einsum('ijn, ijk -> ink', np.tile(basis_val,(nq, 1)), Fq_quad)

	return R_B # [nf, nb, ns]


def calculate_source_term_integral(elem_helpers, elem_helpers_st, Sq):
	'''
	Calculates the source term volume integral for the ADERDG scheme

	Inputs:
	-------
		elem_helpers: helpers defined in ElemHelpers
		elem_helpers_st: space-time helpers defined in ElemHelpers
		Sq: source term array evaluated at the quadrature points 
			[ne, nq, ns]

	Outputs:
	--------
		res_elem: residual contribution (from volume integral of source term) 
			[ne, nb, ns]
	'''
	quad_wts = elem_helpers.quad_wts
	quad_wts_st = elem_helpers_st.quad_wts

	basis_val = elem_helpers.basis_val 
	djac_elems = elem_helpers.djac_elems 

	nb = basis_val.shape[1]
	nq = quad_wts.shape[0]
	nq_st = quad_wts_st.shape[0]

	# Calculate residual from source term
	res_elem = np.einsum('jk, ijl -> ikl', np.tile(basis_val, (nq, 1)), 
			Sq*(quad_wts_st.reshape(nq, nq)*djac_elems).reshape(Sq.shape[0], 
			nq_st, 1)) # [ne, nb, ns]

	return res_elem # [ne, nb, ns]


def predictor_elem_explicit(solver, dt, W, U_pred):
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
		W: previous time step solution in space only [ne, nb, ns]

	Outputs:
	--------
		U_pred: predicted solution in space-time [ne, nb_st, ns]
	'''
	# Unpack
	physics = solver.physics
	ns = physics.NUM_STATE_VARS
	mesh = solver.mesh

	basis = solver.basis
	basis_st = solver.basis_st
	
	elem_helpers = solver.elem_helpers
	ader_helpers = solver.ader_helpers
	
	order = solver.order
	quad_wts = elem_helpers.quad_wts
	basis_val = elem_helpers.basis_val 
	djac_elems = elem_helpers.djac_elems 

	FTR = ader_helpers.FTR
	MM = ader_helpers.MM
	SMS_elems = ader_helpers.SMS_elems
	iK = ader_helpers.iK

	# Calculate the average state for each element in spatial coordinates
	vol_elems = elem_helpers.vol_elems
	Wq = helpers.evaluate_state(W, basis_val, skip_interp=basis.skip_interp)
	W_bar = helpers.get_element_mean(Wq, quad_wts, djac_elems, vol_elems)

	# Initialize space-time coefficients with computed average
	U_pred[:] = W_bar

	# Calculate the source and flux coefficients with initial guess
	source_coeffs = solver.source_coefficients(dt, order, basis_st, 
			U_pred)
	flux_coeffs = solver.flux_coefficients(dt, order, basis_st, 
			U_pred)

	# Iterate using a discrete Picard nonlinear solve for the 
	# updated space-time coefficients.
	niter = 100
	for i in range(niter):

		U_pred_new = np.einsum('jk, ikm -> ijm',iK, 
				np.einsum('jk, ijl -> ijl', MM, source_coeffs) - 
				np.einsum('ijkl, ikml -> ijm', SMS_elems, flux_coeffs) +
				np.einsum('jk, ikm -> ijm', FTR, W))

		err = U_pred_new - U_pred

		if np.amax(np.abs(err)) < 1.e-8:
			U_pred = U_pred_new
			break

		U_pred = U_pred_new
		
		source_coeffs = solver.source_coefficients(dt, order, 
				basis_st, U_pred)
		flux_coeffs = solver.flux_coefficients(dt, order, basis_st, 
				U_pred)

		if i == niter - 1:
			print('Sub-iterations not converging')
			raise ValueError

	return U_pred # [ne, nb_st, ns]


def predictor_elem_implicit(solver, dt, W, U_pred):
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
		W: previous time step solution in space only [ne, nb, 1]

	Outputs:
	--------
		U_pred: predicted solution in space-time [ne, nb_st, 1]
	'''
	# Unpack
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
	x_elems = elem_helpers.x_elems

	FTR = ader_helpers.FTR
	MM = ader_helpers.MM
	SMS_elems = ader_helpers.SMS_elems
	K = ader_helpers.K

	# Calculate the average state for each element in spatial coordinates
	vol_elems = elem_helpers.vol_elems
	Wq = helpers.evaluate_state(W, basis_val, skip_interp=basis.skip_interp)
	W_bar = helpers.get_element_mean(Wq, quad_wts, djac_elems, vol_elems)

	# Calculate the source term Jacobian using average state
	Sjac = np.zeros([U_pred.shape[0], ns, ns])
	Sjac = physics.eval_source_term_jacobians(W_bar, x_elems, solver.time, 
			Sjac) 
	Kp = K - dt * np.einsum('jk, imn -> ijk', MM, Sjac)

	iK = np.linalg.inv(Kp)

	# Initialize space-time coefficients with computed average
	U_pred[:] = W_bar

	# Calculate the source and flux coefficients with initial guess
	source_coeffs = solver.source_coefficients(dt, order, basis_st, 
			U_pred)
	flux_coeffs = solver.flux_coefficients(dt, order, basis_st, 
			U_pred)

	# Iterate using a discrete Picard nonlinear solve for the 
	# updated space-time coefficients.
	niter = 100
	for i in range(niter):

		U_pred_new = np.einsum('ijk, ikm -> ikm',iK, 
				(np.einsum('jk, ijl -> ikl', MM, source_coeffs) -
				np.einsum('ijkl, ikml -> ijm', SMS_elems, flux_coeffs) +
				np.einsum('jk, ikm -> ijm', FTR, W) - 
				np.einsum('jk, ijm -> ikm', MM, dt*Sjac*U_pred)))

		err = U_pred_new - U_pred

		if np.amax(np.abs(err)) < 1.e-8:
			U_pred = U_pred_new
			break

		U_pred = U_pred_new

		source_coeffs = solver.source_coefficients(dt, order, 
				basis_st, U_pred)
		flux_coeffs = solver.flux_coefficients(dt, order, basis_st, 
				U_pred)
		
		if i == niter - 1:
			print('Sub-iterations not converging')
			raise ValueError

	return U_pred # [ne, nb_st, ns]


def predictor_elem_sylvester(solver, dt, W, U_pred):
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
		W: previous time step solution in space only [ne, nb, ns]

	Outputs:
	--------
		U_pred: predicted solution in space-time [ne, nb_st, ns]
	'''
	# Unpack
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
	x_elems = elem_helpers.x_elems

	FTR = ader_helpers.FTR
	iMM = ader_helpers.iMM
	SMS_elems = ader_helpers.SMS_elems
	K = ader_helpers.K

	# Calculate the average state for each element in spatial coordinates
	vol_elems = elem_helpers.vol_elems
	Wq = helpers.evaluate_state(W, basis_val, skip_interp=basis.skip_interp)
	W_bar = helpers.get_element_mean(Wq, quad_wts, djac_elems, vol_elems)

	# Calculate the source term Jacobian using average state
	Sjac = np.zeros([U_pred.shape[0], 1, ns, ns])
	Sjac = physics.eval_source_term_jacobians(W_bar, x_elems, solver.time, 
			Sjac) 
	Sjac = Sjac[:, 0, :, :]

	# Initialize space-time coefficients with computed average
	U_pred[:] = W_bar

	# Calculate the source and flux coefficients with initial guess
	source_coeffs = solver.source_coefficients(dt, order, basis_st,
			U_pred)
	flux_coeffs = solver.flux_coefficients(dt, order, basis_st, 
			U_pred)

	# Iterate using a nonlinear Sylvester solver for the 
	# updated space-time coefficients. Solves for X in the form:
	# 	AX + XB = C
	niter = 100
	U_pred_new = np.zeros_like(U_pred)
	for i in range(niter):

		A = np.zeros([U_pred.shape[0], iMM.shape[0], iMM.shape[1]])
		A[:] = np.matmul(iMM,K)/dt
		B = -1.0*Sjac.transpose(0,2,1)

		Q = np.einsum('jk, ikm -> ijm', FTR, W) - np.einsum(
				'ijkl, ikml -> ijm', SMS_elems, flux_coeffs)

		C = source_coeffs/dt - np.matmul(U_pred[:], 
				Sjac[:].transpose(0,2,1)) + \
				np.einsum('jk, ikl -> ijl',iMM, Q)/dt

		for i in range(U_pred.shape[0]):
			U_pred_new[i, :, :] = solve_sylvester(A[i, :, :], B[i, :, :], 
					C[i, :, :])

		err = U_pred_new - U_pred
		if np.amax(np.abs(err)) < 1.e-8:
			U_pred = U_pred_new
			break

		U_pred = U_pred_new
		
		source_coeffs = solver.source_coefficients(dt, order, 
				basis_st, U_pred)
		flux_coeffs = solver.flux_coefficients(dt, order, basis_st,
				U_pred)

		if i == niter - 1:
			print('Sub-iterations not converging')
			raise ValueError

	return U_pred # [ne, nb_st, ns]


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

	rhs = np.einsum('jk, ijl -> ikl', basis.basis_val, f*quad_wts*djac) 
			# [ne, nb, ns]
	U[:, :, :] = np.einsum('ijk, ijl -> ikl', iMM, rhs)
	

def ref_to_phys_time(mesh, time, dt, tref, basis=None):
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