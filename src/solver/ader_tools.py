# ------------------------------------------------------------------------ #
#
#       File : src/solver/ader_tools.py
#
#       Contains additional functions (tools) for the ADER-DG solver class
#
# ------------------------------------------------------------------------ #
import numpy as np
from scipy.optimize import fsolve, root
from scipy.linalg import solve_sylvester

import general

import meshing.tools as mesh_tools

import numerics.basis.basis as basis_defs
import numerics.helpers.helpers as helpers
from scipy.integrate import ode


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
	elif source_treatment == "Testing":
		# fcn = predictor_elem_testing
		fcn = predictor_elem_sylvester_guess
	elif source_treatment == "Polyjac":
		fcn = predictor_elem_polyjac
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
		Fq: flux array evaluated at the quadrature points [ne, nq, ns, ndims]

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

	nq_t = elem_helpers_st.nq_tile_constant

	tile_basis_phys_grads = np.tile(basis_phys_grad_elems, (1, nq_t, 1, 1))

	quad_wts_st_djac = quad_wts_st * np.tile(djac_elems, (nq_t, 1))

	# integrate
	res_elem = np.einsum('ijkl, ijml -> ikm', tile_basis_phys_grads,
			Fq * np.expand_dims(quad_wts_st_djac, axis=3))# [ne, nb, ns]

	return res_elem # [ne, nb, ns]


def calculate_inviscid_flux_boundary_integral(nq_t, basis_val, 
		quad_wts_st, Fq):
	'''
	Calculates the inviscid flux boundary integral for the ADERDG scheme

	Inputs:
	-------
		basis_val: basis function for the interior element [nf, nq, nb]
		quad_wts_st: space-time quadrature weights [nq, 1]
		Fq: flux array evaluated at the quadrature points [nf, nq, ns, ndims]

	Outputs:
	--------
		resB: residual contribution (from boundary face) [nf, nb, ns]
	'''
	# Calculate the flux quadrature
	Fq_quad = np.einsum('ijk, jm -> ijk', Fq, quad_wts_st)
	# Calculate residual
	resB = np.einsum('ijn, ijk -> ink', np.tile(basis_val,(nq_t, 1)), Fq_quad)

	return resB # [nf, nb, ns]


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

	nq_t = elem_helpers_st.nq_tile_constant

	quad_wts_st_djac = quad_wts_st * np.tile(djac_elems, (nq_t, 1))

	# Calculate residual from source term
	res_elem = np.einsum('jk, ijl -> ikl', np.tile(basis_val, (nq_t, 1)),
			Sq*quad_wts_st_djac) # [ne, nb, ns]

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
				np.einsum('jk, ikl -> ijl', MM, source_coeffs) -
				np.einsum('ijkl, ikml -> ijm', SMS_elems, flux_coeffs) +
				np.einsum('jk, ikm -> ijm', FTR, W))

		# We check when the coefficients are no longer changing.
		# This can lead to differences between NODAL and MODAL solutions.
		# This could be resolved by evaluating at the quadrature points
		# and comparing the error between those values.
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
	elem_helpers_st = solver.elem_helpers_st
	ader_helpers = solver.ader_helpers

	quad_wts = elem_helpers.quad_wts
	quad_wts_st = elem_helpers_st.quad_wts
	quad_pts = elem_helpers.quad_pts
	basis_val = elem_helpers.basis_val
	djac_elems = elem_helpers.djac_elems
	djac_elems_st = ader_helpers.djac_elems
	x_elems = elem_helpers.x_elems

	FTR = ader_helpers.FTR
	MM = ader_helpers.MM
	SMS_elems = ader_helpers.SMS_elems
	K = ader_helpers.K

	# Calculate the average state for each element in spatial coordinates
	vol_elems = elem_helpers.vol_elems
	Wq = helpers.evaluate_state(W, basis_val, skip_interp=basis.skip_interp)
	W_bar = helpers.get_element_mean(Wq, quad_wts, djac_elems, vol_elems)

	res = solver.stepper.res

	# Only evaluate Jacobian for stiff sources
	temp_sources = physics.source_terms.copy()
	physics.source_terms = physics.implicit_sources.copy()

	# Calculate the source term Jacobian using average state
	Sjac = np.zeros([U_pred.shape[0], 1, ns, ns])
	Sjac = physics.eval_source_term_jacobians(W_bar, x_elems, solver.time,
			Sjac)

	Sjac = np.reshape(Sjac, [U_pred.shape[0], ns, ns])
	Kp = K - dt * np.einsum('jk, imn -> ijk', MM, Sjac)

	iK = np.linalg.inv(Kp)

	# Initialize space-time coefficients with computed average
	U_pred[:] = W_bar

	physics.source_terms = temp_sources.copy()
	# Calculate the source and flux coefficients with initial guess
	source_coeffs = solver.source_coefficients(dt, order, basis_st,
			U_pred)
	flux_coeffs = solver.flux_coefficients(dt, order, basis_st,
			U_pred)

	# Iterate using a discrete Picard nonlinear solve for the
	# updated space-time coefficients.
	niter = 100
	for i in range(niter):
		U_pred_new = np.einsum('ijk, ikm -> ijm',iK,
				(np.einsum('jk, ijl -> ikl', MM, source_coeffs) -
				np.einsum('ijkl, ikml -> ijm', SMS_elems, flux_coeffs) +
				np.einsum('jk, ikm -> ijm', FTR, W) -
				np.einsum('jk, ijm -> ikm', MM, dt*Sjac*U_pred)))

		# We check when the coefficients are no longer changing.
		# This can lead to differences between NODAL and MODAL solutions.
		# This could be resolved by evaluating at the quadrature points
		# and comparing the error between those values.
		err = U_pred_new - U_pred
		if np.amax(np.abs(err)) < 1.e-8:
			U_pred = U_pred_new
			print("Predictor iterations: ", i)
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

		for ie in range(U_pred.shape[0]):
			U_pred_new[ie, :, :] = solve_sylvester(A[ie, :, :], B[ie, :, :],
					C[ie, :, :])

		# We check when the coefficients are no longer changing.
		# This can lead to differences between NODAL and MODAL solutions.
		# This could be resolved by evaluating at the quadrature points
		# and comparing the error between those values.
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


def predictor_elem_testing(solver, dt, W, U_pred):
	'''
	Calculates the predicted solution state for the ADER-DG method using a
	nonlinear solve of the weak form of the DG discretization in time.

	This function treats the source term implicitly. Appropriate for
	stiff scalar equations.

	Inputs:
	-------
		solver: solver object
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
	elem_helpers_st = solver.elem_helpers_st
	ader_helpers = solver.ader_helpers

	quad_wts = elem_helpers.quad_wts
	quad_wts_st = elem_helpers_st.quad_wts
	quad_pts_st = elem_helpers_st.quad_pts
	quad_pts = elem_helpers.quad_pts
	basis_val = elem_helpers.basis_val
	basis_val_st = elem_helpers_st.basis_val

	djac_elems = elem_helpers.djac_elems
	djac_elems_st = ader_helpers.djac_elems
	x_elems = elem_helpers.x_elems

	nelem = W.shape[0]
	nq_st = quad_wts_st.shape[0]
	nb_st = basis_val_st.shape[1]
	nq_t = elem_helpers_st.nq_tile_constant


	FTR = ader_helpers.FTR
	MM = ader_helpers.MM
	iMM_elems = ader_helpers.iMM_elems
	SMS_elems = ader_helpers.SMS_elems
	K = ader_helpers.K
	x_elems_ader = ader_helpers.x_elems
	vol_elems = elem_helpers.vol_elems

	# Evaluate spatial coeffs on spatial quadrature points.
	Wq = helpers.evaluate_state(W, basis_val, skip_interp=basis.skip_interp)
	Uq_guess = np.zeros([nelem, nq_st, ns])
	Uq_guess = np.tile(Wq, [1, Wq.shape[1], 1])
	# Uq_guess = 
	# W_bar = helpers.get_element_mean(Wq, quad_wts, djac_elems, vol_elems)

	# Build temporal array for space-time element
	t, elem_helpers_st.basis_time = ref_to_phys_time(
			mesh, solver.time, dt,
			quad_pts[:, -1:], elem_helpers_st.basis_time)

	W0, t0 = Wq.reshape(-1), solver.time

	# import code; code.interact(local=locals())
	def func(t, y, x):
		# Keep track of the number of times func is called
		tvals.append(t)

		# Evaluate the source term at the quadrature points
		Sq = np.zeros([U_pred.shape[0], x.shape[1], ns])
		y = y.reshape(Sq.shape)
		Sq = physics.eval_source_terms(y, x, t, Sq)

		return Sq.reshape(-1)

	# Initialize the integrator
	r = ode(func, jac=None)
	r.set_integrator('lsoda', atol=1e-14, rtol=1e-12)
	r.set_initial_value(W0, t0).set_f_params(x_elems)

	# Set constants for managing data and begin ODE integration loop
	i = t.shape[0]
	j = 1
	while r.successful() and j < t.shape[0]: 
		tvals = []

		value = r.integrate(r.t + (t[j] - r.t))

		Uq_guess[:,i:t.shape[0]*j+t.shape[0],:] = value.reshape([nelem, t.shape[0], ns])
		i+=t.shape[0]
		j+=1
		tvals = np.unique(tvals)

		print("len(tvals) =", len(tvals))
	# import code; code.interact(local=locals())
	# Get space-time average from initial guess
	U_bar = helpers.get_element_mean(Uq_guess, quad_wts_st, np.tile(djac_elems, [1, nq_t, 1])*dt/2., dt*vol_elems)

	# Project the guess at the space-time quadrature points to the 
	# state coefficient's initial guess
	L2_projection(mesh, iMM_elems, solver.basis_st, quad_pts_st,
			quad_wts_st, np.tile(djac_elems, [1, nq_t, 1]), Uq_guess, U_pred)

	# Only evaluate Jacobian for stiff sources
	temp_sources = physics.source_terms.copy()
	physics.source_terms = physics.implicit_sources.copy()

	# # Calculate the source term Jacobian using average state
	Sjac = np.zeros([nelem, 1, ns, ns])
	Sjac = physics.eval_source_term_jacobians(U_bar, x_elems, solver.time,
			Sjac)

	# # # Calculate iK
	Sjac = np.reshape(Sjac, [nelem, ns, ns])
	Kp = K - dt * np.einsum('jk, imn -> ijk', MM, Sjac)
	iK = np.linalg.inv(Kp)

	# # Testing out source term Jacobian on every coefficient
	# Sjac = np.zeros([U_pred.shape[0], U_pred.shape[1], ns, ns])
	# Sjac = physics.eval_source_term_jacobians(U_pred, x_elems, solver.time,
	# 		Sjac)	

	# # Test out calculating iK for all coefficients
	# Kp = K - dt * np.einsum('jk, ikmn -> ijk', MM, Sjac)
	# iK = np.linalg.inv(Kp)

	# Set all sources for source_coeffs calculation
	physics.source_terms = temp_sources.copy()

	# Calculate the source and flux coefficients with initial guess
	source_coeffs = solver.source_coefficients(dt, order, basis_st,
			U_pred)
	flux_coeffs = solver.flux_coefficients(dt, order, basis_st,
			U_pred)

	# Iterate using a discrete Picard nonlinear solve for the
	# updated space-time coefficients.
	# if solver.time > 295798.0:
	# 	import code; code.interact(local=locals())
	niter = 100
	for i in range(niter):

		U_pred_new = np.einsum('ijk, ikm -> ijm',iK,
				(np.einsum('jk, ijl -> ikl', MM, source_coeffs) -
				np.einsum('ijkl, ikml -> ijm', SMS_elems, flux_coeffs) +
				np.einsum('jk, ikm -> ijm', FTR, W) -
				np.einsum('jk, ijm -> ikm', MM, dt * Sjac * U_pred)))#\
				# np.einsum('ijmn, ijn -> ijn', Sjac, U_pred))))
		# if i >= 79:
		# 	import code; code.interact(local=locals())
		# We check when the coefficients are no longer changing.
		# This can lead to differences between NODAL and MODAL solutions.
		# This could be resolved by evaluating at the quadrature points
		# and comparing the error between those values.
		err = U_pred_new - U_pred
		if np.amin(np.abs(err)) < 1.e-14:
			U_pred = U_pred_new
			print('Predictor iterations', i)
			# import code; code.interact(local=locals())
			# if np.abs(W[0,0] - W[0,1]) > 1e-12:
			# if solver.time > 295798.0:
			# 	import code; code.interact(local=locals())
			break

		U_pred = U_pred_new

		source_coeffs = solver.source_coefficients(dt, order,
				basis_st, U_pred)
		flux_coeffs = solver.flux_coefficients(dt, order, basis_st,
				U_pred)

		Uq = helpers.evaluate_state(U_pred, basis_val_st)#, skip_interp=basis.skip_interp)
		U_bar = helpers.get_element_mean(Uq, quad_wts_st, np.tile(djac_elems, [1, nq_t, 1])*dt/2., dt*vol_elems)
			# import code; code.interact(local=locals())
		# # Only evaluate Jacobian for stiff sources
		# temp_sources = physics.source_terms.copy()
		# physics.source_terms = physics.implicit_sources.copy()

		# # Calculate the source term Jacobian using average state
		Sjac = np.zeros([U_pred.shape[0], 1, ns, ns])
		Sjac = physics.eval_source_term_jacobians(U_bar, x_elems, solver.time,
				Sjac)

		Sjac = np.reshape(Sjac, [U_pred.shape[0], ns, ns])
		Kp = K - dt * np.einsum('jk, imn -> ijk', MM, Sjac)
		iK = np.linalg.inv(Kp)
		# if solver.time > 295714.0:#295798.0:
		# 	print('i: ',i,"Sjac: ", Sjac,"Ubar: ", U_bar)
		# if i >= 79:
		# 	import code; code.interact(local=locals())	
		
		# TESTING
		# Sjac = physics.eval_source_term_jacobians(U_pred, x_elems, solver.time,
				# Sjac)	

		# Test out calculating iK for all coefficients
		# Kp = K - dt * np.einsum('jk, ikmn -> ijk', MM, Sjac)
		# iK = np.linalg.inv(Kp)

		# Set all sources for source_coeffs calculation
		# physics.source_terms = temp_sources.copy()

		if i == niter - 1:
			print('Sub-iterations not converging')
			raise ValueError

	return U_pred # [ne, nb_st, ns]

def predictor_elem_polyjac(solver, dt, W, U_pred):
	'''
	Calculates the predicted solution state for the ADER-DG method using a
	nonlinear solve of the weak form of the DG discretization in time.

	This function treats the source term implicitly. Appropriate for
	stiff scalar equations.

	Inputs:
	-------
		solver: solver object
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
	elem_helpers_st = solver.elem_helpers_st
	ader_helpers = solver.ader_helpers

	quad_wts = elem_helpers.quad_wts
	quad_wts_st = elem_helpers_st.quad_wts
	quad_pts_st = elem_helpers_st.quad_pts
	quad_pts = elem_helpers.quad_pts
	basis_val = elem_helpers.basis_val
	basis_val_st = elem_helpers_st.basis_val

	djac_elems = elem_helpers.djac_elems
	djac_elems_st = ader_helpers.djac_elems
	x_elems = elem_helpers.x_elems

	nelem = W.shape[0]
	nq_st = quad_wts_st.shape[0]
	nb_st = basis_val_st.shape[1]
	FTR = ader_helpers.FTR
	MM = ader_helpers.MM
	iMM_elems = ader_helpers.iMM_elems
	SMS_elems = ader_helpers.SMS_elems
	K = ader_helpers.K
	x_elems_ader = ader_helpers.x_elems
	vol_elems = elem_helpers.vol_elems

	# Evaluate spatial coeffs on spatial quadrature points.
	Wq = helpers.evaluate_state(W, basis_val, skip_interp=basis.skip_interp)
	Uq_guess = np.zeros_like(U_pred)
	# W_bar = helpers.get_element_mean(Wq, quad_wts, djac_elems, vol_elems)

	# Build temporal array for space-time element
	t, elem_helpers_st.basis_time = ref_to_phys_time(
			mesh, solver.time, dt,
			quad_pts[:, -1:], elem_helpers_st.basis_time)

	# Wq = np.tile(Wq, [1, Wq.shape[1], 1])
	W0, t0 = Wq.reshape(-1), solver.time

	# import code; code.interact(local=locals())
	def func(t, y, x):
		# Keep track of the number of times func is called
		tvals.append(t)

		# Evaluate the source term at the quadrature points
		Sq = np.zeros([U_pred.shape[0], x.shape[1], ns])
		y = y.reshape(Sq.shape)
		Sq = physics.eval_source_terms(y, x, t, Sq)

		return Sq.reshape(-1)

	# Initialize the integrator
	r = ode(func, jac=None)
	r.set_integrator('lsoda', atol=1e-14, rtol=1e-12)
	r.set_initial_value(W0, t0).set_f_params(x_elems)

	# Set constants for managing data and begin ODE integration loop
	i = 0
	j = 0
	while r.successful() and j < t.shape[0]: 
		tvals = []
		value = r.integrate(r.t + (t[j] - r.t))

		Uq_guess[:,i:t.shape[0]*j+t.shape[0],:] = value.reshape([nelem, t.shape[0], ns])
		i+=t.shape[0]
		j+=1
		tvals = np.unique(tvals)

		print("len(tvals) =", len(tvals))

	# Get space-time average from initial guess
	U_bar = helpers.get_element_mean(Uq_guess, quad_wts_st, djac_elems_st*dt/2., dt*vol_elems)

	# Project the guess at the space-time quadrature points to the 
	# state coefficient's initial guess
	L2_projection(mesh, iMM_elems, solver.basis_st, quad_pts_st,
			quad_wts_st, djac_elems_st, Uq_guess, U_pred)

	# Only evaluate Jacobian for stiff sources
	temp_sources = physics.source_terms.copy()
	physics.source_terms = physics.implicit_sources.copy()
	# if solver.time > 295830:#295298:
	# 	import code; code.interact(local=locals())
	# Evaluate the source term jacobian at the quadrature points
	Sqjac = np.zeros([nelem, nq_st, ns, ns])
	Sqjac = physics.eval_source_term_jacobians(Uq_guess, x_elems, solver.time,
			Sqjac)

	# Project the jacobian to the basis to be represented as polynomial coefficients
	Sjac = np.zeros([nelem, nb_st, ns, ns])
	L2_dSdU_projection(mesh, iMM_elems, solver.basis_st, quad_pts_st,
		quad_wts_st, djac_elems_st, Sqjac, Sjac)

	# Calculate iK
	Kp = K - dt * np.einsum('jk, ikmn -> ijk', MM, Sjac)
	iK = np.linalg.inv(Kp)

	# Set all sources for source_coeffs calculation
	physics.source_terms = temp_sources.copy()

	# Calculate the source and flux coefficients with initial guess
	source_coeffs = solver.source_coefficients(dt, order, basis_st,
			U_pred)
	flux_coeffs = solver.flux_coefficients(dt, order, basis_st,
			U_pred)

	niter = 100
	for i in range(niter):

		# if solver.time > 295830:#295298:
		# 	import code; code.interact(local=locals())
		U_pred_new = np.einsum('ijk, ikm -> ijm',iK,
				(np.einsum('jk, ijl -> ikl', MM, source_coeffs) -
				np.einsum('ijkl, ikml -> ijm', SMS_elems, flux_coeffs) +
				np.einsum('jk, ikm -> ijm', FTR, W) -
				np.einsum('jk, ijm -> ikm', MM, dt * \
				np.einsum('ijmn, ijn -> ijn', Sjac, U_pred))))
		# import code; code.interact(local=locals())
		# We check when the coefficients are no longer changing.
		# This can lead to differences between NODAL and MODAL solutions.
		# This could be resolved by evaluating at the quadrature points
		# and comparing the error between those values.
		err = U_pred_new - U_pred
		if np.amax(np.abs(err)) < 1.e-4:
			U_pred = U_pred_new
			print('Predictor iterations', i)
			break

		U_pred = U_pred_new

		source_coeffs = solver.source_coefficients(dt, order,
				basis_st, U_pred)
		flux_coeffs = solver.flux_coefficients(dt, order, basis_st,
				U_pred)

		# # Interpolate to quadrature points
		# Uq = helpers.evaluate_state(U_pred, basis_val_st)#, skip_interp=basis.skip_interp)

		# # Only evaluate Jacobian for stiff sources
		# temp_sources = physics.source_terms.copy()
		# physics.source_terms = physics.implicit_sources.copy()

		# # Calculate the source term Jacobian using Uq
		# Sqjac = physics.eval_source_term_jacobians(Uq, x_elems, solver.time,
		# 		Sjac)
		# L2_dSdU_projection(mesh, iMM_elems, solver.basis_st, quad_pts_st,
		# 	quad_wts_st, djac_elems_st, Sqjac, Sjac)

		# # Calculate iK
		# Kp = K - dt * np.einsum('jk, ikmn -> ijk', MM, Sjac)
		# iK = np.linalg.inv(Kp)

		# # if solver.time > 295830:#295298:
		# # 	import code; code.interact(local=locals())
		# # Set all sources for source_coeffs calculation
		# physics.source_terms = temp_sources.copy()

		if i == niter - 1: 
			print('Sub-iterations not converging')
			raise ValueError

	return U_pred # [ne, nb_st, ns]


def predictor_elem_sylvester_guess(solver, dt, W, U_pred):
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
	elem_helpers_st = solver.elem_helpers_st
	ader_helpers = solver.ader_helpers

	quad_wts = elem_helpers.quad_wts
	quad_pts = elem_helpers.quad_pts
	basis_val = elem_helpers.basis_val
	basis_val_st = elem_helpers_st.basis_val
	djac_elems = elem_helpers.djac_elems
	x_elems = elem_helpers.x_elems

	FTR = ader_helpers.FTR
	iMM = ader_helpers.iMM
	iMM_elems = ader_helpers.iMM_elems
	SMS_elems = ader_helpers.SMS_elems
	K = ader_helpers.K

	nelem = W.shape[0]
	quad_pts_st = elem_helpers_st.quad_pts
	quad_wts_st = elem_helpers_st.quad_wts
	nq_st = quad_wts_st.shape[0]

	nq_t = elem_helpers_st.nq_tile_constant
	vol_elems = elem_helpers.vol_elems

	# Evaluate spatial coeffs on spatial quadrature points.
	Wq = helpers.evaluate_state(W, basis_val, skip_interp=basis.skip_interp)

	#PRINTING 
	if solver.time > 295700.0 and solver.time < 296200.:
		print_Wq = open('print_Wq.txt', 'a')
		s = str(solver.time)
		s1 = str(Wq[0,1,0])
		print_Wq.write(s)
		print_Wq.write(' , ')
		print_Wq.write(s1)
		print_Wq.write('\n')
		print_Wq.close()


	Uq_guess = np.zeros([nelem, nq_st, ns])
	Uq_guess = np.tile(Wq, [1, Wq.shape[1], 1])
	W_bar = helpers.get_element_mean(Wq, quad_wts, djac_elems, vol_elems)

	# Build temporal array for space-time element
	t, elem_helpers_st.basis_time = ref_to_phys_time(
			mesh, solver.time, dt,
			quad_pts[:, -1:], elem_helpers_st.basis_time)

	tphys, elem_helpers_st.basis_time = ref_to_phys_time(
			mesh, solver.time, dt,
			ader_helpers.x_elems[0,0:2,:], elem_helpers_st.basis_time)

	W0, t0 = Wq.reshape(-1), solver.time

	def func(t, y, x):
		# Keep track of the number of times func is called
		tvals.append(t)

		# Evaluate the source term at the quadrature points
		Sq = np.zeros([U_pred.shape[0], x.shape[1], ns])
		y = y.reshape(Sq.shape)
		Sq = physics.eval_source_terms(y, x, t, Sq)

		return Sq.reshape(-1)

	# Initialize the integrator
	r = ode(func, jac=None)
	r.set_integrator('lsoda', atol=1e-14, rtol=1e-12)
	r.set_initial_value(W0, t0).set_f_params(x_elems)

	# Set constants for managing data and begin ODE integration loop
	i = 0#t.shape[0]
	j = 0#1
	while r.successful() and j < t.shape[0]: 
		tvals = []

		value = r.integrate(r.t + (t[j] - r.t))

		Uq_guess[:,i:t.shape[0]*j+t.shape[0],:] = value.reshape([nelem, t.shape[0], ns])
		i+=t.shape[0]
		j+=1
		tvals = np.unique(tvals)

		print("len(tvals) =", len(tvals))

	# #PRINTING 
	if solver.time > 295700.0 and solver.time < 296200.:
		print_Uq = open('print_Uq_guess.txt', 'a')
		for it in range(t.shape[0]):
			s = str(t[it,0])
			s1 = str(Uq_guess[0,it*t.shape[0],0])
			print_Uq.write(s)
			print_Uq.write(' , ')
			print_Uq.write(s1)
			print_Uq.write('\n')
	# 	print_Uq.close()


	# Get space-time average from initial guess
	# U_bar = helpers.get_element_mean(Uq_guess, quad_wts_st, np.tile(djac_elems, [1, nq_t, 1])*dt/2., dt*vol_elems)

	# Project the guess at the space-time quadrature points to the 
	# state coefficient's initial guess
	L2_projection(mesh, iMM_elems, solver.basis_st, quad_pts_st,
			quad_wts_st, np.tile(djac_elems, [1, nq_t, 1]), Uq_guess, U_pred)

	if solver.time > 295700.0 and solver.time < 296200.:
		print_Uq = open('print_Upred_guess.txt', 'a')
		for it in range(tphys.shape[0]):
			s = str(tphys[it,0])
			s1 = str(U_pred[0,it*t.shape[0],0])
			print_Uq.write(s)
			print_Uq.write(' , ')
			print_Uq.write(s1)
			print_Uq.write('\n')
		print_Uq.close()

	# Only evaluate Jacobian for stiff sources
	temp_sources = physics.source_terms.copy()
	physics.source_terms = physics.implicit_sources.copy()

	# # Calculate the source term Jacobian using average state
	Sjac = np.zeros([nelem, 1, ns, ns])
	Sjac = physics.eval_source_term_jacobians(W_bar, x_elems, solver.time,
			Sjac)
	Sjac = np.reshape(Sjac, [nelem, ns, ns])
	#PRINTING 
	# if solver.time > 295500.0:
	# 	print_Wq = open('print_jac.txt', 'a')
	# 	s = str(solver.time)
	# 	s1 = str(Sjac[0,0,0])
	# 	print_Wq.write(s)
	# 	print_Wq.write(' , ')
	# 	print_Wq.write(s1)
	# 	print_Wq.write('\n')
	# 	print_Wq.close()


	# Set all sources for source_coeffs calculation
	physics.source_terms = temp_sources.copy()

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

		for ie in range(U_pred.shape[0]):
			U_pred_new[ie, :, :] = solve_sylvester(A[ie, :, :], B[ie, :, :],
					C[ie, :, :])

		# We check when the coefficients are no longer changing.
		# This can lead to differences between NODAL and MODAL solutions.
		# This could be resolved by evaluating at the quadrature points
		# and comparing the error between those values.
		err = U_pred_new - U_pred
		if np.amax(np.abs(err)) < 1.e-8:
			U_pred = U_pred_new
			print("Predictor iterations: ", i)
			break

		U_pred = U_pred_new

		source_coeffs = solver.source_coefficients(dt, order,
				basis_st, U_pred)
		flux_coeffs = solver.flux_coefficients(dt, order, basis_st,
				U_pred)

		# # Only evaluate Jacobian for stiff sources
		temp_sources = physics.source_terms.copy()
		physics.source_terms = physics.implicit_sources.copy()

		Uq = helpers.evaluate_state(U_pred, basis_val_st)
		U_bar = helpers.get_element_mean(Uq, quad_wts_st, 
				np.tile(djac_elems, [1, nq_t, 1])*dt/2., dt*vol_elems)

		# # Calculate the source term Jacobian using average state
		Sjac = np.zeros([U_pred.shape[0], 1, ns, ns])
		Sjac = physics.eval_source_term_jacobians(U_bar, x_elems, solver.time,
				Sjac)
		Sjac = np.reshape(Sjac, [nelem, ns, ns])

		# Set all sources for source_coeffs calculation
		physics.source_terms = temp_sources.copy()

		if i == niter - 1:
			print('Sub-iterations not converging')
			raise ValueError

		#PRINTING 
		# Uq = helpers.evaluate_state(U_pred, basis_val_st)
	# if solver.time > 295500.0:
	# 	print('here')
		# Uq = helpers.evaluate_state(U_pred, basis_val_st)
		# import code; code.interact(local=locals())
		# print_Uq = open('print_Upred.txt', 'a')
		# for it in range(t.shape[0]):
		# 	s = str(t[it,0])
		# 	s1 = str(U_pred[0,it*t.shape[0],0])
		# 	print_Uq.write(s)
		# 	print_Uq.write(' , ')
		# 	print_Uq.write(s1)
		# 	print_Uq.write('\n')
	# 	print_Uq.close()

	if solver.time > 295700.0 and solver.time < 296200.:
		print_Uq = open('print_Upred.txt', 'a')
		for it in range(tphys.shape[0]):
			s = str(tphys[it,0])
			s1 = str(U_pred[0,it*tphys.shape[0],0])
			print_Uq.write(s)
			print_Uq.write(' , ')
			print_Uq.write(s1)
			print_Uq.write('\n')
		print_Uq.close()
	if solver.time > 295700.0 and solver.time < 296200.:
		Uq = helpers.evaluate_state(U_pred, basis_val_st)
		print_Uq = open('print_Uq.txt', 'a')
		for it in range(t.shape[0]):
			s = str(t[it,0])
			s1 = str(U_pred[0,it*t.shape[0],0])
			print_Uq.write(s)
			print_Uq.write(' , ')
			print_Uq.write(s1)
			print_Uq.write('\n')
		print_Uq.close()

		# 	print_Uq = open('print_jac_update.txt', 'a')
		# 	for it in range(t.shape[0]):
		# 		s = str(solver.time)
		# 		s1 = str(Sjac[0,0,0])
		# 		print_Uq.write(s)
		# 		print_Uq.write(' , ')
		# 		print_Uq.write(s1)
		# 		print_Uq.write('\n')
		# 	print_Uq.close()
	return U_pred # [ne, nb_st, ns]

def predictor_elem_explicit_split(solver, dt, W, U_pred):
	'''
	Calculates the predicted solution state for the ADER-DG method using a
	nonlinear solve of the weak form of the DG discretization in time.

	This function treats the source term explicitly. Appropriate for
	non-stiff systems.

	Inputs:
	-------
		solver: solver object
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

	# Only evaluate non-stiff sources first
	temp_sources = physics.source_terms.copy()
	physics.source_terms = physics.explicit_sources.copy()
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
				np.einsum('jk, ikl -> ijl', MM, source_coeffs) -
				np.einsum('ijkl, ikml -> ijm', SMS_elems, flux_coeffs) +
				np.einsum('jk, ikm -> ijm', FTR, W))

		# We check when the coefficients are no longer changing.
		# This can lead to differences between NODAL and MODAL solutions.
		# This could be resolved by evaluating at the quadrature points
		# and comparing the error between those values.
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


	# Set up implicit portion
	elem_helpers_st = solver.elem_helpers_st
	quad_wts = elem_helpers.quad_wts
	quad_pts = elem_helpers.quad_pts
	quad_pts_st = elem_helpers_st.quad_pts
	quad_wts_st = elem_helpers_st.quad_wts

	basis_val = elem_helpers.basis_val
	basis_val_st = elem_helpers_st.basis_val
	djac_elems = elem_helpers.djac_elems
	x_elems = elem_helpers.x_elems
	nq_t = elem_helpers_st.nq_tile_constant

	iMM_elems = ader_helpers.iMM_elems

	physics.source_terms = physics.implicit_sources.copy()

	
	# Build temporal array for space-time element
	t, elem_helpers_st.basis_time = ref_to_phys_time(
			mesh, solver.time, dt,
			quad_pts[:, -1:], elem_helpers_st.basis_time)

	def func(t, y, x):
		# Keep track of the number of times func is called
		tvals.append(t)

		# Evaluate the source term at the quadrature points
		Sq = np.zeros([U_pred.shape[0], x.shape[1], ns])
		y = y.reshape(Sq.shape)
		Sq = physics.eval_source_terms(y, x, t, Sq)

		return Sq.reshape(-1)

	Uq = helpers.evaluate_state(U_pred, basis_val_st)
	U0, t0 = Uq.reshape(-1), solver.time

	# Initialize the integrator
	r = ode(func, jac=None)
	r.set_integrator('lsoda', atol=1e-14, rtol=1e-12)
	r.set_initial_value(U0, t0).set_f_params(x_elems)
	


	# Project the guess at the space-time quadrature points to the 
	# state coefficient's initial guess
	L2_projection(mesh, iMM_elems, solver.basis_st, quad_pts_st,
			quad_wts_st, np.tile(djac_elems, [1, nq_t, 1]), value, U_pred)

	# Put sources back
	physics.source_terms = temp_sources.copy()
	# import code; code.interact(local=locals())

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
	U[:, :, :] = np.einsum('ijk, ikl -> ijl', iMM, rhs)


def L2_dSdU_projection(mesh, iMM, basis, quad_pts, quad_wts, djac, f, U):
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

	f_qwts_djac = np.einsum('ijkl, jm, ijm -> ijkl', f, quad_wts, djac)
	rhs = np.einsum('jk, ijlm -> iklm', basis.basis_val, f_qwts_djac)
			# [ne, nb_st, ns]
	U[:, :, :, :] = np.einsum('ijk, ijlm -> iklm', iMM, rhs)

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
