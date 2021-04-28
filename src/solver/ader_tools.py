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
		fcn: the name of the function chosen 
			for the calculate_predictor_elem
	'''
	if source_treatment == "Explicit":
		fcn = predictor_elem_explicit
	elif source_treatment == "Implicit":
		fcn = predictor_elem_implicit
	elif source_treatment == "SylImp":
		fcn = predictor_elem_sylvester
	elif source_treatment == "StiffImplicit":
		fcn = predictor_elem_ode_guess
	else:
		raise NotImplementedError

	return fcn


def set_predictor_guess(predictor_guess):
	'''
	This method sets the appropriate predictor guess function for the 
	ADER-DG scheme given the input deck parameters

	Inputs:
	-------
		predictor_guess: string from input deck to determine how the 
			prediction to the nonlinear equation is conducted

	Outputs:
	--------
		fcn: the name of the function chosen for get_spacetime_guess
	'''
	if predictor_guess == "Average":
		fcn = average_spacetime_guess
	elif predictor_guess == "Zeros": 
		fcn = zeros_spacetime_guess
	elif predictor_guess == "ODEGuess":
		fcn = spacetime_odeguess
	else: 
		raise NotImplementedError

	return fcn


def set_recalculate_jac(recalculate_jacobian):
	'''
	This method sets the jacobian recalculation function for the
	nonlinear subiterations in the predictor step of the ADERDG 
	scheme. If True we recalculate the source term jacobians at
	each subiteration, if false we use the source term jacobian
	from the first iteration. Recalculating has shown some 
	improvement of convergence for very stiff cases.

	Inputs:
	-------
		recalculate_jacobian: Boolean variable

	Outputs:
	--------
		fcn: the name of the function chosen for recalculate_jacobian
	'''
	if recalculate_jacobian:
		fcn = recalculate_jacobian_on
	else:
		fcn = recalculate_jacobian_off

	return fcn

def recalculate_jacobian_on(solver, U_pred, dt, Sjac=None):
	# Unpack
	physics = solver.physics
	elem_helpers = solver.elem_helpers
	elem_helpers_st = solver.elem_helpers_st

	ns = physics.NUM_STATE_VARS
	nelem = U_pred.shape[0]
	quad_wts_st = elem_helpers_st.quad_wts
	basis_val_st = elem_helpers_st.basis_val
	nq_t = elem_helpers_st.nq_tile_constant

	x_elems = elem_helpers.x_elems
	vol_elems = elem_helpers.vol_elems
	djac_elems = elem_helpers.djac_elems

	# Only evaluate Jacobian for stiff sources
	temp_sources = physics.source_terms.copy()
	physics.source_terms = physics.implicit_sources.copy()

	Uq = helpers.evaluate_state(U_pred, basis_val_st)

	U_bar = helpers.get_element_mean(Uq, quad_wts_st, 
			np.tile(djac_elems, [1, nq_t, 1])*dt/2., dt*vol_elems)

	# Calculate the source term Jacobian using average state
	Sjac = Sjac.reshape([U_pred.shape[0], 1, ns, ns])
	Sjac[:] = 0.
	# Sjac = np.zeros([U_pred.shape[0], 1, ns, ns])
	Sjac = physics.eval_source_term_jacobians(U_bar, x_elems, solver.time,
			Sjac)
	Sjac = np.reshape(Sjac, [nelem, ns, ns])

	# Set all sources for source_coeffs calculation
	physics.source_terms = temp_sources.copy()


def recalculate_jacobian_off(solver, U_pred, dt, Sjac=None):
	pass

def zeros_spacetime_guess(solver, W, U_pred, dt=None):
	'''
	This method sets the space-time guess to zeros.

	Inputs:
	-------
		solver: solver object
		W: spatial polynomial solution [ne, nb, ns]
		U_pred: space-time polynomial coefficients [ne, nb_st, ns]

	Outputs:
	--------
		U_pred: Space-time predicted polynomial coefficients [ne, nb_st, ns]
	'''
	U_pred[:] = 0.0

	return U_pred, None


def average_spacetime_guess(solver, W, U_pred, dt=None):
	'''
	This method calculates the average space-time guess for the ADERDG
	predictor step. (This is the default approach for the guess)

	Inputs:
	-------
		solver: solver object
		W: spatial polynomial solution [ne, nb, ns]
		U_pred: space-time polynomial coefficients [ne, nb_st, ns]
		
	Outputs:
	--------
		U_pred: Space-time predicted polynomial coefficients [ne, nb_st, ns]
	'''
	# Unpack
	physics = solver.physics
	basis = solver.basis
	elem_helpers = solver.elem_helpers

	quad_wts = elem_helpers.quad_wts
	basis_val = elem_helpers.basis_val
	djac_elems = elem_helpers.djac_elems
	vol_elems = elem_helpers.vol_elems

	# Calculate the average state for each element in spatial coordinates
	Wq = helpers.evaluate_state(W, basis_val, skip_interp=basis.skip_interp)
	W_bar = helpers.get_element_mean(Wq, quad_wts, djac_elems, vol_elems)

	U_pred[:] = W_bar

	return U_pred, W_bar # [ne, nb_st, ns]


def spacetime_odeguess(solver, W, U_pred, dt=None):
	'''
	This method sets the space-time guess to the predictor step in the 
	ADER-DG scheme by using a built-in ODE solver from scipy. We solve 
	the stationary ODE in time and use the result to construct our guess 
	to the non-linear solver

	NOTE: Current implementation only supports ODEs

	Inputs:
	-------
		solver: solver object
		W: spatial polynomial solution [ne, nb, ns]
		U_pred: space-time polynomial coefficients [ne, nb_st, ns]

	Outputs:
	--------
		U_pred: Space-time predicted polynomial coefficients [ne, nb_st, ns]
		U_bar: Space-time average value [ne, 1, ns]
	'''
	# Unpack
	physics = solver.physics

	ns = physics.NUM_STATE_VARS
	mesh = solver.mesh

	basis = solver.basis
	basis_st = solver.basis_st

	elem_helpers = solver.elem_helpers
	elem_helpers_st = solver.elem_helpers_st
	ader_helpers = solver.ader_helpers
	iMM_elems = ader_helpers.iMM_elems

	quad_wts = elem_helpers.quad_wts
	quad_pts = elem_helpers.quad_pts
	basis_val = elem_helpers.basis_val
	basis_val_st = elem_helpers_st.basis_val
	djac_elems = elem_helpers.djac_elems
	x_elems = elem_helpers.x_elems

	nelem = W.shape[0]
	quad_pts_st = elem_helpers_st.quad_pts
	quad_wts_st = elem_helpers_st.quad_wts
	nq_st = quad_wts_st.shape[0]
	nq_t = elem_helpers_st.nq_tile_constant
	vol_elems = elem_helpers.vol_elems

	# Evaluate spatial coeffs on spatial quadrature points
	Wq = helpers.evaluate_state(W, basis_val, skip_interp=basis.skip_interp)

	# Allocate memory for the guess at the quadrature points
	Uq_guess = np.zeros([nelem, nq_st, ns])
	Uq_guess = np.tile(Wq, [1, Wq.shape[1], 1])

	# Build ref temporal array for space-time element
	t, elem_helpers_st.basis_time = ref_to_phys_time(
			mesh, solver.time, dt,
			quad_pts[:, -1:], elem_helpers_st.basis_time)

	# Build phys time array for space-time element
	tphys, elem_helpers_st.basis_time = ref_to_phys_time(
			mesh, solver.time, dt,
			ader_helpers.x_elems[0,0:2,:], elem_helpers_st.basis_time)

	W0, t0 = Wq.reshape(-1), solver.time

	def func(t, y, x, Sq_exp):
		# Keep track of the number of times func is called
		tvals.append(t)

		# Evaluate the source term at the quadrature points
		Sq = np.zeros([U_pred.shape[0], x.shape[1], ns])
		y = y.reshape(Sq.shape)
		Sq = Sq_exp + physics.eval_source_terms(y, x, t, Sq)

		# NOTE: This will eventually need to include the flux evaluation
		return Sq.reshape(-1)


	# Evaluate source terms to be taken explicitly
	temp_sources = physics.source_terms.copy()
	physics.source_terms = physics.explicit_sources.copy()

	Sq_exp = np.zeros([U_pred.shape[0], x_elems.shape[1], ns])
	Sq_exp = physics.eval_source_terms(Wq, x_elems, t, Sq_exp)

	# Set implicit sources only for stiff ODE evaluation
	physics.source_terms = physics.implicit_sources.copy()

	# Initialize the integrator
	r = ode(func, jac=None)
	r.set_integrator('lsoda', atol=1e-14, rtol=1e-12)
	r.set_initial_value(W0, t0).set_f_params(x_elems, Sq_exp)

	# Set constants for managing data and begin ODE integration loop
	# Note: These commented points after i, j defs mods for 
	# gauss-lobatto quadrature
	i = 0 #t.shape[0]
	j = 0 #1
	# Run the ODEsolver guess

	while r.successful() and j < t.shape[0]: 
		# Length of tvals represents number of ODE interations per
		# timestep between two quadrature points in time
		tvals = []

		# Runs the integrator
		value = r.integrate(r.t + (t[j] - r.t))

		# Populate the data into the guess
		Uq_guess[:,i:t.shape[0]*j+t.shape[0],:] = \
				value.reshape([nelem, t.shape[0], ns])

		i+=t.shape[0]
		j+=1

		tvals = np.unique(tvals)
		solver.count_evaluations += len(tvals)
		# Prints the number of ODE iterations
		print("len(tvals) =", len(tvals))

	physics.source_terms = temp_sources

	# Get space-time average from initial guess
	U_bar = helpers.get_element_mean(Uq_guess, quad_wts_st, 
			np.tile(djac_elems, [1, nq_t, 1])*dt/2., dt*vol_elems)

	# Project the guess at the space-time quadrature points to the 
	# state coefficient's initial guess
	L2_projection(mesh, iMM_elems, solver.basis_st, quad_pts_st,
			quad_wts_st, np.tile(djac_elems, [1, nq_t, 1]), Uq_guess, U_pred)

	return U_pred, U_bar


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
	threshold = solver.params["PredictorThreshold"]
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

	# Initialize space-time coefficients
	U_pred, U_bar = solver.get_spacetime_guess(solver, W, U_pred, dt=dt)

	# Calculate the source and flux coefficients with initial guess
	source_coeffs = solver.source_coefficients(dt, order, basis_st,
			U_pred)
	flux_coeffs = solver.flux_coefficients(dt, order, basis_st,
			U_pred)

	# Iterate using a discrete Picard nonlinear solve for the
	# updated space-time coefficients.
	niter = 1000
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
		if np.amax(np.abs(err)) < threshold:
			U_pred = U_pred_new
			print("Predictor iterations: ", i)
			break

		U_pred = np.copy(U_pred_new)

		source_coeffs = solver.source_coefficients(dt, order,
				basis_st, U_pred)
		flux_coeffs = solver.flux_coefficients(dt, order, basis_st,
				U_pred)

		if i == niter - 1:
			print('Sub-iterations not converging', np.amax(np.abs(err)))

	return U_pred # [ne, nb_st, ns]


def predictor_elem_implicit(solver, dt, W, U_pred):
	'''
	Calculates the predicted solution state for the ADER-DG method using a
	nonlinear solve of the weak form of the DG discretization in time.

	This function treats the source term implicitly. Appropriate for
	stiff scalar equations.

	Note: Currently not used -> Sylvestor approach supercedes this one

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
	threshold = solver.params["PredictorThreshold"]
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

	# Initialize space-time coefficients
	U_pred, U_bar = solver.get_spacetime_guess(solver, W, U_pred, dt=dt)

	res = solver.stepper.res

	# Only evaluate Jacobian for stiff sources
	temp_sources = physics.source_terms.copy()
	physics.source_terms = physics.implicit_sources.copy()

	# Calculate the source term Jacobian using average state
	Sjac = np.zeros([U_pred.shape[0], 1, ns, ns])
	Sjac = physics.eval_source_term_jacobians(U_bar, x_elems, solver.time,
			Sjac)

	Sjac = np.reshape(Sjac, [U_pred.shape[0], ns, ns])
	Kp = K - dt * np.einsum('jk, imn -> ijk', MM, Sjac)

	iK = np.linalg.inv(Kp)

	physics.source_terms = temp_sources.copy()
	# Calculate the source and flux coefficients with initial guess
	source_coeffs = solver.source_coefficients(dt, order, basis_st,
			U_pred)
	flux_coeffs = solver.flux_coefficients(dt, order, basis_st,
			U_pred)
	# Iterate using a discrete Picard nonlinear solve for the
	# updated space-time coefficients.
	niter = 1000
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
		if np.amax(np.abs(err)) < threshold:
			U_pred = U_pred_new
			print("Predictor iterations: ", i)
			break

		U_pred = np.copy(U_pred_new)

		source_coeffs = solver.source_coefficients(dt, order,
				basis_st, U_pred)
		flux_coeffs = solver.flux_coefficients(dt, order, basis_st,
				U_pred)
		
		if i == niter - 1:
			print('Sub-iterations not converging', np.amax(np.abs(err)))

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
	threshold = solver.params["PredictorThreshold"]
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

	# Initialize space-time coefficients
	U_pred, U_bar = solver.get_spacetime_guess(solver, W, U_pred, dt=dt)

	# Get physical average for testing purposes
	vol_elems = elem_helpers.vol_elems
	Wq = helpers.evaluate_state(W, basis_val, skip_interp=basis.skip_interp)
	W_bar = helpers.get_element_mean(Wq, quad_wts, djac_elems, vol_elems)
	

	# Only evaluate Jacobian for stiff sources
	temp_sources = physics.source_terms.copy()
	physics.source_terms = physics.implicit_sources.copy()

	# Calculate the source term Jacobian using average state
	Sjac = np.zeros([U_pred.shape[0], 1, ns, ns])
	Sjac = physics.eval_source_term_jacobians(W_bar, x_elems, solver.time,
			Sjac)
	Sjac = Sjac[:, 0, :, :]

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
	# Update: We now transform AX+XB=C into KX=C using kronecker
	# products.
	niter = 10000

	A = np.matmul(iMM,K)

	U_pred_new = np.zeros_like(U_pred)

	for i in range(niter):

		solver.count_nonlinear_iterations+=1
		
		B = -1.0*dt*Sjac.transpose(0,2,1)

		Q = np.einsum('jk, ikm -> ijm', FTR, W) - np.einsum(
				'ijkl, ikml -> ijm', SMS_elems, flux_coeffs)

		C = source_coeffs - dt*np.matmul(U_pred[:],
				Sjac[:].transpose(0,2,1)) + \
				np.einsum('jk, ikl -> ijl',iMM, Q)


		# Build identity matrices for kronecker procucts
		I2 = np.eye(A.shape[1])
		I1 = np.eye(B.shape[1])

		for ie in range(U_pred.shape[0]):

			# Conduct kronecker products to make system Ax=b
			kronecker = np.kron(I1, A) + np.kron(B[ie, :, :], I2)

			# import code; code.interact(local=locals())
			# U_pred_hold = np.linalg.solve(kronecker, C[ie, :, :].reshape([27,1]))
			# U_pred_new[ie, :, :] = U_pred_hold.reshape(U_pred.shape[1], U_pred.shape[2])

			# import code; code.interact(local=locals())
			# Note: Previous implementaion used sylvester solve directly.
			# This still requires further testing to determine which is 
			# more efficient.
			U_pred_new[ie, :, :] = solve_sylvester(A, B[ie, :, :],
					C[ie, :, :])
			# import code; code.interact(local=locals())
		# We check when the coefficients are no longer changing.
		# This can lead to differences between NODAL and MODAL solutions.
		# This could be resolved by evaluating at the quadrature points
		# and comparing the error between those values.
		err = U_pred_new - U_pred
		# if solver.time>=295900:
		# import code; code.interact(local=locals())

		if np.amax(np.abs(err)) < threshold:
			print("Predictor iterations: ", i)
			U_pred = np.copy(U_pred_new)
			break

		U_pred = np.copy(U_pred_new)

		source_coeffs = solver.source_coefficients(dt, order,
				basis_st, U_pred)
		flux_coeffs = solver.flux_coefficients(dt, order, basis_st,
				U_pred)

		# Recalculate jacobian for subiterations (Default is OFF)
		solver.recalculate_jacobian(solver, U_pred, dt, Sjac)

		if i == niter - 1:
			print('Sub-iterations not converging', np.amax(np.abs(err)))

	return U_pred # [ne, nb_st, ns]


def predictor_elem_ode_guess(solver, dt, W, U_pred):
	'''
	Calculates the predicted solution state for the ADER-DG method using a
	nonlinear solve of the weak form of the DG discretization in time.

	This function applies the source term implicitly. Appropriate for
	stiff systems of equations. The implicit solve utilizes the Sylvester
	equation of the form:

		AX + XB = C

	This is a built-in function via the scipy.linalg library.

	In addition, we use a robust version of the guess to the ODE solve
	where we use an LSODA package to create the first guess to the non-linear
	solver. This often leads to minimal iterations from the non-linear solver 
	and is better suited for very stiff systems.

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

	# Evaluate spatial coeffs on spatial quadrature points
	Wq = helpers.evaluate_state(W, basis_val, skip_interp=basis.skip_interp)

	# Allocate memory for the guess at the quadrature points
	Uq_guess = np.zeros([nelem, nq_st, ns])
	Uq_guess = np.tile(Wq, [1, Wq.shape[1], 1])

	# Evaluate the spacial average from the quadrature points
	W_bar = helpers.get_element_mean(Wq, quad_wts, djac_elems, vol_elems)

	# Build ref temporal array for space-time element
	t, elem_helpers_st.basis_time = ref_to_phys_time(
			mesh, solver.time, dt,
			quad_pts[:, -1:], elem_helpers_st.basis_time)

	# Build phys time array for space-time element
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
	# Note: These commented points after i, j defs mods for gauss-lobatto quadrature
	i = 0 #t.shape[0]
	j = 0 #1

	# Run the ODEsolver guess
	while r.successful() and j < t.shape[0]: 
		# Length of tvals represents number of ODE interations per
		# timestep between two quadrature points in time
		tvals = []

		# Runs the integrator
		value = r.integrate(r.t + (t[j] - r.t))

		# Populate the data into the guess
		Uq_guess[:,i:t.shape[0]*j+t.shape[0],:] = \
				value.reshape([nelem, t.shape[0], ns])

		i+=t.shape[0]
		j+=1
		tvals = np.unique(tvals)

		# Prints the number of ODE iterations
		print("len(tvals) =", len(tvals))

	# Get space-time average from initial guess
	U_bar = helpers.get_element_mean(Uq_guess, quad_wts_st, 
			np.tile(djac_elems, [1, nq_t, 1])*dt/2., dt*vol_elems)

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
	Sjac = np.reshape(Sjac, [nelem, ns, ns])

	# Set all sources for source_coeffs calculation
	physics.source_terms = temp_sources.copy()

	# Calculate the source and flux coefficients with initial guess
	source_coeffs = solver.source_coefficients(dt, order, basis_st,
			U_pred)
	flux_coeffs = solver.flux_coefficients(dt, order, basis_st,
			U_pred)

	A = np.matmul(iMM,K)/dt
	# Iterate using a nonlinear Sylvester solver for the
	# updated space-time coefficients. Solves for X in the form:
	# 	AX + XB = C
	niter = 1000


	U_pred_new = np.zeros_like(U_pred)
	for i in range(niter):

		B = -1.0*Sjac.transpose(0,2,1)

		Q = np.einsum('jk, ikm -> ijm', FTR, W) - np.einsum(
				'ijkl, ikml -> ijm', SMS_elems, flux_coeffs)

		C = source_coeffs/dt - np.matmul(U_pred[:],
				Sjac[:].transpose(0,2,1)) + \
				np.einsum('jk, ikl -> ijl',iMM, Q) / dt

		for ie in range(U_pred.shape[0]):
			U_pred_new[ie, :, :] = solve_sylvester(A, B[ie, :, :],
					C[ie, :, :])

		# We check when the coefficients are no longer changing.
		# This can lead to differences between NODAL and MODAL solutions.
		# This could be resolved by evaluating at the quadrature points
		# and comparing the error between those values.
		err = U_pred_new - U_pred
		if np.amax(np.abs(err)) < 1.e-15:
			U_pred = U_pred_new
			print("Predictor iterations: ", i)
			break

		U_pred = np.copy(U_pred_new)

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


		# Recalculate jacobian for subiterations (Default is OFF)
		# solver.recalculate_jacobian(solver, U_pred, dt, Sjac)
		if i == niter - 1:
			print('Sub-iterations not converging', np.amax(np.abs(err)))


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
