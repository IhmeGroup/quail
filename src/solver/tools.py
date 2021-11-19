# ------------------------------------------------------------------------ #
#
#       File : src/solver/tools.py
#
#       Contains additional methods (tools) for the DG solver class
#
# ------------------------------------------------------------------------ #
import numpy as np
import sys

import general
import numerics.basis.tools as basis_tools
import numerics.helpers.helpers as helpers


def calculate_inviscid_flux_volume_integral(solver, elem_helpers, Fq):
	'''
	Calculates the inviscid flux volume integral for the DG scheme

	Inputs:
	-------
		solver: solver object
		elem_helpers: helpers defined in ElemHelpers
		Fq: flux array evaluated at the quadrature points [ne, nq, ns, ndims]

	Outputs:
	--------
		res_elem: calculated residual array (for volume integral of all elements)
			[ne, nb, ns]
	'''
	quad_wts = elem_helpers.quad_wts # [nq, 1]
	basis_phys_grad_elems = elem_helpers.basis_phys_grad_elems
			# [ne, nq, nb, ndims]
	djac_elems = elem_helpers.djac_elems # [ne, nq, 1]

	# Calculate flux quadrature
	F_quad = np.einsum('ijkl, jm, ijm -> ijkl', Fq, quad_wts, djac_elems)
			# [ne, nq, ns, ndims]
	# Calculate residual
	res_elem = np.einsum('ijnl, ijkl -> ink', basis_phys_grad_elems, F_quad)
			# [ne, nb, ns]

	return res_elem # [ne, nb, ns]


def calculate_inviscid_flux_boundary_integral(basis_val, quad_wts, Fq):
	'''
	Calculates the inviscid flux boundary integral for the DG scheme

	Inputs:
	-------
		basis_val: basis function for the interior element [nf, nq, nb]
		quad_wts: quadrature weights [nq, 1]
		Fq: flux array evaluated at the quadrature points [nf, nq, ns]

	Outputs:
	--------
		resB: residual contribution (from boundary face) [nb, ns]
	'''
	# Calculate flux quadrature
	Fq_quad = np.einsum('ijk, jm -> ijk', Fq, quad_wts) # [nf, nq, ns]

	# Calculate residual
	resB = np.einsum('ijn, ijk -> ink', basis_val, Fq_quad) # [nf, nb, ns]

	return resB # [nf, nb, ns]


def calculate_source_term_integral(elem_helpers, Sq):
	'''
	Calculates the source term volume integral for the DG scheme

	Inputs:
	-------
		elem_helpers: helpers defined in ElemHelpers
		Sq: source term array evaluated at the quadrature points [ne, nq, ns]

	Outputs:
	--------
		res_elem: calculated residual array (for volume integral of all elements)
		[ne, nb, ns]
	'''
	quad_wts = elem_helpers.quad_wts # [nq, 1]
	basis_val = elem_helpers.basis_val # [nq, nb]
	djac_elems = elem_helpers.djac_elems # [ne, nq, 1]

	# Calculate source term quadrature
	Sq_quad = np.einsum('ijk, jm, ijm -> ijk', Sq, quad_wts, djac_elems)
			# [ne, nq, ns]

	# Calculate residual
	res_elem = np.einsum('jn, ijk -> ink', basis_val, Sq_quad) # [ne, nb, ns]

	return res_elem # [ne, nb, ns]

def calculate_artificial_viscosity_integral(physics, elem_helpers, Uc, av_param, p):
	'''
	Calculates the artificial viscosity volume integral, given in:
		Hartmann, R. and Leicht, T, "Higher order and adaptive DG methods for
		compressible flows", p. 92, 2013.

	Inputs:
	-------
		physics: physics object
		elem_helpers: helpers defined in ElemHelpers
		Uc: state coefficients of each element
		av_param: artificial viscosity parameter
		p: solution basis order

	Outputs:
	--------
		res_elem: artificial viscosity residual array for all elements
		[ne, nb, ns]
	'''
	# Unpack
	quad_wts = elem_helpers.quad_wts # [nq, 1]
	basis_phys_grad_elems = elem_helpers.basis_phys_grad_elems
			# [ne, nq, nb, dim]
	basis_val = elem_helpers.basis_val # [nq, nb]
	djac_elems = elem_helpers.djac_elems # [ne, nq, 1]
	vol_elems = elem_helpers.vol_elems # [ne]
	ndims = basis_phys_grad_elems.shape[3]

	# Evaluate solution at quadrature points
	Uq = helpers.evaluate_state(Uc, basis_val)
	# Evaluate solution gradient at quadrature points
	grad_Uq = np.einsum('ijnl, ink -> ijkl', basis_phys_grad_elems, Uc)
	# Compute pressure
	pressure = physics.compute_additional_variable("Pressure", Uq,
			flag_non_physical=False)[:, :, 0]
	# For Euler equations, use pressure as the smoothness variable
	if physics.PHYSICS_TYPE == general.PhysicsType.Euler:
		# Compute pressure gradient
		grad_p = physics.compute_pressure_gradient(Uq, grad_Uq)
		# Compute its magnitude
		norm_grad_p = np.linalg.norm(grad_p, axis = 2)
		# Calculate smoothness switch
		f = norm_grad_p / (pressure + 1e-12)
	# For everything else, use the first solution variable
	else:
		U0 = Uq[:, :, 0]
		grad_U0 = grad_Uq[:, :, 0]
		norm_grad_U0 = np.linalg.norm(grad_U0, axis = 2)
		# Calculate smoothness switch
		f =  norm_grad_U0 / (U0 + 1e-12)

	# Compute s_k
	s = np.zeros((Uc.shape[0], ndims))
	# Loop over dimensions
	for k in range(ndims):
		# Loop over number of faces per element
		for i in range(elem_helpers.normals_elems.shape[1]):
			# Integrate normals
			s[:, k] += np.einsum('jx, ij -> i', elem_helpers.face_quad_wts,
					np.abs(elem_helpers.normals_elems[:, i, :, k]))
		s[:, k] = 2 * vol_elems / s[:, k]
	# Compute h_k (the length scale in the kth direction)
	h = np.empty_like(s)
	# Loop over dimensions
	for k in range(ndims):
		h[:, k] = s[:, k] * (vol_elems / np.prod(s, axis=1))**(1/3)
	# Scale with polynomial order
	h_tilde = h / (p + 1)
	# Compute dissipation scaling
	epsilon = av_param *  np.einsum('ij, il -> ijl', f, h_tilde**3)
	# Calculate integral, with state coeffs factored out
	integral = np.einsum('ijm, ijpm, ijnm, jx, ijx -> ipn', epsilon,
				basis_phys_grad_elems, basis_phys_grad_elems, quad_wts,
				djac_elems)
	# Calculate residual
	res_elem = np.einsum('ipn, ipk -> ink', integral, Uc)

	return res_elem # [ne, nb, ns]


def calculate_dRdU(elem_helpers, Sjac):
	'''
	Helper function for ODE solvers that calculates the derivative of
	the source term integral with respect to the solution state.

	Inputs:
	-------
		elem_helpers: object containing precomputed element helpers
		Sjac: element source term Jacobian [ne, nq, ns, ns]

	Outputs:
	--------
		dRdU: derivative of the source term integral
			[ne, nb, nb, ns, ns]
	'''
	quad_wts = elem_helpers.quad_wts
	basis_val = elem_helpers.basis_val
	djac_elems = elem_helpers.djac_elems

	a = np.einsum('eijk, il, eil -> eijk', Sjac, quad_wts, djac_elems)

	return np.einsum('bq, ql, eqts -> eblts', basis_val.transpose(),
			basis_val, a)
		# [ne, nb, nb, ns, ns]


def mult_inv_mass_matrix(mesh, solver, dt, res):
	'''
	Multiplies the residual array with the inverse mass matrix

	Inputs:
		mesh: mesh object
		solver: solver object (e.g., DG, ADER-DG, etc...)
		dt: time step
		res: residual array

	Outputs:
		U: solution array
	'''
	physics = solver.physics
	iMM_elems = solver.elem_helpers.iMM_elems

	return dt*np.einsum('ijk, ikl -> ijl', iMM_elems, res)


def L2_projection(mesh, iMM, basis, quad_pts, quad_wts, f, U):
	'''
	Performs an L2 projection

	Inputs:
	-------
		mesh: mesh object
		iMM: space-time inverse mass matrix
		basis: basis object
		quad_pts: quadrature coordinates in reference space
		quad_wts: quadrature weights
		f: array of values to be projected from

	Outputs:
	--------
		U: array of values to be projected to
	'''
	if basis.basis_val.shape[0] != quad_wts.shape[0]:
		basis.get_basis_val_grads(quad_pts, get_val=True)

	for elem_ID in range(U.shape[0]):
		djac, _, _ = basis_tools.element_jacobian(mesh, elem_ID, quad_pts,
				get_djac=True)
		rhs = np.matmul(basis.basis_val.transpose(),
				f[elem_ID, :, :]*quad_wts*djac) # [nb, ns]

		U[elem_ID, :, :] = np.matmul(iMM[elem_ID], rhs)


def interpolate_to_nodes(f, U):
	'''
	Interpolates directly to the nodes of the element

	Inputs:
	-------
		f: array of values to be interpolated from

	Outputs:
	--------
		U: array of values to be interpolated onto
	'''
	U[:, :, :] = f


def update_progress(progress):
	'''
	Displays or updates a console progress bar.
	Accepts a float between 0 and 1. Any int will be converted to a float.
	A value under 0 represents a 'halt'.
	A value at 1 or bigger represents 100%.

	Inputs:
	-------
		progress: value representing the progress, scaled from 0 to 1
	'''
	# Length of the progress bar
	bar_length = 55

	status = ""
	# Convert ints
	if isinstance(progress, int):
		progress = float(progress)
	# Make sre it's a number
	if not isinstance(progress, float):
		progress = 0
		status = "error: progress var must be float\r\n"
	# Less than 0 'halts' the progress
	if progress < 0:
		progress = 0
		status = "Halt...\r\n"
	# Cap the progress at 100%
	if progress >= 1:
		progress = 1
		status = "Done...\r\n"

	# Compute number of blocks
	block = int(round(bar_length*progress))
	# Figure out the color
	if progress < .25:
		color = '\033[0;31m' # Dark red
	elif progress < .5:
		color = '\033[1;31m' # Light red
	elif progress < .75:
		color = '\033[0;33m' # Yellow
	elif progress < 1:
		color = '\033[0;32m' # Dark green
	else:
		color = '\033[1;32m' # Light green
	reset_color = '\033[0m'
	# Write out the text
	text = color + '\rPercent: [{0}] {1}% {2}'.format( "#"*block + "-"*(bar_length-block),
			int(round(progress*100)), status) + reset_color
	sys.stdout.write(text)
	sys.stdout.flush()
