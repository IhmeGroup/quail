# ------------------------------------------------------------------------ #
#
#       File : src/solver/tools.py
#
#       Contains additional methods (tools) for the DG solver class
#
# ------------------------------------------------------------------------ #
import numpy as np

import general
import numerics.basis.tools as basis_tools
import numerics.helpers.helpers as helpers
import solver.tools as solver_tools

def set_function_definitions(solver, params):
	'''
	This function sets the necessary functions for the given case 
	dependent upon setter flags in the input deck (primarily for 
	the diffusive flux definitions)

	Test
	Inputs:
	-------
		solver: solver object
		params: dict with solver parameters
	'''
	if solver.physics.diff_flux_fcn:
		solver.evaluate_gradient = helpers.evaluate_gradient
		solver.ref_to_phys_grad = helpers.ref_to_phys_grad
		solver.calculate_flux_boundary_integral_sum = \
			solver_tools.calculate_flux_boundary_integral_sum
	else:
		solver.evaluate_gradient = helpers.pass_evaluate_gradient	
		solver.ref_to_phys_grad = helpers.pass_ref_to_phys_grad
		solver.calculate_flux_boundary_integral_sum = \
			solver_tools.pass_calculate_flux_boundary_integral_sum


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


def pass_calculate_flux_boundary_integral_sum(basis_ref_grad, quad_wts, gFq):
	return 0.


def calculate_flux_boundary_integral_sum(basis_ref_grad, quad_wts, gFq):
	'''
	'''
	# Calculate flux quadrature
	gFq_quad = np.einsum('ijkl, jm -> ijkl', gFq, quad_wts) # [nf, nq, ns]

	# Calculate residual
	resB = np.einsum('ijnl, ijkl -> ink', basis_ref_grad, gFq_quad)

	return resB


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