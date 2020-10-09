# ------------------------------------------------------------------------ #
#
#       File : src/numerics/solver/tools.py
#
#       Contains additional methods (tools) for the DG solver class
#
# ------------------------------------------------------------------------ #
import numpy as np

import general
import numerics.basis.tools as basis_tools

def calculate_inviscid_flux_volume_integral(solver, elem_helpers, Fq):
	'''
	Calculates the inviscid flux volume integral for the DG scheme

	Inputs:
	-------
		solver: solver object
		elem_helpers: helpers defined in ElemHelpers
		Fq: flux array evaluated at the quadrature points [ne, nq, ns, dim]

	Outputs:
	--------
		R: calculated residual array (for volume integral of all elements)
		[ne, nb, ns]
	'''
	quad_wts = elem_helpers.quad_wts # [nq, 1]
	basis_phys_grad_elems = elem_helpers.basis_phys_grad_elems # [ne, nq, nb, dim]
	djac_elems = elem_helpers.djac_elems # [ne, nq, 1]

	# Calculate flux quadrature
	F_quad = np.einsum('ijkl, jm, ijm -> ijkl', Fq, quad_wts, djac_elems) # [ne, nq, ns, dim]
	# Calculate residual
	R = np.einsum('ijnl, ijkl -> ink', basis_phys_grad_elems, F_quad) # [ne, nb, ns]

	return R # [ne, nb, ns]

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
		R_B: residual contribution (from boundary face) [nb, ns]
	'''
	# Calculate flux quadrature
	Fq_quad = np.einsum('ijk, jm -> ijk', Fq, quad_wts) # [nf, nq, ns]
	# Calculate residual
	R_B = np.einsum('ijn, ijk -> ink', basis_val, Fq_quad) # [nf, nb, ns]

	return R_B # [nf, nb, ns]


def calculate_source_term_integral(elem_helpers, Sq):
	'''
	Calculates the source term volume integral for the DG scheme

	Inputs:
	-------
		elem_helpers: helpers defined in ElemHelpers
		Sq: source term array evaluated at the quadrature points [ne, nq, ns]

	Outputs:
	--------
		R: calculated residual array (for volume integral of all elements)
		[ne, nb, ns]
	'''
	quad_wts = elem_helpers.quad_wts # [nq, 1]
	basis_val = elem_helpers.basis_val # [nq, nb]
	djac_elems = elem_helpers.djac_elems # [ne, nq, 1]

	# Calculate source term quadrature
	Sq_quad = np.einsum('ijk, jm, ijm -> ijk', Sq, quad_wts, djac_elems) # [ne, nq, ns]
	# Calculate residual
	R = np.einsum('jn, ijk -> ink', basis_val, Sq_quad) # [ne, nb, ns]

	return R # [ne, nb, ns]

def calculate_dRdU(elem_helpers, elem_ID, Sjac):
	'''
	Helper function for ODE solvers that calculates the derivative of

		integral(basis_val*S(U))dx

	with respect to the solution state

	Inputs:
		elem_helpers: object containing precomputed element helpers
		elem_ID: element index
		Sjac: element source term Jacobian [nq, ns, ns]
	'''
	quad_wts = elem_helpers.quad_wts
	basis_val = elem_helpers.basis_val
	djac_elems = elem_helpers.djac_elems
	djac = djac_elems[elem_ID]

	a = np.einsum('ijk,il->ijk', Sjac, quad_wts*djac)

	return np.einsum('bq,qts->bts',basis_val.transpose(), a) # [nb, ns, ns]


def mult_inv_mass_matrix(mesh, solver, dt, R):
	'''
	Multiplies the residual array with the inverse mass matrix

	Inputs:
		mesh: mesh object
		solver: solver object (e.g., DG, ADER-DG, etc...)
		dt: time step
		R: residual array

	Outputs:
		U: solution array
	'''
	physics = solver.physics
	iMM_elems = solver.elem_helpers.iMM_elems

	return dt*np.einsum('ijk,ikl->ijl', iMM_elems, R)


def L2_projection(mesh, iMM, basis, quad_pts, quad_wts, elem_ID, f, U):
	'''
	Performs an L2 projection

	Inputs:
	-------
		mesh: mesh object
		iMM: space-time inverse mass matrix
		basis: basis object
		quad_pts: quadrature coordinates in reference space
		quad_wts: quadrature weights
		elem_ID: element index
		f: array of values to be projected from

	Outputs:
	--------
		U: array of values to be projected to
	'''
	if basis.basis_val.shape[0] != quad_wts.shape[0]:
		basis.get_basis_val_grads(quad_pts, get_val=True)

	djac, _, _ = basis_tools.element_jacobian(mesh, elem_ID, quad_pts, get_djac=True)

	rhs = np.matmul(basis.basis_val.transpose(), f*quad_wts*djac) # [nb, ns]

	U[:,:] = np.matmul(iMM, rhs)


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
	U[:,:] = f
