# ------------------------------------------------------------------------ #
#
#       File : src/numerics/solver/tools.py
#
#       Contains additional methods (tools) for the DG solver class
#      
# ------------------------------------------------------------------------ #
import copy
import numpy as np

import general
import numerics.basis.tools as basis_tools


def calculate_inviscid_flux_volume_integral(solver, elem_ops, elem, Fq):
	'''
	Calculates the inviscid flux volume integral for the DG scheme

	Inputs:
	-------
		solver: solver object
		elem_ops: helper operators defined in ElemHelpers
		elem: element index
		Fq: flux array evaluated at the quadrature points [nq, ns, dim]

	Outputs:
	--------
		ER: calculated residual array (for volume integral of specified 
		element) [nb, ns]
	'''	
	quad_wts = elem_ops.quad_wts
	basis_val = elem_ops.basis_val 
	basis_phys_grad_elems = elem_ops.basis_phys_grad_elems
	djac_elems = elem_ops.djac_elems 

	basis_phys_grad = basis_phys_grad_elems[elem]
	djac = djac_elems[elem]
	nq = quad_wts.shape[0]

	ER = np.tensordot(basis_phys_grad, Fq*(quad_wts*djac).reshape(nq,1,1), axes=([0,2],[0,2])) # [nb, ns]

	return ER # [nb, ns]


def calculate_inviscid_flux_boundary_integral(basis_val, quad_wts, Fq):
	'''
	Calculates the inviscid flux boundary integral for the DG scheme

	Inputs:
	-------
		basis_val: basis function for the interior element [nq, nb]
		quad_wts_st: space-time quadrature weights [nq, 1]
		Fq: flux array evaluated at the quadrature points [nq, ns, dim]

	Outputs:
	--------
		R: calculated residual array (from boundary face) [nb, ns]
	'''
	R = np.matmul(basis_val.transpose(), Fq*quad_wts) # [nb, ns]

	return R # [nb, ns]


def calculate_source_term_integral(elem_ops, elem, Sq):
	'''
	Calculates the source term volume integral for the DG scheme

	Inputs:
	-------
		elem_ops: helper operators defined in ElemHelpers
		elem: element index
		Sq: source term array evaluated at the quadrature points [nq, ns]

	Outputs:
	--------
		ER: calculated residual array (for volume integral of specified 
		element) [nb, ns]
	'''
	quad_wts = elem_ops.quad_wts
	basis_val = elem_ops.basis_val 
	djac_elems = elem_ops.djac_elems 
	djac = djac_elems[elem]

	ER = np.matmul(basis_val.transpose(), Sq*quad_wts*djac) # [nb, ns]

	return ER # [nb, ns]


def calculate_dRdU(elem_ops, elem, jac):
	'''
	Helper function for ODE solvers that calculates the derivative of 

		integral(basis_val*S(U))dVol
	
	with respect to the solution state 
	
	Inputs: 
		elem_ops: object containing precomputed element operations
		elem: element index
		jac: element source term Jacobian [nq, ns, ns]
	'''
	quad_wts = elem_ops.quad_wts
	basis_val = elem_ops.basis_val 
	djac_elems = elem_ops.djac_elems 
	djac = djac_elems[elem]
	ns = jac.shape[-1]
	nb = basis_val.shape[1]
	nq = quad_wts.shape[0]

	test1 = quad_wts*djac
	test = np.einsum('ijk,il->ijk',jac,test1)
	ER = np.einsum('bq,qts -> bts',basis_val.transpose(),test)

	return ER # [nb, ns, ns]


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
	iMM_elems = solver.elem_operators.iMM_elems

	if dt is None:
		c = 1.
	else:
		c = dt

	return c*np.einsum('ijk,ikl->ijl', iMM_elems, R)


def L2_projection(mesh, iMM, basis, quad_pts, quad_wts, elem, f, U):
	'''
	Performs an L2 projection

	Inputs:
	-------
		mesh: mesh object
		iMM: space-time inverse mass matrix
		basis: basis object
		quad_pts: quadrature coordinates in reference space
		quad_wts: quadrature weights
		elem: element index
		f: array of values to be projected from

	Outputs:
	--------
		U: array of values to be projected to
	'''
	if basis.basis_val.shape[0] != quad_wts.shape[0]:
		basis.get_basis_val_grads(quad_pts, get_val=True)

	djac, _, _ = basis_tools.element_jacobian(mesh, elem, quad_pts, get_djac=True)

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