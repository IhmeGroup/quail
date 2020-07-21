import code
import copy
import numpy as np

import data
import general

import numerics.basis.tools as basis_tools

def calculate_inviscid_flux_volume_integral(solver, elem_ops, elem, Fq):
	
	quad_wts = elem_ops.quad_wts
	basis_val = elem_ops.basis_val 
	basis_phys_grad_elems = elem_ops.basis_phys_grad_elems
	djac_elems = elem_ops.djac_elems 

	basis_phys_grad = basis_phys_grad_elems[elem]
	djac = djac_elems[elem]
	nq = quad_wts.shape[0]

	# for ir in range(ns):
	# 	for jn in range(nb):
	# 		for iq in range(nq):
	# 			gPhi = PhiData.gPhi[iq,jn] # dim
	# 			ER[jn,ir] += np.dot(gPhi, F[iq,ir,:])*wq[iq]*JData.djac[iq*(JData.nq!=1)]

	ER = np.tensordot(basis_phys_grad, Fq*(quad_wts*djac).reshape(nq,1,1), axes=([0,2],[0,2])) # [nb, ns]

	return ER

def calculate_inviscid_flux_boundary_integral(basis_val, quad_wts, Fq):

	R = np.matmul(basis_val.transpose(), Fq*quad_wts) # [nb,sr]

	return R

def calculate_source_term_integral(elem_ops, elem, Sq):
	
	quad_wts = elem_ops.quad_wts
	basis_val = elem_ops.basis_val 
	djac_elems = elem_ops.djac_elems 
	djac = djac_elems[elem]

	# Calculate source term integral
	# for ir in range(sr):
	# 	for jn in range(nn):
	# 		for iq in range(nq):
	# 			Phi = PhiData.Phi[iq,jn]
	# 			ER[jn,ir] += Phi*s[iq,ir]*wq[iq]*JData.djac[iq*(JData.nq!=1)]
	ER = np.matmul(basis_val.transpose(), Sq*quad_wts*djac) # [nb, ns]

	return ER

def calculate_dRdU(elem_ops, elem, jac):
	'''
	Method: calculate_dRdU
	-----------------------
	Helper function for ODE solvers that calculates the derivative of 

		integral(basis_val*S(U))dVol
	
	with respect to the solution state 
	
	INPUTS: 
		elem_ops: object containing precomputed element operations
		elem: element index [int]
		jac: element source term jacobian [nq, ns, ns]
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
	Method: mult_inv_mass_matrix
	-------------------------------
	Multiplies the residual array with the inverse mass matrix

	INPUTS:
		mesh: mesh object
		solver: solver object (i.e. DG, ADER-DG, etc...)
		dt: time step
		R: residual array

	OUTPUTS:
		U: solution array
	'''
	physics = solver.physics
	DataSet = solver.DataSet
	iMM_elems = solver.elem_operators.iMM_elems

	if dt is None:
		c = 1.
	else:
		c = dt

	return c*np.einsum('ijk,ikl->ijl', iMM_elems, R)


def L2_projection(mesh, iMM, basis, quad_pts, quad_wts, elem, f, U):

	if basis.basis_val.shape[0] != quad_wts.shape[0]:
		basis.get_basis_val_grads(quad_pts, get_val=True)

	djac, _, _ = basis_tools.element_jacobian(mesh, elem, quad_pts, get_djac=True)

	rhs = np.matmul(basis.basis_val.transpose(), f*quad_wts*djac) # [nb, ns]

	U[:,:] = np.matmul(iMM, rhs)


def interpolate_to_nodes(f, U):
	U[:,:] = f