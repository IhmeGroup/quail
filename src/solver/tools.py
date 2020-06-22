import code
import copy
import numpy as np

import data
import numerics.basis.tools as basis_tools

def calculate_inviscid_flux_volume_integral(solver, elem_ops, elem, Fq):
	
	quad_wts = elem_ops.quad_wts
	basis_val = elem_ops.basis_val 
	basis_pgrad_elems = elem_ops.basis_pgrad_elems
	djac_elems = elem_ops.djac_elems 

	basis_pgrad = basis_pgrad_elems[elem]
	djac = djac_elems[elem]
	nq = quad_wts.shape[0]

	# for ir in range(ns):
	# 	for jn in range(nb):
	# 		for iq in range(nq):
	# 			gPhi = PhiData.gPhi[iq,jn] # dim
	# 			ER[jn,ir] += np.dot(gPhi, F[iq,ir,:])*wq[iq]*JData.djac[iq*(JData.nq!=1)]

	ER = np.tensordot(basis_pgrad, Fq*(quad_wts*djac).reshape(nq,1,1), axes=([0,2],[0,2])) # [nb, ns]

	return ER

def calculate_inviscid_flux_boundary_integral(basis_val, quad_wts, Fq):

	R = np.matmul(basis_val.transpose(), Fq*quad_wts) # [nb,sr]

	return R

def calculate_source_term_integral(solver, elem_ops, physics, elem, Uq):
	
	quad_wts = elem_ops.quad_wts
	basis_val = elem_ops.basis_val 
	djac_elems = elem_ops.djac_elems 
	x_elems = elem_ops.x_elems

	djac = djac_elems[elem]
	nq = quad_wts.shape[0]
	x = x_elems[elem]


	Sq = elem_ops.Sq
	Sq[:] = 0. # SourceState is an additive function so source needs to be initialized to zero for each time step
	Sq = physics.SourceState(nq, x, solver.Time, Uq, Sq) # [nq,ns]

	# Calculate source term integral
	# for ir in range(sr):
	# 	for jn in range(nn):
	# 		for iq in range(nq):
	# 			Phi = PhiData.Phi[iq,jn]
	# 			ER[jn,ir] += Phi*s[iq,ir]*wq[iq]*JData.djac[iq*(JData.nq!=1)]

	ER = np.matmul(basis_val.transpose(), Sq*quad_wts*djac) # [nb, ns]

	return ER

def mult_inv_mass_matrix(mesh, solver, dt, R, U):
	'''
	Method: mult_inv_mass_matrix
	-------------------------------
	Multiplies the residual array with the inverse mass matrix

	INPUTS:
		mesh: mesh object
		solver: type of solver (i.e. DG, ADER-DG, etc...)
		dt: time step
		R: residual array

	OUTPUTS:
		U: solution array
	'''
	EqnSet = solver.EqnSet
	DataSet = solver.DataSet
	# if MMinv is None:
	# 	MMinv = GetInvMassMatrix(mesh, 0, 0, EqnSet.orders[0])

	try:
		MMinv_all = DataSet.MMinv_all
	except AttributeError:
		# not found; need to compute
		MMinv_all = basis_tools.get_inv_mass_matrices(mesh, EqnSet, solver=solver)


	if dt is None:
		c = 1.
	else:
		c = dt
	# if not U:
	# 	U = copy.deepcopy(R)

	for elem in range(mesh.nElem):
		U_ = U[elem]
		U_[:,:] = c*np.matmul(MMinv_all[elem], R[elem])

def project_state_to_new_basis(solver, mesh, EqnSet, basis_old, order_old):
	''' Old state '''
	U = EqnSet.U

	basis = copy.copy(basis_old)
	''' Allocate new state '''
	# New basis, Order information stored in EqnSet
	# ArrayDims = [[mesh.nElems[egrp],Basis.order_to_num_basis_coeff(EqnSet.Bases[egrp], EqnSet.orders[egrp]), EqnSet.StateRank] \
	# 				for egrp in range(mesh.nElemGroup)]
	# U_new = Data.ArrayList(nArray=mesh.nElemGroup, ArrayDims=ArrayDims)
	U_new = np.zeros([mesh.nElem, basis.get_num_basis_coeff(EqnSet.order), EqnSet.StateRank])

	''' Loop through elements '''
	order = EqnSet.order
	basis.order = order
	basis.nb = basis.get_num_basis_coeff(EqnSet.order)

	## New mass matrix inverse (in reference space)
	iMM = basis_tools.get_elem_inv_mass_matrix(mesh, basis, order)
	## Projection matrix
	PM = basis_tools.get_projection_matrix(mesh, basis, basis_old, order, order_old, iMM)
	for elem in range(mesh.nElem):
		Uc = U[elem]
		Uc_new = U_new[elem]

		# New coefficients
		Uc_new[:] = np.matmul(PM, Uc)

	''' Store in EqnSet '''
	delattr(EqnSet, "U")
	EqnSet.U = U_new