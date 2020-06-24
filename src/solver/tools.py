import code
import copy
import numpy as np

import data
import general

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
		MMinv_all = basis_tools.get_inv_mass_matrices(mesh, EqnSet, solver.basis)
		DataSet.MMinv_all = MMinv_all


	if dt is None:
		c = 1.
	else:
		c = dt
	# if not U:
	# 	U = copy.deepcopy(R)

	for elem in range(mesh.nElem):
		U_ = U[elem]
		U_[:,:] = c*np.matmul(MMinv_all[elem], R[elem])

	# U[:,:,:] = c*np.einsum('ijk,ikl->ijl', MMinv_all, R)


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


# def eval_IC_at_pts(mesh, EqnSet, basis, eval_pts, time=0., fcn_data=None, order_old=-1, U_old=None):
# 	if fcn_data is None and U_old is None:
# 		raise ValueError
# 	elif fcn_data is not None and U_old is not None:
# 		raise ValueError

# 	ns = EqnSet.StateRank

# 	if fcn_data is not None:
# 		order = 2*np.amax([EqnSet.order, 1])
# 		order = EqnSet.QuadOrder(order)
# 	else:
# 		order = 2*np.amax([EqnSet.order, order_old])

# 	QuadOrder,_ = get_gaussian_quadrature_elem(mesh, basis, order)

# 	quadData = QuadData(mesh, mesh.gbasis, general.EntityType.Element, QuadOrder)

# 	quad_pts = quadData.quad_pts
# 	quad_wts = quadData.quad_wts

# 	basis.eval_basis(quad_pts, Get_Phi=True)





# 	if basis.basis_val.shape[0] != eval_pts.shape[0]:
# 		basis.eval_basis(eval_pts, Get_Phi=True)

# 	for elem in range(mesh.nElem):

# 		if fcn_data is not None:
# 			xphys, _ = ref_to_phys(mesh, elem, None, quad_pts)
# 			f = EqnSet.CallFunction(fcn_data, x=xphys, t=time)
# 			f.shape = quad_wts.shape[0],ns
# 		else:
# 			f = 


# 		djac, _, _ = element_jacobian(mesh,elem,quad_pts,get_djac=True)

# 		rhs = np.matmul(basis.basis_val.transpose(), f*quad_wts*djac) # [nb, ns]

# 		U[elem,:,:] = np.matmul(iMM_elems[elem],rhs)



def L2_projection(mesh, iMM, basis, elem, f, U):
	quad_pts = basis.quad_pts
	quad_wts = basis.quad_wts

	if basis.basis_val.shape[0] != quad_wts.shape[0]:
		basis.eval_basis(quad_pts, Get_Phi=True)

	djac, _, _ = basis_tools.element_jacobian(mesh, elem, quad_pts, get_djac=True)

	rhs = np.matmul(basis.basis_val.transpose(), f*quad_wts*djac) # [nb, ns]

	U[:,:] = np.matmul(iMM, rhs)


def interpolate_to_nodes(f, U):
	U[:,:] = f


		
