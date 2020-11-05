# ------------------------------------------------------------------------ #
#
#       File : src/numerics/limiting/tools.py
#
#       Contains helpers functions related to limiters.
#      
# ------------------------------------------------------------------------ #
import numpy as np

import general
import numerics.limiting.positivitypreserving as pp_limiter
import numerics.limiting.wenolimiter as weno_limiter
import numerics.helpers.helpers as helpers


def set_limiter(limiter_type, physics_type):
	'''
	This function instantiates the desired limiter class.

	Inputs:
	-------
	    limiter_type: limiter type (general.LimiterType enum member)
	    physics_type: physics type (general.PhysicsType enum member)

	Outputs:
	--------
	    limiter: limiter object
	'''
	if limiter_type is None:
		return None
	elif general.LimiterType[limiter_type] is \
			general.LimiterType.PositivityPreserving:
		limiter_class = pp_limiter.PositivityPreserving
	elif general.LimiterType[limiter_type] is \
			general.LimiterType.PositivityPreservingChem:
		limiter_class = pp_limiter.PositivityPreservingChem
	elif general.LimiterType[limiter_type] is general.LimiterType.ScalarWENO:
		limiter_class = weno_limiter.ScalarWENO
	else:
		raise NotImplementedError

	# Instantiate class
	limiter = limiter_class(physics_type)

	return limiter


def set_shock_indicator(limiter, shock_indicator_type):
	'''
	This function sets the appropriate shock indicator. 

	Inputs:
	-------
		shock_indicator_type: see general.ShockIndicatorType
	'''
	if shock_indicator_type is None:
		return None
	elif general.ShockIndicatorType[shock_indicator_type] is \
			general.ShockIndicatorType.MinMod:
		limiter.shock_indicator = minmod_shock_indicator
		

def minmod_shock_indicator(limiter, solver, Uc):
	'''
	Minmod calculation used to determine if oscillations are large enough 
	to require limiting.

	Inputs:
	-------
		limiter: limiter object
		solver: solver object
		Uc: state coefficients [ne, nb, ns]

	Outputs:
	--------
		shock_elems: array with IDs of elements flagged for limiting
	'''
	# Unpack
	physics = solver.physics
	mesh = solver.mesh

	elem_helpers = solver.elem_helpers
	int_face_helpers = solver.int_face_helpers
	
	elemP_IDs = limiter.elemP_IDs
	elemM_IDs = limiter.elemM_IDs
	djacs = limiter.djac_elems
	djacP = limiter.djac_elems[elemP_IDs]
	djacM = limiter.djac_elems[elemM_IDs]

	UcP = solver.state_coeffs[elemP_IDs]
	UcM = solver.state_coeffs[elemM_IDs]

	# Interpolate state at quadrature points over element and on faces
	U_elem_faces = helpers.evaluate_state(Uc, limiter.basis_val_elem_faces, 
			skip_interp=solver.basis.skip_interp)
	nq_elem = limiter.quad_wts_elem.shape[0]
	U_elem = U_elem_faces[:, :nq_elem, :]
	U_face = U_elem_faces[:, nq_elem:, :]
		
	# Average value of states
	U_bar = helpers.get_element_mean(U_elem, limiter.quad_wts_elem, djacs, 
			limiter.elem_vols)

	# UcP neighbor evaluated at quadrature points
	Up_elem = helpers.evaluate_state(UcP, elem_helpers.basis_val, 
			skip_interp=solver.basis.skip_interp)
	# Average value of state
	Up_bar = helpers.get_element_mean(Up_elem, limiter.quad_wts_elem, djacP, 
			limiter.elem_vols[elemP_IDs])

	# UcM neighbor evaluated at quadrature points
	Um_elem = helpers.evaluate_state(UcM, elem_helpers.basis_val, 
			skip_interp=solver.basis.skip_interp)
	# Average value of state
	Um_bar = helpers.get_element_mean(Um_elem, limiter.quad_wts_elem, djacM, 
			limiter.elem_vols[elemM_IDs])

	# Store the polynomial coeff values for Up, Um, and U.
	limiter.U_elem = Uc
	limiter.Up_elem = UcP
	limiter.Um_elem = UcM

	# Store the average values for Up, Um, and U.
	limiter.U_bar = U_bar
	limiter.Up_bar = Up_bar
	limiter.Um_bar = Um_bar

	U_tilde = (U_face[:, 1, :] - U_bar[:, 0, :]).reshape(U_bar.shape)
	U_dtilde = (U_bar[:, 0, :] - U_face[:, 0, :]).reshape(U_bar.shape)

	deltaP_u_bar = Up_bar - U_bar
	deltaM_u_bar = U_bar - Um_bar

	aj = np.zeros([U_tilde.shape[0], 3, 1])
	aj[:, 0, :] = U_tilde[:, 0, :]
	aj[:, 1, :] = deltaP_u_bar[:, 0, :]
	aj[:, 2, :] = deltaM_u_bar[:, 0, :]
	u_tilde_mod = minmod(aj)

	aj = np.zeros([U_dtilde.shape[0], 3, 1])
	aj[:, 0, :] = U_dtilde[:, 0, :]
	aj[:, 1, :] = deltaP_u_bar[:, 0, :]
	aj[:, 2, :] = deltaM_u_bar[:, 0, :]
	u_dtilde_mod = minmod(aj)

	shock_elems = np.where((u_tilde_mod != U_tilde) 
			| (u_dtilde_mod != U_dtilde))[0]

	return shock_elems


def minmod(a):
	'''
	Calculates the minmod function for the minmod shock indicator function

	Inputs:
	-------
		a: vector used to evaluate the minmod [ne, 3, ns]

	Outputs:
	--------
		u: evaluated minmod state [ne, 1, 1]
	'''
	s = np.sign(a)
	# Allocate with a large negative number. 
	u = -1234567.*np.ones([a.shape[0], 1, a.shape[-1]])

	elemID_gt = np.where(np.all(s > 0., axis=1))[0]
	elemID_lt = np.where(np.all(s < 0., axis=1))[0]

	if elemID_gt.size != 0:
		u[elemID_gt, 0, :] = s[elemID_gt, 0, :] * np.amin(
				np.abs(a[elemID_gt]), axis=1)
	if elemID_lt.size !=0:
		u[elemID_lt, 0, :] = s[elemID_lt, 0, :] * np.amin(
				np.abs(a[elemID_lt]), axis=1)

	return u # [ne, 1, 1]

def get_hessian(limiter, basis, quad_pts):
	'''
	Calculates the reference hessian of the basis function

	Inputs:
	-------
		limiter: limiter object
		basis: basis object
		quad_pts: quadrature point coordinates [nq, ndims] 

	Outputs:
	--------
		basis_ref_hessian: reference hessian of the basis function 
			[nq, nb, dim]
	'''
	ndims = basis.NDIMS
	p = basis.order
	nb = basis.nb
	nq = quad_pts.shape[0]

	basis_ref_hessian = np.zeros([nq, nb, ndims])

	if p > 0:
		xnodes = basis.get_1d_nodes(-1., 1., p+1)
		get_lagrange_hessian_1D(quad_pts, xnodes, 
				basis_ref_hessian=basis_ref_hessian)
	
	return basis_ref_hessian # [nq, nb, ndims]


def get_lagrange_hessian_1D(xq, xnodes, basis_ref_hessian=None):
	'''
	Calculates the 1D Lagrange hessian

	Inputs:
	-------
		xq: coordinates of quadrature points [nq, 1]
		xnodes: coordinates of nodes in 1D ref space [nb, 1] 

	Outputs:
	-------- 
		basis_hessian: evaluated reference hessian [nq, nb, ndims]
	'''
	nnodes = xnodes.shape[0]

	if basis_ref_hessian is not None:
		basis_ref_hessian[:] = 0.

	for j in range(nnodes):
		for i in range(nnodes):
			if i != j:
				for k in range(nnodes):
					if (k != i) and (k != j):
						h = 1./(xnodes[j]-xnodes[i]) * 1./(xnodes[j]- \
								xnodes[k])
						for l in range(nnodes):
							if (l != i) and (l != j) and (l !=k ):
								h *= (xq - xnodes[l])/(xnodes[j] - xnodes[l])
						basis_ref_hessian[:, j, :] += h

	
def get_phys_hessian(limiter, basis, ijac):
	'''
	Calculates the physical gradient of the hessian

	Inputs:
	-------
		limiter: limiter object
		basis: basis object
		ijac: inverse of the Jacobian [nq, nb, ndims]

	Outputs:
	--------
		basis_phys_hessian: evaluated hessian of the basis function in 
			physical space [nq, nb, ndims]
	'''
	ndims = basis.NDIMS
	nb = basis.nb

	basis_ref_hessian = limiter.basis_ref_hessian
	nq = basis_ref_hessian.shape[0]

	if nq == 0:
		raise ValueError("basis_ref_hessian not evaluated")

	# Check to see if ijac has been passed and has the right shape
	if ijac is None or ijac.shape != (nq, ndims, ndims):
		raise ValueError("basis_ref_hessian and ijac shapes not compatible")
	
	ijac2 = np.einsum('ijk,ijk->ijk', ijac, ijac)
	basis_phys_hessian = np.einsum('ijk, ikk -> ijk', 
			basis_ref_hessian, ijac2)

	return basis_phys_hessian # [nq, nb, ndims]