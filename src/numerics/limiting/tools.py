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
	elif general.LimiterType[limiter_type] is general.LimiterType.WENO:
		limiter_class = weno_limiter.WENO
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
	TVB modified Minmod calculation used to detect shocks

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
	tvb_param = limiter.tvb_param
	ns = physics.NUM_STATE_VARS

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
	
	if solver.basis.skip_interp is True:
		U_face = np.zeros([U_elem.shape[0], 2, ns])
		U_face[:, 0, :] = U_elem[:, 0, :]
		U_face[:, 1, :] = U_elem[:, -1, :]

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

	# Unpack the left and right eigenvector matrices
	right_eigen = limiter.right_eigen
	left_eigen = limiter.left_eigen

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
	
	aj = np.zeros([U_tilde.shape[0], 3, ns])
	aj[:, 0, :] = U_tilde[:, 0, :]
	aj[:, 1, :] = deltaP_u_bar[:, 0, :]
	aj[:, 2, :] = deltaM_u_bar[:, 0, :]
	u_tilde_mod = minmod(aj)

	tvb = np.where(np.abs(aj[:, 0, :]) <= tvb_param * \
			limiter.elem_vols[0]**2)[0]
	u_tilde_mod[tvb, 0] = aj[tvb, 0, :]

	aj = np.zeros([U_dtilde.shape[0], 3, ns])
	aj[:, 0, :] = U_dtilde[:, 0, :]
	aj[:, 1, :] = deltaP_u_bar[:, 0, :]
	aj[:, 2, :] = deltaM_u_bar[:, 0, :]
	u_dtilde_mod = minmod(aj)
	tvd = np.where(np.abs(aj[:, 0, :]) <= tvb_param * \
			limiter.elem_vols[0]**2)[0]
	u_dtilde_mod[tvd, 0] = aj[tvd, 0, :]


	check1 = u_tilde_mod - U_tilde
	check2 = u_dtilde_mod - U_dtilde

	shock_elems = np.where((np.abs(check1[:,:,0]) > 1.e-12)
			| (np.abs(check2[:,:,0]) > 1.e-12))[0]

	# print(shock_elems)
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

	ns = a.shape[2]
	nelem = a.shape[0]
	u = np.zeros([nelem, 1, ns])

	for i in range(ns):
		s1 = np.sign(a[:, 0, i])
		s2 = np.sign(a[:, 1, i])
		s3 = np.sign(a[:, 2, i])
		sign_agreement = np.where(np.logical_and(s1 == s2, s2 == s3))[0]
		if sign_agreement.size !=0:
			u[sign_agreement, 0, i] = s1[sign_agreement] * \
					np.min(np.abs(a[sign_agreement, :, i]))

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
		if basis.MODAL_OR_NODAL == general.ModalOrNodal.Modal:
			get_legendre_hessian_1D(quad_pts, p, 
				basis_ref_hessian=basis_ref_hessian)
		else:
			xnodes = basis.get_1d_nodes(-1., 1., p+1)
			get_lagrange_hessian_1D(quad_pts, xnodes,
					basis_ref_hessian=basis_ref_hessian)

	return basis_ref_hessian # [nq, nb, ndims]


def get_legendre_hessian_1D(xq, p, basis_ref_hessian=None):
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
	# Use numpy legendre polynomials
	leg_poly = np.polynomial.legendre.Legendre

	xq.shape = -1

	xq.shape = -1, 1

	if basis_ref_hessian is not None:
		basis_ref_hessian[:,:] = 0.

		for it in range(p+1):
			dleg = leg_poly.basis(it).deriv(2)
			basis_ref_hessian[:,it] = dleg(xq)


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
