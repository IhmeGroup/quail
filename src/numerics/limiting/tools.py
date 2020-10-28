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

import code


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
	elif general.LimiterType[limiter_type] is general.LimiterType.PositivityPreserving:
		limiter_class = pp_limiter.PositivityPreserving
	elif general.LimiterType[limiter_type] is general.LimiterType.PositivityPreservingChem:
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
	elif general.ShockIndicatorType[shock_indicator_type]  is general.ShockIndicatorType.MinMod:
		limiter.shock_indicator = minmod_shock_indicator
		

def minmod_shock_indicator(limiter, solver, elem_ID, Uc):

	# set for 1D only currently 

	# Unpack
	physics = solver.physics
	mesh = solver.mesh

	elem_helpers = solver.elem_helpers
	int_face_helpers = solver.int_face_helpers
	
	# identify neighboring elements
	elemP_ID = mesh.elements[elem_ID].face_to_neighbors[1]
	elemM_ID = mesh.elements[elem_ID].face_to_neighbors[0]

	djac = limiter.djac_elems[elem_ID]
	djacP = limiter.djac_elems[elemP_ID]
	djacM = limiter.djac_elems[elemM_ID]

	UcP = solver.state_coeffs[elemP_ID]
	UcM = solver.state_coeffs[elemM_ID]

	# Interpolate state at quadrature points over element and on faces
	u_elem_faces = helpers.evaluate_state(Uc, limiter.basis_val_elem_faces, 
			skip_interp=solver.basis.skip_interp)
	nq_elem = limiter.quad_wts_elem.shape[0]
	u_elem = u_elem_faces[:nq_elem, :]
	u_face = u_elem_faces[nq_elem:, :]
	# Average value of state
	u_bar = helpers.get_element_mean(u_elem, limiter.quad_wts_elem, djac, 
			limiter.elem_vols[elem_ID])

	# UcP neighbor average calculation
	up_elem = helpers.evaluate_state(UcP, elem_helpers.basis_val, 
			skip_interp=solver.basis.skip_interp)
	# Average value of state
	up_bar = helpers.get_element_mean(up_elem, limiter.quad_wts_elem, djacP, 
			limiter.elem_vols[elemP_ID])

	# UcM neighbor average calculation
	um_elem = helpers.evaluate_state(UcM, elem_helpers.basis_val, 
			skip_interp=solver.basis.skip_interp)
	# Average value of state
	um_bar = helpers.get_element_mean(um_elem, limiter.quad_wts_elem, djacM, 
			limiter.elem_vols[elemM_ID])

	# Store the polynomial coeff values for Up, Um, and U.
	limiter.u_elem = u_elem
	limiter.up_elem = up_elem
	limiter.um_elem = um_elem

	# Store the average values for Up, Um, and U.
	limiter.u_bar = u_bar
	limiter.up_bar = up_bar
	limiter.um_bar = um_bar

	u_tilde = u_face[1] - u_bar 
	u_dtilde = u_bar - u_face[0]

	deltaP_u_bar = up_bar - u_bar
	deltaM_u_bar = u_bar - um_bar

	aj = np.array([u_tilde, deltaP_u_bar, deltaM_u_bar])
	u_tilde_mod = helpers.minmod(aj)
	aj = np.array([u_dtilde, deltaP_u_bar, deltaM_u_bar])
	u_dtilde_mod = helpers.minmod(aj)

	if u_tilde_mod != u_tilde or u_dtilde_mod != u_dtilde:
		return True
	else:
		return False