# ------------------------------------------------------------------------ #
#
#       File : src/numerics/helpers/helpers.py
#
#       Contains general numerics-related helper functions
#
# ------------------------------------------------------------------------ #
import numpy as np


def get_element_mean(Uq, quad_wts, djac, vol):
	'''
	This function computes element averages of the state.

	Inputs:
	-------
        Uq: values of state at the quadrature points [ne, nq, ns]
	    quad_wts: quadrature weights [nq, 1]
	    djac: Jacobian determinants [ne, nq, 1]
	    vol: element volume [ne]

	Outputs:
	--------
	    U_mean: mean values of state variables [ne, 1, ns]
	'''
	U_mean = np.einsum('ijk, jm, ijm, i -> imk', Uq, quad_wts, djac, 1/vol)

	return U_mean # [ne, 1, ns]


def evaluate_state(Uc, basis_val, skip_interp=False):
	'''
	This function evaluates the state based on the given basis values.

	Inputs:
	-------
	    Uc: state coefficients [ne, nb, ns]
	    basis_val: basis values [ne, nq, nb]
	    skip_interp: if True, then will simply copy the state coefficients;
	    	useful for a colocated scheme, i.e. quadrature points and
	    	solution nodes (for a nodal basis) are the same

	Outputs:
	--------
	    Uq: values of state [ne, nq, ns]
	'''
	if skip_interp:
		Uq = Uc.copy()
	else:
		if basis_val.ndim == 3:
			# For faces, there is a different basis_val for each face
			Uq = np.einsum('ijn, ink -> ijk', basis_val, Uc)
		else:
			# For elements, all elements have the same basis_val
			Uq = np.einsum('jn, ink -> ijk', basis_val, Uc)

	return Uq # [ne, nq, ns]

def get_face_avg(mesh,physics,U_face):
	Uf_avg = np.zeros([U_face.shape[0], 4, U_face.shape[-1]])

	Uf_avg[:,0,:] = 0.5*(U_face[:,0,:] + U_face[:,1,:])
	Uf_avg[:,1,:] = 0.5*(U_face[:,2,:] + U_face[:,3,:])
	Uf_avg[:,2,:] = 0.5*(U_face[:,4,:] + U_face[:,5,:])
	Uf_avg[:,3,:] = 0.5*(U_face[:,6,:] + U_face[:,7,:])
	
	return Uf_avg

def get_face_length(djac_faces,quad_wts_face):
	face_length = np.zeros((djac_faces.shape[0],djac_faces.shape[1]))

	face_length = np.einsum('ijkl,kl->ij',djac_faces,quad_wts_face)

	return face_length

def get_face_mean(U_face,djac_faces,quad_wts_face,face_lengths):
	U_face_bar = np.zeros((U_face.shape[0],face_lengths.shape[1],U_face.shape[-1]))
	nq_face = quad_wts_face.shape[0]
	
	U_face_mod = np.zeros((U_face.shape[0],face_lengths.shape[1],nq_face,U_face.shape[-1]))

	U_face_mod[:,0,:,:] = U_face[:,:nq_face,:]
	U_face_mod[:,1,:,:] = U_face[:,nq_face:2*nq_face,:]
	U_face_mod[:,2,:,:] = U_face[:,2*nq_face:3*nq_face,:]
	U_face_mod[:,3,:,:] = U_face[:,3*nq_face:4*nq_face,:]
	
	#U_face_bar = np.einsum('ijn,ikl,lm,ik->ikn',U_face,djac_faces,quad_wts_face,1/face_lengths)
	U_face_bar = np.einsum('ijkn,ijkm,km,ij->ijn',U_face_mod,djac_faces,quad_wts_face,1/face_lengths)

	return U_face_bar