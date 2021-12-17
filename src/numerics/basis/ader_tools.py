# ------------------------------------------------------------------------ #
#
#       quail: A lightweight discontinuous Galerkin code for
#              teaching and prototyping
#		<https://github.com/IhmeGroup/quail>
#       
#		Copyright (C) 2020-2021
#
#       This program is distributed under the terms of the GNU
#		General Public License v3.0. You should have received a copy
#       of the GNU General Public License along with this program.  
#		If not, see <https://www.gnu.org/licenses/>.
#
# ------------------------------------------------------------------------ #

# ------------------------------------------------------------------------ #
#
#		File : src/numerics/basis/ader_tools.py
#
#		Contains helper definitions for the shape and basis classes with
#		specific application to the ADER-DG solver.
#
# ------------------------------------------------------------------------ #
from abc import ABC, abstractmethod
import numpy as np

from general import BasisType, ShapeType

import meshing.gmsh as mesh_gmsh

import numerics.basis.basis as basis_defs
import numerics.basis.tools as basis_tools


def set_basis_spacetime(mesh, order, basis_name):
	'''
	Sets the space-time basis class given the basis_name string argument

	Inputs:
	-------
		order: solution order
		basis_name: name of the spatial basis function used to determine
			the space-time basis function we wish to instantiate
			as a class

	Outputs:
	--------
		basis_st: instantiated space-time basis class

	Raise:
	------
		If the basis class is not defined returns a NotImplementedError
	'''
	if BasisType[basis_name] == BasisType.LagrangeSeg:
		basis_st = basis_defs.LagrangeQuad(order)
	elif BasisType[basis_name] == BasisType.LegendreSeg:
		basis_st = basis_defs.LegendreQuad(order)
	elif BasisType[basis_name] == BasisType.LagrangeQuad:
		basis_st = basis_defs.LagrangeHex(order)
	elif BasisType[basis_name] == BasisType.LagrangeTri:
		basis_st = basis_defs.LagrangePrism(order)
	else:
		raise NotImplementedError

	return basis_st


def get_elem_inv_mass_matrix_ader(mesh, basis, order, elem_ID=-1,
		physical_space=False):
	'''
	Calculate the inverse mass matrix for ADER-DG prediction step

	Inputs:
	-------
		mesh: mesh object
		basis: basis object
		order: solution order
		elem_ID: element index
		physical_space: [OPTIONAL] Flag to calc matrix in physical or
			reference space (default: False {reference space})

	Outputs:
	--------
		iMM: inverse mass matrix for ADER-DG predictor step [nb_st, nb_st]
	'''
	MM = get_elem_mass_matrix_ader(mesh, basis, order, elem_ID,
			physical_space)

	iMM = np.linalg.inv(MM)

	return iMM # [nb_st, nb_st]


def get_stiffness_matrix_ader(mesh, basis, basis_st, order, dt, elem_ID,
		grad_dir, physical_space=False):
	'''
	Calculate the stiffness matrix for ADER-DG prediction step

	Inputs:
	-------
		mesh: mesh object
		basis: basis object
		basis_st: space-time basis object
		order: solution order
		dt: time step
		elem_ID: element index
		grad_dir: direction of gradient calculation

	Outputs:
	--------
		SM: stiffness matrix for ADER-DG [nb_st, nb_st]
	'''
	ndims = mesh.ndims

	quad_order_st = basis_st.get_quadrature_order(mesh, order*2)
	quad_order = quad_order_st

	quad_pts_st, quad_wts_st = basis_st.get_quadrature_data(quad_order_st)
	quad_pts, quad_wts = basis.get_quadrature_data(quad_order)

	nq_st = quad_pts_st.shape[0]
	nq = quad_pts.shape[0]

	if physical_space:
		djac, jac, ijac = basis_tools.element_jacobian(mesh, elem_ID,
				quad_pts_st, get_djac=True, get_ijac=True)

		ijac_st = np.zeros([nq_st, ndims + 1, ndims + 1])
		ijac_st[:, :ndims, :ndims] = ijac

		# Add the temporal Jacobian in the ndims+1 dimension
		ijac_st[:, ndims, ndims] = 2./dt

	basis_st.get_basis_val_grads(quad_pts_st, get_val=True,
			get_ref_grad=True)

	nb_st = basis_st.basis_val.shape[1]
	basis_st_val = basis_st.basis_val

	if physical_space:
		basis_ref_grad = basis_st.basis_ref_grad
		basis_st_grad = np.transpose(np.matmul(ijac_st.transpose(0, 2, 1),
				basis_ref_grad.transpose(0, 2, 1)), (0, 2, 1))
	else:
		basis_st_grad = basis_st.basis_ref_grad

	# ------------------------------------------------------------------- #
	# Example of ADER Stiffness Matrix calculation using for-loops
	# ------------------------------------------------------------------- #
	#
	# nb_st = basis_st.basis_val.shape[1]
	#
	# for i in range(nb_st):
	#	  for j in range(nb_st):
	#		  a = 0.
	#		  for iq in range(nq_st):
	#			  a += basis_st_grad[iq, i, grad_dir]*basis_st_val[iq, j]
	#					 * quad_wts_st[iq]
	#		  SM[i,j] = a
	#
	# ------------------------------------------------------------------- #
	SM = np.matmul(basis_st_grad[:, :, grad_dir].transpose(),
			 basis_st_val * quad_wts_st)

	return SM # [nb_st, nb_st]


def get_temporal_flux_ader(mesh, basis1, basis2, order,
		physical_space=False):
	'''
	Calculate the temporal flux matrix for ADER-DG prediction step

	Inputs:
	-------
		mesh: mesh object
		basis1: basis object
		basis2: basis object
		order: solution order
		physical_space: [OPTIONAL] Flag to calc matrix in physical or
			reference space (default: False {reference space})

	Outputs:
	--------
		FT: flux matrix for ADER-DG # [nb_st, nb] or [nb_st, nb_st]
			(shape depends on basis objects passed in)

	Notes:
	------
		Can work at tau_n and tau_n+1 depending on basis combinations
	'''

	gbasis = mesh.gbasis
	quad_order = gbasis.get_quadrature_order(mesh, order*2)
	quad_pts, quad_wts = gbasis.get_quadrature_data(quad_order)

	if basis1.BASIS_TYPE == basis2.BASIS_TYPE:
		# If both bases are space-time you are at tau_{n+1} in ref time
		# Evaluate basis functions at tau_{n+1}
		face_ID = basis1.FACE_TIME_MAPPING[1]

		basis2.get_basis_face_val_grads(mesh, face_ID, quad_pts, basis1,
				get_val=True)
	else:
		# If bases are different you are at tau_{n} in ref time
		# Evaluate basis at tau_{n}
		face_ID = basis1.FACE_TIME_MAPPING[0]

		basis2.get_basis_val_grads(quad_pts, get_val=True,
				get_ref_grad=False)

	basis1.get_basis_face_val_grads(mesh, face_ID, quad_pts, basis1,
			get_val=True)

	# ------------------------------------------------------------------- #
	# Example of ADER Flux Matrix calculation using for-loops
	# ------------------------------------------------------------------- #
	#
	# nq = quad_pts.shape[0]
	# nb_st = basis1.basis_val.shape[1]
	# nb = basis2.basis_val.shape[1]
	#
	# for i in range(nb_st):
	#	  for j in range(nb):
	#		  a = 0.
	#		  for iq in range(nq):
	#			  a += basis1.basis_val[iq, i]*basis2.basis_val[iq,
	#					 j]*quad_wts[iq]
	#		  FT[i,j] = a
	#
	# ------------------------------------------------------------------- #
	FT = np.matmul(basis1.basis_val.transpose(), basis2.basis_val * \
			quad_wts) # [nb_st, nb] or [nb_st, nb_st]

	return FT # [nb_st, nb] or [nb_st, nb_st]


def get_elem_mass_matrix_ader(mesh, basis, order, elem_ID=-1,
		physical_space=False):
	'''
	Calculate the mass matrix for ADER-DG prediction step

	Inputs:
	-------
		mesh: mesh object
		basis: basis object
		order: solution order
		elem_ID: [OPTIONAL] element index
		physical_space: [OPTIONAL] Flag to calc matrix in physical or
			reference space (default: False {reference space})

	Outputs:
	--------
		MM: mass matrix for ADER-DG [nb_st, nb_st]
	'''
	if physical_space:
		gbasis = mesh.gbasis
		quad_order = gbasis.get_quadrature_order(mesh, order*2)
	else:
		quad_order = order*2

	quad_pts, quad_wts = basis.get_quadrature_data(quad_order)

	nq = quad_pts.shape[0]

	basis.get_basis_val_grads(quad_pts, get_val=True)

	if physical_space:
		djac, _, _ = basis_tools.element_jacobian(mesh, elem_ID, quad_pts,
				get_djac=True)

		if len(djac) == 1:
			djac = np.full(nq, djac[0])
	else:
		djac = np.full(nq, 1.).reshape(nq, 1)

	# ------------------------------------------------------------------- #
	# Example of ADER Flux Matrix calculation using for-loops
	# ------------------------------------------------------------------- #
	#
	# nb_st = basis.basis_val.shape[1]
	# basis_val = basis.basis_val
	#
	# for i in range(nb_st):
	#	  for j in range(nb_st):
	#		  a = 0.
	#		  for iq in range(nq):
	#			  a += basis_val[iq,i]*basis_val[iq,j]*quad_wts[iq]*djac[iq]
	#		  MM[i,j] = a
	#
	# ------------------------------------------------------------------- #
	MM = np.matmul(basis.basis_val.transpose(), basis.basis_val * \
			quad_wts * djac) # [nb_st, nb_st]

	return MM # [nb_st, nb_st]


def get_tiling_constants_segment(nq):
	'''
	Precomputes the tiling constants for the ADER-DG scheme on Segments. 
	Tiling constants are used with the numpy 'tile' function to build
	arrays to maintain consistency for the tensor multiplications 
	throughout the solver.

	Inputs:
	-------
		nq: number of quadrature points 

	Outputs:
	--------
		nq_t: quadrature points tiling constant
		nb_t: basis coefficients tiling constant
		time_skip: Value to skip when building time
				array for each bface in get_boundary_face_residual
				in src/solver/ADERDG.py
		time_tile: time array tiling constant for each bface in 
				get_boundary_face_residual in src/solver/ADERDG.py
	'''
	return nq, 1, nq


def get_tiling_constants_quad(nq, nq_bface):
	'''
	Precomputes the tiling constants for the ADER-DG scheme on quads. 
	Tiling constants are used with the numpy 'tile' function to build
	arrays to maintain consistency for the tensor multiplications 
	throughout the solver.

	Inputs:
	-------
		nq: number of quadrature points 
		nq_bface: number of quadrature points on boundary face

	Outputs:
	--------
		nq_t: quadrature points tiling constant
		time_skip: Value to skip when building time
				array for each bface in get_boundary_face_residual
				in src/solver/ADERDG.py. Also used for tiling
				in the interior and boundary face integral.
		time_tile: time array tiling constant for each bface in 
				get_boundary_face_residual in src/solver/ADERDG.py
	'''
	return int(np.sqrt(nq)), int(np.sqrt(nq_bface)), int(np.sqrt(nq_bface))


def get_tiling_constants_tri(nq):
	'''
	Precomputes the tiling constants for the ADER-DG scheme on tris. 
	Tiling constants are used with the numpy 'tile' function to build
	arrays to maintain consistency for the tensor multiplications 
	throughout the solver.

	Inputs:
	-------
		nq: number of quadrature points

	Outputs:
	--------
		nq_t: quadrature points tiling constant
		time_skip: Value to skip when building time
				array for each bface in get_boundary_face_residual
				in src/solver/ADERDG.py. Also used for tiling
				in the interior and boundary face integral.
		time_tile: time array tiling constant for each bface in 
				get_boundary_face_residual in src/solver/ADERDG.py
	'''
	return int(np.sqrt(nq)), int(np.sqrt(nq)), int(np.sqrt(nq))