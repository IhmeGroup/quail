import numpy as np
import pytest
import sys
sys.path.append('../src')

import numerics.basis.basis as basis_defs
import numerics.basis.ader_tools as basis_st_tools
import meshing.common as mesh_common

rtol = 1e-15
atol = 1e-15

@pytest.mark.parametrize('order', [
	# Order of Lagrange basis
	1, 2
])
def test_legendre_spacetime_massmatrix(order):
	'''
	This test compares the analytic solution of the space-time MM to 
	the computed one.
	'''
	basis = basis_defs.LegendreQuad(order)
	mesh = mesh_common.mesh_1D(num_elems=1, xmin=-1., xmax=1.)
		
	# Set quadrature
	basis.set_elem_quadrature_type("GaussLegendre")
	basis.set_face_quadrature_type("GaussLegendre")

	MM = basis_st_tools.get_elem_mass_matrix_ader(mesh, basis, order, -1)

	'''
	Reference:
	Dumbser, M., Enaux, C., and Toro, E. JCP 2008
		Vol. 227, Issue 8, Pgs: 3971 - 4001 Appendix B
	'''
	# Multiply by 4.0 caused by reference element shift [0, 1] -> [-1, 1]
	if order == 1:
		expected = 4.0 * np.array([[1., 0., 0., 0.], 
						 	   	  [0., 1./3., 0., 0.],
							      [0., 0., 1./3., 0],
							      [0., 0., 0., 1./9.]])
	elif order == 2:
		expected = 4.0 * np.array([[1., 0., 0., 0., 0., 0., 0., 0., 0.], 
				 	   	  [0., 1./3., 0., 0., 0., 0., 0., 0., 0.],
					      [0., 0., 1./5., 0., 0., 0., 0., 0., 0.],
					      [0., 0., 0., 1./3., 0., 0., 0., 0., 0.],
					      [0., 0., 0., 0., 1./9., 0., 0., 0., 0.],
					      [0., 0., 0., 0., 0., 1./15., 0., 0., 0.],
					      [0., 0., 0., 0., 0., 0., 1./5, 0., 0.],
					      [0., 0., 0., 0., 0., 0., 0., 1./15., 0.],
					      [0., 0., 0., 0., 0., 0., 0., 0., 1./25.]])
	# Assert
	np.testing.assert_allclose(MM, expected, rtol, atol)


@pytest.mark.parametrize('order', [
	# Order of Lagrange basis
	1, 2
])
def test_temporal_flux_ader_tn(order):
	'''
	This test compares the analytic solution of the space-time temporal
	flux evaluated at t^{n} to the computed one.
	'''
	basis = basis_defs.LegendreSeg(order)
	basis_st = basis_defs.LegendreQuad(order)
	mesh = mesh_common.mesh_1D(num_elems=1, xmin=-1., xmax=1.)

	# Set quadrature
	basis.set_elem_quadrature_type("GaussLegendre")
	basis.set_face_quadrature_type("GaussLegendre")
	basis_st.set_elem_quadrature_type("GaussLegendre")
	basis_st.set_face_quadrature_type("GaussLegendre")
	mesh.gbasis.set_elem_quadrature_type("GaussLegendre")
	mesh.gbasis.set_face_quadrature_type("GaussLegendre")
	
	# Get flux matrices in time
	FTR = basis_st_tools.get_temporal_flux_ader(mesh, basis_st, basis,
			order, physical_space=False)

	'''
	Reference:
	Dumbser, M., Enaux, C., and Toro, E. JCP 2008
		Vol. 227, Issue 8, Pgs: 3971 - 4001 Appendix B
	'''
	# Multiply by 2.0 caused by reference element shift [0, 1] -> [-1, 1]
	if order == 1:
		expected = 2.0 * np.array([[1., 0.], 
						 	   	  [0., 1./3.],
							      [-1., 0.],
							      [0., -1./3.]])
	elif order == 2:
		expected = 2.0 * np.array([[1., 0., 0.], 
				 	   	  [0., 1./3., 0.],
					      [0., 0., 1./5.],
					      [-1, 0., 0.  ],
					      [0., -1./3., 0.],
					      [0., 0., -1./5.],
					      [1., 0., 0],
					      [0., 1./3., 0.],
					      [0., 0., 1./5.]])

	# Assert
	np.testing.assert_allclose(FTR, expected, rtol, atol)


@pytest.mark.parametrize('order', [
	# Order of Lagrange basis
	1, 2
])
def test_temporal_flux_ader_tnp1(order):
	'''
	This test compares the analytic solution of the space-time temporal
	flux evaluated at t^{n+1} to the computed one.
	'''
	basis_st = basis_defs.LegendreQuad(order)
	mesh = mesh_common.mesh_1D(num_elems=1, xmin=-1., xmax=1.)

	# Set quadrature
	basis_st.set_elem_quadrature_type("GaussLegendre")
	basis_st.set_face_quadrature_type("GaussLegendre")
	mesh.gbasis.set_elem_quadrature_type("GaussLegendre")
	mesh.gbasis.set_face_quadrature_type("GaussLegendre")
	
	# Get flux matrices in time
	FTL = basis_st_tools.get_temporal_flux_ader(mesh, basis_st, basis_st,
			order, physical_space=False)

	'''
	Reference:
	Dumbser, M., Enaux, C., and Toro, E. JCP 2008
		Vol. 227, Issue 8, Pgs: 3971 - 4001 Appendix B
	'''
	# Multiply by 2.0 caused by reference element shift [0, 1] -> [-1, 1]
	if order == 1:
		expected = 2.0 * np.array([[1., 0., 1., 0.], 
						 	   	  [0., 1./3., 0., 1./3.],
							      [1., 0., 1., 0],
							      [0., 1./3., 0., 1./3.]])
	elif order == 2:
		expected = 2.0 * np.array([[1., 0., 0., 1., 0., 0., 1., 0., 0.], 
				 	   			   [0., 1./3., 0., 0., 1./3., 0., 0., 1./3., 0.],
				 	   			   [0., 0., 1./5., 0., 0., 1./5., 0., 0., 1./5.],
				 	   			   [1., 0., 0., 1., 0., 0., 1., 0., 0.],
				 	   			   [0., 1./3., 0., 0., 1./3., 0., 0., 1./3., 0.],
				 	   			   [0., 0., 1./5., 0., 0., 1./5., 0., 0., 1./5.],
				 	   			   [1., 0., 0., 1., 0., 0., 1., 0., 0.],
				 	   			   [0., 1./3., 0., 0., 1./3., 0., 0., 1./3., 0.],
				 	   			   [0., 0., 1./5., 0., 0., 1./5., 0., 0., 1./5.]])

	# Assert
	np.testing.assert_allclose(FTL, expected, rtol, atol)


@pytest.mark.parametrize('dt', [
	# A few random time step sizes (should not effect matrix)
	1.0, 0.1, 0.0132
])
@pytest.mark.parametrize('order', [
	# Order of Lagrange basis
	1, 2
])
def test_legendre_spacetime_stiffnessmatrix_spacedir(dt, order):
	'''
	This test compares the analytic solution of the space-time 
	stiffness matrix in space to the computed one.
	'''
	basis = basis_defs.LegendreSeg(order)
	basis_st = basis_defs.LegendreQuad(order)
	mesh = mesh_common.mesh_1D(num_elems=1, xmin=-1., xmax=1.)

	# Set quadrature
	basis.set_elem_quadrature_type("GaussLegendre")
	basis.set_face_quadrature_type("GaussLegendre")
	basis_st.set_elem_quadrature_type("GaussLegendre")
	basis_st.set_face_quadrature_type("GaussLegendre")
	mesh.gbasis.set_elem_quadrature_type("GaussLegendre")
	mesh.gbasis.set_face_quadrature_type("GaussLegendre")

	# Get stiffness matrix in space
	SMS = basis_st_tools.get_stiffness_matrix_ader(mesh, basis, basis_st,
			order, dt, elem_ID=0, grad_dir=0, physical_space=False)
	'''
	Reference:
	Dumbser, M., Enaux, C., and Toro, E. JCP 2008
		Vol. 227, Issue 8, Pgs: 3971 - 4001 Appendix B
	'''
	# Multiply by 2.0 caused by reference element shift [0, 1] -> [-1, 1]
	if order == 1:
		expected = 2.0 * np.array([[0., 2., 0., 0.], 
						 	   	  [0., 0., 0., 0.],
							      [0., 0., 0., 2./3.],
							      [0., 0., 0., 0.]])
	elif order == 2:
		expected = 2.0 * np.array([[0., 2., 0., 0., 0., 0., 0., 0., 0.], 
				 	   	  [0., 0., 2., 0., 0., 0., 0., 0., 0.],
					      [0., 0., 0., 0., 0., 0., 0., 0., 0.],
					      [0., 0., 0., 0., 2./3., 0., 0., 0., 0.],
					      [0., 0., 0., 0., 0., 2./3., 0., 0., 0.],
					      [0., 0., 0., 0., 0., 0., 0., 0., 0.],
					      [0., 0., 0., 0., 0., 0., 0., 2./5., 0.],
					      [0., 0., 0., 0., 0., 0., 0., 0., 2./5.],
					      [0., 0., 0., 0., 0., 0., 0., 0., 0.]])
	# Assert
	np.testing.assert_allclose(SMS.transpose(), expected, rtol, atol)


@pytest.mark.parametrize('dt', [
	# A few random time step sizes (should not effect matrix)
	1.0, 0.1, 0.0132
])
@pytest.mark.parametrize('order', [
	# Order of Lagrange basis
	1, 2
])
def test_legendre_spacetime_stiffnessmatrix_timedir(dt, order):
	'''
	This test compares the analytic solution of the space-time 
	stiffness matrix in time to the computed one.
	'''
	basis = basis_defs.LegendreSeg(order)
	basis_st = basis_defs.LegendreQuad(order)
	mesh = mesh_common.mesh_1D(num_elems=1, xmin=-1., xmax=1.)

	# Set quadrature
	basis.set_elem_quadrature_type("GaussLegendre")
	basis.set_face_quadrature_type("GaussLegendre")
	basis_st.set_elem_quadrature_type("GaussLegendre")
	basis_st.set_face_quadrature_type("GaussLegendre")
	mesh.gbasis.set_elem_quadrature_type("GaussLegendre")
	mesh.gbasis.set_face_quadrature_type("GaussLegendre")

	# Get stiffness matrix in time
	SMT = basis_st_tools.get_stiffness_matrix_ader(mesh, basis, basis_st,
			order, dt, elem_ID=0, grad_dir=-1, physical_space=False)
	'''
	Reference:
	Dumbser, M., Enaux, C., and Toro, E. JCP 2008
		Vol. 227, Issue 8, Pgs: 3971 - 4001 Appendix B
	'''
	# Multiply by 2.0 caused by reference element shift [0, 1] -> [-1, 1]
	if order == 1:
		expected = 2.0 * np.array([[0., 0., 0., 0.], 
						 	   	  [0., 0., 0., 0.],
							      [2., 0., 0., 0.],
							      [0., 2./3., 0., 0.]])
	elif order == 2:
		expected = 2.0 * np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0.], 
				 	   	  [0., 0., 0., 0., 0., 0., 0., 0., 0.],
					      [0., 0., 0., 0., 0., 0., 0., 0., 0.],
					      [2., 0., 0., 0., 0., 0., 0., 0., 0.],
					      [0., 2./3., 0., 0., 0., 0., 0., 0., 0.],
					      [0., 0., 2./5., 0., 0., 0., 0., 0., 0.],
					      [0., 0., 0., 2., 0., 0., 0., 0., 0.],
					      [0., 0., 0., 0., 2./3., 0., 0., 0., 0.],
					      [0., 0., 0., 0., 0., 2./5., 0., 0., 0.]])
	# Assert
	np.testing.assert_allclose(SMT, expected, rtol, atol)