import numpy as np
import pytest
import os
import pickle
import subprocess
import sys
sys.path.append('../src')

import general
import meshing.common as mesh_common
import meshing.tools as mesh_tools
import numerics.helpers.helpers as helpers
import numerics.limiting.positivitypreserving as positivitypreserving
import physics.euler.euler as euler
import solver.DG as DG

rtol = 1e-15
atol = 1e-15


def create_solver_object():
	'''
	This function creates a solver object that stores the positivity-
	preserving limiter object needed for the tests here.
	'''
	mesh = mesh_common.mesh_1D(num_elems=1, xmin=0., xmax=1.)

	mesh_tools.make_periodic_translational(mesh, x1="x1", x2="x2")

	params = general.set_solver_params(SolutionOrder=1, 
			SolutionBasis="LagrangeSeg",
			ApplyLimiters=["PositivityPreserving"])

	physics = euler.Euler1D()
	physics.set_conv_num_flux("Roe")
	physics.set_physical_params()
	U = np.array([1., 0., 1.])
	physics.set_IC(IC_type="Uniform", state=U)

	solver = DG.DG(params, physics, mesh)

	return solver


def compute_variable(solver, Uc, var_name):
	'''
	This function computes the desired variable at the element and
	face quadrature points, as well as the mean of the variable over
	the element.
	'''
	# Interpolate state at quadrature points over element and on faces
	limiter = solver.limiters[0]
	basis = solver.basis
	physics = solver.physics
	U_elem_faces = helpers.evaluate_state(Uc, limiter.basis_val_elem_faces,
			skip_interp=basis.skip_interp)
	nq_elem = limiter.quad_wts_elem.shape[0]
	U_elem = U_elem_faces[:, :nq_elem, :]
	# Average value of state
	U_bar = helpers.get_element_mean(U_elem, limiter.quad_wts_elem, 
			limiter.djac_elems, limiter.elem_vols)
	# Compute variable at quadrature points
	var_elem_faces = physics.compute_variable(var_name,
			U_elem_faces)
	# Compute mean
	var_bar = physics.compute_variable(var_name,
			U_bar)

	return var_elem_faces, var_bar


def test_positivity_preserving_limiter_solution_already_positive():
	'''
	This test ensures that the limiter does not modify an Euler solution
	that already has positive density and pressure.
	'''
	solver = create_solver_object()
	Uc = solver.state_coeffs

	# Copy original solution
	Uc_orig = Uc.copy()

	# Apply limiter
	solver.apply_limiter(Uc)

	np.testing.assert_allclose(Uc, Uc_orig, rtol, atol)


def test_positivity_preserving_limiter_solution_positive_density():
	'''
	This test ensures that the limiter enforces positive density
	and maintains conservation.
	'''
	solver = create_solver_object()
	Uc = solver.state_coeffs

	# Modify solution to have negative density
	srho = solver.physics.get_state_slice("Density")
	Uc[:, 1:2, srho] = -0.1

	# Compute mean density of original solution
	Uc_orig = Uc.copy()
	rho_orig_elem_faces, rho_orig_bar = compute_variable(solver, Uc_orig,
			"Density")

	# Apply limiter
	solver.apply_limiter(Uc)

	# Compute mean density and density at elem and face quadrature points
	rho_elem_faces, rho_bar = compute_variable(solver, Uc, "Density")

	# Minimum density
	pos_tol = positivitypreserving.POS_TOL

	np.testing.assert_allclose(rho_bar, rho_orig_bar, rtol, atol)
	assert(np.amin(rho_elem_faces) >= pos_tol - atol)


def test_positivity_preserving_limiter_solution_positive_pressure():
	'''
	This test ensures that the limiter enforces positive pressure
	and maintains conservation.
	'''
	solver = create_solver_object()
	Uc = solver.state_coeffs

	# Modify solution to have negative pressure
	srhoE = solver.physics.get_state_slice("Energy")
	Uc[:, 1:2, srhoE] = -0.25

	# Compute mean pressure of original solution
	Uc_orig = Uc.copy()
	p_orig_elem_faces, p_orig_bar = compute_variable(solver, Uc_orig,
			"Pressure")

	# Apply limiter
	solver.apply_limiter(Uc)

	# Compute mean pressure and pressure at elem and face quadrature points
	p_elem_faces, p_bar = compute_variable(solver, Uc, "Pressure")

	# Minimum pressure
	pos_tol = positivitypreserving.POS_TOL

	np.testing.assert_allclose(p_bar, p_orig_bar, rtol, atol)
	assert(np.amin(p_elem_faces) >= pos_tol - atol)
	# np.testing.assert_allclose(np.amin(p_elem_faces), pos_tol, rtol, 
	# 		atol)
