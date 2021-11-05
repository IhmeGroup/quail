import numpy as np
import pytest
import os
import pickle
import subprocess
import sys
sys.path.append('../src')
sys.path.append('../../../../src/')

import numerics.helpers.helpers as helpers
import general

import meshing.common as mesh_common

rtol = 1e-15
atol = 1e-15


# def test_create_limiter(order):
# 	'''
# 	This test ensures that a limiter object can be successfully created
# 	and stored. It also generates a data file whose solver object will
# 	be utilized in the tests below.

# 	Inputs:
# 	-------
# 		order: quadrature order
# 	'''
# 	# Get current directory
# 	cwd = os.getcwd()

# 	# Get test directory
# 	test_dir = os.path.dirname(os.path.abspath(__file__))

# 	# Enter test directory
# 	os.chdir(test_dir)

# 	# Run quail with a barebones input file just so we have a solver object
# 	subprocess.check_output([f'quail', 'input_file.py',
# 			], stderr=subprocess.STDOUT)

# 	# Extract solver object and solution array
# 	with open('Data_final.pkl', 'rb') as f:
# 		# Read final solution from file
# 		solver = pickle.load(f)
# 		Uc = solver.state_coeffs

# 	# Back to original directory
# 	os.chdir(cwd)


def test_positivity_preserving_limiter_solution_already_positive():
	'''
	This test ensures that the limiter does not modify an Euler solution
	that already has positive density and pressure.
	'''
	# Get current directory
	cwd = os.getcwd()

	# Get test directory
	test_dir = os.path.dirname(os.path.abspath(__file__))

	# Enter test directory
	os.chdir(test_dir)

	# Run quail with a barebones input file to create solver object
	subprocess.check_output([f'quail', 'input_file.py',
			], stderr=subprocess.STDOUT)

	# Extract solver object and solution array
	with open('Data_final.pkl', 'rb') as f:
		# Read final solution from file
		solver = pickle.load(f)
		Uc = solver.state_coeffs

	# Back to original directory
	os.chdir(cwd)

	# Copy original solution
	Uc_orig = Uc.copy()

	# Apply limiter
	solver.apply_limiter(Uc)

	np.testing.assert_allclose(Uc, Uc_orig, rtol, atol)


def compute_density(solver, Uc):
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
	# Compute density at quadrature points
	rho_elem_faces = physics.compute_variable("Density",
			U_elem_faces)
	# Compute mean
	rho_bar = physics.compute_variable("Density",
			U_bar)

	return rho_elem_faces, rho_bar


def test_positivity_preserving_limiter_solution_positive_density():
	'''
	This test ensures that the limiter enforces positive density
	and maintains conservation.
	'''
	# Get current directory
	cwd = os.getcwd()

	# Get test directory
	test_dir = os.path.dirname(os.path.abspath(__file__))

	# Enter test directory
	os.chdir(test_dir)

	# Run quail with a barebones input file to create solver object
	# subprocess.check_output([f'quail', 'input_file.py',
	# 		], stderr=subprocess.STDOUT)

	# Extract solver object and solution array
	with open('Data_final.pkl', 'rb') as f:
		# Read final solution from file
		solver = pickle.load(f)
		Uc = solver.state_coeffs

	# Back to original directory
	os.chdir(cwd)

	# Modify solution to have negative density
	srho = solver.physics.get_state_slice("Density")
	Uc[:, 1:2, srho] = -0.1

	# Compute mean density of original solution
	Uc_orig = Uc.copy()
	rho_orig_elem_faces, rho_orig_bar = compute_density(solver, Uc_orig)

	# np.testing.assert_allclose(Uc.shape, Uc_orig.shape, rtol, atol)

	# Apply limiter
	solver.apply_limiter(Uc)

	# Compute mean density and density at elem and face quadrature points
	rho_elem_faces, rho_bar = compute_density(solver, Uc)

	rho_elem_faces_expected = np.array([[[7.098076210775963e-01],
	        [1.901923789224034e-01],
	        [8.999999998999996e-01],
	        [1.000000082740371e-10]]])

	np.testing.assert_allclose(rho_bar, rho_orig_bar, rtol, atol)
	np.testing.assert_allclose(rho_elem_faces, rho_elem_faces_expected, rtol, 
			atol)
