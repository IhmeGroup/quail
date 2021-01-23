import numpy as np

# Calculate the difference between the maximum and minimum values
def custom_user_function(solver):
	# Unpack
	Uc = solver.state_coeffs
	elem_helpers = solver.elem_helpers
	basis_val = elem_helpers.basis_val

	# Interpolate state at quad points
	Uq = np.einsum('jn, ink -> ijk', basis_val, Uc) # [ne, nq, ns]

	# Get min and max
	solver.get_min_max_state(Uq)

	# Print difference
	print("Max - min = %g" % (solver.max_state[0] - solver.min_state[0]))