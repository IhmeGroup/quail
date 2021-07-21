import numerics.helpers.helpers as helpers

def custom_user_function(solver):
	'''
	Provides a custom interface for the model psr problem into the solver.
	This particular case saves the physical time and solution in time for 
	the "1D" element. The file written is called 'time_hist.txt'.

	Inputs:
	-------
		solver: solver object
	'''
	# User specified time to start custom processing
	tstart = 0.
	if solver.time>=tstart:
		# Unpack
		Uc = solver.state_coeffs
		# jac = solver.physics.jac
		basis_val = solver.elem_helpers.basis_val
		Uq = helpers.evaluate_state(Uc, basis_val)
		time_hist = open('time_hist.txt', 'a')
		
		# Convert applicable variables to strings
		s = str(solver.time)
		s1 = str(Uq[0,0,0])

		# write to the file at each iteration
		time_hist.write(s)
		time_hist.write(' , ')
		time_hist.write(s1)
		time_hist.write('\n')
		time_hist.close()
