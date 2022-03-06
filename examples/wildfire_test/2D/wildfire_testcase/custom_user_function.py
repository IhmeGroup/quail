import numerics.helpers.helpers as helpers
import pickle
import numpy as np

def custom_user_function(solver):
	'''
	Provides a custom interface for the model problem into the solver.
	This particular case saves the physical time and solution in time for 
	the "1D" element. The file written is called 'time_hist.txt'.

	Inputs:
	-------
		solver: solver object
	'''
	tstart = 0.
	if solver.time>=tstart:
		# Unpack
		Uc = solver.state_coeffs
		basis_val = solver.elem_helpers.basis_val
		Uq = helpers.evaluate_state(Uc, basis_val)
		time_hist = open('state_variables_time_hist_TsSource_Tgas.txt', 'a') # Opens a file for appending, creates the file if it does not exist
		
		# Convert applicable variables to strings
		s = str(solver.time)
		s1 = str(Uq[0, 0, 0]) # rho_wood 
		s2 = str(Uq[0, 0, 1]) # rho_water
		s3 = str(Uq[0, 0, 2]) # Temperature 
		# breakpoint()
		# write to the file at each iteration
		time_hist.write(s)
		time_hist.write(' , ')
		time_hist.write(s1)
		time_hist.write(' , ')
		time_hist.write(s2)
		time_hist.write(' , ')
		time_hist.write(s3)
		time_hist.write('\n')
		time_hist.close()
