import numerics.helpers.helpers as helpers
import processing.readwritedatafiles as readwritedatafiles

import importlib
import numpy as np
import os
import fileinput
import sys
import pickle
import time

def search_and_replace(filename, text_to_search, replacement_text):
	with fileinput.FileInput(filename, inplace=True, backup='.bak') as file:
	    for line in file:
	        print(line.replace(text_to_search, replacement_text), end='')

def write_file(fname, solution):
	with open(fname, 'wb') as fo:
		# Save solver
		pickle.dump(solution, fo, pickle.HIGHEST_PROTOCOL)

# ------------------ USER INPUTS -------------------------------------- #
# Change the following inputs for the given problem
filename = 'pendulum.py'
tfinal = 6.0 # final solution time
dt = np.array([0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 
	0.00390625, 0.001953125, 9.765625e-4, 4.8828125e-4])
dtinit = dt[0]
scheme_name = 'Trapezoidal'
model_psr = importlib.import_module(filename.replace('.py',''))
# -------------- END USER INPUTS -------------------------------------- #

if model_psr.timestep != dtinit:
	search_and_replace(filename, f'timestep = {model_psr.timestep}', 
			f'timestep = {dtinit}')

time.sleep(1)

for j in range(dt.shape[0]):

	importlib.reload(model_psr)
	if model_psr.tfinal != tfinal:
		search_and_replace(filename, f'tfinal = {model_psr.tfinal}', 
			f'tfinal = {tfinal}')
	time.sleep(1)

	solution = []
	importlib.reload(model_psr)

	# Run the simulation
	os.system("quail " + filename)

	# Access and process the final time step
	final_file = model_psr.prefix + '_final.pkl'

	# Read data file
	solver = readwritedatafiles.read_data_file(final_file)

	Uc = solver.state_coeffs
	basis_val = solver.elem_helpers.basis_val
	Uq = helpers.evaluate_state(Uc, basis_val)

	solution.append(Uq[0, 0, 0]) # Assumes 0D

	order = model_psr.order

	file_out = f'convergence_testing/{scheme_name}/{str(j)}.pkl'
	write_file(file_out, solution)

	# text from previous case
	if j+1 < dt.shape[0]:
		text_to_search = f'timestep = {str(dt[j])}'
		replacement_text = f'timestep = {str(dt[j+1])}'
		search_and_replace(filename, text_to_search, replacement_text)

	time.sleep(1)
