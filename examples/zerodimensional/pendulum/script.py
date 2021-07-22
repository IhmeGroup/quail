import sys; sys.path.append('/Users/brettbornhoft/utilities/quail_dev/src'); sys.path.append('./src')
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

filename = 'pendulum.py'
model_psr = importlib.import_module(filename.replace('.py',''))

if model_psr.timestep != 0.5:
	search_and_replace(filename, 'timestep = '+str(model_psr.timestep), 'timestep = 0.5')

time.sleep(1)

# dt = np.array([0.01, 0.001, 0.0001])
# dt = np.array([40.0, 20.0, 10.0, 5.0, 2.5, 1.25])
dt = np.array([0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 
	0.00390625, 0.001953125, 9.765625e-4, 4.8828125e-4])
for j in range(dt.shape[0]):

	# tin = 40.0
	importlib.reload(model_psr)
	if model_psr.tfinal != 6.0:
		search_and_replace(filename, 'tfinal = '+str(model_psr.tfinal), 'tfinal = 6.0')
	time.sleep(1)

	solution = []
	# for i in range(1):

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

	solution.append(Uq[0,1,0])

	# text from previous case
	# text_to_search = 'tfinal = ' + str(tin)
	# adjust for replacement text
	# tin = tin + 40.0
	# replacement_text = 'tfinal = ' + str(tin)
	# search_and_replace(filename, text_to_search, replacement_text)
	# time.sleep(1)

	order = model_psr.order

	# file_out = 'time_accuracy_study/RK4/'+ str(j)+'.pkl'
	# file_out = 'time_accuracy_study/Strang/' + str(int(dt[j]))+'.pkl'
	file_out = 'time_accuracy_study/ADER/p'+str(order)+'/'+str(j)+'.pkl'
	write_file(file_out, solution)

	# text from previous case
	if j+1 < dt.shape[0]:
		text_to_search = 'timestep = ' + str(dt[j])
		replacement_text = 'timestep = ' + str(dt[j+1])
		search_and_replace(filename, text_to_search, replacement_text)

	time.sleep(1)