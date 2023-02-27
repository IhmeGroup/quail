import numerics.helpers.helpers as helpers
import processing.readwritedatafiles as readwritedatafiles
import processing.post as post

import importlib
import json
import numpy as np
import os
import fileinput
import sys
import pickle
import time

def print_errors(N, errors):
	'''
	Prints the errors for convergence studies

	Inputs:
	-------
		N: array whose elements are related to the size of the case
		errors: rate of convergence
	'''
	for i in range(errors.shape[0]-1):
		err = np.log(errors[i+1]/errors[i]) / np.log(N[i+1]/N[i])
		print(err)


def write_file(fname, solution):
	'''
	Writes a pickle file with the solution at the specified final time

	Inputs:
	-------
		fname: filname
		solution: list with final solution for each dt 
	'''
	with open(fname, 'wb') as fo:
		# Save solver
		pickle.dump(solution, fo, pickle.HIGHEST_PROTOCOL)

# ------------------ USER INPUTS -------------------------------------- #
# Change the following inputs for the given problem
filename = 'manufactured_solution.py'
outfile = 'inputdg.py'
tfinal = 1.e-2 # final solution time
dtinit = np.array([0.01]) # initial dt for each p
timestepper = 'SSPRK3'
inputdeck = importlib.import_module(filename.replace('.py',''))

order = np.array([1, 2, 3])
meshx = np.array([2, 4, 8, 16, 32])
meshy = np.array([2, 4, 8, 16, 32])
# -------------- END USER INPUTS -------------------------------------- #
inputdeck.TimeStepping['FinalTime'] = tfinal
inputdeck.TimeStepping['TimeStepper'] = timestepper

inputdeck.Output['AutoPostProcess'] = False

l2err = np.zeros([order.shape[0], meshx.shape[0]])
j=0
while (j < order.shape[0]):

	i = 0
	while (i < meshx.shape[0]):

		inputdeck.Numerics['SolutionOrder'] = order[j]
		inputdeck.Mesh['NumElemsX'] = int(meshx[i])
		inputdeck.Mesh['NumElemsY'] = int(meshy[i])
		# write quail inputdeck 
		with open(outfile, 'w') as f:
			print(f'TimeStepping = {inputdeck.TimeStepping}', file=f)
			print(f'Numerics = {inputdeck.Numerics}', file=f)
			print(f'Output = {inputdeck.Output}', file=f)
			print(f'Mesh = {inputdeck.Mesh}', file=f)
			print(f'InitialCondition = {inputdeck.InitialCondition}', file=f)
			print(f'ExactSolution = {inputdeck.ExactSolution}', file=f)
			print(f'BoundaryConditions = {inputdeck.BoundaryConditions}', file=f)
			print(f'Physics = {inputdeck.Physics}', file=f)
			print(f'SourceTerms = {inputdeck.SourceTerms}', file=f)


		inputdeck = importlib.import_module(outfile.replace('.py',''))

		# Run the simulation
		os.system("quail " + outfile)

		# Access and process the final time step
		final_file = inputdeck.Output['Prefix'] + '_final.pkl'

		# Read data file
		solver = readwritedatafiles.read_data_file(final_file)
		# Unpack
		mesh = solver.mesh
		physics = solver.physics

		# Compute L2 error
		tot_err,_ = post.get_error( mesh, physics, solver, "Density", normalize_by_volume=False)
		l2err[j, i] = tot_err
		print(f'Nelems in x: {meshx[i]}')
		if (int(np.sqrt(solver.mesh.num_elems)) != meshx[i]) or (tot_err > 1.0) or (np.isnan(tot_err)):
			inputdeck.TimeStepping['TimeStepSize'] = inputdeck.TimeStepping['TimeStepSize'] / 3.
			i-=1
		i+=1

	print(f'Output file P{(order[0] + j)}: Solution Order is {solver.order}')
	file_out = f'convergence_testing/P{str(order[j])}.pkl'
	write_file(file_out, l2err[j, :])	
	if solver.order != order[0] + j:
		inputdeck.TimeStepping['TimeStepSize'] = inputdeck.TimeStepping['TimeStepSize'] / 3.
		j-=1
	# update while loop
	j+=1

print('')
for j in range(order.shape[0]):
	print('------------------------------')
	print(f'Errors for p={order[j]}')
	print_errors(meshx, l2err[j])
	print('------------------------------')