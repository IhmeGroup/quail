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
	for i in range(errors.shape[0]-1):
		err = np.log(errors[i+1]/errors[i]) / np.log(N[i+1]/N[i])
		print(err)

def array_errors(N, errors):
	err=np.zeros([N.shape[0]-1])
	for i in range(errors.shape[0]-1):
		err[i] = np.log(errors[i+1]/errors[i]) / np.log(N[i+1]/N[i])
	return err

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
filename = 'convergence_inputs.py'
outfile = 'inputdg.py'
tfinal = 1.0 # final solution time
dtinit = np.array([0.02, 0.01, 0.005, 0.0025]) # initial dt for each p
timestepper = 'RK4'
inputdeck = importlib.import_module(filename.replace('.py',''))

order = np.array([1, 2, 3, 4])
meshx = np.array([2, 4, 8, 16, 32, 64, 128])
meshy = np.array([2, 4, 8, 16, 32, 64, 128])
# -------------- END USER INPUTS -------------------------------------- #


inputdeck.TimeStepping['FinalTime'] = tfinal
inputdeck.TimeStepping['TimeStepper'] = timestepper

inputdeck.Output['AutoPostProcess'] = False

l2err = np.zeros([order.shape[0], meshx.shape[0]])

for j in range(order.shape[0]):

	inputdeck.TimeStepping['TimeStepSize'] = dtinit[j]

	for i in range(meshx.shape[0]):

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
		tot_err,_ = post.get_error( mesh, physics, solver, "Scalar")
		l2err[j, i] = tot_err

		inputdeck.TimeStepping['TimeStepSize'] = inputdeck.TimeStepping['TimeStepSize'] / 2.


	file_out = f'convergence_testing/{str(timestepper)}/P{str(order[j])}.pkl'
	write_file(file_out, l2err[j, :])	

print('')
for j in range(order.shape[0]):
	print('------------------------------')
	print(f'Errors for p={order[j]}')
	print_errors(meshx, l2err[j])
	print('------------------------------')






