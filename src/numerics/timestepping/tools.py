# ------------------------------------------------------------------------ #
#
#       File : src/numerics/timestepping/tools.py
#
#       Contains helper functions for the stepper class
#
#       Authors: Eric Ching and Brett Bornhoft
#
#       Created: January 2020
#      
# ------------------------------------------------------------------------ #
from abc import ABC, abstractmethod
import code
import math
import numpy as np 

from data import ArrayList
from general import StepperType, ODESolverType
from solver.tools import mult_inv_mass_matrix
import numerics.basis.tools as basis_tools
import numerics.timestepping.stepper as stepper_defs
import solver.tools as solver_tools

def set_stepper(params, U):
	'''
	Given the TimeStepper parameter, set the stepper object

	Inputs:
	-------
		params: list of parameters for solver
		U: solution vector for instantiaing stepper class [nelem, nb, ns]

	Outputs:
	-------- 
	    stepper: instantiated stepper object
	'''
	TimeStepper = params["TimeStepper"]
	if StepperType[TimeStepper] == StepperType.FE:
		stepper = stepper_defs.FE(U)
	elif StepperType[TimeStepper] == StepperType.RK4:
		stepper = stepper_defs.RK4(U)
	elif StepperType[TimeStepper] == StepperType.LSRK4:
		stepper = stepper_defs.LSRK4(U)
	elif StepperType[TimeStepper] == StepperType.SSPRK3:
		stepper = stepper_defs.SSPRK3(U)
	# if setting a splitting scheme select solvers for the splits
	elif StepperType[TimeStepper] == StepperType.Strang:
		stepper = stepper_defs.Strang(U)
		stepper.set_split_schemes(params["OperatorSplittingExplicit"], 
			params["OperatorSplittingImplicit"], U)
	elif StepperType[TimeStepper] == StepperType.Simpler:
		stepper = stepper_defs.Simpler(U)
		stepper.set_split_schemes(params["OperatorSplittingExplicit"], 
			params["OperatorSplittingImplicit"], U)
	else:
		raise NotImplementedError("Time scheme not supported")
	return stepper


def set_time_stepping_approach(stepper, params):
	'''
	Sets stepper.get_time_step method given input parameters

	Inputs:
	-------
		stepper: stepper object (e.g., FE, RK4, etc...)
		params: list of parameters for solver

	Outputs:
	-------- 
		get_time_step: method selected to calculate dt
		num_time_steps: number of time steps for the solution
	'''

	# unpack time stepping settings
	cfl = params["CFL"]
	timestepsize = params["TimeStepSize"]
	num_time_steps = params["NumTimeSteps"]
	FinalTime = params["FinalTime"]

	'''
	Hierarchy for cases goes:
		1. number of time steps
		2. time step size
		3. CFL number
	'''
	if num_time_steps != None:
		stepper.get_time_step = get_dt_from_num_time_steps
		stepper.num_time_steps = num_time_steps
	elif timestepsize != None:
		stepper.get_time_step = get_dt_from_timestepsize
		stepper.num_time_steps = math.ceil(FinalTime/timestepsize)
	elif cfl != None:
		stepper.get_time_step = get_dt_from_cfl
		stepper.num_time_steps = 1


def get_dt_from_num_time_steps(stepper, solver):
	'''
	Calculates dt from the specified number of time steps

	Inputs:
	-------
		stepper: stepper object (e.g., FE, RK4, etc...)
		solver: solver object (e.g., DG, ADERDG, etc...)

	Outputs:
	-------- 
		dt: time step for the solver
	'''
	num_time_steps = stepper.num_time_steps
	tfinal = solver.params["FinalTime"]
	time = solver.time

	# only needs to be set once per simulation
	if stepper.dt == 0.0:
		return (tfinal - time) / num_time_steps
	else:
		return stepper.dt


def get_dt_from_timestepsize(stepper, solver):
	'''
	Sets dt directly based on input deck specification of 
	params["TimeStepSize"].

	Inputs:
	-------
		stepper: stepper object (e.g., FE, RK4, etc...)
		solver: solver object (e.g., DG, ADERDG, etc...)

	Outputs:
	-------- 
		dt: time step for the solver
	'''
	time = solver.time
	timestepsize = solver.params["TimeStepSize"]
	tfinal = solver.params["FinalTime"]

	# logic to ensure final time step yields FinalTime
	if time + timestepsize < tfinal:
		return timestepsize
	else:
		return tfinal - time


def get_dt_from_cfl(stepper, solver):
	'''
	Calculates dt using a specified CFL number. Updates at everytime step to 
	ensure solution remains within the CFL bound.

	Inputs:
	-------
		stepper: stepper object (e.g., FE, RK4, etc...)
		solver: solver object (e.g., DG, ADERDG, etc...)

	Outputs:
	-------- 
		dt: time step for the solver
	'''	
	mesh = solver.mesh
	dim = mesh.dim

	physics = solver.physics
	Up = physics.U

	time = solver.time
	tfinal = solver.params["FinalTime"]
	cfl = solver.params["CFL"]
	
	elem_ops = solver.elem_operators
	vol_elems = elem_ops.vol_elems
	a = np.zeros([mesh.num_elems,Up.shape[1],1])

	# get the maximum wave speed per element
	for i in range(mesh.num_elems):
		a[i] = physics.compute_variable("MaxWaveSpeed", Up[i], 
				flag_non_physical=True)

	# calculate the dt for each element
	dt_elems = cfl*vol_elems**(1./dim)/a

	# take minimum to set appropriate dt
	dt = np.min(dt_elems)

	# logic to ensure final time step yields FinalTime
	if time + dt < tfinal:
		stepper.num_time_steps += 1
		return dt
	else:
		return tfinal - time