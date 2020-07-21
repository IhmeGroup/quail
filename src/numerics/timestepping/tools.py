# ------------------------------------------------------------------------ #
#
#       File : numerics/timestepping/tools.py
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

def set_stepper(Params, U):
	'''
	Method: set_stepper
	---------------------
	Given the timescheme parameter, set the stepper object

	INPUTS:
		Params: list of parameters for solver [list]
		U: solution vector for instantiaing stepper class [nelem, nb, ns]

	OUTPUTS: 
	    stepper: instantiated stepper object
	'''
	TimeScheme = Params["TimeScheme"]
	if StepperType[TimeScheme] == StepperType.FE:
		stepper = stepper_defs.FE(U)
	elif StepperType[TimeScheme] == StepperType.RK4:
		stepper = stepper_defs.RK4(U)
	elif StepperType[TimeScheme] == StepperType.LSRK4:
		stepper = stepper_defs.LSRK4(U)
	elif StepperType[TimeScheme] == StepperType.SSPRK3:
		stepper = stepper_defs.SSPRK3(U)
	# if setting a splitting scheme select solvers for the splits
	elif StepperType[TimeScheme] == StepperType.Strang:
		stepper = stepper_defs.Strang(U)
		stepper.set_split_schemes(Params["OperatorSplitting_Exp"], 
			Params["OperatorSplitting_Imp"], U)
	elif StepperType[TimeScheme] == StepperType.Simpler:
		stepper = stepper_defs.Simpler(U)
		stepper.set_split_schemes(Params["OperatorSplitting_Exp"], 
			Params["OperatorSplitting_Imp"], U)
	else:
		raise NotImplementedError("Time scheme not supported")
	return stepper


def set_time_stepping_approach(stepper, Params):
	'''
	Method: set_time_stepping_approach
	-----------------------------------
	Sets stepper.get_time_step method given input parameters

	INPUTS:
		stepper: stepper object (i.e. FE, RK4, etc...)
		Params: list of parameters for solver [list]

	OUTPUTS: 
		get_time_step: method selected to calculate dt
		numtimesteps: number of time steps for the solution
	'''

	# unpack time stepping settings
	cfl = Params["CFL"]
	timestepsize = Params["TimeStepSize"]
	numtimesteps = Params["NumTimeSteps"]
	endtime = Params["EndTime"]

	if numtimesteps != None:
		stepper.get_time_step = get_dt_from_numtimesteps
		stepper.numtimesteps = numtimesteps
	elif timestepsize != None:
		stepper.get_time_step = get_dt_from_timestepsize
		stepper.numtimesteps = math.ceil(endtime/timestepsize)
	elif cfl != None:
		stepper.get_time_step = get_dt_from_cfl
		stepper.numtimesteps = 1


def get_dt_from_numtimesteps(stepper, solver):
	'''
	Method: get_dt_from_numtimesteps
	-----------------------------------
	Calculates dt from the specified number of time steps

	INPUTS:
		stepper: stepper object (i.e. FE, RK4, etc...)
		solver: solver object (i.e. DG, ADERDG, etc...)

	OUTPUTS: 
		dt: time step for the solver
	'''
	numtimesteps = stepper.numtimesteps
	tfinal = solver.Params["EndTime"]
	time = solver.Time

	# only needs to be set once per simulation
	if stepper.dt == 0.0:
		return (tfinal - time) / numtimesteps
	else:
		return stepper.dt


def get_dt_from_timestepsize(stepper, solver):
	'''
	Method: get_dt_from_timestepsize
	-----------------------------------
	Sets dt directly based on input deck specification of 
	Params["TimeStepSize"].

	INPUTS:
		stepper: stepper object (i.e. FE, RK4, etc...)
		solver: solver object (i.e. DG, ADERDG, etc...)

	OUTPUTS: 
		dt: time step for the solver
	'''
	time = solver.Time
	timestepsize = solver.Params["TimeStepSize"]
	tfinal = solver.Params["EndTime"]

	# logic to ensure final time step yields EndTime
	if time + timestepsize < tfinal:
		return timestepsize
	else:
		return tfinal - time


def get_dt_from_cfl(stepper, solver):
	'''
	Method: get_dt_from_cfl
	-----------------------------------
	Calculates dt using a specified CFL number. Updates at everytime step to 
	ensure solution remains within the CFL bound.

	INPUTS:
		stepper: stepper object (i.e. FE, RK4, etc...)
		solver: solver object (i.e. DG, ADERDG, etc...)

	OUTPUTS: 
		dt: time step for the solver
	'''	
	mesh = solver.mesh
	dim = mesh.Dim

	physics = solver.physics
	Up = physics.U

	time = solver.Time
	tfinal = solver.Params["EndTime"]
	cfl = solver.Params["CFL"]
	
	elem_ops = solver.elem_operators
	vol_elems = elem_ops.vol_elems
	a = np.zeros([mesh.nElem,Up.shape[1],1])

	# get the maximum wave speed per element
	for i in range(mesh.nElem):
		a[i] = physics.ComputeScalars("MaxWaveSpeed", Up[i], 
			flag_non_physical=True)

	# calculate the dt for each element
	dt_elems = cfl*vol_elems**(1./dim)/a

	# take minimum to set appropriate dt
	dt = np.min(dt_elems)

	# logic to ensure final time step yields EndTime
	if time + dt < tfinal:
		stepper.numtimesteps += 1
		return dt
	else:
		return tfinal - time