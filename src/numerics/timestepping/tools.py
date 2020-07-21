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
	TimeScheme = Params["TimeScheme"]
	if StepperType[TimeScheme] == StepperType.FE:
		stepper = stepper_defs.FE(U)
	elif StepperType[TimeScheme] == StepperType.RK4:
		stepper = stepper_defs.RK4(U)
	elif StepperType[TimeScheme] == StepperType.LSRK4:
		stepper = stepper_defs.LSRK4(U)
	elif StepperType[TimeScheme] == StepperType.SSPRK3:
		stepper = stepper_defs.SSPRK3(U)
	elif StepperType[TimeScheme] == StepperType.Strang:
		stepper = stepper_defs.Strang(U)
		stepper.set_split_schemes(Params["OperatorSplitting_Exp"], Params["OperatorSplitting_Imp"], U)
	elif StepperType[TimeScheme] == StepperType.Simpler:
		stepper = stepper_defs.Simpler(U)
		stepper.set_split_schemes(Params["OperatorSplitting_Exp"], Params["OperatorSplitting_Imp"], U)
	else:
		raise NotImplementedError("Time scheme not supported")
	return stepper


def set_time_stepping_approach(stepper, Params):

	# unpack time stepping settings
	cfl = Params["CFL"]
	timestepsize = Params["TimeStepSize"]
	numtimesteps = Params["NumTimeSteps"]

	endtime = Params["EndTime"]
	# code.interact(local=locals())
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

	numtimesteps = stepper.numtimesteps
	tfinal = solver.Params["EndTime"]
	time = solver.Time

	# if solver.Time == 0.0:
	if stepper.dt == 0.0:
		return (tfinal-time)/numtimesteps
	else:
		return stepper.dt

def get_dt_from_timestepsize(stepper, solver):

	time = solver.Time
	timestepsize = solver.Params["TimeStepSize"]
	tfinal = solver.Params["EndTime"]

	if time + timestepsize < tfinal:
		return timestepsize
	else:
		return tfinal - time


def get_dt_from_cfl(stepper, solver):
	
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
	for i in range(mesh.nElem):
		a[i] = physics.ComputeScalars("MaxWaveSpeed", Up[i], flag_non_physical=True)
	dt_elems = cfl*vol_elems**(1./dim)/a
	dt = np.min(dt_elems)

	if time + dt < tfinal:
		stepper.numtimesteps += 1
		return dt
	else:
		return tfinal - time
