import numpy as np
import pytest
import sys
sys.path.append('../src')

import numerics.timestepping.stepper as stepper_defs
import numerics.timestepping.tools as stepper_tools
import meshing.common as mesh_common
from general import StepperType
import general

import physics.scalar.scalar as scalar
import solver.DG as DG

rtol = 1e-14
atol = 1e-14

def create_solver_object():
	'''
	This function creates a solver object that stores the 
	timestepper and its members
	'''
	mesh = mesh_common.mesh_1D(num_elems=1, xmin=0., xmax=1.)

	params = general.set_solver_params(SolutionOrder=1, 
			FinalTime=10.0, NumTimeSteps=1000, ApplyLimiters=[])
	
	physics = scalar.ConstAdvScalar1D(mesh)
	physics.set_conv_num_flux("LaxFriedrichs")
	physics.set_physical_params()
	U = np.array([1.])
	physics.set_IC(IC_type="Uniform", state=U)

	solver = DG.DG(params, physics, mesh)

	return solver


def test_set_stepper_FE():
	'''
	Checks setter function for stepper FE
	'''
	time_scheme = "FE"
	params = {'TimeStepper' : time_scheme}
	stepper = stepper_tools.set_stepper(params, None)
	expected = stepper_defs.FE(None)
	assert stepper == expected


def test_set_stepper_not_FE():
	'''
	Checks setter function for stepper not FE
	'''
	time_scheme = "RK4"
	params = {'TimeStepper' : time_scheme}
	stepper = stepper_tools.set_stepper(params, None)
	expected = stepper_defs.FE(None)
	assert stepper != expected


def test_set_stepper_RK4():
	'''
	Checks setter function for stepper RK4
	'''
	time_scheme = "RK4"
	params = {'TimeStepper' : time_scheme}
	stepper = stepper_tools.set_stepper(params, None)
	expected = stepper_defs.RK4(None)
	assert stepper == expected


def test_set_stepper_LSRK4():
	'''
	Checks setter function for stepper LSRK4
	'''
	time_scheme = "LSRK4"
	params = {'TimeStepper' : time_scheme}
	stepper = stepper_tools.set_stepper(params, None)
	expected = stepper_defs.LSRK4(None)
	assert stepper == expected


def test_set_stepper_SSPRK3():
	'''
	Checks setter function for stepper SSPRK3
	'''
	time_scheme = "SSPRK3"
	params = {'TimeStepper' : time_scheme}
	stepper = stepper_tools.set_stepper(params, None)
	expected = stepper_defs.SSPRK3(None)
	assert stepper == expected


def test_set_stepper_Strang():
	'''
	Checks setter function for stepper Strang
	'''
	time_scheme = "Strang"
	params = {'TimeStepper' : time_scheme, 
			'OperatorSplittingExplicit' : 'SSPRK3',
			'OperatorSplittingImplicit' : 'BDF1',
			}
	stepper = stepper_tools.set_stepper(params, None)
	expected = stepper_defs.Strang(None)
	assert stepper == expected


def test_set_stepper_Simpler():
	'''
	Checks setter function for stepper Simpler
	'''
	time_scheme = "Simpler"
	params = {'TimeStepper' : time_scheme, 
			'OperatorSplittingExplicit' : 'SSPRK3',
			'OperatorSplittingImplicit' : 'BDF1',
			}
	stepper = stepper_tools.set_stepper(params, None)
	expected = stepper_defs.Simpler(None)
	assert stepper == expected


def test_get_dt_from_num_time_steps():
	'''
	Verifies the time step from number of time steps
	'''
	solver = create_solver_object()
	dt = stepper_tools.get_dt_from_num_time_steps(
			solver.stepper, solver)

	np.testing.assert_allclose(dt, 0.01, rtol, atol)


def test_get_dt_from_num_time_steps_zero():
	'''
	Verifies the time step when number of times steps is zero
	'''
	solver = create_solver_object()
	solver.stepper.num_time_steps = 0
	dt = stepper_tools.get_dt_from_num_time_steps(
			solver.stepper, solver)

	np.testing.assert_allclose(dt, 1.0, rtol, atol)


def test_get_dt_from_num_time_steps_not_active_with_dt():
	'''
	Verifies the time step when stepper.dt is not zero
	'''
	solver = create_solver_object()
	solver.stepper.dt = 0.001
	dt = stepper_tools.get_dt_from_num_time_steps(
			solver.stepper, solver)

	np.testing.assert_allclose(dt, 0.001, rtol, atol)


def test_get_dt_from_cfl_yields_one():
	'''
	Checks unity CFL with all unity conditions in scalar
	'''
	solver = create_solver_object()
	solver.params["CFL"] = 1.0
	dt = stepper_tools.get_dt_from_cfl(solver.stepper, solver)

	np.testing.assert_allclose(dt, 1.0, rtol, atol)


def test_get_dt_from_timestepsize_and_finaltime():
	'''
	Checks the correct dt from time step size and tfinal
	when time + dt < tfinal
	'''
	solver = create_solver_object()
	solver.params["TimeStepSize"] = 2.0
	dt = stepper_tools.get_dt_from_timestepsize(solver.stepper, solver)
	np.testing.assert_allclose(dt, 2.0, rtol, atol)


def test_get_dt_from_timestepsize_and_finaltime():
	'''
	Checks the correct dt from time step size and tfinal
	when time + dt > tfinal
	'''
	solver = create_solver_object()
	solver.params["TimeStepSize"] = 2.0
	solver.time = 9.995
	dt = stepper_tools.get_dt_from_timestepsize(solver.stepper, solver)
	np.testing.assert_allclose(dt, 0.005, rtol, atol)


def test_get_dt_from_timestepsize_and_numtimesteps():
	'''
	Checks that the final time has not been reached
	'''
	solver = create_solver_object()
	solver.params["TimeStepSize"] = 2.0
	solver.stepper.dt = 2.0
	dt = stepper_tools.get_dt_from_timestepsize_and_numtimesteps(
			solver.stepper, solver)

	np.testing.assert_allclose(dt, 2.0, rtol, atol)


def test_get_dt_from_timestepsize_and_numtimesteps_finalstep():
	'''
	Checks the change to the final time step to make sure we get the 
	correct final time
	'''
	solver = create_solver_object()
	solver.params["TimeStepSize"] = 0.01
	solver.stepper.dt = 0.01
	# FinalTime is 10.0 therefore the last time-step should be 
	# 1.0 not 2.0.
	solver.time = 9.995
	dt = stepper_tools.get_dt_from_timestepsize_and_numtimesteps(
			solver.stepper, solver)

	np.testing.assert_allclose(dt, 0.005, rtol, atol)


def test_check_addition_of_explicit_and_implicit_sources():
	'''
	This test adds an explicit and implicit source term and checks
	that they were properly added to the list
	'''
	solver = create_solver_object()
	# Add an implicit and explicit source term
	sparams = {"nu" : 1.0, "source_treatment" : "Explicit"}
	solver.physics.set_source(source_type='SimpleSource', **sparams)
	sparams = {"nu" : 1.0, "source_treatment" : "Implicit"}
	solver.physics.set_source(source_type='SimpleSource', **sparams)
	sparams = {"nu" : 1.0}
	solver.physics.set_source(source_type='SimpleSource', **sparams)

	stepper_tools.set_source_treatment(solver.physics)

	# Assert the correct source treatment for each source (default is implicit)
	assert(solver.physics.source_terms[0].source_treatment == 'Explicit')
	assert(solver.physics.source_terms[1].source_treatment == 'Implicit')
	assert(solver.physics.source_terms[2].source_treatment == 'Implicit')