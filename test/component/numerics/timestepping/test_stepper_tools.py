import numpy as np
import pytest
import sys
sys.path.append('../src')

import numerics.timestepping.stepper as stepper_defs
import numerics.timestepping.tools as stepper_tools
import meshing.common as mesh_common
from general import StepperType

rtol = 1e-14
atol = 1e-14


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