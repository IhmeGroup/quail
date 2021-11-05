import numpy as np

TimeStepping = {
	"InitialTime" : 0.,
	"NumTimeSteps" : 0,
	"TimeStepper" : "FE",
}

Numerics = {
	"SolutionOrder" : 1,
	"SolutionBasis" : "LagrangeSeg",
	"Solver" : "DG",
	"ApplyLimiters" : "PositivityPreserving",
}

Output = {
	"AutoPostProcess" : False,
}

Mesh = {
	"File" : None,
	"ElementShape" : "Segment",
	"NumElemsX" : 1,
	"xmin" : 0.,
	"xmax" : 1.,
	"PeriodicBoundariesX" : ["x1", "x2"],
}

Physics = {
	"Type" : "Euler",
}

Uinflow = np.array([1., 0., 1.])
InitialCondition = {
	"Function" : "Uniform",
	"state" : Uinflow,
}
