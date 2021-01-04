import numpy as np

TimeStepping = {
	"InitialTime" : 0.,
	"FinalTime" : 0.5,
	"NumTimeSteps" : 40,
	"TimeStepper" : "Strang",
}

Numerics = {
	"SolutionOrder" : 2,
	"SolutionBasis" : "LegendreSeg",
	"Solver" : "DG",
}

Mesh = {
	"File" : None,
	"ElementShape" : "Segment",
	"NumElemsX" : 16,
	"xmin" : -1.,
	"xmax" : 1.,
	"PeriodicBoundariesX" : ["x1", "x2"],
}

Physics = {
	"Type" : "ConstAdvScalar",
	"ConvFluxNumerical" : "LaxFriedrichs",
	"ConstVelocity" : 1.,
}

# Stiffness parameter
nu = -3.
# nu = -1000.

InitialCondition = {
	"Function" : "DampingSine",
	"omega" : 2*np.pi,
	"nu" : nu ,
}

ExactSolution = InitialCondition.copy()

SourceTerms = {
	"Source1" : { # Name of source term ("Source1") doesn't matter
		"Function" : "SimpleSource",
		"nu" : nu,
	},
}

Output = {
	"AutoPostProcess" : False,
}
