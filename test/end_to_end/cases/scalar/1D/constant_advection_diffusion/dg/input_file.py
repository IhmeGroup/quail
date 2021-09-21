import numpy as np

TimeStepping = {
	"InitialTime" : 0.,
	"FinalTime" : 1.0,
	"TimeStepSize" : 1e-3,
	"TimeStepper" : "RK4",
}

Numerics = {
	"SolutionOrder" : 3,
	"SolutionBasis" : "LagrangeSeg",
	"Solver" : "DG",
}

Mesh = {
	"File" : None,
	"ElementShape" : "Segment",
	"NumElemsX" : 64,
	"xmin" : 0.,
	"xmax" : 9.,
	"PeriodicBoundariesX" : ["x1","x2"]
}

Physics = {
	"Type" : "ConstAdvDiffScalar",
	"ConvFluxNumerical" : "LaxFriedrichs",
	"DiffFluxNumerical" : "SIP",
	"ConstVelocity" : 0.8,
	"DiffCoefficient" : 0.005,
}


InitialCondition = {
	"Function" : "DiffGaussian",
	"xo" : 1.0,
}

ExactSolution = InitialCondition.copy()

Output = {
	"AutoPostProcess" : False,
}
