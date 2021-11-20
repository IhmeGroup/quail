import numpy as np

TimeStepping = {
	"InitialTime" : 0.,
	"FinalTime" : 0.5,
	"TimeStepSize" : 1e-3,
	"TimeStepper" : "ADER",
}

Numerics = {
	"SolutionOrder" : 3,
	"SolutionBasis" : "LagrangeSeg",
	"Solver" : "ADERDG",
	"InterpolateFluxADER" : False,
}

Mesh = {
	"File" : None,
	"ElementShape" : "Segment",
	"NumElemsX" : 16,
	"xmin" : 0.,
	"xmax" : 2.25,
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
