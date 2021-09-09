import numpy as np

TimeStepping = {
	"InitialTime" : 0.,
	"FinalTime" : 5.0,
	"TimeStepSize" : 8.78906e-05,
	# "CFL" : 0.002,
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
	"NumElemsX" : 256,
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

# Output = {
# 	"Prefix" : "Data",
# 	"WriteInterval" : 50,
# 	"WriteInitialSolution" : True,
# 	"WriteFinalSolution" : True,
# 	"AutoPostProcess" : True,
# }
