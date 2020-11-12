import numpy as np

TimeStepping = {
	"InitialTime" : 0.,
	"FinalTime" : 0.1,
	"nTimeStep" : 40,
	"TimeStepper" : "RK4",
}

Numerics = {
	"SolutionOrder" : 2,
	"SolutionBasis" : "LagrangeEqSeg",
	"Solver" : "DG",
}

Output = {
	"AutoProcess" : True,
}

Mesh = {
	"File" : None,
	"ElementShape" : "Segment",
	"nElem_x" : 25,
	"xmin" : -1.,
	"xmax" : 1.,
	"PeriodicBoundariesX" : ["x1", "x2"],
}

Physics = {
	"Type" : "Chemistry",
	"ConvFlux" : "LaxFriedrichs",
	# "GasConstant" : 1.,
	# "SpecificHeatRatio" : 3.,
}

InitialCondition = {
	"Function" : "DensityWave",
	"p" : 1.0,
}

ExactSolution = InitialCondition.copy()

# BoundaryConditions = {
#	  "Left" : {
#		  "BCType" : "StateAll",
#		"Function" : "SmoothIsentropicFlow",
#		  "a" : a,
#	  },
#	  "Right" : {
#		  "BCType" : "StateAll",
#		  "Function" : "SmoothIsentropicFlow",
#		  "a" : a,
#	  },
# }
