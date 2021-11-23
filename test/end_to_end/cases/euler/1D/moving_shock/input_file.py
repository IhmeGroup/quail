import numpy as np

TimeStepping = {
	"InitialTime" : 0.,
	"FinalTime" : 2.e-4,
	"CFL" : 0.1,
	"TimeStepper" : "SSPRK3",
}

Numerics = {
	"SolutionOrder" : 2,
	"SolutionBasis" : "LagrangeSeg",
	"Solver" : "DG",
	"L2InitialCondition" : True,
	"ApplyLimiters" : "PositivityPreserving",
	"NodeType" : "GaussLobatto",
	"ElementQuadrature" : "GaussLobatto",
	"FaceQuadrature" : "GaussLobatto",
	"ColocatedPoints" : True,
}

Output = {
	"AutoPostProcess" : False,
	"ProgressBar" : True,
}

Mesh = {
	"File" : None,
	"ElementShape" : "Segment",
	"NumElemsX" : 50,
	"xmin" : 0.,
	"xmax" : 1.,
}

Physics = {
	"Type" : "Euler",
	"ConvFluxNumerical" : "LaxFriedrichs",
	"GasConstant" : 287,
	"SpecificHeatRatio" : 1.4,
}

M = 5.0
xshock = 0.2

InitialCondition = {
	"Function" : "MovingShock",
	"M" : M,
	"xshock" : xshock,
}

ExactSolution = InitialCondition.copy()

BoundaryConditions = {
	"x1" : {
		"BCType" : "StateAll",
		"Function" : "MovingShock",
		"M" : M,
		"xshock" : xshock,
	},
	"x2" : {
		"BCType" : "StateAll",
		"Function" : "MovingShock",
		"M" : M,
		"xshock" : xshock,
	},
}
