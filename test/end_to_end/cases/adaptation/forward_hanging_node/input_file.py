import numpy as np

TimeStepping = {
	"FinalTime" : .001,
	"NumTimeSteps" : 1,
	"TimeStepper" : "FE",
}

Numerics = {
	"SolutionOrder" : 2,
	"SolutionBasis" : "LagrangeTri",
	"ElementQuadrature" : "Dunavant",
	"FaceQuadrature" : "GaussLegendre",
}

Mesh = {
	"ElementShape" : "Triangle",
	"NumElemsX" : 1,
	"NumElemsY" : 1,
	"xmin" : -2.5,
	"xmax" : 7.5,
	"ymin" : -2.5,
	"ymax" : 7.5,
}

Physics = {
	"Type" : "Euler",
	"ConvFluxNumerical" : "LaxFriedrichs",
	"GasConstant" : 1.,
}

InitialCondition = {
	"Function" : "IsentropicVortex",
}

ExactSolution = InitialCondition.copy()

d = {
	"BCType" : "StateAll",
	"Function" : "IsentropicVortex",
}

BoundaryConditions = {
	"x1" : d,
	"x2" : d,
	"y1" : d,
	"y2" : d,
}

Output = {
	"Prefix" : "Data",
	"AutoPostProcess" : False,
	"WriteInitialSolution" : True,
	"WriteFinalSolution" : True,
	"Verbose" : True,
}
