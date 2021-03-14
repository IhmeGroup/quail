import numpy as np

TimeStepping = {
	"FinalTime" : 2,
	"TimeStepSize" : .01,
	"TimeStepper" : "RK4",
}

Numerics = {
	"SolutionOrder" : 10,
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
	#"PeriodicBoundariesX" : ["x2", "x1"],
}

Physics = {
	"Type" : "Euler",
	"ConvFluxNumerical" : "LaxFriedrichs",
	"GasConstant" : 1.,
}

#U = np.array([1.0, 0.0, 0.0, 2])
#InitialCondition = {
#	"Function" : "Uniform",
#	"state" : U,
#}

InitialCondition = {
	"Function" : "IsentropicVortex",
}

ExactSolution = InitialCondition.copy()

d = {
	"BCType" : "StateAll",
	"Function" : "IsentropicVortex",
}
#d = {"BCType" : "SlipWall"}

BoundaryConditions = {
	"x1" : d,
	"x2" : d,
	"y1" : d,
	"y2" : d,
}

Output = {
	"WriteInterval" : 50,
	"Prefix" : "Data",
	"AutoPostProcess" : False,
	"WriteInitialSolution" : True,
	"WriteFinalSolution" : True,
	"Verbose" : True,
}
