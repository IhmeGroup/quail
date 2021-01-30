TimeStepping = {
	"FinalTime" : .002,
	"TimeStepSize" : .002,
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
	"xmin" : -5.,
	"xmax" : 5.,
	"ymin" : -5.,
	"ymax" : 5.,
	"PeriodicBoundariesX" : ["x2", "x1", "y2", "y1"],
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

#BoundaryConditions = {
#	"y1" : d,
#	"y2" : d,
#}

Output = {
	"AutoPostProcess" : False,
}
