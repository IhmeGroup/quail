TimeStepping = {
	"FinalTime" : 1.0,
	"TimeStepSize" : 0.05,
	"TimeStepper" : "LSRK4",
}

Numerics = {
	"SolutionOrder" : 2,

	# Basis when running the curved meshes
	# "SolutionBasis" : "LagrangeQuad",

	# Basis and settings when running the triangular meshes
	"SolutionBasis" : "LagrangeTri",
	"ElementQuadrature" : "Dunavant",
	"FaceQuadrature" : "GaussLegendre",
}

Mesh = {

	# Mesh setting for curved mesh case
	# "File" : "meshes/box3.msh"

	# Mesh settings for triangle mesh case
	"ElementShape" : "Triangle",
	"NumElemsX" : 5,
	"NumElemsY" : 5,
	"xmin" : -5.,
	"xmax" : 5.,
	"ymin" : -5.,
	"ymax" : 5.,
	# "PeriodicBoundariesX" : ["x2", "x1"],
}

Physics = {
	"Type" : "Euler",
	"ConvFluxNumerical" : "Roe",
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
	"AutoPostProcess" : True,
	"Verbose" : True,
}
