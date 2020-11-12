dt = 0.05
tfinal = 1.0
nt = int(tfinal/dt + 1.e-12)

TimeStepping = {
	"FinalTime" : tfinal,
	"NumTimeSteps" : nt,
	"TimeStepper" : "LSRK4",
}

Numerics = {
	"SolutionOrder" : 2,
	"SolutionBasis" : "LagrangeTri",
	#"SolutionBasis" : "LagrangeQuad",
	"ElementQuadrature" : "Dunavant",
	"FaceQuadrature" : "GaussLegendre",
}

Mesh = {
	# "File" : "meshes/box_5x5.msh",
	"ElementShape" : "Triangle",
	#"ElementShape" : "Quadrilateral",
	"NumElemsX" : 5,
	"NumElemsY" : 5,
	"xmin" : -5.,
	"xmax" : 5.,
	"ymin" : -5.,
	"ymax" : 5.,
	"PeriodicBoundariesX" : ["x2", "x1"],
	# "PeriodicBoundariesY" : ["y1", "y2"],
}

Physics = {
	"Type" : "Euler",
	"ConvFluxNumerical" : "LaxFriedrichs",
	"GasConstant" : 1.,
}

InitialCondition = {
	"Function" : "IsentropicVortex",
	# "State" : [1.0],
	# "SetAsExact" : False,
}

ExactSolution = InitialCondition.copy()

d = {
		"BCType" : "StateAll",
		"Function" : "IsentropicVortex",
}

BoundaryConditions = {
	# "x1" : d,
	# "x2" : d,
	"y1" : d,
	"y2" : d,
}

Output = {
	"AutoPostProcess" : True,
	"Verbose" : True,
}
