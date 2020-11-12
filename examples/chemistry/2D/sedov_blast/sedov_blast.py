dt = 0.05
tfinal = 1.0
nt = int(tfinal/dt + 1.e-12)

TimeStepping = {
	"FinalTime" : tfinal,
	"NumTimeSteps" : nt,
	"TimeStepper" : "SSPRK3",
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
	"nElem_x" : 5,
	"nElem_y" : 5,
	"xmin" : -5.,
	"xmax" : 5.,
	"ymin" : -5.,
	"ymax" : 5.,
	#"PeriodicBoundariesX" : ["x2", "x1"],
	# "PeriodicBoundariesY" : ["y1", "y2"],
}

Physics = {
	"Type" : "Euler",
	"ConvFlux" : "LaxFriedrichs",
	"GasConstant" : 1.,
}

InitialCondition = {
	"Function" : "IsentropicVortex",
	# "State" : [1.0],
	# "SetAsExact" : False,
}

ExactSolution = InitialCondition.copy()

d1 = {
		"BCType" : "SlipWall",
}
d2 = {
	"BCType" : "Extrapolate",
}
BoundaryConditions = {
	# "x1" : d,
	# "x2" : d,
	"y1" : d,
	"y2" : d,
}

Output = {
	"AutoProcess" : True,
}
