TimeStepping = {
	"FinalTime" : 12,
	"TimeStepSize" : .01,
	"TimeStepper" : "RK4",
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
	"xmin" : 0,
	"xmax" : 10,
	"ymin" : 0,
	"ymax" : 10,
	#"PeriodicBoundariesX" : ["x2", "x1", "y2", "y1"],
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

Restart = {
	"File" : "initial_mesh_final.pkl",
	"StartFromInitialCondition" : True,
}

Output = {
	"WriteInterval" : 10,
	"Prefix" : "Data",
	"AutoPostProcess" : True,
	"WriteInitialSolution" : True,
	"WriteFinalSolution" : True,
	"Verbose" : True,
}
