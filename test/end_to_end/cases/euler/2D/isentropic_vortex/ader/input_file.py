TimeStepping = {
	"FinalTime" : 1.0,
	"TimeStepSize" : 0.05,
	"TimeStepper" : "ADER",
}

Numerics = {
	"Solver" : "ADERDG",
	"SolutionOrder" : 2,
	"SolutionBasis" : "LagrangeTri",
	"ElementQuadrature" : "Dunavant",
	"FaceQuadrature" : "GaussLegendre",
	"PredictorThreshold" : 1e-14,
}

Mesh = {
	"ElementShape" : "Triangle",
	"NumElemsX" : 5,
	"NumElemsY" : 5,
	"xmin" : -5.,
	"xmax" : 5.,
	"ymin" : -5.,
	"ymax" : 5.,
	"PeriodicBoundariesX" : ["x2", "x1"],
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
	"y1" : d,
	"y2" : d,
}

Output = {
	"AutoPostProcess" : False,
	"ProgressBar" : True,
}
