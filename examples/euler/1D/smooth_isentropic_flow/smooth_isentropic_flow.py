TimeStepping = {
	"InitialTime" : 0.,
	"FinalTime" : 0.1,
	"NumTimeSteps" : 100,
	"TimeStepper" : "RK4",
}

Numerics = {
	"SolutionOrder" : 2,
	"SolutionBasis" : "LagrangeSeg",
	"Solver" : "DG",
}

Output = {
	"AutoPostProcess" : True,
}

Mesh = {
	"File" : None,
	"ElementShape" : "Segment",
	"NumElemsX" : 25,
	"xmin" : -1.,
	"xmax" : 1.,
}

Physics = {
	"Type" : "Euler",
	"ConvFluxNumerical" : "LaxFriedrichs",
	"GasConstant" : 1.,
	"SpecificHeatRatio" : 3.,
}

a = 0.9
InitialCondition = {
	"Function" : "SmoothIsentropicFlow",
	"a" : a,
}

ExactSolution = InitialCondition.copy()

BoundaryConditions = {
	"x1" : {
		"BCType" : "StateAll",
		"Function" : "SmoothIsentropicFlow",
		"a" : a,
	},
	"x2" : {
		"BCType" : "StateAll",
		"Function" : "SmoothIsentropicFlow",
		"a" : a,
	},
}
