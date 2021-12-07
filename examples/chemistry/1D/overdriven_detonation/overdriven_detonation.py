FinalTime = 10.0
TimeStepping = {
	"FinalTime" : FinalTime,
	"TimeStepSize" : 0.001,
	"TimeStepper" : "SSPRK3",
}

Numerics = {
	"SolutionOrder" : 2,
	"SolutionBasis" : "LagrangeSeg",
	"Solver" : "DG",
	"ApplyLimiters" : ["WENO"],
	"ShockIndicator" : "MinMod",
	"TVBParameter" : 0.01,
}

Output = {
	"WriteInterval" : 400,
	"WriteInitialSolution" : True,
}

Mesh = {
	"File" : None,
	"ElementShape" : "Segment",
	"NumElemsX" : 2000,
	"xmin" : 0.,
	"xmax" : 100.,
}

Physics = {
	"Type" : "Chemistry",
	"ConvFluxNumerical" : "LaxFriedrichs",
	"GasConstant" : 1.,
	"SpecificHeatRatio" : 1.2,
	"HeatRelease": 50.,
}

xshock = 5.
InitialCondition = {
	"Function" : "OverdrivenDetonation",
	"xshock" : xshock,
}

ExactSolution = InitialCondition.copy()

BoundaryConditions = {
	"x1" : {
		"BCType" : "StateAll",
		"Function" : "OverdrivenDetonation",
		"xshock" : xshock,
	},
	"x2" : {
		"BCType" : "StateAll",
		"Function" : "OverdrivenDetonation",
		"xshock" : xshock,  },
}

SourceTerms = {
	"source1" : {
		"Function" : "Arrhenius",
		"A" : 230.75,
		"b" : 0.,
		"Tign" : 50.,
	},
}
