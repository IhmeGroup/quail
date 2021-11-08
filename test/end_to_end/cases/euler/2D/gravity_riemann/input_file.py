TimeStepping = {
	"FinalTime" : 0.028,
	"TimeStepSize" : 0.004,
	"TimeStepper" : "SSPRK3",
}

Numerics = {
	"SolutionOrder" : 2,
	"SolutionBasis" : "LagrangeQuad",
#	"ApplyLimiters" : "PositivityPreserving",
}

Mesh = {
	"ElementShape" : "Quadrilateral",
	"NumElemsX" : 10,
	"NumElemsY" : 10,
	"xmin" : 0.,
	"xmax" : 2.,
	"ymin" : 0.,
	"ymax" : 2.,
}

Physics = {
	"Type" : "Euler",
	"ConvFluxNumerical" : "LaxFriedrichs",
	"GasConstant" : 1.,
}

InitialCondition = {
	"Function" : "GravityRiemann",
}

outflow = {
		"BCType" : "PressureOutlet",
		"p" : 0.1,
}
BoundaryConditions = {
	"x1" : outflow,
	"x2" : outflow,
	"y1" : {
		"BCType" : "SlipWall",
	},
	"y2" : {
		"BCType" : "SlipWall",
	}
}

SourceTerms = {
	"source1" : {
		"Function" : "GravitySource",
		"gravity" : 1.,
	},
}

Output = {
	"AutoPostProcess" : False,
	"ProgressBar" : False,
}
