TimeStepping = {
	"InitialTime" : 0.,
	"FinalTime" : 5.,
	# "CFL" : 0.0025,
	"TimeStepSize" : 0.001,
	# "NumTimeSteps" : 1,
	"TimeStepper" : "RK4",
}

Numerics = {
	"SolutionOrder" : 3,
	"SolutionBasis" : "LagrangeQuad",
	"Solver" : "DG",
}

Output = {
	"Prefix" : "Data",
	"AutoPostProcess" : True,
	"WriteInterval" : 150,
	"WriteInitialSolution" : True,

}

nelem = 8
Mesh = {
	"File" : None,
	"ElementShape" : "Quadrilateral",
	"NumElemsX" : nelem,
	"NumElemsY" : nelem,
	"xmin" : 0.,
	"xmax" : 2.,
	"ymin" : 0.,
	"ymax" : 2.,
}

Physics = {
	"Type" : "ConstAdvDiffScalar",
	"ConvFluxNumerical" : "LaxFriedrichs",
	"DiffFluxNumerical" : "SIP",
	"ConstXVelocity" : 0.2,
	"ConstYVelocity" : 0.2,
	"DiffCoefficientX" : 0.01,
	"DiffCoefficientY" : 0.01,
}

InitialCondition = {
	"Function" : "DiffGaussian2D",
	"xo" : 0.5,
	"yo" : 0.5,
}

d = {
		"BCType" : "StateAll",
		"Function" : "DiffGaussian2D",
		"xo" : 0.5,
		"yo" : 0.5,
}


BoundaryConditions = {
	"x1" : d,
	"x2" : d,
	"y1" : d,
	"y2" : d,
}

ExactSolution = InitialCondition.copy()
