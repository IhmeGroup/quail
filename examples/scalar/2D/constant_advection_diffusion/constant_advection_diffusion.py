TimeStepping = {
	"InitialTime" : 0.,
	"FinalTime" : 5.,
	"CFL" : 0.0025,
	"TimeStepper" : "RK4",
}

Numerics = {
	"SolutionOrder" : 3,
	"SolutionBasis" : "LagrangeTri",
	"Solver" : "DG",
}

Output = {
	"Prefix" : "Data",
	"WriteInterval" : 150,
	"WriteInitialSolution" : True,
	"WriteFinalSolution" : True,
	"AutoPostProcess" : True,
}

Mesh = {
	"File" : None,
	"ElementShape" : "Triangle",
	"NumElemsX" : 8,
	"NumElemsY" : 8,
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
