TimeStepping = {
	"InitialTime" : 0.,
	"FinalTime" : 2.,
	"TimeStepSize" : 0.025,
	"TimeStepper" : "ADER",
}

Numerics = {
	"SolutionOrder" : 3,
	"SolutionBasis" : "LagrangeQuad",
	"Solver" : "ADERDG",
	"ElementQuadrature" : "GaussLegendre",
	"FaceQuadrature" : "GaussLegendre",
}

Output = {
	"Prefix" : "Data",
	"AutoPostProcess" : False,
}

Mesh = {
	"File" : None,
	"ElementShape" : "Quadrilateral",
	"NumElemsX" : 8,
	"NumElemsY" : 8,
	"xmin" : -5.,
	"xmax" : 5.,
	"ymin" : -5.,
	"ymax" : 5.,
}

Physics = {
	"Type" : "ConstAdvScalar",
	"ConvFluxNumerical" : "LaxFriedrichs",
	"ConstXVelocity" : 1.,
	"ConstYVelocity" : 1.,
}

x0 = [0., 0.]
InitialCondition = {
	"Function" : "Gaussian",
	"x0" : x0,
}

ExactSolution = InitialCondition.copy()


d = {
		"BCType" : "StateAll",
		"Function" : "Gaussian",
		"x0" : x0,
}


BoundaryConditions = {
	"x1" : d,
	"x2" : d,
	"y1" : d,
	"y2" : d,
}