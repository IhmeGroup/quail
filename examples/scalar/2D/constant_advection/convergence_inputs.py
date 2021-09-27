TimeStepping = {
	"InitialTime" : 0.,
	"FinalTime" : 1.,
	"TimeStepSize" : 1.,
	"TimeStepper" : "RK4",
}

Numerics = {
	"SolutionOrder" : 1,
	"SolutionBasis" : "LagrangeQuad",
	"Solver" : "DG",
}

Output = {
	"Prefix" : "Data",
}

Mesh = {
	"File" : None,
	"ElementShape" : "Quadrilateral",
	"NumElemsX" : 2,
	"NumElemsY" : 2,
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
