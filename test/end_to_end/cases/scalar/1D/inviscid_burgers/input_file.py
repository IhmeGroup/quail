import numpy as np

TimeStepping = {
	"InitialTime" : 0.,
	"FinalTime" : 0.5,
	"CFL" : 0.1,
	"TimeStepper" : "SSPRK3",
}

Numerics = {
	"SolutionOrder" : 2,
	"SolutionBasis" : "LagrangeSeg",
	"Solver" : "DG",
}

Mesh = {
	"File" : None,
	"ElementShape" : "Segment",
	"NumElemsX" : 80,
	"xmin" : 0.,
	"xmax" : 2*np.pi,
}

Physics = {
	"Type" : "Burgers",
	"ConvFluxNumerical" : "LaxFriedrichs",
}

InitialCondition = {
	"Function" : "SineBurgers",
	"omega" : 1.,
}

ExactSolution = {
	"Function" : "SineBurgers",
	"omega" : 1.,
}

Output = {
	"Prefix" : "Data",
	"AutoPostProcess" : False,
}

BoundaryConditions = {
   "x1" : {
		"Function" : "SineBurgers",
		"omega" : 2*np.pi,
	"BCType" : "StateAll",
   },
   "x2" : {
	"BCType" : "Extrapolate",
   },
}
