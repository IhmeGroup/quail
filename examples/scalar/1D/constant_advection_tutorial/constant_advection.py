import numpy as np

Mesh = {
    "File" : None,
    "ElementShape" : "Segment",
    "NumElems_x" : 16,
    "xmin" : -1.,
    "xmax" : 1.,
    "PeriodicBoundariesX" : ["xmin","xmax"]
}

Physics = {
	"Type" : "ConstAdvScalar",
	"ConvFluxNumerical" : "LaxFriedrichs", 
	"ConstVelocity" : 1.,
}

InitialCondition = {
	"Function" : "Sine",
	"omega" : 2.*np.pi,
}

ExactSolution = InitialCondition.copy()

Numerics = {
	"SolutionOrder" : 2,
	"SolutionBasis" : "LagrangeSeg",
	"Solver" : "DG",
}

TimeStepping = {
	"InitialTime" : 0.,
	"FinalTime" : 0.5,
	"CFL" : 0.1,
	"TimeStepper" : "RK4"
}




















