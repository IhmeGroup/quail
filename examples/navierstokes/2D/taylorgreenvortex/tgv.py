import numpy as np

TimeStepping = {
	"FinalTime" : 2e-4,
	"TimeStepSize" : 2e-5,
	"TimeStepper" : "SSPRK3",
}

Numerics = {
	"Solver" : "DG",
	"SolutionOrder" : 3,
	"SolutionBasis" : "LagrangeQuad",
}

nelem = 16 
num_elem_x = nelem
num_elem_y = nelem

Mesh = {
	"ElementShape" : "Quadrilateral",
	"NumElemsX" : num_elem_x,
	"NumElemsY" : num_elem_y,
	"xmin" : 0.,
	"xmax" : 1.,
	"ymin" : 0.,
	"ymax" : 1.,
	"PeriodicBoundariesX" : ["x2", "x1"],
	"PeriodicBoundariesY" : ["y1", "y2"],
}

Physics = {
	"Type" : "NavierStokes",
	"ConvFluxNumerical" : "Roe",
	"DiffFluxNumerical" : "SIP",
	"Transport" : "Constant",
	"GasConstant" : 7142.857142857142,
	"Viscosity" : 1.863e-5,
}

InitialCondition = {
	"Function" : "TaylorGreenVortexNS",
}

ExactSolution = InitialCondition.copy()
