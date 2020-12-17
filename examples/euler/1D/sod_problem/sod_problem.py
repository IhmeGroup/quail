import numpy as np
import copy

TimeStepping = {
	"InitialTime" : 0.,
	"FinalTime" : 0.25,
	"NumTimeSteps" : 4000,
	"TimeStepper" : "RK4",
}

Numerics = {
	"SolutionOrder" : 2,
	"SolutionBasis" : "LagrangeSeg",
	"Solver" : "DG",
	"ApplyLimiters" : "PositivityPreserving",
	"NodeType" : "GaussLobatto",
	"ElementQuadrature" : "GaussLobatto",
	"FaceQuadrature" : "GaussLobatto",
}

Output = {
	"AutoPostProcess" : True,
	"Prefix" : "Data",
}

Mesh = {
	"File" : None,
	"ElementShape" : "Segment",
	"NumElemsX" : 60,
	"xmin" : 0.,
	"xmax" : 1.,
}

Physics = {
	"Type" : "Euler",
	"ConvFluxNumerical" : "Roe",
	"GasConstant" : 1.,
	"SpecificHeatRatio" : 1.4,
}

state = {
	"Function" : "RiemannProblem",
	"rhoL" : 1.,
	"uL" : 0.,
	"pL" : 1.,
	"rhoR" : 0.125,
	"uR" : 0.,
	"pR" : 0.1,
	"xd" : 0.5,
}

state_exact = {
	"Function" : "ExactRiemannSolution",
	"rhoL" : 1.,
	"uL" : 0.,
	"pL" : 1.,
	"rhoR" : 0.125,
	"uR" : 0.,
	"pR" : 0.1,
	"xd" : 0.5,
}
InitialCondition = state
state2 = state.copy()
state2.update({"BCType":"StateAll"})
ExactSolution = state_exact

BoundaryConditions = {
	"x1" : {
		"BCType" : "SlipWall"
		},
	"x2" : {
		"BCType" : "SlipWall"
		}
}
