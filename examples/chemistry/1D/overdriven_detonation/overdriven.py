import numpy as np

cfl = 0.1
dx = 0.02
u = 10.
dt = cfl*dx/u

FinalTime = 1.5
nTimeSteps =  int(FinalTime/dt)
print(nTimeSteps)
TimeStepping = {
	"InitialTime" : 0.,
	"FinalTime" : FinalTime,
	"nTimeStep" : nTimeSteps,
	"TimeStepper" : "SSPRK3",
}

Numerics = {
	"SolutionOrder" : 2,
	"SolutionBasis" : "LagrangeEqSeg",
	"InterpolateIC" : False,
	"Solver" : "DG",
	"ApplyLimiters" : "PositivityPreservingChem",
	# "NodeType" : "GaussLobatto",
	# "ElementQuadrature" : "GaussLobatto",
	# "FaceQuadrature" : "GaussLobatto",
	# "ColocatedPoints" : True,
}

Output = {
	"WriteInterval" : 50,
	"WriteInitialSolution" : True,
	"AutoProcess" : True,
}

Mesh = {
	"File" : None,
	"ElementShape" : "Segment",
	"nElem_x" : 150,
	"xmin" : 0.,
	"xmax" : 30.,
	# "PeriodicBoundariesX" : ["x1", "x2"],
}

Physics = {
	"Type" : "Chemistry",
	"ConvFlux" : "LaxFriedrichs",
	"GasConstant" : 1.,
	"SpecificHeatRatio" : 1.2,
	"HeatRelease": 50.,
}

InitialCondition = {
	"Function" : "SimpleDetonation1",
	"rho_u" : 1.0,
	"u_u" : 0.0,
	"p_u" : 1.0,
	"Y_u" : 1.0,
	"xshock" : 10.,
}

ExactSolution = InitialCondition.copy()

BoundaryConditions = {
	"Left" : {
		"BCType" : "StateAll",
		"Function" : "SimpleDetonation1",
		"rho_u" : 1.0,
		"u_u" : 0.0,
		"p_u" : 1.0,
		"Y_u" : 1.0,
		"xshock" : 10.,
	},
	"Right" : {
		"BCType" : "StateAll",
		"Function" : "SimpleDetonation1",
		"rho_u" : 1.0,
		"u_u" : 0.0,
		"p_u" : 1.0,
		"Y_u" : 1.0,
		"xshock" : 10.,    },
}

SourceTerms = {

	"source1" : {
		"Function" : "Arrhenius",
		"A" : 230.75.,
		"b" : 0.0,
		"Tign" : 50.,
	},
}
