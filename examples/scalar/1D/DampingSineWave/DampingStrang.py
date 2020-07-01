import numpy as np
cfl = .1
dx = 0.125
dt = cfl*dx
EndTime = 0.5
nTimeStep = int(EndTime/dt)
TimeStepping = {
    "StartTime" : 0.,
    "EndTime" : EndTime,
    "nTimeStep" : nTimeStep,
    "TimeScheme" : "Strang",
    "OperatorSplitting_Imp" : "BDF1",

}

Numerics = {
    "InterpOrder" : 2,
    # "InterpolateIC" : True,
    "InterpBasis" : "LagrangeEqSeg",
    "Solver" : "DG",
    "ConvFluxSwitch" : True,
    "SourceTreatment" : "Implicit"
}

Output = {
    "AutoProcess" : True
}

Mesh = {
    "File" : None,
    "ElementShape" : "Segment",
    "nElem_x" : 16,
    "nElem_y" : 2,
    "xmin" : -1.,
    "xmax" : 1.,
    "PeriodicBoundariesX" : ["x1", "x2"],
}

Physics = {
    "Type" : "ConstAdvScalar",
    "ConvFlux" : "LaxFriedrichs",
    "ConstVelocity" : 1.,
}

nu = -100000.
InitialCondition = {
    "Function" : "DampingSine",
    "omega" : 2*np.pi,
    "nu" : nu ,
    # "state" : [1.0],
}

ExactSolution = InitialCondition.copy()

BoundaryConditions = {
    # "Left" : {
	   #  "Function" : "DampingSine",
	   #  "omega" : 2*np.pi,
	   #  "nu" : nu,
    # 	"BCType" : "StateAll",
    # },
    # "Right" : {
    # 	#"Function" : None,
    # 	"BCType" : "Extrapolate",
    # },
}

SourceTerms = {
	"source1" : {
		"Function" : "SimpleSource",
		"nu" : nu,
	},
}
