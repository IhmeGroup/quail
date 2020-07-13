import numpy as np
cfl = 0.002
nElem = 128
dx = float(1./nElem)
dt = cfl*dx
EndTime = 0.5
nTimeStep = int(EndTime/dt)
print(nTimeStep)
TimeStepping = {
    "StartTime" : 0.,
    "EndTime" : EndTime,
    "nTimeStep" : nTimeStep,
    "TimeScheme" : "Strang",
    "OperatorSplitting_Imp" : "Trapezoidal",

}

Numerics = {
    "InterpOrder" : 3,
    # "InterpolateIC" : True,
    "InterpBasis" : "LagrangeEqSeg",
    "Solver" : "DG",
    "ConvFluxSwitch" : True,
    # "SourceTreatment" : "Implicit"
}

Output = {
    # "WriteInterval" : 2,
    # "WriteInitialSolution" : True,
    "AutoProcess" : True,
}

Mesh = {
    "File" : None,
    "ElementShape" : "Segment",
    "nElem_x" : nElem,
    # "nElem_y" : 2,
    "xmin" : -1.,
    "xmax" : 1.,
    "PeriodicBoundariesX" : ["x1", "x2"],
}

Physics = {
    "Type" : "ConstAdvScalar",
    "ConvFlux" : "LaxFriedrichs",
    "ConstVelocity" : 1.,
}

nu = -3.
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
