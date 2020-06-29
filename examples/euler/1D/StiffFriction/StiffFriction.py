import numpy as np

cfl = 0.05
dx = 0.05
EndTime = 0.2
nTimeStep = int(EndTime/(cfl*dx))
TimeStepping = {
    "StartTime" : 0.,
    "EndTime" : EndTime,
    "nTimeStep" : nTimeStep,
    "TimeScheme" : "Strang",
}

Numerics = {
    "InterpOrder" : 2,
    "InterpBasis" : "LagrangeEqSeg",
    "Solver" : "DG",
    "SourceTreatment" : "Implicit"
}

Output = {
    "AutoProcess" : True,
}

Mesh = {
    "File" : None,
    "ElementShape" : "Segment",
    "nElem_x" : 20,
    "xmin" : 0.,
    "xmax" : 1.,
    "PeriodicBoundariesX" : ["x1", "x2"],
}

Physics = {
    "Type" : "Euler",
    "ConvFlux" : "Roe",
    "SpecificHeatRatio" : 1.4,
}

nu = -3.
InitialCondition = {
    "Function" : "DensityWave",
    "p" : 1.,
}

SourceTerms = {
	"source1" : {
		"Function" : "StiffFriction",
		"nu" : nu,
	},
}
