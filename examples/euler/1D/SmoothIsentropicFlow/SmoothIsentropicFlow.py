import numpy as np

TimeStepping = {
    "StartTime" : 0.,
    "EndTime" : 0.1,
    "nTimeStep" : 100,
    "TimeScheme" : "RK4",
}

Numerics = {
    "InterpOrder" : 2,
    "InterpBasis" : "LagrangeEqSeg",
    "Solver" : "DG",
}

Output = {}


Output = {
    "AutoProcess" : True,
}

Mesh = {
    "File" : None,
    "ElementShape" : "Segment",
    "nElem_x" : 25,
    "xmin" : -1.,
    "xmax" : 1.,
    # "PeriodicBoundariesX" : ["x1", "x2"],
}

Physics = {
    "Type" : "Euler",
    "ConvFlux" : "Roe",
    "GasConstant" : 1.,
    "SpecificHeatRatio" : 3.,
}

a = 0.9
InitialCondition = {
    "Function" : "SmoothIsentropicFlow",
    "a" : a,
    "SetAsExact" : True,
}

BoundaryConditions = {
    "Left" : {
	    "Function" : "SmoothIsentropicFlow",
        "a" : a,
    	"BCType" : "StateAll",
    },
    "Right" : {
        "Function" : "SmoothIsentropicFlow",
        "a" : a,
        "BCType" : "StateAll",
    },
}

SourceTerms = {}
