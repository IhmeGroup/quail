import numpy as np
import copy

FinalTime = 4.0e-5
NumTimeSteps = 1500

TimeStepping = {
    "InitialTime" : 0.,
    "FinalTime" : FinalTime,
#    "CFL" : 0.1,
    "TimeStepSize" : 2.5e-8,
    "TimeStepper" : "ADER",
#    "TimeStepper" : "SSPRK3",
}


Numerics = {
    "SolutionOrder" : 2,
    "SolutionBasis" : "LagrangeSeg",
    "Solver" : "ADERDG",
#    "Solver" : "DG",
    "ArtificialViscosity" : True,
    "AVParameter" : 1e7,
    "PredictorThreshold" : 1e-10,
}

Output = {
    "AutoPostProcess" : False,
    "Prefix" : "Data",
#    "WriteInterval" : 2,
    "WriteInitialSolution" : True,
}

Mesh = {
    "File" : None,
    "ElementShape" : "Segment",
    "NumElemsX" : 400,
    "xmin" : 0.0,
    "xmax" : 0.1,
#    "PeriodicBoundariesX" : ["x1", "x2"]
}


Physics = {
    "Type" : "EulerMultispecies1D_3sp_H2O2_inert",
    "ConvFluxNumerical" : "LaxFriedrichs",
}

state = { 
    "Function" : "InertShockTube",
}

InitialCondition = state

BoundaryConditions = {
    "x1" : {
        "BCType" : "StateAll",
	"Function" : "InertShockTube",
       },
    "x2" : { 
        "BCType" : "StateAll",
	"Function" : "InertShockTube",
       }
}
