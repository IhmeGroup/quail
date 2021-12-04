import numpy as np
import copy

FinalTime = 17.0e-5
NumTimeSteps = 1500

TimeStepping = {
    "InitialTime" : 0.,
    "FinalTime" : FinalTime,
    "CFL" : 0.5,
#    "TimeStepSize" : 1.0e-5,
#    "TimeStepper" : "ADER",
    "TimeStepper" : "SSPRK3",
}


Numerics = {
    "SolutionOrder" : 0,
    "SolutionBasis" : "LagrangeSeg",
#   "Solver" : "ADERDG",
    "Solver" : "DG",
    "ArtificialViscosity" : True,
    "AVParameter" : 1e4,
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
    "NumElemsX" : 200,
    "xmin" : 0.0,
    "xmax" : 0.12,
#    "PeriodicBoundariesX" : ["x1", "x2"]
}


Physics = {
    "Type" : "EulerMultispecies1D_9sp_H2O2",
    "ConvFluxNumerical" : "LaxFriedrichs",
}

state = { 
    "Function" : "ReactingShockTube",
}

InitialCondition = state

BoundaryConditions = {
    "x1" : {
        "BCType" : "SlipWall",
       },
    "x2" : { 
        "BCType" : "StateAll",
	"Function" : "ReactingShockTube",
       }
}
