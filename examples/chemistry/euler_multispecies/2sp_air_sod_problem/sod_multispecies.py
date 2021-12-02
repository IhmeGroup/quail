import numpy as np
import copy

FinalTime = 0.01
NumTimeSteps = 1500

TimeStepping = {
    "InitialTime" : 0.,
    "FinalTime" : FinalTime,
    "TimeStepSize" : 1.0e-5,
    "TimeStepper" : "ADER",
#    "TimeStepper" : "SSPRK3",
}


Numerics = {
    "SolutionOrder" : 2,
    "SolutionBasis" : "LagrangeSeg",
    "Solver" : "ADERDG",
    #"Solver" : "DG",
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
    "xmin" : -10.,
    "xmax" : 10.,
#    "PeriodicBoundariesX" : ["x1", "x2"]
}


Physics = {
    "Type" : "EulerMultispecies_2sp_air",
    "ConvFluxNumerical" : "LaxFriedrichs",
}

state = { 
    "Function" : "SodMultispeciesAir",
}

InitialCondition = state

BoundaryConditions = {
    "x1" : {
        "BCType" : "StateAll",
	"Function" : "SodMultispeciesAir",
       },
    "x2" : { 
        "BCType" : "StateAll",
	"Function" : "SodMultispeciesAir",
       }
}
