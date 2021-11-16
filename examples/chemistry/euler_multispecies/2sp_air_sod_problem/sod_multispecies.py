import numpy as np
import copy

FinalTime = 0.01
NumTimeSteps = 1500

TimeStepping = {
    "InitialTime" : 0.,
    "FinalTime" : FinalTime,
    "TimeStepSize" : 1e-5,
#    "CFL" : 0.05,
    "TimeStepper" : "SSPRK3",
}


Numerics = {
    "SolutionOrder" : 0,
    "SolutionBasis" : "LagrangeSeg",
    "Solver" : "DG",
#    "ApplyLimiters" : "PositivityPreserving",
}

Output = {
    "AutoPostProcess" : True,
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
ExactSolution = state
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
