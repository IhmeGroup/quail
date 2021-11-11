import numpy as np
import copy

FinalTime = 2.0
NumTimeSteps = 1500

TimeStepping = {
    "InitialTime" : 0.,
    "FinalTime" : 0.,#FinalTime,
    "TimeStepSize" : 0.005,
	#"CFL" : 0.2,
    "TimeStepper" : "SSPRK3",
}


Numerics = {
    "SolutionOrder" : 1,
    "SolutionBasis" : "LagrangeSeg",
    "Solver" : "DG",
#    "ApplyLimiters" : "PositivityPreserving",
}

Output = {
    "AutoPostProcess" : True,
    "Prefix" : "Data",
}

Mesh = {
    "File" : None,
    "ElementShape" : "Segment",
    "NumElemsX" : 200,
    "xmin" : -5.,
    "xmax" : 5.,
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
