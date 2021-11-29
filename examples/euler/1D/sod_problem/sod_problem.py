import numpy as np
import copy

FinalTime = 0.01
NumTimeSteps = 1500

TimeStepping = {
    "InitialTime" : 0.,
    "FinalTime" : FinalTime,
    "TimeStepSize" : 1e-5,
#    "CFL" : 0.05,
    "TimeStepper" : "ADER",
}


Numerics = {
    "SolutionOrder" : 3,
    "SolutionBasis" : "LagrangeSeg",
    "Solver" : "ADERDG",
#    "ApplyLimiters" : "PositivityPreserving",
    "NodeType" : "Equidistant",
    "ElementQuadrature" : "GaussLegendre",
    "FaceQuadrature" : "GaussLegendre",
    "ArtificialViscosity" : True,
    "AVParameter" : 1000,
    "PredictorThreshold" : 1e-9,

}

Output = {
    "AutoPostProcess" : False,
    "Prefix" : "Data",
#    "WriteInterval" : 500
}

Mesh = {
    "File" : None,
    "ElementShape" : "Segment",
    "NumElemsX" : 100,
    "xmin" : -10.,
    "xmax" : 10.,
}


Physics = {
    "Type" : "Euler",
    "ConvFluxNumerical" : "LaxFriedrichs",
    #"GasConstant" :  189.036442535717,
    "SpecificHeatRatio" : 1.4,
}


state = { 
    "Function" : "RiemannProblem",
    "rhoL" : 1., 
    "uL" : 0.,
    "pL" : 100000.,
    "rhoR" : 0.125, 
    "uR" : 0.,
    "pR" : 10000.,
    "xd" : 0.0,
}

InitialCondition = state
ExactSolution = state

BoundaryConditions = {
    "x1" : {
        "Function" : "RiemannProblem",
        "rhoL" : 1., 
        "uL" : 0.,
        "pL" : 100000.,
        "rhoR" : 0.125, 
        "uR" : 0.,
        "pR" : 10000.,
        "xd" : 0.0,
        "BCType" : "StateAll" 
        },
    "x2" : { 
        "Function" : "RiemannProblem",
        "rhoL" : 1., 
        "uL" : 0.,
        "pL" : 100000.,
        "rhoR" : 0.125, 
        "uR" : 0.,
        "pR" : 10000.,
        "xd" : 0.0,
        "BCType" : "StateAll",
        }
}
