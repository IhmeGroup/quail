import numpy as np
import copy

TimeStepping = {
    "InitialTime" : 0.,
    "FinalTime" : 1.3,
    "NumTimeSteps" : 4000,
    "TimeStepper" : "RK4",
}

Numerics = {
    "SolutionOrder" : 2,
    "SolutionBasis" : "LagrangeSeg",
    "Solver" : "DG",
    "ApplyLimiters" : ["PositivityPreserving"],
    # "NodeType" : "Equidistant",
#    "ElementQuadrature" : "GaussLobatto",
#    "FaceQuadrature" : "GaussLobatto",
#    "ColocatedPoints" : True,

}

Mesh = {
    "File" : None,
    "ElementShape" : "Segment",
    "NumElemsX" : 200,
    "xmin" : 0.,
    "xmax" : 1.,
}

Physics = {
    "Type" : "Euler",
    "ConvFluxNumerical" : "Roe",
    "GasConstant" : 1.,
    "SpecificHeatRatio" : 1.4,
}

state = { 
    "Function" : "RiemannProblem",
    "rhoL" : .445, 
    "uL" : 0.698,
    "pL" : 3.528,
    "rhoR" : 0.5, 
    "uR" : 0.,
    "pR" : 0.571,
    "xd" : 0.5,
}

state_exact = {
    "Function" : "ExactRiemannSolution",
    "rhoL" : .445, 
    "uL" : 0.698,
    "pL" : 3.528,
    "rhoR" : 0.5, 
    "uR" : 0.,
    "pR" : 0.571,
    "xd" : 0.5,
}
InitialCondition = state

state2 = state.copy()
state2.update({"BCType":"StateAll"})
ExactSolution = state_exact

BoundaryConditions = {
    "x1" : {
        "BCType" : "SlipWall"
        },
    "x2" : { 
        "BCType" : "SlipWall"
        }
}
