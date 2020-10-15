import numpy as np
import copy

FinalTime = 0.
NumTimeSteps = 0  

TimeStepping = {
    "InitialTime" : 0.,
    "FinalTime" : FinalTime,
    "NumTimeSteps" :NumTimeSteps,
    # "CFL" : 0.2,
    "TimeStepper" : "RK4",
}

Numerics = {
    "SolutionOrder" : 2,
    "SolutionBasis" : "LagrangeSeg",
    "Solver" : "DG",
    "ApplyLimiters" : "PositivityPreserving",
    "NodeType" : "GaussLobatto",
    "ElementQuadrature" : "GaussLobatto",
    "FaceQuadrature" : "GaussLobatto",
    "ColocatedPoints" : True,

}

Output = {
    # "WriteInterval" : 2,
    # "WriteInitialSolution" : True,
    "AutoPostProcess" : True,
    "Prefix" : "data",
}

Mesh = {
    "File" : None,
    "ElementShape" : "Segment",
    "NumElemsX" : 200,
    "xmin" : 0.,
    "xmax" : 1.,
    # "PeriodicBoundariesX" : ["x1", "x2"],
}

Physics = {
    "Type" : "Euler",
    "ConvFluxNumerical" : "Roe",
    "GasConstant" : 1.,
    "SpecificHeatRatio" : 1.4,
}

# a = 0.9
uL = np.array([1.,0.,1.])
uR = np.array([0.125,0.,0.1])

state = { 
     "Function" : "RiemannProblem",
     "uL" : uL,
     "uR" : uR,
     # "w" : 0.05,
     "xd" : 0.5,
}

state_exact = {
    "Function" : "ExactRiemannSolution",
    "uL" : uL,
    "uR" : uR,
    "xmin" : 0.,
    "xmax" : 1.,
    "xd" : 0.5,
}
InitialCondition = state
state2 = state_exact.copy()
state2.update({"BCType":"StateAll"})

# ExactSolution = state_exact

BoundaryConditions = {
    "x1" : {
        "BCType" : "SlipWall"
        },
    "x2" : { 
        "BCType" : "SlipWall"
        }
}
