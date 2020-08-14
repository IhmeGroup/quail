import numpy as np
import copy

EndTime = 0.25
nTimeSteps = 4000
#nTimeSteps = 0

TimeStepping = {
    "StartTime" : 0.,
    "EndTime" : EndTime,
    "num_time_steps" : nTimeSteps,
    #"CFL" : 0.025,
    "TimeScheme" : "RK4",
}

Numerics = {
    "InterpOrder" : 2,
    "InterpBasis" : "LagrangeSeg",
    "Solver" : "DG",
    "ApplyLimiter" : "PositivityPreserving",
    "InterpolateIC" : False,
    #"NodeType" : "GaussLobatto",
    #"ElementQuadrature" : "GaussLobatto",
    #"FaceQuadrature" : "GaussLobatto",
    #"NodesEqualQuadpts" : True,

}

Output = {
    # "WriteInterval" : 2,
    # "WriteInitialSolution" : True,
    "AutoProcess" : True,
    "Prefix" : "Test2",
}

Mesh = {
    "File" : None,
    "ElementShape" : "Segment",
    "NumElems_x" : 60,
    "xmin" : 0.,
    "xmax" : 1.,
    # "PeriodicBoundariesX" : ["x1", "x2"],
}

Physics = {
    "Type" : "Euler",
    "ConvFlux" : "Roe",
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
    "xshock" : 0.5,
}

state_exact = {
    "Function" : "ExactRiemannSolution",
    "uL" : uL,
    "uR" : uR,
    "xmin" : 0.,
    "xmax" : 1.,
    "xshock" : 0.5,
}
InitialCondition = state
state2 = state.copy()
state2.update({"BCType":"StateAll"})
ExactSolution = state_exact
# ExactSolution = state_exact

BoundaryConditions = {
    "Left" : {
        "BCType" : "SlipWall"
        },
    "Right" : { 
        "BCType" : "SlipWall"
        }
}
