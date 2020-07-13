import numpy as np
import copy
cfl = 1.0
u = 1.
dx = 0.01

dt = cfl*dx/u
EndTime = 2.
nTimeSteps =int(EndTime/dt)
print(nTimeSteps)
TimeStepping = {
    "StartTime" : 0.,
    "EndTime" : EndTime,
    "nTimeStep" : nTimeSteps,
    "TimeScheme" : "SSPRK3",
}

Numerics = {
    "InterpOrder" : 2,
    "InterpBasis" : "LagrangeEqSeg",
    "Solver" : "DG",
    "ApplyLimiter" : "PositivityPreserving",
    "InterpolateIC" : False,
    # "NodeType" : "GaussLobatto",
    # "ElementQuadrature" : "GaussLobatto",
    # "FaceQuadrature" : "GaussLobatto",
    # "NodesEqualQuadpts" : False,

}

Output = {
    "WriteInterval" : 5,
    "WriteInitialSolution" : True,
    "AutoProcess" : True,
}

Mesh = {
    "File" : None,
    "ElementShape" : "Segment",
    "nElem_x" : 100,
    "xmin" : -5.,
    "xmax" : 5.,
    # "PeriodicBoundariesX" : ["x1", "x2"],
}

Physics = {
    "Type" : "Euler",
    "ConvFlux" : "LaxFriedrichs",
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
    "xshock" : 0.0,
}

state_exact = {
    "Function" : "ExactRiemannSolution",
    "uL" : uL,
    "uR" : uR,
    "length" : 10.,
}
InitialCondition = state
state2 = state.copy()
state2.update({"BCType":"StateAll"})
ExactSolution = state
# ExactSolution = state_exact

BoundaryConditions = {
    "Left" : {
        "BCType" : "SlipWall"
        },
    "Right" : { 
        "BCType" : "SlipWall"
        }
}
