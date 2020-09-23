import numpy as np
import copy

#FinalTime = 0.25
#nTimeSteps = 4000
FinalTime = 6.2500e-03
nTimeSteps = 100
#nTimeSteps = 0

TimeStepping = {
    "InitialTime" : 0.,
    "FinalTime" : FinalTime,
    "NumTimeSteps" : nTimeSteps,
    #"CFL" : 0.025,
    "TimeStepper" : "RK4",
}

Numerics = {
    "SolutionOrder" : 2,
    "SolutionBasis" : "LagrangeSeg",
    "Solver" : "DG",
    "ApplyLimiter" : "PositivityPreserving",
    #"NodeType" : "GaussLobatto",
    #"ElementQuadrature" : "GaussLobatto",
    #"FaceQuadrature" : "GaussLobatto",
    #"CollocatedPoints" : True,

}

Output = {
    # "WriteInterval" : 2,
    # "WriteInitialSolution" : True,
    "AutoPostProcess" : True,
    "Prefix" : "Test2",
}

Mesh = {
    "File" : None,
    "ElementShape" : "Segment",
    "NumElemsX" : 60,
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
    "rhoL" : 1., 
    "uL" : 0.,
    "pL" : 1.,
    "rhoR" : 0.125, 
    "uR" : 0.,
    "pR" : 0.1,
    # "w" : 0.05,
    "xd" : 0.5,
}

state_exact = {
    "Function" : "ExactRiemannSolution",
    "rhoL" : 1.,
    "uL" : 0.,
    "pL" : 1.,
    "rhoR" : 0.125,
    "uR" : 0.,
    "pR" : 0.1,
#    "xmin" : 0.,
#    "xmax" : 1.,
    "xd" : 0.5,
}
InitialCondition = state
state2 = state.copy()
state2.update({"BCType":"StateAll"})
ExactSolution = state_exact
# ExactSolution = state_exact

BoundaryConditions = {
    "x1" : {
        "BCType" : "SlipWall"
        },
    "x2" : { 
        "BCType" : "SlipWall"
        }
}
