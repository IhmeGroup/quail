import numpy as np
import copy
cfl = 0.02
u = 1.8
dx = 0.1

dt = cfl*dx/u
FinalTime = 1.3
nTimeSteps =int(FinalTime/dt)
print(nTimeSteps)
NumTimeSteps = 4000
TimeStepping = {
    "InitialTime" : 0.,
    "FinalTime" : FinalTime,
    "NumTimeSteps" : NumTimeSteps,
    "TimeStepper" : "RK4",
}

Numerics = {
    "SolutionOrder" : 2,
    "SolutionBasis" : "LagrangeSeg",
    "Solver" : "DG",
    "ApplyLimiter" : "PositivityPreserving",
    "L2InitialCondition" : False,
    "NodeType" : "Equidistant",
#    "ElementQuadrature" : "GaussLobatto",
#    "FaceQuadrature" : "GaussLobatto",
#    "ColocatedPoints" : True,

}

Output = {
    # "WriteInterval" : 1,
    # "WriteInitialSolution" : True,
    "AutoPostProcess" : False,
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

uL = np.array([0.445, 0.698, 3.528])
uR = np.array([0.5, 0., 0.571])

state = {
    "Function" : "ExactRiemannSolution",
    "uL" : uL,
    "uR" : uR,
    "xmin" : -5.,
    "xmax" : 5.,
    "xd" : 0.0,
}
InitialCondition = state

state2 = state.copy()
state2.update({"BCType":"StateAll"})
ExactSolution = InitialCondition.copy()

BoundaryConditions = {
    "x1" : state2,
    "x2" : state2,
}
