import numpy as np

cfl = 0.25
num_elems = 360
dx = float(1/num_elems)
u = 5.
dt = cfl*dx/u

EndTime = np.pi/10.
num_time_stepss =  int(EndTime/dt)
print(num_time_stepss)
TimeStepping = {
    "StartTime" : 0.,
    "EndTime" : EndTime,
    "num_time_steps" : num_time_stepss,
    "TimeScheme" : "Simpler",
    "OperatorSplitting_Imp" : "Trapezoidal",

}

Numerics = {
    "InterpOrder" : 2,
    "InterpBasis" : "LagrangeSeg",
    "L2InitialCondition" : False,
    "Solver" : "DG",
    "ApplyLimiter" : "PositivityPreservingChem",
    "SourceTreatment" : "Implicit",
#     "NodeType" : "GaussLobatto",
#     "ElementQuadrature" : "GaussLobatto",
#     "FaceQuadrature" : "GaussLobatto",
#     "NodesEqualQuadpts" : True,
}

Output = {
    # "WriteInterval" : 10,
    # "WriteInitialSolution" : True,
    "AutoPostProcess" : False,
    "Prefix" : "TestSimpler",

}

Mesh = {
    "File" : None,
    "ElementShape" : "Segment",
    "NumElemsX" : num_elems,
    "xmin" : 0.,
    "xmax" : 2.*np.pi,
    # "PeriodicBoundariesX" : ["x1", "x2"],
}

Physics = {
    "Type" : "Chemistry",
    "ConvFluxNumerical" : "LaxFriedrichs",
    "GasConstant" : 1.,
    "SpecificHeatRatio" : 1.2,
    "HeatRelease": 50.,
}

uL = np.array([2.,4.,40.,0.])
uR = np.array([0.,0.,1.,1.])

state = {
    "Function" : "SimpleDetonation3",
    "uL" : uL,
    "uR" : uR,
    "xshock" : np.pi/2.,
}

InitialCondition = state
state2 = state.copy()
state2.update({"BCType":"StateAll"})
# ExactSolution = InitialCondition.copy()

BoundaryConditions = {
    "x1" : state2,
    # "x2" : state2,
    "x2" : { 
        "BCType" : "SlipWall"
        }
}

SourceTerms = {
    "source1" : {
        "Function" : "Heaviside",
        "Da" : 3000.,
        "Tign" : 2.,
    },
}
