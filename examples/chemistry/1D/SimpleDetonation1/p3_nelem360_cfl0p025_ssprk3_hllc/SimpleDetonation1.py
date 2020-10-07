import numpy as np

cfl = 0.01

FinalTime = 1.8
TimeStepping = {
    "InitialTime" : 0.,
    "FinalTime" : FinalTime,
    "CFL" : cfl,
    # "NumTimeSteps" : 0,
    "TimeStepper" : "SSPRK3",
}

Numerics = {
    "SolutionOrder" : 3,
    "SolutionBasis" : "LagrangeSeg",
    "InterpolateIC" : False,
    "Solver" : "DG",
    "ApplyLimiter" : "PositivityPreservingChem",
    # "NodeType" : "GaussLobatto",
    # "ElementQuadrature" : "GaussLobatto",
    # "FaceQuadrature" : "GaussLobatto",
    # "ColocatedPoints" : True,
}

Output = {
    "WriteInterval" : 50,
    "WriteInitialSolution" : True,
    "AutoProcess" : True,
}

Mesh = {
    "File" : None,
    "ElementShape" : "Segment",
    "nElem_x" : 360,
    "xmin" : 0.,
    "xmax" : 30.,
    # "PeriodicBoundariesX" : ["x1", "x2"],
}

Physics = {
    "Type" : "Chemistry",
    "ConvFlux" : "HLLC",
    "GasConstant" : 1.,
    "SpecificHeatRatio" : 1.4,
    "HeatRelease": 25.,
}

InitialCondition = {
    "Function" : "SimpleDetonation1",
    "rho_u" : 1.0,
    "u_u" : 0.0,
    "p_u" : 1.0,
    "Y_u" : 1.0,
    "xshock" : 10.,
}

ExactSolution = InitialCondition.copy()

BoundaryConditions = {
    "Left" : {
        "BCType" : "StateAll",
	    "Function" : "SimpleDetonation1",
        "rho_u" : 1.0,
        "u_u" : 0.0,
        "p_u" : 1.0,
        "Y_u" : 1.0,
        "xshock" : 10.,  
    },
    "Right" : {
        "BCType" : "StateAll",
        "Function" : "SimpleDetonation1",
        "rho_u" : 1.0,
        "u_u" : 0.0,
        "p_u" : 1.0,
        "Y_u" : 1.0,
        "xshock" : 10.,    },
}

SourceTerms = {

    # "source1" : {
    #     "Function" : "Heaviside",
    #     "Da" : 20.,
    #     "Tign" : 0.22,
    # },
    "source1" : {
        "Function" : "Arrhenius",
         "A" : 16418.,
        #"A" : 0.
        "b" : 0.0,
        "Tign" : 25.,
    },
}
