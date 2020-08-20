import numpy as np

TimeStepping = {
    "StartTime" : 0.,
    "EndTime" : 0.5,
    "CFL" : 0.1,
    # "TimeStepSize" : 0.0125,
    # "num_time_steps" : 40,
    "TimeScheme" : "RK4",
}

Numerics = {
    "InterpOrder" : 2,
    "InterpBasis" : "LagrangeSeg",
    "Solver" : "DG",
    "ElementQuadrature" : "GaussLobatto",
    "FaceQuadrature" : "GaussLobatto",
    "NodeType" : "GaussLobatto",
    "NodesEqualQuadpts" : True,
    "InterpolateFlux" : True,
}

Mesh = {
    "File" : None,
    "ElementShape" : "Segment",
    "NumElemsX" : 16,
    "xmin" : -1.,
    "xmax" : 1.,
    # "PeriodicBoundariesX" : ["xmin","xmax"]
}

Physics = {
    "Type" : "ConstAdvScalar",
    "ConvFluxNumerical" : "LaxFriedrichs",
    "ConstVelocity" : 1.,
}

InitialCondition = {
    "Function" : "Sine",
    "omega" : 2*np.pi,
}

ExactSolution = InitialCondition.copy()

BoundaryConditions = {
    "Left" : {
	    "Function" : "Sine",
	    "omega" : 2*np.pi,
    	"BCType" : "StateAll",
    },
    "Right" : {
    	"BCType" : "Extrapolate",
    },
}
