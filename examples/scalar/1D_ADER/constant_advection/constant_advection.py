import numpy as np

TimeStepping = {
    "InitialTime" : 0.,
    "FinalTime" : 0.5,
    "NumTimeSteps" : 40,
    "TimeStepper" : "ADER",
}

Numerics = {
    "SolutionOrder" : 3,
    "SolutionBasis" : "LagrangeSeg",
    "L2InitialCondition" : False,
    "Solver" : "ADERDG",
    "ElementQuadrature" : "GaussLobatto",
    "FaceQuadrature" : "GaussLobatto",
    "NodeType" : "GaussLobatto",
    "ColocatedPoints" : True,
    "InterpolateFluxADER" : True,
}

Output = {
    # "WriteInterval" : 1,
    # "WriteInitialSolution" : True,
    "AutoPostProcess" : True,
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
    "x1" : {
	    "Function" : "Sine",
	    "omega" : 2*np.pi,
    	"BCType" : "StateAll",
    },
    "x2" : {
    	"BCType" : "Extrapolate",
    },
}
