import numpy as np

TimeStepping = {
    "InitialTime" : 0.,
    "FinalTime" : 1.5,
    "CFL" : 0.1,
    # "TimeStepSize" : 0.0125,
    # "NumTimeSteps" : 40,
    "TimeStepper" : "SSPRK3",
}

Numerics = {
    "SolutionOrder" : 2,
    "SolutionBasis" : "LagrangeSeg",
    "Solver" : "DG",
    "ElementQuadrature" : "GaussLegendre",
    "FaceQuadrature" : "GaussLegendre",
   "ApplyLimiters" : ["ScalarWENO"],
   "ShockIndicator" : "MinMod",
    # "NodeType" : "GaussLobatto",
    # "ColocatedPoints" : True,
    # "InterpolateFluxADER" : True,
}

Output = {
    "WriteInterval" : 10,
    "WriteInitialSolution" : True,
    "AutoPostProcess" : True,
#    "Prefix" : "WENO",

}

Mesh = {
    "File" : None,
    "ElementShape" : "Segment",
    "NumElemsX" : 80,
    "xmin" : 0.,
    "xmax" : 2*np.pi,
    "PeriodicBoundariesX" : ["x1","x2"]
}

Physics = {
    "Type" : "Burgers",
    "ConvFluxNumerical" : "LaxFriedrichs",
    # "ConstVelocity" : 1.,
}

InitialCondition = {
    "Function" : "SineBurgers",
    "omega" : 1.,
}

# ExactSolution = {
#     "Function" : "SineBurgers",
#     "omega" : 1.,
# }
#BoundaryConditions = {
#    "x1" : {
#	    "Function" : "Sine",
#	    "omega" : 2*np.pi,
#    	"BCType" : "StateAll",
#    },
#    "x2" : {
#    	"BCType" : "Extrapolate",
#    },
#}
