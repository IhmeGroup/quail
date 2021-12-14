import numpy as np
import cantera as ct
import copy

def get_state():

    gas = ct.Solution("h2o2_11sp_23rxn.xml")
    gas.TPX = 1000.0, ct.one_atm,  \
            "H2:{}, O2:{}, N2:{}".format(2, 1, 7)
    rho = gas.DP[0]
    U = np.zeros([13])
    U[0] = rho
    U[1] = 0.0
    U[2] = rho * gas.UV[0] # ignore KE since u = 0
    for isp in range(10):
        U[isp+3] = rho * gas.DPY[2][isp]
    return U

init_state = get_state()

FinalTime = 0.00012

TimeStepping = {
    "InitialTime" : 0.,
    "FinalTime" : FinalTime,
    "CFL" : 0.1,
#    "TimeStepSize" : 1.0e-5,
#    "TimeStepper" : "ADER",
    "TimeStepper" : "Strang",
    "OperatorSplittingImplicit" : "LSODA",

}


Numerics = {
    "SolutionOrder" : 0,
    "SolutionBasis" : "LagrangeSeg",
#   "Solver" : "ADERDG",
    "Solver" : "DG",
    # "ArtificialViscosity" : True,
    # "AVParameter" : 1e4,
    "PredictorThreshold" : 1e-10,
}

Output = {
    "AutoPostProcess" : True,
    "Prefix" : "Data",
#    "WriteInterval" : 2,
    "WriteInitialSolution" : True,
}

Mesh = {
    "File" : None,
    "ElementShape" : "Segment",
    "NumElemsX" : 1,
    "xmin" : 0.0,
    "xmax" : 0.005,
    "PeriodicBoundariesX" : ["x1", "x2"]
}


Physics = {
    "Type" : "EulerMultispecies1D_11sp_H2O2",
    "ConvFluxNumerical" : "LaxFriedrichs",
    "ReactorFlag" : True,
}

state = { 
    "Function" : "Uniform",
    "state" : init_state,
}

InitialCondition = state

# BoundaryConditions = {
#     "x1" : {
#         "BCType" : "SlipWall",
#        },
#     "x2" : { 
#         "BCType" : "StateAll",
# 	"Function" : "ReactingShockTube",
#        }
# }

SourceTerms = {
	"Reacting" : { # Name of source term ("Source1") doesn't matter
		"Function" : "Reacting",
		"source_treatment" : "Implicit",
	},
}
