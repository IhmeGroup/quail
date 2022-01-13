import numpy as np
import cantera as ct
import copy

def get_state():

    gas = ct.Solution("h2o2_10sp_40rxn.cti")
    # gas = ct.Solution("h2o2.xml")

    # gas.TPX = 1000.0, ct.one_atm,  \
    #         "H2:{}, O2:{}, AR:{}".format(2, 1, 7)
    # rho = gas.DP[0]
    P = 1e5
    YH2 = 1.1189834407236525e-01
    YO2 = 8.8810165592763479e-01
    gas.TPY = 1100., P,  \
        "H2:{}, O2:{}, Ar:{}".format(YH2, YO2, 1.0-YH2-YO2) 
    rho = gas.DP[0]
    U = np.zeros([12])
    U[0] = rho
    U[1] = 0.0
    U[2] = rho * gas.UV[0] # ignore KE since u = 0
    for isp in range(9):
        U[isp+3] = rho * gas.DPY[2][isp]
    
    return U

init_state = get_state()
FinalTime = 0.0001
#FinalTime = 3.503e-5
Restart = {
#	"File" : 'Data_restart.pkl',
		# If file name provided (str), then will restart from said data file
		# (pickle format)
#	"StartFromFileTime" : True
		# If True, then will restart from time saved in restart file
}

TimeStepping = {
    "InitialTime" : 0.,
    "FinalTime" : FinalTime,
    # "CFL" : 0.001,
    "TimeStepSize" : 1.0e-7,
    "TimeStepper" : "ADER",
    #"TimeStepper" : "Strang",
    "OperatorSplittingImplicit" : "LSODA",

}


Numerics = {
    "SolutionOrder" : 2,
    "SolutionBasis" : "LagrangeSeg",
    "Solver" : "ADERDG",
    "SourceTreatmentADER" : "StiffImplicit",
    #"Solver" : "DG",
    # "ArtificialViscosity" : True,
    # "AVParameter" : 1e4,
    "PredictorThreshold" : 1e-12,
    "PredictorGuessADER" : "ODEGuess",

}

Output = {
    "AutoPostProcess" : False,
    "Prefix" : "Data",
#    "WriteInterval" : 325,
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
    "Type" : "EulerMultispecies1D_10sp_H2O2",
    "ConvFluxNumerical" : "LaxFriedrichs",
}

state = { 
    "Function" : "Uniform",
    "state" : init_state,
}

InitialCondition = state

SourceTerms = {
	"Reacting" : { # Name of source term ("Source1") doesn't matter
		"Function" : "Reacting",
		"source_treatment" : "Implicit",
	},
}
