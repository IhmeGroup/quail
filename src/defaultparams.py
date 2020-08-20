Restart = {
    "File" : None,
    "StartFromFileTime" : True
}

TimeStepping = {
    "InitialTime" : 0.,
    "FinalTime" : 1.
    "num_time_steps" : None, # NumTimeSteps
    "TimeStepSize" : None, 
    "CFL" : None,
    "TimeScheme" : "RK4", # TimeStepper
    "OperatorSplitting_Exp" : "SSPRK3", # OperatorSplittingExplicit
    "OperatorSplitting_Imp" : "BDF1", # OperatorSplittingImplicit
}

Numerics = {
    "InterpOrder" : 1, # SolutionOrder
    "InterpBasis" : "LagrangeSeg", # SolutionBasis
    "Solver" : "DG",
    "ElementQuadrature" : "GaussLegendre",
    "FaceQuadrature" : "GaussLegendre",
    "NodeType" : "Equidistant",
    "NodesEqualQuadpts" : False, # CollocatedPoints
    "L2InitialCondition" : True,
    "ApplyLimiter" : None, 
    "SourceTreatment" : "Explicit", # SourceTreatmentADER
    "InterpolateFlux" : True, # InterpolateFluxADER, True
    "ConvFluxSwitch" : True,
    "SourceSwitch" : True,
}
# eric 
Mesh = {
    "File" : None,
    "ElementShape" : "Segment",
    "NumElemsX" : 10,
    "NumElemsY" : 10,
    "xmin" : -1.,
    "xmax" : 1.,
    "ymin" : -1.,
    "ymax" : 1.,
    "PeriodicBoundariesX" : [],
    "PeriodicBoundariesY" : [],
}

Physics = {
    "Type" : "ConstAdvScalar",
    "ConvFluxNumerical" : "LaxFriedrichs", 
}

InitialCondition = {
    "Function" : "Uniform",
}

ExactSolution = {}

BoundaryConditions = {}

SourceTerms = {}

Output = {
    "Prefix" : "Data",
    "WriteInterval" : -1,
    "WriteInitialSolution" : False,
    "WriteFinalSolution" : True,
    "AutoPostProcess" : True,
}