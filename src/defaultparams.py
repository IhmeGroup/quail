Restart = {
    "File" : None,
    "StartFromFileTime" : True
}

TimeStepping = {
    "StartTime" : 0., # InitialTime
    "EndTime" : 1., # FinalTime
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
    "InterpolateIC" : False, # L2InitialCondition, True - eric
    "OrderSequencing" : False, # remove - eric
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
    "NumElems_x" : 10, # NumElemsX
    "NumElems_y" : 10, # NumElemsY
    "xmin" : -1.,
    "xmax" : 1.,
    "ymin" : -1.,
    "ymax" : 1.,
    "PeriodicBoundariesX" : [],
    "PeriodicBoundariesY" : [],
}

Physics = {
    "Type" : "ConstAdvScalar",
    "ConvFlux" : "LaxFriedrichs", # ConvFluxNumerical
}

InitialCondition = {
    "Function" : "Uniform",
    # "State" : [1.0],
    # "SetAsExact" : False,
}

ExactSolution = {}

BoundaryConditions = {}

SourceTerms = {}

Output = {
    "TrackOutput" : None, # remove
    "WriteTimeHistory" : False, # remove
    "Prefix" : "Data",
    "WriteInterval" : -1,
    "WriteInitialSolution" : False,
    "WriteFinalSolution" : True,
    "AutoProcess" : True, # AutoPostProcess
}