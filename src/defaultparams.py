Restart = {
    "File" : None,
    "StartFromFileTime" : True
}

TimeStepping = {
    "StartTime" : 0.,
    "EndTime" : 1.,
    "nTimeStep" : 100,
    "TimeScheme" : "RK4",
}

Numerics = {
    "InterpOrder" : 1,
    "InterpBasis" : "LagrangeEqSeg",
    "Solver" : "DG",
    "ElementQuadrature" : "GaussLegendre",
    "FaceQuadrature" : "GaussLegendre",
    "InterpolateIC" : False,
    "OrderSequencing" : False,
    "ApplyLimiter" : None, 
    "SourceTreatment" : "Explicit",
    "InterpolateFlux" : True,
    "ConvFluxSwitch" : True,
    "SourceSwitch" : True,
}

Mesh = {
    "File" : None,
    "ElementShape" : "Segment",
    "nElem_x" : 10,
    "nElem_y" : 10,
    "xmin" : -1.,
    "xmax" : 1.,
    "ymin" : -1.,
    "ymax" : 1.,
    "PeriodicBoundariesX" : [],
    "PeriodicBoundariesY" : [],
}

Physics = {
    "Type" : "ConstAdvScalar",
    "ConvFlux" : "LaxFriedrichs",
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
    "TrackOutput" : None,
    "WriteTimeHistory" : False,
    "Prefix" : "Data",
    "WriteInterval" : -1,
    "WriteInitialSolution" : False,
    "WriteFinalSolution" : True,
    "AutoProcess" : True,
}