import general


TimeStepping = {
    "StartTime" : 0.,
    "EndTime" : 1.,
    "nTimeStep" : 100.,
    "TimeScheme" : "RK4",
}

Numerics = {
    "InterpOrder" : 1,
    "InterpBasis" : "LagrangeEqSeg",
    "Solver" : "DG",
    "InterpolateIC" : False,
    "InterpolateFlux" : False,
    "OrderSequencing" : False,
    "ApplyLimiter" : None, 
}

Output = {
    "TrackOutput" : None,
    "WriteTimeHistory" : False,
    "Prefix" : "Data",
    "WriteInterval" : -1,
    "WriteInitialSolution" : False,
    "WriteFinalSolution" : False,
    "RestartFile" : None,
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
    "SetAsExact" : False,
}

BoundaryConditions = {}

SourceTerms = {}