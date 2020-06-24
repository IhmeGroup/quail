TimeStepping = {
    "StartTime" : 0.,
    "EndTime" : 10.,
    "nTimeStep" : 200,
    "TimeScheme" : "RK4",
}

Numerics = {
    "InterpOrder" : 10,
    "InterpBasis" : "HierarchicH1Tri",
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
    "WriteInterval" : 50,
    "WriteInitialSolution" : False,
    "WriteFinalSolution" : True,
    "AutoProcess" : True,
}

Mesh = {
    "File" : None,
    "ElementShape" : "Triangle",
    "nElem_x" : 2,
    "nElem_y" : 2,
    "xmin" : -5.,
    "xmax" : 5.,
    "ymin" : -5.,
    "ymax" : 5.,
    "PeriodicBoundariesX" : ["x1", "x2"],
    "PeriodicBoundariesY" : ["y1", "y2"],
}

Physics = {
    "Type" : "ConstAdvScalar",
    "ConvFlux" : "LaxFriedrichs",
    "ConstXVelocity" : 1.,
    "ConstYVelocity" : 1.,
}

x0 = [0., 0.]
InitialCondition = {
    "Function" : "Gaussian",
    "x0" : x0,
}

ExactSolution = InitialCondition.copy()


bparams = {
    "Function" : "Gaussian",
    "x0" : x0,
    "BCType" : "StateAll"
}
BoundaryConditions = {
    # "x1" : bparams,
    # "x2" : bparams,
    # "y1" : bparams,
    # "y2" : bparams,
}