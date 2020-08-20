
TimeStepping = {
    "InitialTime" : 0.,
    "FinalTime" : 10.,
    "CFL" : 0.01,
    "TimeScheme" : "RK4",
}

Numerics = {
    "InterpOrder" : 10,
    "InterpBasis" : "HierarchicH1Tri",
    "Solver" : "DG",
    "ElementQuadrature" : "GaussLegendre",
    "FaceQuadrature" : "GaussLegendre",
    "InterpolateFlux" : False,
    "ApplyLimiter" : None, 
}

Output = {
    "Prefix" : "Data",
    "WriteInterval" : 4,
    "WriteInitialSolution" : True,
    "WriteFinalSolution" : True,
    "AutoPostProcess" : True,
}

Mesh = {
    "File" : None,
    "ElementShape" : "Triangle",
    "NumElemsX" : 2,
    "NumElemsY" : 2,
    "xmin" : -5.,
    "xmax" : 5.,
    "ymin" : -5.,
    "ymax" : 5.,
    "PeriodicBoundariesX" : ["x1", "x2"],
    "PeriodicBoundariesY" : ["y1", "y2"],
}

Physics = {
    "Type" : "ConstAdvScalar",
    "ConvFluxNumerical" : "LaxFriedrichs",
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
