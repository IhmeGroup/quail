Restart = {
    "File" : None,
    "StartFromFileTime" : True
}

TimeStepping = {
    "InitialTime" : 0.,
    "FinalTime" : 1.,
    "NumTimeSteps" : None,
    "TimeStepSize" : None, 
    "CFL" : None,
    "TimeStepper" : "RK4",
    "OperatorSplittingExplicit" : "SSPRK3",
    "OperatorSplittingImplicit" : "BDF1",
}

Numerics = {
    "SolutionOrder" : 1, 
    "SolutionBasis" : "LagrangeSeg",
    "Solver" : "DG",
    "ElementQuadrature" : "GaussLegendre",
    "FaceQuadrature" : "GaussLegendre",
    "NodeType" : "Equidistant",
    "CollocatedPoints" : False,
    "L2InitialCondition" : True,
    "ApplyLimiter" : None, 
    "SourceTreatmentADER" : "Explicit",
    "InterpolateFluxADER" : True,
    "ConvFluxSwitch" : True,
    "SourceSwitch" : True,
}

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