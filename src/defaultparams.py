# ------------------------------------------------------------------------ #
#
#       File : src/defaultparams.py
#
#       Contains default inputs parameters.
#      
# ------------------------------------------------------------------------ #

'''
Restart information
'''
Restart = {
    "File" : None,
        # If file name provided (str), then will restart from said file
    "StartFromFileTime" : True
        # If True, then will restart from time saved in restart file
}


'''
Time stepping information
'''
TimeStepping = {
    "InitialTime" : 0.,
        # Initial time
    "FinalTime" : 1.,
        # Final time
    "NumTimeSteps" : None,
        # Number of time steps (1st priority)
    "TimeStepSize" : None, 
        # Time step size (2nd priority)
    "CFL" : None,
        # CFL number (3rd priority)
    "TimeStepper" : "RK4",
        # Time stepping scheme
        # See general.StepperType
    "OperatorSplittingExplicit" : "SSPRK3",
        # Explicit time stepping scheme for source terms if doing operator 
        # splitting
        # See general.StepperType
    "OperatorSplittingImplicit" : "BDF1",
        # Implicit time stepping scheme for source terms if doing operator 
        # splitting
        # See general.ODESolverType
}

Numerics = {
    "SolutionOrder" : 1, 
        # Polynomial order of solution approximation
    "SolutionBasis" : "LagrangeSeg",
        # Basis type of solution approximation
        # See general.BasisType
    "Solver" : "DG",
        # Solver type
        # See general.SolverType
    "ElementQuadrature" : "GaussLegendre",
        # Quadrature type for integration over elements
        # See general.QuadratureType
    "FaceQuadrature" : "GaussLegendre",
        # Quadrature type for integration over faces
        # See general.QuadratureType
    "NodeType" : "Equidistant",
        # Node location type (for nodal basis functions)
        # See general.NodeType
    "CollocatedPoints" : False,
        # If True, quadrature points will be the solution nodes. Otherwise,
        # overintegration is used, i.e. quadrature of high order, determined
        # by the physics type and the geometric order
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