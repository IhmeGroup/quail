import code
import numpy as np
from enum import IntEnum, Enum

import errors



np.set_printoptions(precision=15)

# Constants
eps = 1.e-15


###
class EntityType(IntEnum):
    Element = 0
    IFace = 1
    BFace = 2


class ShapeType(IntEnum):
    Point = 0
    Segment = 1
    Quadrilateral = 2
    Triangle = 3


class BasisType(IntEnum):
    LagrangeEqSeg = 0
    LagrangeEqQuad = 1
    LagrangeEqTri = 2
    LegendreSeg = 3
    LegendreQuad = 4
    HierarchicH1Tri= 5


class LimiterType(IntEnum):
    PositivityPreserving = 0
    ScalarPositivityPreserving = 1


class SolverType(IntEnum):
    DG = 0
    ADERDG = 1


class StepperType(IntEnum):
    FE = 0
    RK4 = 1
    LSRK4 = 2
    SSPRK3 = 3
    ADER = 4


class PhysicsType(IntEnum):
    ConstAdvScalar = 0
    Burgers = 1
    Euler = 2


INTERIORFACE = -1
NULLFACE = -2


# Default solver parameters
SolverParams = {
	"StartTime" : 0.,
	"EndTime" : 1.,
	"nTimeStep" : 100.,
    "InterpOrder" : 1,
    "InterpBasis" : BasisType.LagrangeEqSeg,
    "TimeScheme" : "RK4",
    "InterpolateIC" : False,
    "InterpolateFlux": True,
    "LinearGeomMapping" : False,
    "UniformMesh" : False,
    "UseNumba" : False,
    "OrderSequencing" : False,
    "TrackOutput" : None,
    "WriteTimeHistory" : False,
    "ApplyLimiter" : None, 
    "Prefix" : "Data",
    "WriteInterval" : -1,
    "WriteInitialSolution" : False,
    "WriteFinalSolution" : False,
    "RestartFile" : None,
}


def SetSolverParams(Params=None, **kwargs):
    if Params is None:
        Params = SolverParams
    for key in kwargs:
    	if key not in Params.keys(): raise KeyError
    	Params[key] = kwargs[key]
    return Params



