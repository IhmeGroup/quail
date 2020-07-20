import code
from enum import Enum, auto
import numpy as np

import defaultparams
import errors



np.set_printoptions(precision=15)

# Constants
eps = 1.e-15

class ShapeType(Enum):
    Point = auto()
    Segment = auto()
    Quadrilateral = auto()
    Triangle = auto()


class BasisType(Enum):
    LagrangeSeg = auto()
    LagrangeQuad = auto()
    LagrangeTri = auto()
    LegendreSeg = auto()
    LegendreQuad = auto()
    HierarchicH1Tri = auto()


class LimiterType(Enum):
    PositivityPreserving = auto()
    ScalarPositivityPreserving = auto()
    PositivityPreservingChem = auto()


class SolverType(Enum):
    DG = auto()
    ADERDG = auto()


class StepperType(Enum):
    FE = auto()
    RK4 = auto()
    LSRK4 = auto()
    SSPRK3 = auto()
    ADER = auto()
    Strang = auto()
    Simpler = auto()

class ODESolverType(Enum):
    BDF1 = auto()
    Trapezoidal = auto()

class PhysicsType(Enum):
    ConstAdvScalar = auto()
    Burgers = auto()
    Euler = auto()
    Chemistry = auto()

class ModalOrNodal(Enum):
    Modal = auto()
    Nodal = auto()
    Neither = auto()


class QuadratureType(Enum):
    GaussLegendre = auto()
    GaussLobatto = auto()
    Dunavant = auto()

class NodeType(Enum):
    Equidistant = auto()
    GaussLobatto = auto()


INTERIORFACE = -1
NULLFACE = -2


# Default solver parameters
# SolverParams = {
# 	"StartTime" : 0.,
# 	"EndTime" : 1.,
# 	"nTimeStep" : 100.,
#     "InterpOrder" : 1,
#     "InterpBasis" : BasisType.LagrangeEqSeg,
#     "TimeScheme" : "RK4",
#     "InterpolateIC" : False,
#     "InterpolateFlux": True,
#     "LinearGeomMapping" : False,
#     "UniformMesh" : False,
#     "UseNumba" : False,
#     "OrderSequencing" : False,
#     "TrackOutput" : None,
#     "WriteTimeHistory" : False,
#     "ApplyLimiter" : None, 
#     "Prefix" : "Data",
#     "WriteInterval" : -1,
#     "WriteInitialSolution" : False,
#     "WriteFinalSolution" : False,
#     "RestartFile" : None,
#     "AutoProcess" : False,
# }

SolverParams = {**defaultparams.TimeStepping, **defaultparams.Numerics, **defaultparams.Output, **defaultparams.Restart}


def SetSolverParams(Params=None, **kwargs):
    if Params is None:
        Params = SolverParams
        Params["RestartFile"] = SolverParams["File"]
    for key in kwargs:
    	if key not in Params.keys(): 
            raise KeyError
    	Params[key] = kwargs[key]
    return Params



