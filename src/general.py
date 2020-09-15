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

solver_params = {**defaultparams.TimeStepping, **defaultparams.Numerics, **defaultparams.Output, **defaultparams.Restart}


def set_solver_params(params=None, **kwargs):
    if params is None:
        params = solver_params
        params["RestartFile"] = solver_params["File"]
    for key in kwargs:
    	if key not in params.keys(): 
            raise KeyError
    	params[key] = kwargs[key]
    return params



