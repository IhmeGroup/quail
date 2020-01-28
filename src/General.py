from enum import IntEnum, Enum
import Errors
import numpy as np
import code


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
    SegLagrange = 0
    QuadLagrange = 1
    TriLagrange = 2


class LimiterType(IntEnum):
    PositivityPreserving = 0
    ScalarPositivityPreserving = 1


INTERIORFACE = -1
NULLFACE = -2


# Default solver parameters
SolverParams = {
	"StartTime" : 0.,
	"EndTime" : 1.,
	"nTimeStep" : 100.,
    "InterpOrder" : 1,
    "InterpBasis" : BasisType.SegLagrange,
    "TimeScheme" : "RK4",
    "InterpolateIC" : False,
    "LinearGeomMapping" : False,
    "UniformMesh" : False,
    "UseNumba" : False,
    "OrderSequencing" : False,
    "TrackOutput" : None,
    "WriteTimeHistory" : False,
    "ApplyLimiter" : None, 
}


def SetSolverParams(Params=None, **kwargs):
    if Params is None:
        Params = SolverParams
	for key in kwargs:
		if key not in Params.keys(): raise KeyError
		Params[key] = kwargs[key]
    if Params["UseNumba"]:
        try:
            import numba
        except:
            Params["UseNumba"] = False
    return Params



