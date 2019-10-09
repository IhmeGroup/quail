from enum import IntEnum, Enum


class EntityType(IntEnum):
    Element = 0
    IFace = 1
    BFace = 2


class ShapeType(IntEnum):
    Point = 0
    Segment = 1
    Triangle = 2
    Quadrilateral = 3


class BasisType(IntEnum):
    SegLagrange = 0


INTERIORFACE = -1


# Default solver parameters
SolverParams = {
	"StartTime" : 0.,
	"EndTime" : 1.,
	"nTimeStep" : 100.,
    "InterpOrder" : 1,
    "InterpBasis" : BasisType.SegLagrange,
    "TimeScheme" : "RK4"
}


def SetSolverParams(Params=None, **kwargs):
    if Params is None:
        Params = SolverParams
	for key in kwargs:
		if key not in Params.keys(): raise Exception("Input error")
		Params[key] = kwargs[key]
    return Params