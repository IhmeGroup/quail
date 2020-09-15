from abc import ABC, abstractmethod
from data import GenericData
import numpy as np


def set_key(data, **kwargs):
    for key in kwargs:
        if hasattr(data, key):
            setattr(data, key, kwargs[key])
        else: 
            raise KeyError

class FcnBase(ABC):
    '''
    Class: ICData
    --------------------------------------------------------------------------
    This is a class that encompases the initial conditions

    ATTRIBUTES: 
        Function: function that describes the initial condition. Options located in Scalar.py and Euler.py
        x: coordinates for initial conditions
        Time: establish the initial time
        U: solution array at the initial condition
        Data: generic data needed for specific initial conditions
    '''
    @abstractmethod
    def get_state(self, physics, x, t):
        pass


# class ExactData(ICData):
#     pass


class BCBase(ABC):

    # def __init__(self):
    #     self.UqB = np.zeros(0)
    #     self.F = np.zeros(0)

    # def alloc_helpers(self, shape):
    #     self.UqB.resize(shape)
    #     self.F.resize(shape)

    @abstractmethod
    def get_boundary_state(self, physics, x, t, normals, UqI):
        pass

    @abstractmethod
    def get_boundary_flux(self, physics, x, t, normals, UqI):
        pass


class BCWeakRiemann(ABC):

    def get_boundary_flux(self, physics, x, t, normals, UqI):

        UqB = self.get_boundary_state(physics, x, t, normals, UqI)
        F = physics.get_conv_flux_numerical(UqI, UqB, normals)

        return F


class BCWeakPrescribed(BCWeakRiemann):

    def get_boundary_flux(self, physics, x, t, normals, UqI):

        UqB = self.get_boundary_state(physics, x, t, normals, UqI)
        F = physics.get_conv_flux_projected(UqB, normals)

        return F

    # def __init__(self):
    #     super().__init__()
    #     self.Name = ""
    #     self.BCType = 0
    #     self.uI = np.zeros(0)

    # def Set(self, Function=None, Name="", BCType=0, **kwargs):
    #     self.Function = Function
    #     self.Name = Name
    #     self.BCType = BCType
    #     for key in kwargs:
    #         self.Data.__dict__[key] = kwargs[key]


class SourceBase(ABC):

    @abstractmethod
    def get_source(self, physics, Uq, x, t):
        pass

    def get_jacobian(self, physics, Uq, x, t):
        raise NotImplementedError


class ConvNumFluxBase(ABC):

    def __init__(self, Uq=None):
        pass

    def alloc_helpers(self, Uq):
        self.__init__(Uq)

    @abstractmethod
    def compute_flux(self, physics, UqL, UqR, normals):
        pass


# class ICData(object):
#     '''
#     Class: ICData
#     --------------------------------------------------------------------------
#     This is a class that encompases the initial conditions

#     ATTRIBUTES: 
#         Function: function that describes the initial condition. Options located in Scalar.py and Euler.py
#         x: coordinates for initial conditions
#         Time: establish the initial time
#         U: solution array at the initial condition
#         Data: generic data needed for specific initial conditions
#     '''
#     def __init__(self):
#         self.Function = None
#         self.x = None
#         self.time = 0.
#         self.U = None
#         self.Data = GenericData()

#     def Set(self, **kwargs):
#         for key in kwargs:
#             # if key in self.__dict__.keys(): self.__dict__[key] = kwargs[key]
#                 ## NOTE: __dict__ doesn't work this way for inherited classes
#             # if key in dir(self): 
#             if hasattr(self, key):
#                 setattr(self, key, kwargs[key])
#             else: 
#                 setattr(self.Data, key, kwargs[key])


# class BCData(ICData):
#     def __init__(self):
#         BCData.Name = ""
#         BCData.BCType = 0
#         ICData.__init__(self)

#     # def Set(self, Function=None, Name="", BCType=0, **kwargs):
#     #     self.Function = Function
#     #     self.Name = Name
#     #     self.BCType = BCType
#     #     for key in kwargs:
#     #         self.Data.__dict__[key] = kwargs[key]

# class SourceData(ICData):
#     def __init__(self):
#         SourceData.Name = ""
#         SourceData.S = None
#         ICData.__init__(self)

# class ExactData(ICData):
#     def __init__(self):
#         ICData.__init__(self)
