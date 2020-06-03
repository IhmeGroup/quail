import numpy as np
import code



class GenericData(object):
    '''
    Class: GenericData
    --------------------------------------------------------------------------
    This is a class designed to store random amounts of generic data
    '''
    def __init__(self):
        '''
        Method: __init__
        --------------------------------------------------------------------------
        This method initializes the GenericData class
        '''
        pass


class ICData(object):
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
    def __init__(self):
        self.Function = None
        self.x = None
        self.Time = 0.
        self.U = None
        self.Data = GenericData()

    def Set(self, **kwargs):
        for key in kwargs:
            # if key in self.__dict__.keys(): self.__dict__[key] = kwargs[key]
                ## NOTE: __dict__ doesn't work this way for inherited classes
            # if key in dir(self): 
            if hasattr(self, key):
                setattr(self, key, kwargs[key])
            else: 
                setattr(self.Data, key, kwargs[key])


class BCData(ICData):
    def __init__(self):
        BCData.Name = ""
        BCData.BCType = 0
        ICData.__init__(self)

    # def Set(self, Function=None, Name="", BCType=0, **kwargs):
    #     self.Function = Function
    #     self.Name = Name
    #     self.BCType = BCType
    #     for key in kwargs:
    #         self.Data.__dict__[key] = kwargs[key]

class SourceData(ICData):
    def __init__(self):
        SourceData.Name = ""
        SourceData.S = None
        ICData.__init__(self)

class ExactData(ICData):
    def __init__(self):
        ICData.__init__(self)


class ArrayList(object):
    '''
    Class: IFace
    --------------------------------------------------------------------------
    This is a class defined to encapsulate the temperature table with the 
    relevant methods
    '''
    def __init__(self,nArray=0,ArrayDims=[],SimilarArray=None,CopyValues=False):
        '''
        Method: __init__
        --------------------------------------------------------------------------
        This method initializes the temperature table. The table uses a
        piecewise linear function for the constant pressure specific heat 
        coefficients. The coefficients are selected to retain the exact 
        enthalpies at the table points.
        '''


        if nArray != len(ArrayDims):
            raise Exception("Input error")

        if SimilarArray is not None:
            ArrayDims = SimilarArray.ArrayDims
            nArray = SimilarArray.nArray

        self.Arrays = []
        for n in range(nArray):
            self.Arrays.append(np.zeros(ArrayDims[n]))

        self.ArrayDims = ArrayDims
        self.nArray = nArray

        if CopyValues:
            for n in range(nArray):
                self.Arrays[n][:] = SimilarArray.Arrays[n][:]

        # if nArray != len(nEntriesPerArray):
        #     raise Exception("Input error")

        # if SimilarArray is not None:
        #     FullDim = SimilarArray.FullDim
        #     nArray = SimilarArray.nArray
        #     nEntriesPerArray = SimilarArray.nEntries

        # self.FullArray = np.zeros(FullDim)
        # self.Arrays = [self.FullArray[0:nEntriesPerArray[0]]]
        # for i in range(1,nArray):
        #     self.Arrays.append(self.FullArray[nEntriesPerArray[i-1]:nEntriesPerArray[i]])

        # self.FullDim = FullDim
        # self.nArray = nArray
        # self.nEntries = nEntriesPerArray

    def SetUniformValue(self, value = 0.):
        for n in range(self.nArray):
            self.Arrays[n][:] = value

    def ScaleByFactor(self, c = 1.):
        for n in range(self.nArray):
            self.Arrays[n][:] *= c

    def AddToSelf(self, ArrayListToAdd, c = 1.):
        A = ArrayListToAdd
        if self.nArray != A.nArray or self.ArrayDims != A.ArrayDims:
            raise Exception("ArrayList sizes don't match")

        for n in range(self.nArray):
            self.Arrays[n][:] += c*A.Arrays[n][:]
    
    def CopyToSelf(self, ArrayListToCopy):
        A = ArrayListToCopy
        if self.nArray != A.nArray or self.ArrayDims != A.ArrayDims:
            raise Exception("ArrayList sizes don't match")

        for n in range(self.nArray):
            self.Arrays[n][:] = A.Arrays[n][:]

    def SetToSum(self, A1, A2, c1=1., c2=1.):
        if self.nArray != A1.nArray or self.ArrayDims != A1.ArrayDims:
            raise Exception("ArrayList sizes don't match")
        if self.nArray != A2.nArray or self.ArrayDims != A2.ArrayDims:
            raise Exception("ArrayList sizes don't match")

        self.CopyToSelf(A1)
        self.ScaleByFactor(c1)
        self.AddToSelf(A2, c2)

    def Max(self):
        maxvalue = -np.inf 
        for n in range(self.nArray):
            value = np.max(self.Arrays[n])
            if maxvalue < value:
                maxvalue = value

        return maxvalue

    def Min(self):
        minvalue = np.inf 
        for n in range(self.nArray):
            value = np.max(self.Arrays[n])
            if minvalue > value:
                minvalue = value

        return minvalue

    def VectorNorm(self, ord=2, sdim=-1, sidx=-1):
        if sdim*sidx < 0:
            raise IndexError
        norm = 0.
        for n in range(self.nArray):
            if sdim < 0:
                # Don't slice array
                Array = self.Arrays[n]
            else:
                # Slice array based on sdim, sidx
                indices = []
                ArrayDim = ArrayDims[n]
                broadcasted = False
                for dim in range(len(ArrayDim)):
                    if dim == sdim:
                        indices.append(sidx)
                    elif not broadcasted:
                        indices.append(np.arange(ArrayDim[dim])[:,np.newaxis])
                        broadcasted = True
                    else:
                        indices.append(np.arange(ArrayDim[dim]))
                Array = self.Arrays[n][indices]

            norm += np.sum(np.abs(Array)**ord)

        norm = norm**(1./ord)

        return norm






