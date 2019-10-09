import numpy as np
from enum import IntEnum
from General import *
from Quadrature import *
from Math import *
import Mesh
import code
from Data import ArrayList, GenericData


Basis2Shape = {
    BasisType.SegLagrange : ShapeType.Segment 
}


Shape2Dim = {
    ShapeType.Point : 0,
    ShapeType.Segment : 1
}


FaceShape = {
    ShapeType.Segment : ShapeType.Point
}


def Order2nNode(Basis, p):
    Shape = Basis2Shape[Basis]
    if Shape == ShapeType.Point:
    	nn = 1
    elif Shape == ShapeType.Segment:
    	nn = p + 1
    else:
    	raise Exception("Shape not supported")

    return nn


# def Shape2Dim(Shape):
# 	if Shape == ShapeType.Point:
# 		dim = 0
# 	elif Shape == ShapeType.Segment:
# 		dim = 1
# 	else:
# 		raise Exception("Shape not supported")

# 	return dim


# def FaceShape(ElemShape):
#     if ElemShape == ShapeType.Segment:
#         FShape = ShapeType.Point
#     else:
#         raise Exception("Shape not supported")

#     return FShape


def GetMassMatrix(mesh, egrp, elem, Basis, Order, StaticData=None):
    if StaticData is None:
        pnq = -1
        quadData = None
        PhiData = None
        JData = JacobianData(mesh)
        StaticData = GenericData()
    else:
        nq = StaticData.pnq
        quadData = StaticData.quadData
        PhiData = StaticData.PhiData
        JData = StaticData.JData

    QuadOrder,QuadChanged = GetQuadOrderElem(egrp, Order*2, mesh.ElemGroups[egrp].QBasis, mesh, quadData=quadData)
    if QuadChanged:
        quadData = QuadData(QuadOrder, EntityType.Element, egrp, mesh)

    nq = quadData.nquad
    xq = quadData.xquad
    wq = quadData.wquad

    if QuadChanged:
        # PhiData = BasisData(egrp=0,Order,entity=EntityType.Element,nq,xq,mesh,GetPhi=True,GetGPhi=False)
        PhiData = BasisData(Basis,Order,nq,mesh)
        PhiData.EvalBasis(xq, Get_Phi=True)

    JData.ElemJacobian(egrp,elem,nq,xq,mesh,Get_detJ=True)
    nn = PhiData.nn

    phi = PhiData.Phi
    MM = np.zeros([nn,nn])
    for iq in range(nq):
    	for i in range(nn):
    		for j in range(nn):
    			t = 0.
    			for iq in range(nq):
    				t += phi[iq,i]*phi[iq,j]*wq[iq]*JData.detJ[iq*(JData.nq != 1)]
    			MM[i,j] = t

    StaticData.pnq = nq
    StaticData.quadData = quadData
    StaticData.PhiData = PhiData
    StaticData.JData = JData

    return MM, StaticData


def GetInvMassMatrix(mesh, egrp, elem, Basis, Order, StaticData=None):
	MM, StaticData = GetMassMatrix(mesh, egrp, elem, Basis, Order, StaticData)

	MMinv = np.linalg.inv(MM) 

	return MMinv, StaticData


def ComputeInvMassMatrices(mesh, EqnSet, solver=None):
    ## Allocate MMinv_all
    # Currently calculating inverse mass matrix for every single element,
    # even if uniform mesh
    ArrayDims = [None]*mesh.nElemGroup
    for egrp in range(mesh.nElemGroup):
        Basis = EqnSet.Bases[egrp]
        Order = EqnSet.Orders[egrp]
        nn = Order2nNode(Basis, Order)
        ArrayDims[egrp] = [mesh.nElems[egrp], nn, nn]
    MMinv_all = ArrayList(nArray=mesh.nElemGroup,ArrayDims=ArrayDims)

    StaticData = None

    for egrp in range(mesh.nElemGroup):
        EGroup = mesh.ElemGroups[egrp]
        Basis = EqnSet.Bases[egrp]
        Order = EqnSet.Orders[egrp]
        for elem in range(EGroup.nElem):
            MMinv,StaticData = GetInvMassMatrix(mesh, egrp, elem, Basis, Order, StaticData)
            MMinv_all.Arrays[egrp][elem] = MMinv

    if solver is not None:
        solver.DataSet.MMinv_all = MMinv_all

    return MMinv_all


def GetShapes(Basis, Order, nq, xq, phi=None):
    nn = Order2nNode(Basis, Order)

    if phi is None or phi.shape != (nq,nn):
        phi = np.zeros([nq,nn])
    else:
        phi[:] = 0.

    if Basis == BasisType.SegLagrange:
    	for iq in range(nq): 
    		Shape_TensorLagrange(1, Order, xq[iq], phi[iq,:])
    else:
        raise Exception("Basis not supported")

    return phi


def GetGrads(Basis, Order, dim, nq, xq, GPhi=None):
    nn = Order2nNode(Basis, Order)

    if GPhi is None or GPhi.shape != (nq,nn,dim):
        GPhi = np.zeros([nq,nn,dim])
    else: 
        GPhi[:] = 0.

    if Basis == BasisType.SegLagrange:
    	for iq in range(nq): 
    		Grad_TensorLagrange(1, Order, xq[iq], GPhi[iq,:,:])
    else:
        raise Exception("Basis not supported")

    return GPhi


def BasisLagrange1D(x, xnode, nnode, phi, gphi):
	for j in range(nnode):
		if phi is not None:
			pj = 1.
			for i in range(j): pj *= (x-xnode[i])/(xnode[j]-xnode[i])
			for i in range(j+1,nnode): pj *= (x-xnode[i])/(xnode[j]-xnode[i])
			phi[j] = pj
		if gphi is not None:
			gphi[j] = 0.0;
			for i in range(nnode):
				if i != j:
					g = 1./(xnode[j] - xnode[i])
					for k in range(nnode):
						if k != i and k != j:
							g *= (x - xnode[k])/(xnode[j] - xnode[k])
					gphi[j] += g


def Shape_TensorLagrange(dim, p, x, phi):
	if p == 0:
		phi[0] = 1.
		return phi

	nnode = p + 1
	xnode = np.zeros(nnode)
	dx = 1./float(p)
	for i in range(nnode): xnode[i] = float(i)*dx
	if dim == 1:
		BasisLagrange1D(x, xnode, nnode, phi, None)


def Grad_TensorLagrange(dim, p, x, gphi):
	if p == 0:
		gphi[0,:] = 0.
		return gphi 

	nnode = p + 1
	xnode = np.zeros(nnode)
	dx = 1./float(p)
	for i in range(nnode): xnode[i] = float(i)*dx
	if dim == 1:
		BasisLagrange1D(x, xnode, nnode, None, gphi)


class BasisData(object):
    '''
    Class: IFace
    --------------------------------------------------------------------------
    This is a class defined to encapsulate the temperature table with the 
    relevant methods
    '''
    def __init__(self,Basis,Order,nq,mesh):
        '''
        Method: __init__
        --------------------------------------------------------------------------
        This method initializes the temperature table. The table uses a
        piecewise linear function for the constant pressure specific heat 
        coefficients. The coefficients are selected to retain the exact 
        enthalpies at the table points.
        '''
        self.Basis = Basis
        self.Order = Order
        self.nn = Order2nNode(self.Basis, self.Order)
        self.nq = nq
        self.nnqmax = self.nn * self.nq
        self.dim = Shape2Dim[Basis2Shape[self.Basis]]
        self.Phi = None
        self.GPhi = None
        self.gPhi = None
        self.face = -1

    def PhysicalGrad(self, JData):
        nq = self.nq
        if nq != JData.nq and JData.nq != 1:
            raise Exception("Quadrature doesn't match")
        dim = JData.dim
        if dim != self.dim:
            raise Exception("Dimensions don't match")
        nn = self.nn
        GPhi = self.GPhi 
        if GPhi is None:
            raise Exception("GPhi is an empty list")

        if self.gPhi is None or self.gPhi.shape != (nq,nn,dim):
            self.gPhi = np.zeros([nq,nn,dim])
        else:
            self.gPhi *= 0.

        gPhi = self.gPhi

        if gPhi.shape != GPhi.shape:
            raise Exception("gPhi and GPhi are different sizes")

        for iq in range(nq):
            G = GPhi[iq,:,:]
            g = gPhi[iq,:,:]
            iJ = JData.iJ[iq*(JData.nq != 1),:,:]
            g[:] = np.transpose(np.matmul(iJ.transpose(),G.transpose()))

        return gPhi

    def EvalBasis(self, xq, Get_Phi=True, Get_GPhi=False, Get_gPhi=False, JData=None):
        if Get_Phi:
            self.Phi = GetShapes(self.Basis, self.Order, self.nq, xq, self.Phi)
        if Get_GPhi:
            self.GPhi = GetGrads(self.Basis, self.Order, self.dim, self.nq, xq, self.GPhi)
        if Get_gPhi:
            if not JData:
                raise Exception("Need jacobian data")
            self.gPhi = self.PhysicalGrad(JData)

    def EvalBasisOnFace(self, mesh, egrp, face, xq, xelem, Get_Phi=True, Get_GPhi=False, Get_gPhi=False, JData=False):
        self.face = face
        Basis = mesh.ElemGroups[egrp].QBasis
        xelem = Mesh.RefFace2Elem(Basis2Shape[Basis], face, self.nq, xq, xelem)
        self.EvalBasis(xelem, Get_Phi, Get_GPhi, Get_gPhi, JData)

        return xelem

    	

class JacobianData(object):
    '''
    Class: IFace
    --------------------------------------------------------------------------
    This is a class defined to encapsulate the temperature table with the 
    relevant methods
    '''
    def __init__(self,mesh):
        '''
        Method: __init__
        --------------------------------------------------------------------------
        This method initializes the temperature table. The table uses a
        piecewise linear function for the constant pressure specific heat 
        coefficients. The coefficients are selected to retain the exact 
        enthalpies at the table points.
        '''
        self.nq = 0
        self.dim = mesh.Dim
        self.detJ = None
        self.J = None
        self.iJ = None
        self.A = None
        self.GPhi = None

        # self.ElemJacobian(self,egrp,elem,nq,xq,mesh,Get_detJ,Get_J,Get_iJ)

    def ElemJacobian(self,egrp,elem,nq,xq,mesh,Get_detJ=False,Get_J=False,Get_iJ=False):
        EGroup = mesh.ElemGroups[egrp]
        Basis = EGroup.QBasis
        Order = EGroup.QOrder
        if Order == 1:
            nq = 1

        nn = Order2nNode(Basis, Order)
        dim = Shape2Dim[Basis2Shape[Basis]]

        ## Check if we need to resize or recalculate 
        if self.dim != dim or self.nq != nq: Resize = True
        else: Resize = False

        # if self.GPhi.shape != (nq,nn,dim):
        #     self.GPhi = GetGrads(Basis, Order, dim, nq, xq)
        self.GPhi = GetGrads(Basis, Order, dim, nq, xq, self.GPhi)
        GPhi = self.GPhi

        self.dim = dim
        if dim != mesh.Dim:
            raise Exception("Dimensions don't match")

        self.nq = nq

        if Get_J and (Resize or self.J is None): 
            self.J = np.zeros([nq,dim,dim])
        if Get_detJ and (Resize or self.detJ is None): 
            self.detJ = np.zeros([nq,1])
        if Get_iJ and (Resize or self.iJ is None): 
            self.iJ = np.zeros([nq,dim,dim])

        if Resize or self.A is None:
            self.A = np.zeros([dim,dim])
        else:
            self.A[:] = 0.
        A = self.A
        Elem2Nodes = EGroup.Elem2Nodes[elem]
        for iq in range(nq):
            G = GPhi[iq,:,:]
            for i in range(nn):
                for j in range(dim):
                    for k in range(dim):
                        A[j,k] += mesh.Coords[Elem2Nodes[i],j]*G[i,k]
            if Get_J:
                self.J[iq,:] = A[:]
                # for i in range(dim):
                #     for j in range(dim):
                #         self.J[iq,i,j] = A[i,j]
            if Get_detJ: detJ_ = self.detJ[iq]
            else: detJ_ = None
            if Get_iJ: iJ_ = self.iJ[iq,:,:]
            else: iJ_ = None 
            MatDetInv(A, dim, detJ_, iJ_)

            if detJ_ is not None and detJ_ <= 0.:
                raise Exception("Nonpositive Jacobian (egrp = %d, elem = %d)" % (egrp, elem))

