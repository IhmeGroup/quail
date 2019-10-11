import numpy as np
from General import *
import Mesh
import code
import Basis
from QuadratureRules import *


def GetQuadOrderElem(mesh, egrp, basis, Order, EqnSet=None, quadData=None):
	# assumes uniform QOrder and QBasis
	QOrder = mesh.ElemGroups[egrp].QOrder
	# QuadOrder = 2*Order + 1
	if EqnSet is not None:
		QuadOrder = EqnSet.QuadOrder(Order)
	else:
		QuadOrder = Order

	if QOrder > 1:
		dim = mesh.Dim
		QuadOrder += dim*(QOrder-1)

	Shape = Basis.Basis2Shape[basis]
	QuadChanged = True
	if quadData is not None:
		if QuadOrder == quadData.Order and Shape == quadData.Shape:
			QuadChanged = False

	return QuadOrder, QuadChanged


def GetQuadOrderIFace(mesh, IFace, basis, Order, EqnSet=None, quadData=None):
	# assumes uniform QOrder and QBasis
	QOrderL = mesh.ElemGroups[IFace.ElemGroupL].QOrder
	QOrderR = mesh.ElemGroups[IFace.ElemGroupR].QOrder
	QOrder = np.amax([QOrderL, QOrderR])
	# QuadOrder = 2*Order + 1
	if EqnSet is not None:
		QuadOrder = EqnSet.QuadOrder(Order)
	else:
		QuadOrder = Order
	if QOrder > 1:
		dim = mesh.Dim - 1
		QuadOrder += dim*(QOrder-1)

	Shape = Basis.Basis2Shape[basis]
	FShape = Basis.FaceShape[Shape]
	QuadChanged = True
	if quadData is not None:
		if QuadOrder == quadData.Order and FShape == quadData.Shape:
			QuadChanged = False

	return QuadOrder, QuadChanged


def GetQuadOrderBFace(mesh, BFace, basis, Order, EqnSet=None, quadData=None):
	# assumes uniform QOrder and QBasis
	QOrder = mesh.ElemGroups[BFace.ElemGroup].QOrder
	# QuadOrder = 2*Order + 1
	if EqnSet is not None:
		QuadOrder = EqnSet.QuadOrder(Order)
	else:
		QuadOrder = Order
	if QOrder > 1:
		dim = mesh.Dim - 1
		QuadOrder += dim*(QOrder-1)

	Shape = Basis.Basis2Shape[basis]
	FShape = Basis.FaceShape[Shape]
	QuadChanged = True
	if quadData is not None:
		if QuadOrder == quadData.Order and FShape == quadData.Shape:
			QuadChanged = False

	return QuadOrder, QuadChanged


class QuadData(object):
	'''
	Class: IFace
	--------------------------------------------------------------------------
	This is a class defined to encapsulate the temperature table with the 
	relevant methods
	'''
	def __init__(self,mesh,egrp,entity,Order):
		'''
		Method: __init__
		--------------------------------------------------------------------------
		This method initializes the temperature table. The table uses a
		piecewise linear function for the constant pressure specific heat 
		coefficients. The coefficients are selected to retain the exact 
		enthalpies at the table points.
		'''
		### assumes 1D
		# QOrder = mesh.ElemGroups[egrp].QOrder
		dim = Mesh.GetEntityDim(mesh, entity)
		self.Shape = Basis.Basis2Shape[mesh.ElemGroups[egrp].QBasis]
		self.Order = Order
		self.xquad = QuadLinePoints[Order]
		self.wquad = QuadLineWeights[Order]
		self.qdim = mesh.Dim
		self.nvec = None

		if entity > EntityType.Element: 
			self.Shape = Basis.FaceShape[self.Shape]
			self.xquad = np.zeros([1,1])
			self.wquad = np.ones([1,1])

		self.nquad = len(self.xquad)

		# self.xquad = np.copy(QuadLinePoints[Order])
		# self.wquad = np.copy(QuadLineWeights[Order])
		# reshape
		# self.xquad.shape = self.nquad,self.qdim
		# self.wquad.shape = self.nquad,self.qdim
