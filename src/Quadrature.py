import numpy as np
from General import *
import Mesh
import code
import Basis
from QuadratureRules import *


def GetQuadOrderElem(mesh, egrp, basis, Order, EqnSet=None, quadData=None):
	# assumes uniform QOrder and QBasis

	#Add logic to add 1 to the dimension of an ADER case?
	
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
	if Shape is ShapeType.Quadrilateral:
		QuadOrder += mesh.Dim
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
	if Shape is ShapeType.Quadrilateral:
		QuadOrder += Basis.Shape2Dim[FShape]

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
	if Shape is ShapeType.Quadrilateral:
		QuadOrder += Basis.Shape2Dim[FShape]

	QuadChanged = True
	if quadData is not None:
		if QuadOrder == quadData.Order and FShape == quadData.Shape:
			QuadChanged = False

	return QuadOrder, QuadChanged


class QuadData(object):
	'''
	Class: IFace
	--------------------------------------------------------------------------
	This is a class defined to define the quadrature data for a given entity 
	(i.e. element, iface, or bface)
	'''
	def __init__(self,mesh,basis,entity,Order):
		'''
		Method: __init__
		--------------------------------------------------------------------------
		This method initializes the quadrature data for a given entity. It uses the 
		mesh and basis to determine the shape of the entity as well as its dimension.
		It also uses the order of the method to look up the following:
			xquad: Locations of points in reference space.
			wquad: Weight of the points in reference space.
			nquad: Number of points to be evaluated.
		'''
		# QOrder = mesh.ElemGroups[egrp].QOrder
		dim = Mesh.GetEntityDim(mesh, entity)
		self.Order = Order
		self.qdim = mesh.Dim
		self.nvec = None

		Shape = Basis.Basis2Shape[basis]

		if entity == EntityType.Element:
			self.Shape = Shape
		else:
			self.Shape = Basis.FaceShape[Shape]

		if self.Shape == ShapeType.Point:
			self.xquad = np.zeros([1,1])
			self.wquad = np.ones([1,1])
		elif self.Shape == ShapeType.Segment:
			self.xquad = QuadLinePoints[Order]
			self.wquad = QuadLineWeights[Order]
		elif self.Shape == ShapeType.Quadrilateral:
			self.xquad = QuadQuadrilateralPoints[Order]
			self.wquad = QuadQuadrilateralWeights[Order]
		elif self.Shape == ShapeType.Triangle:
			self.xquad = QuadTrianglePoints[Order]
			self.wquad = QuadTriangleWeights[Order]
		else:
			raise NotImplementedError


		self.nquad = len(self.xquad)

class QuadDataADER(object):
	'''
	Class: IFace
	--------------------------------------------------------------------------
	This class is identical to QuadData's class other than it automatically adds
	1 to the dimension of the entity and mesh. This is done to extend a 1D scheme
	to time.
	'''
	def __init__(self,mesh,basis,entity,Order):
		'''
		Method: __init__
		--------------------------------------------------------------------------
		This method initializes the quadrature data for a given entity. It uses the 
		mesh and basis to determine the shape of the entity as well as its dimension.
		It also uses the order of the method to look up the following:
			xquad: Locations of points in reference space.
			wquad: Weight of the points in reference space.
			nquad: Number of points to be evaluated.
			'''
		# QOrder = mesh.ElemGroups[egrp].QOrder
		dim = Mesh.GetEntityDim(mesh, entity) + 1
		self.Order = Order
		self.qdim = mesh.Dim + 1
		self.nvec = None

		Shape = Basis.Basis2Shape[basis]

		if entity == EntityType.Element:
			self.Shape = Shape
		else:
			self.Shape = Basis.FaceShape[Shape]

		if self.Shape == ShapeType.Point:
			self.xquad = np.zeros([1,1])
			self.wquad = np.ones([1,1])
		elif self.Shape == ShapeType.Segment:
			self.xquad = QuadLinePoints[Order]
			self.wquad = QuadLineWeights[Order]
		elif self.Shape == ShapeType.Quadrilateral:
			self.xquad = QuadQuadrilateralPoints[Order]
			self.wquad = QuadQuadrilateralWeights[Order]
		elif self.Shape == ShapeType.Triangle:
			self.xquad = QuadTrianglePoints[Order]
			self.wquad = QuadTriangleWeights[Order]
		else:
			raise NotImplementedError

		self.nquad = len(self.xquad)

		# self.xquad = np.copy(QuadLinePoints[Order])
		# self.wquad = np.copy(QuadLineWeights[Order])
		# reshape
		# self.xquad.shape = self.nquad,self.qdim
		# self.wquad.shape = self.nquad,self.qdim