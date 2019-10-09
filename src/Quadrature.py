import numpy as np
from General import *
import Mesh
import code
import Basis as Bs


### QuadLine ###
QuadLinePoints = {}
QuadLineWeights = {}
for key in [0,1]:
	# Order 1
	QuadLinePoints[key] = np.array([0.5])
	QuadLineWeights[key] = np.array([1.0])
for key in [2,3]:
	# Order 3
	QuadLinePoints[key] = np.array([0.211324865405187, 0.788675134594813])
	QuadLineWeights[key] = np.array([0.500000000000000, 0.500000000000000])
for key in [4,5]:
	# Order 5
	QuadLinePoints[key] = np.array([0.112701665379258, 0.500000000000000, 0.887298334620742])
	QuadLineWeights[key] = np.array([0.277777777777778, 0.444444444444444, 0.277777777777778])
for key in [6,7]:
	# Order 7
	QuadLinePoints[key] = np.array([0.069431844202974, 0.330009478207572, 0.669990521792428, 0.930568155797026])
	QuadLineWeights[key] = np.array([0.173927422568727, 0.326072577431273, 0.326072577431273, 0.173927422568727])
for key in [8,9]:
	# Order 9
	QuadLinePoints[key] = np.array([0.046910077030668, 0.230765344947158, 0.500000000000000, 0.769234655052841,\
    0.953089922969332])
	QuadLineWeights[key] = np.array([0.118463442528095, 0.239314335249683, 0.284444444444444, 0.239314335249683,\
    0.118463442528095])
for key in [10,11]:
	# Order 11
	QuadLinePoints[key] = np.array([0.033765242898424, 0.169395306766868, 0.380690406958402, 0.619309593041598,\
    0.830604693233132, 0.966234757101576])
	QuadLineWeights[key] = np.array([0.085662246189585, 0.180380786524069, 0.233956967286345, 0.233956967286345,\
    0.180380786524069, 0.085662246189585])
for key in [12,13]:
	# Order 13
	QuadLinePoints[key] = np.array([0.025446043828621, 0.129234407200303, 0.297077424311301, 0.500000000000000,\
    0.702922575688699, 0.870765592799697, 0.974553956171379])
	QuadLineWeights[key] = np.array([0.064742483084435, 0.139852695744638, 0.190915025252560, 0.208979591836735,\
    0.190915025252560, 0.139852695744638, 0.064742483084435])
for key in [14,15]:
	# Order 15
	QuadLinePoints[key] = np.array([0.019855071751232, 0.101666761293187, 0.237233795041836, 0.408282678752175,\
    0.591717321247825, 0.762766204958164, 0.898333238706813, 0.980144928248768])
	QuadLineWeights[key] = np.array([0.050614268145188, 0.111190517226687, 0.156853322938944, 0.181341891689181,\
    0.181341891689181, 0.156853322938944, 0.111190517226687, 0.050614268145188])
for key in [16,17]:
	# Order 17
	QuadLinePoints[key] = np.array([0.015919880246187, 0.081984446336682, 0.193314283649705, 0.337873288298096,\
    0.500000000000000, 0.662126711701905, 0.806685716350295, 0.918015553663318,
    0.984080119753813])
	QuadLineWeights[key] = np.array([0.040637194180787, 0.090324080347429, 0.130305348201468, 0.156173538520001,\
    0.165119677500630, 0.156173538520001, 0.130305348201468, 0.090324080347429,
    0.040637194180787])
# reshape
for key in range(len(QuadLinePoints)):
	QuadLinePoints[key].shape = -1,1
	QuadLineWeights[key].shape = -1,1


def GetQuadOrderElem(egrp, Order, Basis, mesh, EqnSet=None, quadData=None):
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

	Shape = Bs.Basis2Shape[Basis]
	QuadChanged = True
	if quadData is not None:
		if QuadOrder == quadData.Order and Shape == quadData.Shape:
			QuadChanged = False

	return QuadOrder, QuadChanged


def GetQuadOrderIFace(IFace, Order, Basis, mesh, EqnSet=None, quadData=None):
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

	Shape = Bs.Basis2Shape[Basis]
	FShape = Bs.FaceShape[Shape]
	QuadChanged = True
	if quadData is not None:
		if QuadOrder == quadData.Order and FShape == quadData.Shape:
			QuadChanged = False

	return QuadOrder, QuadChanged


def GetQuadOrderBFace(BFace, Order, Basis, mesh, EqnSet=None, quadData=None):
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

	Shape = Bs.Basis2Shape[Basis]
	FShape = Bs.FaceShape[Shape]
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
	def __init__(self,Order,entity,egrp,mesh):
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
		dim = Mesh.GetEntityDim(entity, mesh)
		self.Shape = Bs.Basis2Shape[mesh.ElemGroups[egrp].QBasis]
		self.Order = Order
		self.xquad = QuadLinePoints[Order]
		self.wquad = QuadLineWeights[Order]
		self.qdim = mesh.Dim
		self.nvec = None

		if entity > EntityType.Element: 
			self.Shape = Bs.FaceShape[self.Shape]
			self.xquad = np.zeros([1,1])
			self.wquad = np.ones([1,1])

		self.nquad = len(self.xquad)

		# self.xquad = np.copy(QuadLinePoints[Order])
		# self.wquad = np.copy(QuadLineWeights[Order])
		# reshape
		# self.xquad.shape = self.nquad,self.qdim
		# self.wquad.shape = self.nquad,self.qdim
