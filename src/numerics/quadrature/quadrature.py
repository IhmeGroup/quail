import code
import numpy as np

from general import ShapeType, EntityType
import meshing.meshbase as Mesh
from numerics.quadrature import segment, quadrilateral, triangle

def get_gaussian_quadrature_elem(mesh, basis, order, EqnSet=None, quadData=None):
	'''
	Method: get_gaussian_quadrature_elem
	----------------------------------------
	Determines the quadrature order for the given element based on
	the basis functions, order, and equations.

	INPUTS:
		mesh: mesh object
		basis: polynomial basis function
		order: solution order 
		EqnSet: set of equations to be solved

	OUTPUTS:
		QuadOrder: quadrature order
		QuadChanged: boolean flag for a changed quadrature
	'''
	gorder = mesh.gorder
	shape = basis.shape_type

	# QuadOrder = 2*Order + 1
	if EqnSet is not None:
		QuadOrder = EqnSet.QuadOrder(order)
	else:
		QuadOrder = order
	if gorder > 1:
		dim = mesh.Dim
		QuadOrder += dim*(gorder-1)

	#is instance instead?
	if shape is ShapeType.Quadrilateral:
		QuadOrder += mesh.Dim
	QuadChanged = True

	if quadData is not None:
		if QuadOrder == quadData.order and shape == quadData.Shape:
			QuadChanged = False

	return QuadOrder, QuadChanged


def get_gaussian_quadrature_face(mesh, IFace, basis, order, EqnSet=None, quadData=None):
	'''
	Method: get_gaussian_quadrature_face
	----------------------------------------
	Determines the quadrature order for the given face based on
	the basis functions, order, and equations.

	INPUTS:
		mesh: mesh object
		IFace: face object
		basis: polynomial basis function
		order: solution order 
		EqnSet: set of equations to be solved
		
	OUTPUTS:
		QuadOrder: quadrature order
		QuadChanged: boolean flag for a changed quadrature
	'''

	gorder = mesh.gorder
	shape = basis.shape_type
	face_shape_type = basis.face_shape_type
	# QuadOrder = 2*Order + 1
	if EqnSet is not None:
		QuadOrder = EqnSet.QuadOrder(order)
	else:
		QuadOrder = order
	if gorder > 1:
		dim = mesh.Dim - 1
		QuadOrder += dim*(gorder-1)

	if shape is ShapeType.Quadrilateral:
		QuadOrder += 1

	QuadChanged = True
	if quadData is not None:
		if QuadOrder == quadData.order and face_shape_type == quadData.Shape:
			QuadChanged = False

	return QuadOrder, QuadChanged

class QuadData(object):
	'''
	Class: QuadData
	--------------------------------------------------------------------------
	This is a class defined to define the quadrature data for a given entity 
	(i.e. element, iface, or bface)
	'''
	def __init__(self,mesh,basis,entity,order):
		'''
		Method: __init__
		--------------------------------------------------------------------------
		This method initializes the quadrature data for a given entity. It uses the 
		mesh and basis to determine the shape of the entity as well as its dimension.
		It also uses the Order of the method to look up the following.

		ATTRIBUTES:
			quad_pts: Locations of points in reference space.
			quad_wts: Weight of the points in reference space.
			Shape: shape of the basis function
		'''
		dim = Mesh.get_entity_dim(mesh, entity)
		self.order = order
		
		self.qdim = mesh.Dim
		self.nvec = None

		# shape = Basis.Basis2Shape[basis]
		shape = basis.shape_type
		face_shape_type = basis.face_shape_type

		if entity == EntityType.Element:
			self.Shape = shape
		else:
			self.Shape = face_shape_type

		if self.Shape == ShapeType.Point:
			self.quad_pts = np.zeros([1,1])
			self.quad_wts = np.ones([1,1])
		elif self.Shape == ShapeType.Segment:
			self.quad_pts, self.quad_wts = segment.get_quadrature_points_weights(order, 0)
			# self.quad_pts = QuadLinePoints[order]
			# self.quad_wts = QuadLineWeights[order]
		elif self.Shape == ShapeType.Quadrilateral:
			self.quad_pts, self.quad_wts = quadrilateral.get_quadrature_points_weights(order, 0)
			# self.quad_pts = QuadQuadrilateralPoints[order]
			# self.quad_wts = QuadQuadrilateralWeights[order]
		elif self.Shape == ShapeType.Triangle:
			self.quad_pts, self.quad_wts = triangle.get_quadrature_points_weights(order, 0)
			# self.quad_pts = QuadTrianglePoints[order]
			# self.quad_wts = QuadTriangleWeights[order]
		else:
			raise NotImplementedError