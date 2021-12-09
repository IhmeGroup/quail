# ------------------------------------------------------------------------ #
#
#       quail: A lightweight discontinuous Galerkin code for
#              teaching and prototyping
#		<https://github.com/IhmeGroup/quail>
#       
#		Copyright (C) 2020-2021
#
#       This program is distributed under the terms of the GNU
#		General Public License v3.0. You should have received a copy
#       of the GNU General Public License along with this program.  
#		If not, see <https://www.gnu.org/licenses/>.
#
# ------------------------------------------------------------------------ #

# ------------------------------------------------------------------------ #
#
#		File : src/numerics/basis/basis.py
#
#		Contains class definitions for each shape and basis function.
#
# ------------------------------------------------------------------------ #
from abc import ABC, abstractmethod
import numpy as np

from general import BasisType, ShapeType, ModalOrNodal, \
	QuadratureType, NodeType

import meshing.gmsh as mesh_gmsh

import numerics.basis.tools as basis_tools
import numerics.basis.basis as basis_defs

from numerics.quadrature import segment, quadrilateral, triangle, \
		hexahedron, prism


class ShapeBase(ABC):
	'''
	This is a Mixin class used to represent a shape. Supported shapes
	include point, segment, quadrilateral, and triangle.

	Abstract Constants:
	-------------------
	SHAPE_TYPE
		defines an enum from ShapeType to identify the element's shape
	FACE_SHAPE
		defines an enum from ShapeType to identify the element's face shape
	NFACES
		defines the number of faces per element as an int
	NDIMS
		defines the dimension of the shape as an int
	PRINCIPAL_NODE_COORDS
		defines coordinates of the reference element for each shape type as
		a numpy array
	CENTROID
		defines a coordinate (as a numpy array) for the centroid of the
		reference element
	FACE_TIME_MAPPING
		maps the face ID from the spacial element to the temporal frame

	Attributes:
	-----------
	quadrature_type: enum
		specifies the type of quadrature to be used on the element

	Abstract Methods:
	-----------------
	get_num_basis_coeff
		sets the number of basis coefficients given a polynomial order
	get_equidistant_nodes
		takes nb nodes and places then in equidistant positions along the
		reference element (as a numpy array)
	get_quadrature_data
		gets arrays of quad_pts and quad_wts

	Methods:
	--------
	get_local_face_principal_node_nums
		get local IDs of principal nodes on face
	get_elem_ref_from_face_ref
		get coordinates of element in reference space
	set_elem_quadrature_type
		sets the enum for the element's quadrature type given a str
	set_face_quadrature_type
		sets the enum for the element's face quadrature type given a str
	get_quadrature_order
		conducts logic to specify the quadrature order for an element
	'''

	@property
	@abstractmethod
	def SHAPE_TYPE(self):
		'''
		Stores the location of the ShapeType enum to define the element's
		shape
		'''
		pass

	@property
	@abstractmethod
	def FACE_SHAPE(self):
		'''
		Stores the location of the ShapeType enum to define the element's
		face shape
		'''
		pass

	@property
	@abstractmethod
	def NFACES(self):
		'''
		Stores the number of faces per element as an int
		'''
		pass

	@property
	@abstractmethod
	def NDIMS(self):
		'''
		Stores the dimension of the element
		'''
		pass

	@property
	@abstractmethod
	def PRINCIPAL_NODE_COORDS(self):
		'''
		Stores the node coordinates for the reference element
		'''
		pass

	@property
	@abstractmethod
	def CENTROID(self):
		'''
		Stores the coordinate for the reference element's centroid
		'''
		pass

	@property
	@abstractmethod
	def FACE_TIME_MAPPING(self):
		'''
		Stores the mapping to reference time for ADERDG
		'''
		pass

	@abstractmethod
	def get_num_basis_coeff(self, p):
		'''
		Sets the number of basis coefficients given a polynomial order

		Inputs:
		-------
			p: order of polynomial space

		Outputs:
		--------
			nb: number of basis coefficients
		'''
		pass

	@abstractmethod
	def equidistant_nodes(self, p):
		'''
		Defines an array of equidistant points based on the number of
		basis coefficients

		Inputs:
		-------
			p: order of polynomial space

		Outputs:
		--------
			xnodes: array of nodes equidistantly spaced [nb, ndims]
		'''
		pass

	@abstractmethod
	def get_quadrature_data(self, order):
		'''
		Given the quadrature order, this method returns quadrature points
		and weights to the user. Details of quadrature calculations can be
		found in src/numerics/quadrature.

		Inputs:
		-------
			order: quadrature order (typically obtained using
				get_quadrature_order method)

		Outputs:
		--------
			quad_pts: quadrature point coordinates [nq, ndims]
			quad_wts: quadrature weights [nq, 1]
		'''
		pass

	def get_local_face_principal_node_nums(self, p, face_ID):
		'''
		Gets local IDs of principal nodes on face

		Inputs:
		-------
			p: order of polynomial space
			face_ID: reference element face value

		Outputs:
		--------
			fnode_nums: local IDs of principal nodes on face
		'''
		pass

	def get_elem_ref_from_face_ref(self, face_ID, face_pts):
		'''
		Defines element reference nodes

		Inputs:
		-------
			face_ID: face value
			face_pts: coordinates for face pts

		Outputs:
		--------
			elem_pts: coordinates in element reference space
		'''
		pass

	def set_elem_quadrature_type(self, quadrature_name):
		'''
		Sets the quadrature type based on the QuadratureType enum. Available
		quadrature types in general.py

		Inputs:
		-------
			quadrature_name: name of the quadrature type

		Outputs:
		--------
			self.quadrature_type: set based on name
		'''
		self.quadrature_type = QuadratureType[quadrature_name]

	def set_face_quadrature_type(self, quadrature_name):
		'''
		Sets the quadrature type for the element face based on the
		QuadratureType enum. Available quadrature types in general.py.

		Inputs:
		-------
			quadrature_name: name of the quadrature type

		Outputs:
		--------
			self.FACE_SHAPE.quadrature_type: set based on name
		'''
		self.FACE_SHAPE.quadrature_type = QuadratureType[quadrature_name]

	def get_quadrature_order(self, mesh, order, physics=None):
		'''
		Given the inputs, this function returns the quadrature order,
		which is used to obtain quadrature points and weights.

		Inputs:
		-------
			mesh: mesh for the solution domain
			order: solution order
			physics: [OPTIONAL] instance of physics class

		Outputs:
		--------
			qorder: quadrature order
		'''
		ndims = self.NDIMS
		gorder = mesh.gorder

		if physics is not None:
			qorder = physics.get_quadrature_order(order)
		else:
			qorder = order
		if gorder > 1:
			qorder += ndims * (gorder-1)
		return qorder


class PointShape(ShapeBase):
	'''
	PointShape inherits attributes and methods from the ShapeBase class.
	See ShapeBase for detailed comments of attributes and methods.

	Additional methods and attributes are commented below.
	'''
	SHAPE_TYPE = ShapeType.Point
	FACE_SHAPE = None
	NFACES = 0
	NDIMS = 0
	PRINCIPAL_NODE_COORDS = np.array([0.])
	CENTROID = np.array([[0.]])
	FACE_TIME_MAPPING = None

	def get_num_basis_coeff(self, p):
		return 1
	def equidistant_nodes(self, p):
		pass

	def get_quadrature_data(self, order):
		quad_pts = np.zeros([1, 1])
		quad_wts = np.ones([1, 1])

		return quad_pts, quad_wts # [nq, ndims] and [nq, 1]


class SegShape(ShapeBase):
	'''
	SegShape inherits attributes and methods from the ShapeBase class.
	See ShapeBase for detailed comments of attributes and methods.

	Additional methods and attributes are commented below
	'''
	SHAPE_TYPE = ShapeType.Segment
	FACE_SHAPE = PointShape()
	NFACES = 2
	NDIMS = 1
	PRINCIPAL_NODE_COORDS = np.array([[-1.], [1.]])
	CENTROID = np.array([[0.]])
	FACE_TIME_MAPPING = np.empty([2])

	def get_num_basis_coeff(self, p):
		return p + 1

	def equidistant_nodes(self, p):
		nb = self.get_num_basis_coeff(p)
		ndims = self.NDIMS

		if p == 0:
			xnodes = np.zeros([nb, 1])
		else:
			xnodes = basis_tools.equidistant_nodes_1D_range(-1., 1., nb) \
					.reshape(-1, 1)
		return xnodes  # [nb, ndims]

	def get_local_face_principal_node_nums(self, p, face_ID):
		if face_ID == 0:
			fnode_nums = np.zeros(1, dtype=int)
		elif face_ID == 1:
			fnode_nums = np.full(1, p)
		else:
			raise ValueError

		return fnode_nums

	def get_elem_ref_from_face_ref(self, face_ID, face_pts):
		if face_ID == 0:
			elem_pts = -np.ones([1, 1])
		elif face_ID == 1:
			elem_pts = np.ones([1, 1])
		else:
			raise ValueError

		return elem_pts # [1, 1]

	def get_quadrature_data(self, order):
		quad_pts, quad_wts = segment.get_quadrature_points_weights(order,
				self.quadrature_type, self.num_pts_colocated)

		return quad_pts, quad_wts # [nq, ndims], [nq, 1]
	
	def get_tiling_constants(self, null):
		'''
		Precomputes the tiling constants for the ADER-DG scheme. Tiling
		constants are used with the numpy 'tile' function to build arrays
		to maintain consistency for the tensor multiplications throughout
		the solver.

		Inputs:
		-------
			null: passed input not needed for SegShape class

		Outputs:
		--------
			nq_t: quadrature points tiling constant
			nb_t: basis coefficients tiling constant
			time_skip: Value to skip when building time
					array for each bface in get_boundary_face_residual
					in src/solver/ADERDG.py
			time_tile: time array tiling constant for each bface in 
					get_boundary_face_residual in src/solver/ADERDG.py
		'''
		return self.basis_val.shape[0], 1, self.basis_val.shape[1]

class QuadShape(ShapeBase):
	'''
	QuadShape inherits attributes and methods from the ShapeBase class.
	See ShapeBase for detailed comments of attributes and methods.

	Additional methods and attributes are commented below.
	'''
	SHAPE_TYPE = ShapeType.Quadrilateral
	FACE_SHAPE = SegShape()
	NFACES = 4
	NDIMS = 2
	PRINCIPAL_NODE_COORDS = np.array([[-1., -1.], [1., -1.], [-1., 1.],
			[1., 1.]])
	CENTROID = np.array([[0., 0.]])
	FACE_TIME_MAPPING = np.array([0, 2])
	
	def get_num_basis_coeff(self, p):
		return (p + 1)**2

	def equidistant_nodes(self, p):
		nb = self.get_num_basis_coeff(p)
		ndims = self.NDIMS

		xnodes = np.zeros([nb, ndims])
		if p > 0:
			xseg = basis_tools.equidistant_nodes_1D_range(-1., 1., p+1)

			xnodes[:, 0] = np.tile(xseg, (p+1, 1)).reshape(-1)
			xnodes[:, 1] = np.repeat(xseg, p+1, axis=0).reshape(-1)

		return xnodes # [nb, ndims]

	def get_local_face_principal_node_nums(self, p, face_ID):
		if face_ID == 0:
			fnode_nums = np.array([0, p])
		elif face_ID == 1:
			fnode_nums = np.array([p, (p+2)*p])
		elif face_ID == 2:
			fnode_nums = np.array([(p+2)*p, (p+1)*p])
		elif face_ID == 3:
			fnode_nums = np.array([(p+1)*p, 0])
		else:
			 raise ValueError

		return fnode_nums

	def get_elem_ref_from_face_ref(self, face_ID, face_pts):
		fnodes = self.get_local_face_principal_node_nums(1, face_ID)

		xn0 = self.PRINCIPAL_NODE_COORDS[fnodes[0]]
		xn1 = self.PRINCIPAL_NODE_COORDS[fnodes[1]]

		xf1 = (face_pts + 1.)/2.
		xf0 = 1. - xf1

		elem_pts = xf0*xn0 + xf1*xn1

		return elem_pts # [face_pts.shape[0], ndims]

	def get_quadrature_order(self, mesh, order, physics=None):
		# Add two to qorder for ndims = 2 with quads
		qorder = super().get_quadrature_order(mesh, order, physics)
		qorder += 2

		return qorder

	def get_quadrature_data(self, order):
		quad_pts, quad_wts = quadrilateral.get_quadrature_points_weights(
				order, self.quadrature_type, self.num_pts_colocated)

		return quad_pts, quad_wts # [nq, ndims] and [nq, 1]

	def get_tiling_constants(self, bface_quad_pts_st):
		'''
		Precomputes the tiling constants for the ADER-DG scheme. Tiling
		constants are used with the numpy 'tile' function to build arrays
		to maintain consistency for the tensor multiplications throughout
		the solver.

		Inputs:
		-------
			bface_quad_pts_st: boundary face quad_pts used to define
					the time_skip and time_tile value for 2D ADER 
					approaches using Quads [nq_st, ndims]

		Outputs:
		--------
			nq_t: quadrature points tiling constant
			time_skip: Value to skip when building time
					array for each bface in get_boundary_face_residual
					in src/solver/ADERDG.py. Also used for tiling
					in the interior and boundary face integral.
			time_tile: time array tiling constant for each bface in 
					get_boundary_face_residual in src/solver/ADERDG.py
		'''
		return int(np.sqrt(self.basis_val.shape[0])), \
				int(np.sqrt(bface_quad_pts_st.shape[0])), \
				int(np.sqrt(bface_quad_pts_st.shape[0]))


class TriShape(ShapeBase):
	'''
	TriShape inherits attributes and methods from the ShapeBase class.
	See ShapeBase for detailed comments of attributes and methods.

	Additional methods and attributes are commented below.
	'''
	SHAPE_TYPE = ShapeType.Triangle
	FACE_SHAPE = SegShape()
	NFACES = 3
	NDIMS = 2
	PRINCIPAL_NODE_COORDS = np.array([[0., 0.], [1., 0.], [0., 1.]])
	CENTROID = np.array([[1./3., 1./3.]])
	FACE_TIME_MAPPING = np.empty([2])

	def get_num_basis_coeff(self, p):
		return (p + 1)*(p + 2)//2

	def equidistant_nodes(self, p):
		nb = self.get_num_basis_coeff(p)
		ndims = self.NDIMS

		xnodes = np.zeros([nb, ndims])
		if p > 0:
			n = 0
			xseg = basis_tools.equidistant_nodes_1D_range(0., 1., p+1)
			for j in range(p+1):
				xnodes[n:n+p+1-j, 0] = xseg[:p+1-j]
				xnodes[n:n+p+1-j, 1] = xseg[j]
				n += p+1-j

		return xnodes # [nb, ndims]

	def get_local_face_principal_node_nums(self, p, face_ID):
		'''
		Additional Notes:
		-----------------
		Constructs map for face nodes on triangles for q1 elements only
		'''
		if face_ID == 0:
			fnode_nums = np.array([p, (p+1)*(p+2)//2 - 1])
		elif face_ID == 1:
			fnode_nums = np.array([(p+1)*(p+2)//2 - 1, 0])
		elif face_ID == 2:
			fnode_nums = np.array([0, p])
		else:
			raise ValueError

		return fnode_nums

	def get_elem_ref_from_face_ref(self, face_ID, face_pts):
		fnodes = self.get_local_face_principal_node_nums(1, face_ID)

		# coordinates of local q = 1 nodes on face
		xn0 = self.PRINCIPAL_NODE_COORDS[fnodes[0]]
		xn1 = self.PRINCIPAL_NODE_COORDS[fnodes[1]]

		xf1 = (face_pts + 1.) / 2.
		xf0 = 1. - xf1

		elem_pts = xf0*xn0 + xf1*xn1

		return elem_pts # [face_pts.shape[0], ndims]

	def get_quadrature_data(self, order):
		'''
		Additional Notes:
		-----------------
		Colocated scheme cannot be used with triangles
		'''
		quad_pts, quad_wts = triangle.get_quadrature_points_weights(order,
				self.quadrature_type)

		return quad_pts, quad_wts # [nq, ndims] and [nq, 1]

	def get_tiling_constants(self, bface_quad_pts_st):
		'''
		NEEDS TO BE FIXED


		Precomputes the tiling constants for the ADER-DG scheme. Tiling
		constants are used with the numpy 'tile' function to build arrays
		to maintain consistency for the tensor multiplications throughout
		the solver.

		Inputs:
		-------
			bface_quad_pts_st: boundary face quad_pts used to define
					the time_skip and time_tile value for 2D ADER 
					approaches using Quads [nq_st, ndims]

		Outputs:
		--------
			nq_t: quadrature points tiling constant
			time_skip: Value to skip when building time
					array for each bface in get_boundary_face_residual
					in src/solver/ADERDG.py. Also used for tiling
					in the interior and boundary face integral.
			time_tile: time array tiling constant for each bface in 
					get_boundary_face_residual in src/solver/ADERDG.py
		'''
		return int(np.sqrt(self.basis_val.shape[0])), \
				int(np.sqrt(bface_quad_pts_st.shape[0])), \
				int(np.sqrt(bface_quad_pts_st.shape[0]))


class HexShape(ShapeBase):
	'''
	HexShape inherits attributes and methods from the ShapeBase class.
	See ShapeBase for detailed comments of attributes and methods.

	Additional methods and attributes are commented below.
	'''
	SHAPE_TYPE = ShapeType.Hexahedron
	FACE_SHAPE = QuadShape()
	NFACES = 4 # Only include space flux faces (NEED EXTRA EXPLANATION FROM BRETT) 
	NDIMS = 3
	PRINCIPAL_NODE_COORDS = np.array([[-1., -1., -1.],[1., -1., -1.],
			[-1., 1., -1.],[1., 1., -1.],[-1., -1., 1.],[1., -1., 1.],
			[-1., 1., 1.],[1., 1., 1.]])
	CENTROID = np.array([[0., 0., 0.]])
	FACE_TIME_MAPPING = np.array([4, 5])

	def get_num_basis_coeff(self, p):
		return (p + 1)**3

	def equidistant_nodes(self, p):
		nb = self.get_num_basis_coeff(p)
		ndims = self.NDIMS

		xnodes = np.zeros([nb, ndims])
		if p > 0:
			xseg = basis_tools.equidistant_nodes_1D_range(-1., 1., p+1)

			xnodes[:, 0] = np.tile(xseg, (p+1, p+1)).reshape(-1)
			xnodes_hold = np.zeros([xseg.shape[0]*xseg.shape[0],1])
			xnodes_hold = np.tile(xseg, (xseg.shape[0],1)).reshape(-1)

			xnodes[:, 1] = np.repeat(xn_hold, xseg.shape[0], 
					axis=0).reshape(-1)
			xnodes[:, 2] = np.repeat(xseg, xseg.shape[0]*xseg.shape[0], 
					axis=0).reshape(-1)

		return xnodes # [nb, ndims]

	def get_elem_ref_from_face_ref(self, face_ID, face_pts):
		
		nq = face_pts.shape[0]
		ndims = self.NDIMS

		# Instantiate a lagrange quad basis for face_ID 0-3
		lagrange_eq_quad = LagrangeQuad(self.order)
		
		# Face_ID's 4-5 are prescriptive since face_ID 4 is 
		# always when tau=-1 and face_ID 5 is always when
		# tau=1 in reference time.
		if face_ID < 4:
			fnodes = lagrange_eq_quad.get_local_face_principal_node_nums(1, face_ID)

			xn0_quad = lagrange_eq_quad.PRINCIPAL_NODE_COORDS[fnodes[0]]
			xn1_quad = lagrange_eq_quad.PRINCIPAL_NODE_COORDS[fnodes[1]]

			x0 = np.append(xn0_quad, -1.)
			x1 = np.append(xn1_quad, -1.)
			x2 = np.append(xn1_quad, 1.)
			x3 = np.append(xn0_quad, 1.)

		elem_pts = np.zeros([face_pts.shape[0], ndims])
		if face_ID == 0:
			elem_pts[:, 0] = np.reshape((face_pts[:, 0] * x1[0] - \
					face_pts[:, 0] * x0[0]) / 2., nq)
			elem_pts[:, 1] = -1.
			elem_pts[:, 2] = np.reshape((face_pts[:, 1] * x3[2] - \
					face_pts[:, 1] * x0[2]) / 2., nq)
		elif face_ID == 1:
			elem_pts[:, 1] = np.reshape((face_pts[:, 0] * x1[1] - \
					face_pts[:, 0] * x0[1]) / 2., nq)
			elem_pts[:, 0] = 1.
			elem_pts[:, 2] = np.reshape((face_pts[:, 1] * x3[2] - \
					face_pts[:, 1] * x0[2]) / 2., nq)
		elif face_ID == 2:
			elem_pts[:, 0] = np.reshape((face_pts[:, 0] * x1[0] - \
					face_pts[:, 0] * x0[0]) / 2., nq)
			elem_pts[:, 1] = 1.
			elem_pts[:, 2] = np.reshape((face_pts[:, 1] * x3[2] - \
					face_pts[:, 1] * x0[2]) / 2., nq)
		elif face_ID == 3:
			elem_pts[:, 1] = np.reshape((face_pts[:, 0] * x1[1] - \
					face_pts[:, 0] * x0[1]) / 2., nq)
			elem_pts[:, 0] = -1.
			elem_pts[:, 2] = np.reshape((face_pts[:, 1] * x3[2] - \
					face_pts[:, 1] * x0[2]) / 2., nq)
		# Bottom face (tau = -1 in ref time)
		elif face_ID == 4:
			x0 = [-1., -1., -1.]
			x1 = [1., -1., -1.]
			x2 = [1., 1., -1.]
			x3 = [-1., 1., -1.]
			elem_pts[:, 2] = -1.
			elem_pts[:, 0] = np.reshape((face_pts[:, 0] * x1[0] - \
					face_pts[:, 0] * x0[0]) / 2., nq)
			elem_pts[:, 1] = np.reshape((face_pts[:, 1] * x3[1] - \
					face_pts[:, 1] * x0[1]) / 2., nq)
		# Top face (tau = 1 in ref time)
		elif face_ID == 5: 
			x0 = [-1., -1., 1.]
			x1 = [1., -1., 1.]
			x2 = [1., 1., 1.]
			x3 = [-1., 1., 1.]
			elem_pts[:, 2] = 1.
			elem_pts[:, 0] = np.reshape((face_pts[:, 0] * x1[0] - \
					face_pts[: , 0] * x0[0]) / 2., nq)
			elem_pts[:, 1] = np.reshape((face_pts[:, 1] * x3[1] - \
					face_pts[:, 1] * x0[1]) / 2., nq)
		else:
			raise NotImplementedError

		return elem_pts # [face_pts.shape[0], ndims]

	def get_quadrature_order(self, mesh, order, physics=None):
		qorder = super().get_quadrature_order(mesh, order, physics)

		return qorder

	def get_quadrature_data(self, order):
		quad_pts, quad_wts = hexahedron.get_quadrature_points_weights(
				order, self.quadrature_type, self.num_pts_colocated)

		return quad_pts, quad_wts # [nq, ndims] and [nq, 1]


class PrismShape(ShapeBase):
	'''
	PrismShape inherits attributes and methods from the ShapeBase class.
	See ShapeBase for detailed comments of attributes and methods.

	Additional methods and attributes are commented below.
	'''
	SHAPE_TYPE = ShapeType.Prism
	FACE_SHAPE = QuadShape()
	NFACES = 3 #Explin why not 5 
	NDIMS = 3
	PRINCIPAL_NODE_COORDS = np.array([[0., 0., -1], [1., 0., -1], 
			[0., 1., -1], [0., 0., 1.], [1., 0., 1.], [0., 1., 1]])
	CENTROID = np.array([[1./3., 1./3., 0.]])
	FACE_TIME_MAPPING = np.array([3, 4])

	def get_num_basis_coeff(self, p):
		return (p + 1)*(p + 1)*(p + 2)//2

	def equidistant_nodes(self, p):
		nb_tri = TriShape().get_num_basis_coeff(p)
		nb = self.get_num_basis_coeff(p)
		ndims = self.NDIMS

		xnodes = np.zeros([nb_tri, ndims])

		xnodes[:nb_tri, :ndims-1] = TriShape().equidistant_nodes(p)
		xnodes = np.tile(xnodes, [(p + 1), 1])
		xseg = basis_tools.equidistant_nodes_1D_range(-1., 1., p+1)

		# this could be vectorized (but its also only run for initialization)
		for iseg in range(xseg.shape[0]):
			for ib in range(nb_tri):
				xnodes[iseg * nb_tri + ib, -1] = xseg[iseg]

		return xnodes # [nb, ndims]

	def get_elem_ref_from_face_ref(self, face_ID, face_pts):
		# Need to update
		nq = face_pts.shape[0]
		nq_s = int(np.sqrt(nq))
		ndims = self.NDIMS

		# Instantiate a lagrange quad basis for face_ID 0-2
		lagrange_eq_quad = LagrangeQuad(self.order)
		lagrange_eq_tri = LagrangeTri(self.order)
		# Face_ID's 3-4 are prescriptive since face_ID 3 is 
		# always when tau=-1 and face_ID 4 is always when
		# tau=1 in reference time.
		if face_ID < 3:
			fnodes = lagrange_eq_tri.get_local_face_principal_node_nums(1, face_ID)

			xn0_tri = lagrange_eq_tri.PRINCIPAL_NODE_COORDS[fnodes[0]]
			xn1_tri = lagrange_eq_tri.PRINCIPAL_NODE_COORDS[fnodes[1]]

			x0 = np.append(xn0_tri, -1.)
			x1 = np.append(xn1_tri, -1.)
			x2 = np.append(xn1_tri, 1.)
			x3 = np.append(xn0_tri, 1.)

		elem_pts = np.zeros([face_pts.shape[0], ndims])
		if face_ID != 3 and face_ID !=4:
			elem_pts_tri = lagrange_eq_tri.get_elem_ref_from_face_ref(
				face_ID, face_pts[:nq_s, [0]])
			elem_pts[:, :-1] = np.tile(elem_pts_tri, [nq_s, 1])
			elem_pts[:, -1] = face_pts[:, -1]
		elif face_ID ==3:
			elem_pts[:, :-1] = face_pts
			elem_pts[:, -1] = -1.0
		elif face_ID ==4:
			elem_pts[:, :-1] = face_pts
			elem_pts[:, -1] = 1.0
		else:
			raise NotImplementedError

		return elem_pts # [face_pts.shape[0], ndims]

	def get_quadrature_order(self, mesh, order, physics=None):
		qorder = super().get_quadrature_order(mesh, order, physics)

		return qorder

	def get_quadrature_data(self, order):
		quad_pts, quad_wts = prism.get_quadrature_points_weights(
				order, self.quadrature_type)

		return quad_pts, quad_wts # [nq, ndims] and [nq, 1]


class BasisBase(ABC):
	'''
	This is an abstract base class used for the base attributes and methods
	of all available basis functions.
	Child classes (available basis functions) of this base class include:

		- Lagrange basis [support for segments, quadrilaterals, and
		  triangles]
		- Legendre basis [support for segments and quadrilaterals]
		- Hierarchical basis [support for triangles]
			Ref: Solin, P, Segeth, K. and Dolezel, I., "Higher-Order Finite
			Element Methods" (Boca Raton, FL: Chapman and Hall/CRC). 2004.
			pp. 55-60.

	Abstract Constants:
	-------------------
	BASIS_TYPE
		defines an enum from ShapeType to identify the element's shape
	MODAL_OR_NODAL
		defines whether the basis function is a modal or nodal type

	Attributes:
	-----------
	order: int
		specifies the polynomial or geometric order
	basis_val: numpy array
		evaluated basis function
	basis_ref_grad: numpy array
		evaluated gradient of the basis function in reference space
	basis_phys_grad: numpy array
		evaluated gradient of the basis function in physical space
	nb: int
		number of polynomial coefficients
	get_1d_nodes: method
		method to obtain the 1d nodes [options in src/numerics/basis/
		tools.py]
	calculate_normals: method
		method to obtain normals for element faces [options in
		src/numerics/basis/tools.py]
	skip_interp: boolean
		if True, then interpolation to the quadrature points is skipped;
		useful for a collocated scheme in which the quadrature points
		are the same as the solution nodes
	num_pts_colocated: int
		for a collocated scheme, the number of quadrature points (same
		as the number of solution nodes)

	Methods:
	--------
	get_values
		calculates the basis values
	get_grads
		calculates the gradient of the basis function in reference space
	get_physical_grads
		calculates the physical gradient of the basis function
	get_basis_val_grads
		function that gets the basis values and either the phys or ref
		gradient for the basis depending on the optional arguments
	force_colocated_nodes_quad_pts
		if flag is True, method forces node pts equal to quadrature pts
	'''
	@property
	@abstractmethod
	def BASIS_TYPE(self):
		'''
		Stores the BasisType enum to define the element's basis function.
		'''
		pass

	@property
	@abstractmethod
	def MODAL_OR_NODAL(self):
		'''
		Stores the ModalOrNodal enum to define the basis function's behavior.
		'''
		pass

	@abstractmethod
	def __init__(self, order):
		self.order = order
		self.nb = self.get_num_basis_coeff(order)
		self.basis_val = np.zeros(0)
		self.basis_ref_grad = np.zeros(0)
		self.basis_phys_grad = np.zeros(0)
		self.quadrature_type = -1
		self.get_1d_nodes = basis_tools.set_1D_node_calc("Equidistant")
		self.calculate_normals = None
		self.skip_interp = False
		self.num_pts_colocated = 0

	def __repr__(self):
		return '{self.__class__.__name__}(order={self.order})'.format(
				self=self)

	def __eq__(self, other): 
		if not isinstance(other, BasisBase):
			# don't attempt to compare against unrelated types
			return NotImplementedError
		return self.BASIS_TYPE == other.BASIS_TYPE and \
			self.order == other.order

	@abstractmethod
	def get_values(self, quad_pts):
		'''
		Calculates basis values

		Inputs:
		-------
			quad_pts: coordinates of quadrature points [nb, ndims]

		Outputs:
		--------
			basis_val: evaluated basis function [nq, nb]
		'''
		pass

	@abstractmethod
	def get_grads(self, quad_pts):
		'''
		Calculates basis gradient (in reference space)

		Inputs:
		-------
			quad_pts: coordinates of quadrature points [nb, ndims]

		Outputs:
		--------
			basis_ref_grad: evaluated gradient of basis function in
				reference space [nq, nb, ndims]
		'''
		pass

	def get_physical_grads(self, ijac):
		'''
		Calculates the physical gradient of the basis function

		Inputs:
		-------
			ijac: inverse of the Jacobian [nq, ndims, ndims]

		Outputs:
		--------
			basis_phys_grad: evaluated gradient of the basis function in
				physical space [nq, nb, ndims]
		'''
		ndims = self.NDIMS
		nb = self.nb

		basis_ref_grad = self.basis_ref_grad
		nq = basis_ref_grad.shape[0]

		if nq == 0:
			raise ValueError("basis_ref_grad not evaluated")

		# check to see if ijac has been passed and has the right shape
		if ijac is None or ijac.shape != (nq, ndims, ndims):
			raise ValueError("basis_ref_grad and ijac shapes not compatible")

		basis_phys_grad = np.transpose(np.matmul(ijac.transpose(0, 2, 1),
				basis_ref_grad.transpose(0, 2, 1)), (0, 2, 1))

		return basis_phys_grad # [nq, nb, ndims]

	def get_basis_val_grads(self, quad_pts, get_val=True, get_ref_grad=False,
			get_phys_grad=False, ijac=None):
		'''
		Evaluates the basis function and if applicable evaluates the
		gradient in reference and/or physical space

		Inputs:
		-------
			quad_pts: coordinates of quadrature points
			get_val: [OPTIONAL] flag to calculate basis functions
			get_ref_grad: [OPTIONAL] flag to calculate gradient of basis
				functions in ref space
			get_phys_grad: [OPTIONAL] flag to calculate gradient of basis
				functions in phys space
			ijac: [OPTIONAL] inverse Jacobian (needed if calculating
				physical gradients) [nq, ndims, ndims]

		Outputs:
		--------
			Sets the following attributes of the BasisBase class:

			basis_val: evaluated basis function [nq, nb]
			basis_ref_grad: evaluated gradient of the basis function in
				reference space [nq, nb, ndims]
			basis_phys_grad: evaluated gradient of the basis function in
				physical space [nq, nb, ndims]
		'''
		if get_val:
			self.basis_val = self.get_values(quad_pts)
		if get_ref_grad:
			self.basis_ref_grad = self.get_grads(quad_pts)
		if get_phys_grad:
			if ijac is None:
				raise Exception("Need Jacobian data")
			self.basis_phys_grad = self.get_physical_grads(ijac)

	def get_basis_face_val_grads(self, mesh, face_ID, face_pts, basis=None,
			get_val=True, get_ref_grad=False, get_phys_grad=False,
			ijac=None):
		'''
		Evaluates the basis function and if applicable evaluates the
		gradient in reference and/or physical space on the element face

		Inputs:
		-------
			mesh: mesh object
			face_ID: index of face in reference element
			face_pts: coordinates of quadrature points on the face
			basis: basis object
			get_val: [OPTIONAL] flag to calculate basis functions
			get_ref_grad: [OPTIONAL] flag to calculate gradient of basis
				functions in ref space
			get_phys_grad: [OPTIONAL] flag to calculate gradient of basis
				functions in phys space
			ijac: [OPTIONAL] inverse Jacobian (needed if calculating
				physical gradients) [nq, nq, ndims]

		Outputs:
		--------
			Sets the following attributes of the BasisBase class:

			basis_val: evaluated basis function [nq, nb]
			basis_ref_grad: evaluated gradient of the basis function in
				reference space [nq, nb, ndims]
			basis_phys_grad: evaluated gradient of the basis function in
				physical space [nq, nb, ndims]
		'''
		if basis is None:
			basis = self

		# Convert from face ref space to element ref space
		elem_pts = basis.get_elem_ref_from_face_ref(face_ID, face_pts)

		self.get_basis_val_grads(elem_pts, get_val, get_ref_grad,
				get_phys_grad, ijac)

	def force_colocated_nodes_quad_pts(self, use_colocated_scheme):
		'''
		This method sets appropriate attributes if using a colocated
		scheme, i.e. quadrature points and solution nodes are the same.

		Inputs:
		-------
			use_colocated_scheme: if True, then we are using a
				colocated scheme

		Outputs:
		--------
			self.num_pts_colocated: number of solution nodes (if
				use_colocated_scheme is True)
			self.skip_interp: True (use_colocated_scheme)

		Notes:
		------
			Can only be used when GaussLobatto is specified for both the
			quadrature and node type;
			FACE_SHAPE.num_pts_colocated also modified
		'''
		if use_colocated_scheme:
			self.num_pts_colocated = self.nb
			self.skip_interp = True
		try:
			if use_colocated_scheme:
				self.FACE_SHAPE.num_pts_colocated = \
						self.FACE_SHAPE.get_num_basis_coeff(self.order)
			else:
				self.FACE_SHAPE.num_pts_colocated = 0
		except AttributeError:
			pass


class LagrangeSeg(BasisBase, SegShape):
	'''
	LagrangeSeg inherits attributes and methods from the BasisBase class
	and SegShape class. See BaseShape and SegShape for detailed comments of
	attributes and methods.

	Additional methods and attributes are commented below.
	'''
	BASIS_TYPE =  BasisType.LagrangeSeg
	MODAL_OR_NODAL = ModalOrNodal.Nodal

	def __init__(self, order):
		super().__init__(order)
		self.calculate_normals = basis_tools.calculate_1D_normals
	

	def get_nodes(self, p):
		'''
		Calculate the coordinates in ref space for a Lagrange segment

		Inputs:
		-------
			p: order of polynomial space

		Outputs:
		--------
			xnodes: coordinates of nodes in ref space [nb, ndims]

		Notes:
		------
			This function differs from get_equidistant_nodes by also allowing
			for other NodeTypes (such as GaussLobatto nodes)
		'''
		nb = self.get_num_basis_coeff(p)
		ndims = self.NDIMS

		xnodes = np.zeros([nb, ndims])
		if p > 0:
			xnodes[:, 0] = self.get_1d_nodes(-1., 1., nb)

		return xnodes # [nb, ndims]

	def get_values(self, quad_pts):
		p = self.order
		nb = self.nb
		nq = quad_pts.shape[0]

		basis_val = np.zeros([nq, nb])

		if p == 0:
			basis_val[:] = 1.
		else:
			xnodes = self.get_1d_nodes(-1., 1., p+1)
			basis_tools.get_lagrange_basis_1D(quad_pts, xnodes, basis_val)

		return basis_val # [nq, nb]

	def get_grads(self, quad_pts):
		ndims = self.NDIMS
		p = self.order
		nb = self.nb
		nq = quad_pts.shape[0]

		basis_ref_grad = np.zeros([nq, nb, ndims])

		if p > 0:
			xnodes = self.get_1d_nodes(-1., 1., p+1)
			basis_tools.get_lagrange_basis_1D(quad_pts, xnodes,
					basis_ref_grad=basis_ref_grad)

		return basis_ref_grad # [nq, nb, ndims]

	def get_local_face_node_nums(self, p, face_ID):
		'''
		Returns local IDs of all nodes on face

		Inputs:
		-------
			p: order of polynomial space
			face_ID: reference element face value

		Outputs:
		--------
			fnode_nums: local IDs of all nodes on face
		'''
		fnode_nums = self.get_local_face_principal_node_nums(p, face_ID)

		return fnode_nums


class LagrangeQuad(BasisBase, QuadShape):
	'''
	LagrangeQuad inherits attributes and methods from the BasisBase class
	and QuadShape class. See BaseShape and QuadShape for detailed comments
	of attributes and methods.

	Additional methods and attributes are commented below.
	'''
	BASIS_TYPE = BasisType.LagrangeQuad
	MODAL_OR_NODAL = ModalOrNodal.Nodal

	def __init__(self, order):
		super().__init__(order)
		self.calculate_normals = basis_tools.calculate_2D_normals

	def get_nodes(self, p):
		'''
		Method: get_nodes
		--------------------
		Calculate the coordinates in ref space for a Lagrange segment

		Inputs:
		-------
			p: order of polynomial space

		Outputs:
		--------
			xnodes: coordinates of nodes in ref space [nb, ndims]

		Notes:
		------
			This function differs from get_equidistant_nodes by also allowing
			for other NodeTypes (such as GaussLobatto nodes)
		'''
		nb = self.get_num_basis_coeff(p)
		ndims = self.NDIMS

		xnodes = np.zeros([nb, ndims])

		if p > 0:
			xseg = self.get_1d_nodes(-1., 1., p+1)

			xnodes[:, 0] = np.tile(xseg, (p+1, 1)).reshape(-1)
			xnodes[:, 1] = np.repeat(xseg, p+1, axis=0).reshape(-1)

		return xnodes # [nb, ndims]

	def get_values(self, quad_pts):
		p = self.order
		nb = self.nb
		nq = quad_pts.shape[0]

		basis_val = np.zeros([nq, nb])

		if p == 0:
			basis_val[:] = 1.
		else:
			xnodes = self.get_1d_nodes(-1., 1., p+1)

			basis_tools.get_lagrange_basis_2D(quad_pts, xnodes, basis_val)

		return basis_val # [nq, nb]

	def get_grads(self, quad_pts):
		ndims = self.NDIMS
		p = self.order
		nb = self.nb
		nq = quad_pts.shape[0]

		basis_ref_grad = np.zeros([nq, nb, ndims])

		if p > 0:
			xnodes = self.get_1d_nodes(-1., 1., p + 1)
			basis_tools.get_lagrange_basis_2D(quad_pts, xnodes,
					basis_ref_grad=basis_ref_grad)

		return basis_ref_grad # [nq, nb, ndims]

	def get_local_face_node_nums(self, p, face_ID):
		'''
		Returns local IDs of all nodes on face

		Inputs:
		-------
			p: order of polynomial space
			face_ID: reference element face value

		Outputs:
		--------
			fnode_nums: local IDs of all nodes on face
		'''
		if p < 1:
			raise ValueError

		if face_ID == 0:
			fnode_nums = np.arange(p+1, dtype=int)
		elif face_ID == 1:
			fnode_nums = p + (p+1)*np.arange(p+1, dtype=int)
		elif face_ID == 2:
			fnode_nums = p*(p+2) - np.arange(p+1, dtype=int)
		elif face_ID == 3:
			fnode_nums = p*(p+1) - (p+1)*np.arange(p+1, dtype=int)
		else:
			 raise IndexError

		return fnode_nums


class LagrangeTri(BasisBase, TriShape):
	'''
	LagrangeTri inherits attributes and methods from the BasisBase class
	and TriShape class. See BaseShape and TriShape for detailed comments
	of attributes and methods.

	Additional methods and attributes are commented below.
	'''
	BASIS_TYPE = BasisType.LagrangeTri
	MODAL_OR_NODAL = ModalOrNodal.Nodal

	def __init__(self, order):
		super().__init__(order)
		self.calculate_normals = basis_tools.calculate_2D_normals

	def get_nodes(self, p):
		# get_nodes only has equidistant_nodes option for triangles
		return self.equidistant_nodes(p)

	def get_values(self, quad_pts):
		p = self.order
		nb = self.nb
		nq = quad_pts.shape[0]

		basis_val = np.zeros([nq, nb])

		if p == 0:
			basis_val[:] = 1.
		else:
			xnodes = self.equidistant_nodes(p)
			basis_tools.get_lagrange_basis_tri(quad_pts, p, xnodes,
					basis_val)

		return basis_val # [nq, nb]

	def get_grads(self, quad_pts):
		ndims = self.NDIMS
		p = self.order
		nb = self.nb
		nq = quad_pts.shape[0]

		basis_ref_grad = np.zeros([nq, nb, ndims])

		if p > 0:
			xnodes = self.equidistant_nodes(p)
			basis_tools.get_lagrange_grad_tri(quad_pts, p, xnodes,
					basis_ref_grad)

		return basis_ref_grad # [nq, nb, ndims]

	def get_local_face_node_nums(self, p, face_ID):
		'''
		Returns local IDs of all nodes on face

		Inputs:
		-------
			p: order of polynomial space
			face_ID: reference element face value

		Outputs:
		--------
			fnode_nums: local IDs of all nodes on face
		'''
		if p < 1:
			raise ValueError

		nn = p + 1
		fnode_nums = np.zeros(nn, dtype=int)

		if face_ID == 0:
			nstart = p
			j = p
			k = -1
		elif face_ID == 1:
			nstart = (p+1)*(p+2)//2 - 1
			j = -2
			k = -1
		elif face_ID == 2:
			nstart = 0
			j = 1
			k = 0
		else:
			raise ValueError

		fnode_nums[0] = nstart
		for i in range(1, p+1):
			fnode_nums[i] = fnode_nums[i-1] + j
			j += k

		return fnode_nums

class LagrangeHex(BasisBase, HexShape):
	'''
	LagrangeHex inherits attributes and methods from the BasisBase class
	and HexShape class. See BaseShape and HexShape for detailed comments
	of attributes and methods.

	Additional methods and attributes are commented below.

	Note: This basis is only appropriate for use with 2D ADERDG. It 
		  is not general for a 3D implentation of the DG solver.
	'''
	BASIS_TYPE = BasisType.LagrangeHex
	MODAL_OR_NODAL = ModalOrNodal.Nodal

	def __init__(self, order):
		super().__init__(order)
		self.calculate_normals = basis_tools.calculate_2D_normals

	def get_nodes(self, p):
		'''
		Method: get_nodes
		--------------------
		Calculate the coordinates in ref space for a Lagrange hex

		Inputs:
		-------
			p: order of polynomial space

		Outputs:
		--------
			xnodes: coordinates of nodes in ref space [nb, ndims]

		Notes:
		------
			This function differs from get_equidistant_nodes by also allowing
			for other NodeTypes (such as GaussLobatto nodes)
		'''
		nb = self.get_num_basis_coeff(p)
		ndims = self.NDIMS

		xnodes = np.zeros([nb, ndims])

		if p > 0:
			xseg = self.get_1d_nodes(-1., 1., p+1)

			xnodes[:, 0] = np.tile(xseg, (p+1, p+1)).reshape(-1)
			xnodes_hold = np.zeros([xseg.shape[0]*xseg.shape[0],1])
			xnodes_hold = np.tile(xseg, (xseg.shape[0],1)).reshape(-1)

			xnodes[:, 1] = np.repeat(xnodes_hold, xseg.shape[0], 
					axis=0).reshape(-1)
			xnodes[:, 2] = np.repeat(xseg, xseg.shape[0]*xseg.shape[0], 
					axis=0).reshape(-1)

		return xnodes # [nb, ndims]


	def get_values(self, quad_pts):
		p = self.order
		nb = self.nb
		nq = quad_pts.shape[0]

		basis_val = np.zeros([nq, nb])

		if p == 0:
			basis_val[:] = 1.
		else:
			xnodes = self.get_1d_nodes(-1., 1., p+1)
			
			basis_tools.get_lagrange_basis_3D(quad_pts, xnodes, basis_val)

		return basis_val # [nq, nb]

	def get_grads(self, quad_pts):
		ndims = self.NDIMS
		p = self.order
		nb = self.nb
		nq = quad_pts.shape[0]

		basis_ref_grad = np.zeros([nq, nb, ndims])

		if p > 0:
			xnodes = self.get_1d_nodes(-1., 1., p + 1)
			basis_tools.get_lagrange_basis_3D(quad_pts, xnodes,
					basis_ref_grad=basis_ref_grad)

		return basis_ref_grad # [nq, nb, ndims]


class LagrangePrism(BasisBase, PrismShape):
	'''
	LagrangePrism inherits attributes and methods from the BasisBase class
	and PrismShape class. See BaseShape and PrismShape for detailed comments
	of attributes and methods.

	Additional methods and attributes are commented below.
	'''
	BASIS_TYPE = BasisType.LagrangePrism
	MODAL_OR_NODAL = ModalOrNodal.Nodal

	def __init__(self, order):
		super().__init__(order)
		self.calculate_normals = basis_tools.calculate_2D_normals

	def get_nodes(self, p):
		# get_nodes only has equidistant_nodes option for triangles
		return self.equidistant_nodes(p)

	def get_values(self, quad_pts):
		p = self.order
		nb = self.nb
		nq = quad_pts.shape[0]

		basis_val = np.zeros([nq, nb])

		# Get the triangle basis
		basis_tri = LagrangeTri(p)
		nb_tri = basis_tri.get_num_basis_coeff(p)
		
		unique = np.unique(quad_pts[:, -1], return_counts=True)
		nq_seg = len(unique)

		nq_tri = int(nq / nq_seg)

		basis_val_tri = np.zeros([int(nq / nq_seg), nb_tri])

		if p == 0:
			basis_val[:] = 1.
		else:
			xnodes = self.equidistant_nodes(p)
			xnodes_seg = self.get_1d_nodes(-1., 1., p + 1)
			xnodes_tri = basis_tri.equidistant_nodes(p)

			basis_tools.get_lagrange_basis_tri(quad_pts[:nq_tri, :-1], 
					p, xnodes_tri, basis_val_tri)

			basis_tools.get_lagrange_basis_prism(quad_pts, p, nq_seg, xnodes,
					xnodes_seg, np.tile(basis_val_tri, 
					[nq_seg, 1]), basis_val)

		return basis_val # [nq, nb]

	def get_grads(self, quad_pts):
		ndims = self.NDIMS
		p = self.order
		nb = self.nb
		nq = quad_pts.shape[0]

		basis_ref_grad = np.zeros([nq, nb, ndims])

		if p > 0:
			xnodes = self.equidistant_nodes(p)
			basis_tools.get_lagrange_grad_tri(quad_pts, p, xnodes,
					basis_ref_grad)

		return basis_ref_grad # [nq, nb, ndims]

	def get_local_face_node_nums(self, p, face_ID):
		'''
		Returns local IDs of all nodes on face

		Inputs:
		-------
			p: order of polynomial space
			face_ID: reference element face value

		Outputs:
		--------
			fnode_nums: local IDs of all nodes on face
		'''
		if p < 1:
			raise ValueError

		nn = p + 1
		fnode_nums = np.zeros(nn, dtype=int)

		if face_ID == 0:
			nstart = p
			j = p
			k = -1
		elif face_ID == 1:
			nstart = (p+1)*(p+2)//2 - 1
			j = -2
			k = -1
		elif face_ID == 2:
			nstart = 0
			j = 1
			k = 0
		else:
			raise ValueError

		fnode_nums[0] = nstart
		for i in range(1, p+1):
			fnode_nums[i] = fnode_nums[i-1] + j
			j += k

		return fnode_nums


class LegendreSeg(BasisBase, SegShape):
	'''
	LegendreSeg inherits attributes and methods from the BasisBase class
	and SegShape class. See BaseShape and SegShape for detailed comments
	of attributes and methods.

	Additional methods and attributes are commented below.
	'''
	BASIS_TYPE = BasisType.LegendreSeg
	MODAL_OR_NODAL = ModalOrNodal.Modal

	def __init__(self, order):
		super().__init__(order)

	def get_values(self, quad_pts):
		p = self.order
		nb = self.nb
		nq = quad_pts.shape[0]

		basis_val = np.zeros([nq, nb])

		if p == 0:
			basis_val[:] = 1.
		else:
			basis_tools.get_legendre_basis_1D(quad_pts, p, basis_val)

		return basis_val # [nq, nb]

	def get_grads(self, quad_pts):
		ndims = self.NDIMS
		p = self.order
		nb = self.nb
		nq = quad_pts.shape[0]

		basis_ref_grad = np.zeros([nq, nb, ndims])

		if p > 0:
			basis_tools.get_legendre_basis_1D(quad_pts, p,
					basis_ref_grad=basis_ref_grad)

		return basis_ref_grad # [nq, nb, ndims]


class LegendreQuad(BasisBase, QuadShape):
	'''
	LegendreQuad inherits attributes and methods from the BasisBase class
	and QuadShape class. See BaseShape and QuadShape for detailed comments
	of attributes and methods.

	Additional methods and attributes are commented below.
	'''
	BASIS_TYPE = BasisType.LegendreQuad
	MODAL_OR_NODAL = ModalOrNodal.Modal

	def __init__(self, order):
		super().__init__(order)

	def get_values(self, quad_pts):
		p = self.order
		nb = self.nb
		nq = quad_pts.shape[0]

		basis_val = np.zeros([nq, nb])

		if p == 0:
			basis_val[:] = 1.
		else:
			basis_tools.get_legendre_basis_2D(quad_pts, p, basis_val)

		return basis_val # [nq, nb]

	def get_grads(self, quad_pts):
		ndims = self.NDIMS
		p = self.order
		nb = self.nb
		nq = quad_pts.shape[0]

		basis_ref_grad = np.zeros([nq, nb, ndims])

		if p > 0:
			basis_tools.get_legendre_basis_2D(quad_pts, p,
					basis_ref_grad=basis_ref_grad)

		return basis_ref_grad # [nq, nb, ndims]


class HierarchicH1Tri(BasisBase, TriShape):
	'''
	HierarchicH1Tri inherits attributes and methods from the BasisBase class
	and TriShape class. See BaseShape and TriShape for detailed comments
	of attributes and methods.

	Details of this basis function can be found in the following reference:
		Ref: Solin, P, Segeth, K. and Dolezel, I., "Higher-Order Finite
			Element Methods" (Boca Raton, FL: Chapman and Hall/CRC). 2004.
			pp. 55-60.

	Additional methods and attributes are commented below.
	'''
	BASIS_TYPE = BasisType.HierarchicH1Tri
	MODAL_OR_NODAL = ModalOrNodal.Modal

	def __init__(self, order):
		super().__init__(order)

	def get_values(self, quad_pts):
		p = self.order
		nb = self.nb
		nq = quad_pts.shape[0]

		basis_val = np.zeros([nq, nb])

		if p == 0:
			basis_val[:] = 1.
		else:
			xnodes = self.equidistant_nodes(p)
			basis_tools.get_modal_basis_tri(quad_pts, p, xnodes, basis_val)

		return basis_val # [nq, nb]

	def get_grads(self, quad_pts):
		ndims = self.NDIMS
		p = self.order
		nb = self.nb
		nq = quad_pts.shape[0]

		basis_ref_grad = np.zeros([nq, nb, ndims])

		if p > 0:
			xnodes = self.equidistant_nodes(p)
			basis_tools.get_modal_grad_tri(quad_pts, p, xnodes,
					basis_ref_grad)

			basis_ref_grad = 2.*basis_ref_grad

		return basis_ref_grad # [nq, nb, ndims]
