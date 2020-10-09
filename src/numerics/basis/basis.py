# ------------------------------------------------------------------------ #
#
#       File : src/numerics/basis/basis.py
#
#       Contains class definitions for each shape and basis function.
#      
# ------------------------------------------------------------------------ #
from abc import ABC, abstractmethod
import numpy as np

from general import BasisType, ShapeType, ModalOrNodal, \
    QuadratureType, NodeType

import meshing.gmsh as mesh_gmsh

import numerics.basis.tools as basis_tools
import numerics.basis.basis as basis_defs

from numerics.quadrature import segment, quadrilateral, triangle


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
    DIM
        defines the dimension of the shape as an int
    PRINCIPAL_NODE_COORDS
        defines coordinates of the reference element for each shape type as
        a numpy array
    CENTROID
        defines a coordinate (as a numpy array) for the centroid of the 
        reference element

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
        get index location of local faces 
    get_elem_ref_from_face_ref
        get coordinates of element in reference space
    set_elem_quadrature_type
        sets the enum for the element's quadrature type given a str
    set_face_quadrature_type
        sets the enum for the element's face quadrature type given a str
    get_quadrature_order
        conducts logic to specify the quadrature order for an element
    force_nodes_equal_quad_pts
        if flag is True, method forces node pts equal to quadrature pts
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
    def DIM(self):
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
            xn: array of nodes equidistantly spaced
                [shape: [nb, dim]]
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
            quad_pts: quadrature point coordinates [nq, dim]
            quad_wts: quadrature weights [nq, 1]
        '''        
        pass

    def get_local_face_principal_node_nums(self, p, faceID):
        '''
        Constructs the map for face nodes

        Inputs:
        -------
            p: order of polynomial space
            faceID: reference element face value

        Outputs:
        -------- 
            fnode_nums: index of face nodes
        '''
        pass
    
    def get_elem_ref_from_face_ref(self, faceID, face_pts):
        '''
        Defines element reference nodes

        Inputs:
        -------
            faceID: face value
            face_pts: coordinates for face pts

        Outputs:
        --------
            elem_pts: coordinates in element reference space
        '''

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
        dim = self.DIM
        gorder = mesh.gorder

        if physics is not None:
            qorder = physics.get_quadrature_order(order)
        else:
            qorder = order
        if gorder > 1:
            qorder += dim * (gorder-1)
        return qorder

    def force_nodes_equal_quad_pts(self, force_flag):
        '''
        If the input flag is True, this method forces the number of nodes
        per element to be equal to the number of quadrature points. This 
        eliminates aliasing by removing any interpolation of quadrature
        points to nodes. 

        Inputs:
        -------
            force_flag: Flag used to determine if the method is activated

        Outputs:
        --------
            self.forced_pts: set to nb if force_flag is True, left as None 
                otherwise
        
        Notes:
        ------ 
            can only be used when GaussLobatto is specified for both the 
            quadrature and node type
        '''
        if force_flag:
            self.forced_pts = self.get_num_basis_coeff(self.order)
            self.skip_interp = True


class PointShape(ShapeBase):
    '''
    PointShape inherits attributes and methods from the ShapeBase class.
    See ShapeBase for detailed comments of attributes and methods.

    Additional methods and attributes are commented below.
    '''
    SHAPE_TYPE = ShapeType.Point
    FACE_SHAPE = None
    NFACES = 0
    DIM = 0
    PRINCIPAL_NODE_COORDS = np.array([0.])
    CENTROID = np.array([[0.]])
    
    def get_num_basis_coeff(self, p):
        return 1
    def equidistant_nodes(self, p):
        pass

    def get_quadrature_data(self, order):

        quad_pts = np.zeros([1, 1])
        quad_wts = np.ones([1, 1])

        return quad_pts, quad_wts # [nq, dim] and [nq, 1]


class SegShape(ShapeBase):
    '''
    SegShape inherits attributes and methods from the ShapeBase class.
    See ShapeBase for detailed comments of attributes and methods.

    Additional methods and attributes are commented below
    '''
    SHAPE_TYPE = ShapeType.Segment
    FACE_SHAPE = PointShape()
    NFACES = 2
    DIM = 1
    PRINCIPAL_NODE_COORDS = np.array([[-1.], [1.]])
    CENTROID = np.array([[0.]])

    def get_num_basis_coeff(self, p):
        return p + 1

    def equidistant_nodes(self, p):

        nb = self.get_num_basis_coeff(p)
        dim = self.DIM

        if p == 0:
            xn = np.zeros([nb, 1])
        else:
            xn = basis_tools.equidistant_nodes_1D_range(-1., 1., nb) \
                    .reshape(-1, 1)
        return xn  # [nb, dim]

    def get_local_face_principal_node_nums(self, p, faceID):

        if faceID == 0:
            fnode_nums = np.zeros(1, dtype=int)
        elif faceID == 1:
            fnode_nums = np.full(1, p)
        else:
            raise ValueError

        return fnode_nums

    def get_elem_ref_from_face_ref(self, faceID, face_pts):

        if faceID == 0: 
            elem_pts = -np.ones([1, 1])
        elif faceID == 1: 
            elem_pts = np.ones([1, 1])
        else: 
            raise ValueError

        return elem_pts # [1, 1]

    def get_quadrature_data(self, order):

        try:
            fpts = self.forced_pts
        except:
            fpts = None

        quad_pts, quad_wts = segment.get_quadrature_points_weights(order,
                self.quadrature_type, forced_pts=fpts)

        return quad_pts, quad_wts # [nq, dim], [nq, 1]


class QuadShape(ShapeBase):
    '''
    QuadShape inherits attributes and methods from the ShapeBase class.
    See ShapeBase for detailed comments of attributes and methods.

    Additional methods and attributes are commented below.
    ''' 
    SHAPE_TYPE = ShapeType.Quadrilateral
    FACE_SHAPE = SegShape()
    NFACES = 4
    DIM = 2
    PRINCIPAL_NODE_COORDS = np.array([[-1., -1.], [1., -1.], [-1., 1.], 
            [1., 1.]])
    CENTROID = np.array([[0., 0.]])

    def get_num_basis_coeff(self, p):
        return (p + 1)**2

    def equidistant_nodes(self, p):

        nb = self.get_num_basis_coeff(p)
        dim = self.DIM

        xn = np.zeros([nb, dim])
        if p > 0:

            xseg = basis_tools.equidistant_nodes_1D_range(-1., 1., p+1)

            xn[:, 0] = np.tile(xseg, (p+1, 1)).reshape(-1)
            xn[:, 1] = np.repeat(xseg, p+1, axis=0).reshape(-1)

        return xn # [nb, dim]

    def get_local_face_principal_node_nums(self, p, faceID):

        if faceID == 0:
            fnode_nums = np.array([0, p])
        elif faceID == 1:
            fnode_nums = np.array([p, (p+2)*p])
        elif faceID == 2:
            fnode_nums = np.array([(p+2)*p, (p+1)*p])
        elif faceID == 3:
            fnode_nums = np.array([(p+1)*p, 0])
        else:
             raise ValueError

        return fnode_nums

    def get_elem_ref_from_face_ref(self, faceID, face_pts):

        fnodes = self.get_local_face_principal_node_nums(1, faceID)

        xn0 = self.PRINCIPAL_NODE_COORDS[fnodes[0]]
        xn1 = self.PRINCIPAL_NODE_COORDS[fnodes[1]]

        xf1 = (face_pts + 1.)/2.
        xf0 = 1. - xf1

        elem_pts = xf0*xn0 + xf1*xn1

        return elem_pts # [face_pts.shape[0], dim]
    
    def get_quadrature_order(self, mesh, order, physics=None):

        # add two to qorder for dim = 2 with quads
        qorder = super().get_quadrature_order(mesh, order, physics)
        qorder += 2 

        return qorder

    def get_quadrature_data(self, order):

        try:
            fpts = self.forced_pts
        except:
            fpts = None

        quad_pts, quad_wts = quadrilateral.get_quadrature_points_weights(
                order, self.quadrature_type, forced_pts=fpts)

        return quad_pts, quad_wts # [nq, dim] and [nq, 1]


class TriShape(ShapeBase):
    '''
    TriShape inherits attributes and methods from the ShapeBase class.
    See ShapeBase for detailed comments of attributes and methods.

    Additional methods and attributes are commented below.
    ''' 
    SHAPE_TYPE = ShapeType.Triangle
    FACE_SHAPE = SegShape()
    NFACES = 3
    DIM = 2
    PRINCIPAL_NODE_COORDS = np.array([[0., 0.], [1., 0.], [0., 1.]])
    CENTROID = np.array([[1./3., 1./3.]])

    def get_num_basis_coeff(self, p):
        return (p + 1)*(p + 2)//2

    def equidistant_nodes(self, p):

        nb = self.get_num_basis_coeff(p)
        dim = self.DIM

        xn = np.zeros([nb, dim])
        if p > 0:
            n = 0
            xseg = basis_tools.equidistant_nodes_1D_range(0., 1., p+1)
            for j in range(p+1):
                xn[n:n+p+1-j,0] = xseg[:p+1-j]
                xn[n:n+p+1-j,1] = xseg[j]
                n += p+1-j

        return xn # [nb, dim]

    def get_local_face_principal_node_nums(self, p, faceID):
        '''
        Additional Notes:
        -----------------
        Constructs map for face nodes on triangles for q1 elements only
        '''
        if faceID == 0:
            fnode_nums = np.array([p, (p+1)*(p+2)//2 - 1])
        elif faceID == 1:
            fnode_nums = np.array([(p+1)*(p+2)//2 - 1, 0])
        elif faceID == 2:
            fnode_nums = np.array([0, p])
        else:
            raise ValueError

        return fnode_nums

    def get_elem_ref_from_face_ref(self, faceID, face_pts):

        fnodes = self.get_local_face_principal_node_nums(1, faceID)

        # coordinates of local q = 1 nodes on face
        xn0 = self.PRINCIPAL_NODE_COORDS[fnodes[0]]
        xn1 = self.PRINCIPAL_NODE_COORDS[fnodes[1]]

        xf1 = (face_pts + 1.) / 2.
        xf0 = 1. - xf1

        elem_pts = xf0*xn0 + xf1*xn1

        return elem_pts # [face_pts.shape[0], dim]

    def get_quadrature_data(self, order):
        '''
        Additional Notes:
        -----------------
        Forced points cannot be used with triangles
        '''
        quad_pts, quad_wts = triangle.get_quadrature_points_weights(order, 
            self.quadrature_type)

        return quad_pts, quad_wts # [nq, dim] and [nq, 1]


class BasisBase(ABC): 
    '''
    This is an abstract base class used for the base attributes and methods 
    of all basis functions available in the DG Python framework. Child
    classes (available basis functions) of this base class include:

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
    skip_interp: boolean
        when forcing nodes to be the same as quadrature this flag is 
        used to skip the interpolation routines as they are not needed
    nb: int
        number of polynomial coefficients
    get_1d_nodes: method
        method to obtain the 1d nodes [options in src/numerics/basis/
        tools.py]
    calculate_normals: method
        method to obtain normals for element faces [options in 
        src/numerics/basis/tools.py]
    
    Methods:
    --------
    get_physical_grad
        calculates the physical gradient of the basis function
    get_basis_val_grads
        function that gets the basis values and either the phys or ref 
        gradient for the basis depending on the optional arguments
    get_values
        calculates the basis values
    get_grads
        calculates the gradient of the basis function in reference space    
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
        self.nb = 0
        self.basis_val = np.zeros(0)
        self.basis_ref_grad = np.zeros(0)
        self.basis_phys_grad = np.zeros(0)
        self.skip_interp = False
        self.quadrature_type = -1
        self.get_1d_nodes = basis_tools.set_1D_node_calc("Equidistant")
        self.calculate_normals = None

    def __repr__(self):
        return '{self.__class__.__name__}(order={self.order})'.format(
                self=self)

    @abstractmethod
    def get_values(self, quad_pts):
        '''
        Calculates Lagrange basis

        Inputs:
        -------
            quad_pts: coordinates of quadrature points [nb, dim]

        Outputs:
        -------- 
            basis_val: evaluated basis function [nq, nb] 
        '''
        pass

    @abstractmethod
    def get_grads(self, quad_pts):
        '''
        Calculates gradient of Lagrange basis for a segment shape

        Inputs:
        -------
            quad_pts: coordinates of quadrature points [nb, dim]

        Outputs:
        -------- 
            basis_ref_grad: evaluated gradient of basis function in 
                reference space [nq, nb, dim]
        '''
        pass

    def get_physical_grad(self, ijac):
        '''
        Calculates the physical gradient of the basis function

        Inputs:
        -------
            ijac: inverse of the Jacobian [nq, dim, dim]

        Outputs:
        --------
            basis_phys_grad: evaluated gradient of the basis function in 
                physical space [nq, nb, dim]
        '''
        dim = self.DIM
        nb = self.nb

        basis_ref_grad = self.basis_ref_grad 
        nq = basis_ref_grad.shape[0]

        if nq == 0:
            raise ValueError("basis_ref_grad not evaluated")

        # check to see if ijac has been passed and has the right shape
        if ijac is None or ijac.shape != (nq, dim, dim):
            raise ValueError("basis_ref_grad and ijac shapes not compatible")

        basis_phys_grad = np.transpose(np.matmul(ijac.transpose(0,2,1), 
                basis_ref_grad.transpose(0,2,1)), (0,2,1))

        return basis_phys_grad # [nq, nb, dim]

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
                physical gradients) [nq, dim, dim]
        
        Outputs:
        --------
            Sets the following attributes of the BasisBase class:

            basis_val: evaluated basis function [nq, nb]
            basis_ref_grad: evaluated gradient of the basis function in 
                reference space [nq, nb, dim]
            basis_phys_grad: evaluated gradient of the basis function in 
                physical space [nq, nb, dim]
        '''
        if get_val:
            self.basis_val = self.get_values(quad_pts)
        if get_ref_grad:
            self.basis_ref_grad = self.get_grads(quad_pts)
        if get_phys_grad:
            if ijac is None:
                raise Exception("Need Jacobian data")
            self.basis_phys_grad = self.get_physical_grad(ijac)

    def get_basis_face_val_grads(self, mesh, faceID, face_pts, basis=None, 
            get_val=True, get_ref_grad=False, get_phys_grad=False, 
            ijac=None):
        '''
        Evaluates the basis function and if applicable evaluates the 
        gradient in reference and/or physical space on the element face 

        Inputs:
        -------
            mesh: mesh object
            faceID: index of face in reference element
            face_pts: coordinates of quadrature points on the face
            basis: basis object
            get_val: [OPTIONAL] flag to calculate basis functions
            get_ref_grad: [OPTIONAL] flag to calculate gradient of basis 
                functions in ref space
            get_phys_grad: [OPTIONAL] flag to calculate gradient of basis 
                functions in phys space
            ijac: [OPTIONAL] inverse Jacobian (needed if calculating 
                physical gradients) [nq, nq, dim]
        
        Outputs:
        --------
            elem_pts: element reference coordinates [nq, dim]

            Sets the following attributes of the BasisBase class:

            basis_val: evaluated basis function [nq, nb]
            basis_ref_grad: evaluated gradient of the basis function in 
                reference space [nq, nb, dim]
            basis_phys_grad: evaluated gradient of the basis function in 
                physical space [nq, nb, dim]            
        '''
        if basis is None:
            basis = self

        elem_pts = basis.get_elem_ref_from_face_ref(faceID, face_pts)

        self.get_basis_val_grads(elem_pts, get_val, get_ref_grad, 
                get_phys_grad, ijac)

        return elem_pts


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
        self.nb = self.get_num_basis_coeff(order)
        self.calculate_normals = basis_tools.calculate_1D_normals
 
    def get_nodes(self, p):
        '''
        Calculate the coordinates in ref space for a Lagrange segment

        Inputs:
        -------
            p: order of polynomial space
            
        Outputs:
        -------- 
            xn: coordinates of nodes in ref space [nb, dim]

        Notes:
        ------
            this function differs from get_equidistant_nodes by also allowing
            for other NodeTypes (such as GaussLobatto nodes)
        '''
        nb = self.get_num_basis_coeff(p)
        dim = self.DIM

        xn = np.zeros([nb, dim])
        if p > 0:
            xn[:, 0] = self.get_1d_nodes(-1., 1., nb)

        return xn # [nb, dim]

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

        dim = self.DIM
        p = self.order
        nb = self.nb
        nq = quad_pts.shape[0]

        basis_ref_grad = np.zeros([nq, nb, dim])

        if p > 0:
            xnodes = self.get_1d_nodes(-1., 1., p+1)

            basis_tools.get_lagrange_basis_1D(quad_pts, xnodes, 
                    gphi=basis_ref_grad)

        return basis_ref_grad # [nq, nb, dim]

    def get_local_face_node_nums(self, p, faceID):
        '''
        Returns the local face node mapping

        Inputs:
        ------- 
            p: order of polynomial space
            faceID: reference element face value

        Outputs:
        --------
            fnode_nums: index of nodes from reference element 
        '''
        fnode_nums = self.get_local_face_principal_node_nums(p, faceID)

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
        self.nb = self.get_num_basis_coeff(order)
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
            xn: coordinates of nodes in ref space [nb, dim]

        Notes:
        ------
            this function differs from get_equidistant_nodes by also allowing
            for other NodeTypes (such as GaussLobatto nodes)
        '''
        nb = self.get_num_basis_coeff(p)
        dim = self.DIM

        xn = np.zeros([nb, dim])

        if p > 0:
            xseg = self.get_1d_nodes(-1., 1., p+1)

            xn[:,0] = np.tile(xseg, (p+1,1)).reshape(-1)
            xn[:,1] = np.repeat(xseg, p+1, axis=0).reshape(-1)

        return xn # [nb, dim]

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

        dim = self.DIM
        p = self.order
        nb = self.nb
        nq = quad_pts.shape[0]

        basis_ref_grad = np.zeros([nq, nb, dim])

        if p > 0:
            xnode = self.get_1d_nodes(-1., 1., p + 1)

            basis_tools.get_lagrange_basis_2D(quad_pts, xnode, 
                    gphi=basis_ref_grad)

        return basis_ref_grad # [nq, nb, dim]

    def get_local_face_node_nums(self, p, faceID):
        '''
        Returns the local face node mapping

        Inputs:
        ------- 
            p: order of polynomial space
            faceID: reference element face value

        Outputs:
        --------
            fnode_nums: index of nodes from reference element 
        '''
        if p < 1:
            raise ValueError

        if faceID == 0:
            fnode_nums = np.arange(p+1, dtype=int)
        elif faceID == 1:
            fnode_nums = p + (p+1)*np.arange(p+1, dtype=int)
        elif faceID == 2:
            fnode_nums = p*(p+2) - np.arange(p+1, dtype=int)
        elif faceID == 3:
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
        self.nb = self.get_num_basis_coeff(order)
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
            xn = self.equidistant_nodes(p)
            basis_tools.get_lagrange_basis_tri(quad_pts, p, xn, basis_val)

        return basis_val # [nq, nb]

    def get_grads(self, quad_pts):

        dim = self.DIM
        p = self.order
        nb = self.nb
        nq = quad_pts.shape[0]

        basis_ref_grad = np.zeros([nq, nb, dim])

        if p > 0:
            xn = self.equidistant_nodes(p)
            basis_tools.get_lagrange_grad_tri(quad_pts, p, xn, 
                    basis_ref_grad)

        return basis_ref_grad # [nq, nb, dim]

    def get_local_face_node_nums(self, p, faceID):
        '''
        Returns the local face node mapping

        Inputs:
        ------- 
            p: order of polynomial space
            faceID: reference element face value

        Outputs:
        --------
            fnode_nums: index of nodes from reference element 
        '''
        if p < 1:
            raise ValueError

        nn = p + 1
        fnode_nums = np.zeros(nn, dtype=int)

        if faceID == 0:
            nstart = p
            j = p
            k = -1
        elif faceID == 1:
            nstart = (p+1)*(p+2)//2 - 1
            j = -2
            k = -1
        elif faceID == 2:
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
        self.nb = self.get_num_basis_coeff(order)

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

        dim = self.DIM
        p = self.order
        nb = self.nb
        nq = quad_pts.shape[0]

        basis_ref_grad = np.zeros([nq, nb, dim])

        if p > 0:

            basis_tools.get_legendre_basis_1D(quad_pts, p, 
                    gphi=basis_ref_grad)

        return basis_ref_grad # [nq, nb, dim]


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
        self.nb = self.get_num_basis_coeff(order)

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

        dim = self.DIM
        p = self.order
        nb = self.nb
        nq = quad_pts.shape[0]

        basis_ref_grad = np.zeros([nq, nb, dim])

        if p > 0:

            basis_tools.get_legendre_basis_2D(quad_pts, p, 
                    gphi=basis_ref_grad)

        return basis_ref_grad # [nq, nb, dim]


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
        self.nb = self.get_num_basis_coeff(order)

    def get_values(self, quad_pts):

        p = self.order
        nb = self.nb
        nq = quad_pts.shape[0]

        basis_val = np.zeros([nq, nb])

        if p == 0:
            basis_val[:] = 1.
        else:
            xn = self.equidistant_nodes(p)
            basis_tools.get_modal_basis_tri(quad_pts, p, xn, basis_val)

        return basis_val # [nq, nb]

    def get_grads(self, quad_pts):

        dim = self.DIM
        p = self.order
        nb = self.nb
        nq = quad_pts.shape[0]

        basis_ref_grad = np.zeros([nq, nb, dim])

        if p > 0:
            xn = self.equidistant_nodes(p)
            basis_tools.get_modal_grad_tri(quad_pts, p, xn, basis_ref_grad)

            basis_ref_grad = 2.*basis_ref_grad

        return basis_ref_grad # [nq, nb, dim]