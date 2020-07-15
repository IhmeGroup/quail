from abc import ABC, abstractmethod
import code
import numpy as np

from data import ArrayList, GenericData
from general import BasisType, ShapeType, ModalOrNodal, \
    QuadratureType, NodeType

import meshing.gmsh as mesh_gmsh

import numerics.basis.tools as basis_tools
import numerics.basis.basis as basis_defs

from numerics.quadrature import segment, quadrilateral, triangle

# RefQ1Coords = {
#     BasisType.LagrangeSeg : np.array([[-1.],[1.]]),
#     BasisType.LagrangeQuad : np.array([[-1.,-1.],[1.,-1.],
#                                 [-1.,1.],[1.,1.]]),
#     BasisType.LagrangeTri : np.array([[0.,0.],[1.,0.],
#                                 [0.,1.]]),
#     BasisType.LegendreSeg : np.array([[-1.],[1.]]),
#     BasisType.LegendreQuad : np.array([[-1.,-1.],[1.,-1.],
#                                 [-1.,1.],[1.,1.]]),
#     BasisType.HierarchicH1Tri : np.array([[0.,0.],[1.,0.],
#                                 [0.,1.]])
# }

class ShapeBase(ABC):
    @property
    @abstractmethod
    def SHAPE_TYPE(self):
        pass

    @property
    @abstractmethod
    def FACE_SHAPE(self):
        pass

    @property
    @abstractmethod
    def NFACES(self):
        pass

    @property
    @abstractmethod
    def DIM(self):
        pass

    @property
    @abstractmethod
    def PRINCIPAL_NODE_COORDS(self):
        pass

    @property
    @abstractmethod
    def CENTROID(self):
        pass
    
    @abstractmethod
    def get_num_basis_coeff(self, p):
        pass

    @abstractmethod
    def equidistant_nodes(self, p):
        pass

    def set_elem_quadrature_type(self, quadrature_name):
        self.quadrature_type = QuadratureType[quadrature_name]

    def set_face_quadrature_type(self, quadrature_name):
        self.FACE_SHAPE.quadrature_type = QuadratureType[quadrature_name]

    def get_quadrature_order(self, mesh, order, physics=None):
        dim = self.DIM
        gorder = mesh.gorder
        if physics is not None:
            qorder = physics.QuadOrder(order)
        else:
            qorder = order
        if gorder > 1:
            qorder += dim*(gorder-1)

        return qorder

    def force_nodes_equal_quadpts(self, force_flag):
        if force_flag == True:
            self.forced_pts = self.get_num_basis_coeff(self.order)


class PointShape(ShapeBase):

    SHAPE_TYPE = ShapeType.Point
    FACE_SHAPE = None
    NFACES = 0
    DIM = 0
    PRINCIPAL_NODE_COORDS = np.array([0.])
    CENTROID = np.array([[0.]])
    
    def get_num_basis_coeff(self,p):
        return 1
    def equidistant_nodes(self, p):
        pass

    def get_quadrature_data(self, order):
        quad_pts = np.zeros([1, 1])
        quad_wts = np.ones([1, 1])

        return quad_pts, quad_wts


class SegShape(ShapeBase):
    SHAPE_TYPE = ShapeType.Segment
    FACE_SHAPE = PointShape()
    NFACES = 2
    DIM = 1
    PRINCIPAL_NODE_COORDS = np.array([[-1.], [1.]])
    CENTROID = np.array([[0.]])

    # get_face_quadrature = FACE_SHAPE.get_quadrature
    # get_face_quad_data = FACE_SHAPE.get_quad_data

    def get_num_basis_coeff(self,p):
        return p + 1

    def equidistant_nodes(self, p):
        '''
        Method: equidistant_nodes
        --------------------------
        Calculate the coordinates in ref space

        INPUTS:
            p: order of polynomial space
            
        OUTPUTS: 
            xn: coordinates of nodes in ref space
        '''
        nb = self.get_num_basis_coeff(p)
        dim = self.DIM

        # adim = nb, dim
        # if xn is None or xn.shape != adim:
        #     xn = np.zeros(adim)

        if p == 0:
            # xn[:] = 0.0 # 0.5
            xn = np.zeros([nb, 1])
            # return xn
        else:
            # xn[:, 0] = basis_tools.equidistant_nodes_1D_range(-1., 1., nb)
            xn = basis_tools.equidistant_nodes_1D_range(-1., 1., nb).reshape(-1, 1)

        return xn

    def get_elem_ref_from_face_ref(self, faceID, face_pts):
        '''
        Function: get_elem_ref_from_face_ref
        ----------------------------
        This function converts coordinates in face reference space to
        element reference space

        INPUTS:
            Shape: element shape
            face: local face number
            nq: number of points to convert 
            xface: coordinates in face reference space
            xelem: pre-allocated storage for output coordinates (optional)

        OUTPUTS:
            xelem: coordinates in element reference space
        '''
        # if xelem is None: xelem = np.zeros([1,1])
        if faceID == 0: 
            elem_pts = -np.ones([1, 1])
        elif faceID == 1: 
            elem_pts = np.ones([1, 1])
        else: 
            raise ValueError

        return elem_pts

    # def get_quadrature(self, mesh, order, physics=None):
        
    #     dim = self.DIM
    #     gorder = mesh.gorder
    #     if physics is not None:
    #         qorder = physics.QuadOrder(order)
    #     else:
    #         qorder = order
    #     if gorder > 1:
    #         qorder += dim*(gorder-1)

    #     return qorder

    def get_quadrature_data(self, order):

        try:
            fpts = self.forced_pts
        except:
            fpts = None

        quad_pts, quad_wts = segment.get_quadrature_points_weights(order,
            self.quadrature_type, forced_pts=fpts)

        return quad_pts, quad_wts


class QuadShape(ShapeBase):
    SHAPE_TYPE = ShapeType.Quadrilateral
    FACE_SHAPE = SegShape()
    NFACES = 4
    DIM = 2
    PRINCIPAL_NODE_COORDS = np.array([[-1., -1.], [1., -1.], [-1., 1.], 
            [1., 1.]])
    CENTROID = np.array([[0., 0.]])

    # get_face_quadrature = FACE_SHAPE.get_quadrature
    # get_face_quad_data = FACE_SHAPE.get_quad_data

    def get_num_basis_coeff(self,p):
        return (p + 1)**2

    def equidistant_nodes(self, p):
        '''
        Method: equidistant_nodes
        --------------------------
        Calculate the coordinates in ref space

        INPUTS:
            basis: type of basis function
            p: order of polynomial space
            
        OUTPUTS: 
            xn: coordinates of nodes in ref space
        '''
        nb = self.get_num_basis_coeff(p)
        dim = self.DIM

        # adim = nb, dim
        # if xn is None or xn.shape != adim:
        #     xn = np.zeros(adim)

        xn = np.zeros([nb, dim])
        if p > 0:
            # xn[:] = 0.0 # 0.5
            # return xn, nb

            xseg = basis_tools.equidistant_nodes_1D_range(-1., 1., p+1)

            xn[:, 0] = np.tile(xseg, (p+1, 1)).reshape(-1)
            xn[:, 1] = np.repeat(xseg, p+1, axis=0).reshape(-1)

        return xn

    def get_elem_ref_from_face_ref(self, faceID, face_pts):
        '''
        Function: get_elem_ref_from_face_ref
        ----------------------------
        This function converts coordinates in face reference space to
        element reference space

        INPUTS:
            Shape: element shape
            face: local face number
            nq: number of points to convert 
            xface: coordinates in face reference space
            xelem: pre-allocated storage for output coordinates (optional)

        OUTPUTS:
            xelem: coordinates in element reference space
        '''
        # if xelem is None: xelem = np.zeros([nq,2])

        # xelem = np.zeros([xface.shape[0], 2])

        fnodes, nfnode = self.local_q1_face_nodes(1, faceID)

        xn0 = self.PRINCIPAL_NODE_COORDS[fnodes[0]]
        xn1 = self.PRINCIPAL_NODE_COORDS[fnodes[1]]

        xf1 = (face_pts + 1.)/2.
        xf0 = 1. - xf1

        elem_pts = xf0*xn0 + xf1*xn1

        # elem_pts[2*i] = (b0*x0 + b1*x1);
        # elem_pts[2*i+1] = (b0*y0 + b1*y1);

        # xelem = (xface*x1 - xface*x0)/2.
        # code.interact(local=locals())

        # if face == 0:
        #     # xelem[:,0] = np.reshape((xface*x1[0] - xface*x0[0])/2., nq)
        #     xelem[:,0:1] = (xface*x1[0] - xface*x0[0])/2.
        #     xelem[:,1] = -1.
        # elif face == 1:
        #     # xelem[:,1] = np.reshape((xface*x1[1] - xface*x0[1])/2., nq)
        #     xelem[:,1:2] = (xface*x1[1] - xface*x0[1])/2.
        #     xelem[:,0] = 1.
        # elif face == 2:
        #     # xelem[:,0] = np.reshape((xface*x1[0] - xface*x0[0])/2., nq)
        #     xelem[:,0:1] = (xface*x1[0] - xface*x0[0])/2.
        #     xelem[:,1] = 1.
        # else:
        #     # xelem[:,1] = np.reshape((xface*x1[1] - xface*x0[1])/2., nq)
        #     xelem[:,1:2] = (xface*x1[1] - xface*x0[1])/2.
        #     xelem[:,0] = -1.

        return elem_pts
    
    def get_quadrature_order(self, mesh, order, physics=None):
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

        return quad_pts, quad_wts

class TriShape(ShapeBase):

    SHAPE_TYPE = ShapeType.Triangle
    FACE_SHAPE = SegShape()
    NFACES = 3
    DIM = 2
    PRINCIPAL_NODE_COORDS = np.array([[0., 0.], [1., 0.], [0., 1.]])
    CENTROID = np.array([[1./3., 1./3.]])

    # get_face_quadrature = FACE_SHAPE.get_quadrature
    # get_face_quad_data = FACE_SHAPE.get_quad_data

    def get_num_basis_coeff(self,p):
        return (p + 1)*(p + 2)//2

    def equidistant_nodes(self, p):
        '''
        Method: equidistant_nodes
        --------------------------
        Calculate the coordinates in ref space

        INPUTS:
            basis: type of basis function
            p: order of polynomial space
            
        OUTPUTS: 
            xn: coordinates of nodes in ref space
        '''

        nb = self.get_num_basis_coeff(p)
        dim = self.DIM
        

        # adim = nb,dim
        # if xn is None or xn.shape != adim:
        #     xn = np.zeros(adim)

        # if p == 0:
        #     xn[:] = 0.0 # 0.5
        #     return xn, nb


        xn = np.zeros([nb, dim])
        if p > 0:
            n = 0
            xseg = basis_tools.equidistant_nodes_1D_range(0., 1., p+1)
            for j in range(p+1):
                xn[n:n+p+1-j,0] = xseg[:p+1-j]
                xn[n:n+p+1-j,1] = xseg[j]
                n += p+1-j

        return xn

    def get_elem_ref_from_face_ref(self, faceID, face_pts):
        '''
        Function: get_elem_ref_from_face_ref
        ----------------------------
        This function converts coordinates in face reference space to
        element reference space

        INPUTS:
            Shape: element shape
            face: local face number
            nq: number of points to convert 
            xface: coordinates in face reference space
            xelem: pre-allocated storage for output coordinates (optional)

        OUTPUTS:
            xelem: coordinates in element reference space
        '''
        # if xelem is None: xelem = np.zeros([nq,2])
        # xf = np.zeros(nq)
        # xf = xf.reshape((nq,1))

        # xelem = np.zeros([xface.shape[0], 2])
        # local q = 1 nodes on face
        fnodes, nfnode = self.local_q1_face_nodes(1, faceID)
        # coordinates of local q = 1 nodes on face
        xn0 = self.PRINCIPAL_NODE_COORDS[fnodes[0]]
        xn1 = self.PRINCIPAL_NODE_COORDS[fnodes[1]]
        # for i in range(nq):
        #     xf[i] = (xface[i] + 1.)/2.
        #     xelem[i,:] = (1. - xf[i])*x0 + xf[i]*x1


        xf1 = (face_pts + 1.)/2.
        xf0 = 1. - xf1

        elem_pts = xf0*xn0 + xf1*xn1

        # xelem = (1. - xf)*x0 + xf*x1

        return elem_pts

    # def get_quadrature(self, mesh, order, physics = None):
        
    #     dim = self.DIM
    #     gorder = mesh.gorder
    #     if physics is not None:
    #         qorder = physics.QuadOrder(order)
    #     else:
    #         qorder = order
    #     if gorder > 1:
    #         qorder += dim*(gorder-1)
                    
    #     return qorder

    def get_quadrature_data(self, order):

        quad_pts, quad_wts = triangle.get_quadrature_points_weights(order, 
            self.quadrature_type)

        return quad_pts, quad_wts


class BasisBase(ABC): 
    @property
    @abstractmethod
    def basis_type(self):
        pass

    @property
    @abstractmethod
    def MODAL_OR_NODAL(self):
        pass

    @abstractmethod
    def __init__(self, order):

        self.order = order
        self.basis_val = np.zeros(0)
        self.basis_grad = np.zeros(0)
        self.basis_pgrad = np.zeros(0)
        self.face = -1
        self.nb = 0
        self.quadrature_type = -1
        self.get_1d_nodes = basis_tools.set_node_type("Equidistant")

    def __repr__(self):
        return '{self.__class__.__name__}(Order={self.order})'.format(self=
            self)

    def get_physical_grad(self, ijac):
        '''
        Method: get_physical_grad
        --------------------------
        Calculate the physical gradient

        INPUTS:
            JData: jacobian data

        OUTPUTS:
            gPhi: gradient of basis in physical space
        '''
        nq = ijac.shape[0]

        # if nq != JData.nq and JData.nq != 1:
            # raise Exception("Quadrature doesn't match")
        # dim = JData.dim
        dim = self.DIM
        # if dim != self.dim:
            # raise Exception("Dimensions don't match")
        nb = self.nb
        basis_grad = self.basis_grad 
        if basis_grad.shape[0] == 0:
            raise Exception("basis_grad not evaluated")

        basis_pgrad = np.zeros([nq, nb, dim])

        if basis_pgrad.shape != basis_grad.shape:
            raise Exception("basis_pgrad and basis_grad are different sizes")

        basis_pgrad[:] = np.transpose(np.matmul(ijac.transpose(0,2,1), basis_grad.transpose(0,2,1)), (0,2,1))

        return basis_pgrad

    def eval_basis(self, quad_pts, Get_Phi=True, Get_GPhi=False, 
        Get_gPhi=False, ijac=None):
        '''
        Method: eval_basis
        --------------------
        Evaluate the basis functions

        INPUTS:
            quad_pts: coordinates of quadrature points
            Get_Phi: flag to calculate basis functions (Default: True)
            Get_GPhi: flag to calculate gradient of basis functions in ref space (Default: False)
            Get_gPhi: flag to calculate gradient of basis functions in phys space (Default: False)
            JData: jacobian data (needed if calculating physical gradients)
        '''
        if Get_Phi:
            self.basis_val = self.get_values(quad_pts)
        if Get_GPhi:
            self.basis_grad = self.get_grads(quad_pts)
        if Get_gPhi:
            if ijac is None:
                raise Exception("Need jacobian data")
            self.basis_pgrad = self.get_physical_grad(ijac)

    def eval_basis_on_face(self, mesh, face, quad_pts, xelem=None, basis=None
        , Get_Phi=True, Get_GPhi=False, Get_gPhi=False, ijac=False):
        '''
        Method: eval_basis_on_face
        ----------------------------
        Evaluate the basis functions on faces

        INPUTS:
            mesh: mesh object
            face: index of face in reference space
            quad_pts: coordinates of quadrature points
            Get_Phi: flag to calculate basis functions (Default: True)
            Get_GPhi: flag to calculate gradient of basis functions in ref space (Default: False)
            Get_gPhi: flag to calculate gradient of basis functions in phys space (Default: False)
            JData: jacobian data (needed if calculating physical gradients)

        OUTPUTS:
            xelem: coordinate of face
        '''
        self.face = face
        nq = quad_pts.shape[0]
        if basis is None:
            basis = mesh.gbasis

        #Note: This logic is for ADER-DG when using a modal basis function
        if basis.MODAL_OR_NODAL != ModalOrNodal.Nodal:
            basis = basis_defs.LagrangeQuad(1,mesh=mesh)

        # if xelem is None or xelem.shape != (nq, self.DIM):
        #     xelem = np.zeros([nq, self.DIM])
        # xelem = basis.get_elem_ref_from_face_ref(face, nq, quad_pts, xelem)
        xelem = basis.get_elem_ref_from_face_ref(face, quad_pts)
        self.eval_basis(xelem, Get_Phi, Get_GPhi, Get_gPhi, ijac)

        return xelem


class LagrangeSeg(BasisBase, SegShape):

    basis_type =  BasisType.LagrangeSeg
    MODAL_OR_NODAL = ModalOrNodal.Nodal

    def __init__(self, order):
        super().__init__(order)
        self.nb = self.get_num_basis_coeff(order)
        self.calculate_normals = basis_tools.calculate_1D_normals
 
    def get_nodes(self, p, xn=None):
        '''
        Method: equidistant_nodes
        --------------------------
        Calculate the coordinates in ref space

        INPUTS:
            p: order of polynomial space
            
        OUTPUTS: 
            xn: coordinates of nodes in ref space
        '''
        nb = self.get_num_basis_coeff(p)

        dim = self.DIM

        adim = nb, dim
        if xn is None or xn.shape != adim:
            xn = np.zeros(adim)

        if p == 0:
            xn[:] = 0.0 # 0.5
            return xn, nb

        # xn[:,0] = basis_tools.equidistant_nodes_1D_range(-1., 1., nb)
        xn[:,0] = self.get_1d_nodes(-1., 1., nb)

        return xn, nb

    def get_values(self, quad_pts):
        '''
        Method: get_values
        ------------------------------
        Calculates lagrange basis

        INPUTS:
            x: coordinate of current node

        OUTPUTS: 
            phi: evaluated basis 
        '''
        p = self.order
        nb = self.nb
        nq = quad_pts.shape[0]

        basis_val = np.zeros([nq, nb])

        if p == 0:
            basis_val[:] = 1.
            return basis_val

        xnodes = self.get_1d_nodes(-1., 1., p + 1)

        basis_tools.get_lagrange_basis_1D(quad_pts, xnodes, basis_val)
        # nnode = p+1
        # xnode = basis_tools.equidistant_nodes_1D_range(-1., 1., nnode)

        return basis_val

    def get_grads(self, quad_pts):
        '''
        Method: get_grads
        ------------------------------
        Calculates the lagrange basis gradients

        INPUTS:
            x: coordinate of current node
            
        OUTPUTS: 
            gphi: evaluated gradient of basis
        '''
        dim = self.DIM
        p = self.order
        nb = self.nb
        nq = quad_pts.shape[0]

        basis_grad = np.zeros([nq, nb, dim])

        if p == 0:
            return basis_grad

        # nnode = p+1
        # xnode = basis_tools.equidistant_nodes_1D_range(-1., 1., nnode)
        xnodes = self.get_1d_nodes(-1., 1., p + 1)

        basis_tools.get_lagrange_basis_1D(quad_pts, xnodes, gphi=basis_grad)

        return basis_grad

    def local_q1_face_nodes(self, p, face, fnodes=None):
        '''
        Method: local_q1_face_nodes
        -------------------
        Constructs the map for face nodes in 1D

        INPUTS:
            p: order of polynomial space
            face: face value in ref space

        OUTPUTS: 
            fnodes: index of face nodes
            nfnode: number of face nodes
        '''
        nfnode = 1
        if fnodes is None: fnodes = np.zeros(nfnode, dtype=int)
        if face == 0:
            fnodes[0] = 0
        elif face == 1:
            fnodes[0] = p
        else:
            raise IndexError

        return fnodes, nfnode

    def local_face_nodes(self, p, face, fnodes=None):
        fnodes, nfnode = self.local_q1_face_nodes(p, face, fnodes=None)

        return fnodes, nfnode


class LagrangeQuad(BasisBase, QuadShape):

    basis_type = BasisType.LagrangeQuad
    MODAL_OR_NODAL = ModalOrNodal.Nodal

    def __init__(self, order):
        super().__init__(order)
        self.nb = self.get_num_basis_coeff(order)
        self.calculate_normals = basis_tools.calculate_2D_normals

    def get_nodes(self, p, xn=None):
        '''
        Method: equidistant_nodes
        --------------------------
        Calculate the coordinates in ref space

        INPUTS:
            basis: type of basis function
            p: order of polynomial space
            
        OUTPUTS: 
            xn: coordinates of nodes in ref space
        '''
        nb = self.get_num_basis_coeff(p)
        dim = self.DIM

        adim = nb, dim
        if xn is None or xn.shape != adim:
            xn = np.zeros(adim)

        if p == 0:
            xn[:] = 0.0 # 0.5
            return xn, nb

        # xseg = basis_tools.equidistant_nodes_1D_range(-1., 1., p+1)
        xseg = self.get_1d_nodes(-1., 1., p + 1)

        xn[:,0] = np.tile(xseg, (p+1,1)).reshape(-1)
        xn[:,1] = np.repeat(xseg, p+1, axis=0).reshape(-1)

        return xn, nb

    def get_values(self, quad_pts):
        '''
        Method: get_values
        ------------------------------
        Calculates Lagrange basis for 2D quads

        INPUTS:
            x: coordinate of current node

        OUTPUTS: 
            phi: evaluated basis 
        '''
        p = self.order
        nb = self.nb
        nq = quad_pts.shape[0]

        # if basis_val is None or basis_val.shape != (nq,nb):
        #     basis_val = np.zeros([nq,nb])
        # else:
        #     basis_val[:] = 0.
        basis_val = np.zeros([nq, nb])

        if p == 0:
            basis_val[:] = 1.
            return basis_val

        # nnode = p + 1
        # xnodes = basis_tools.equidistant_nodes_1D_range(-1., 1., p+1)

        xnodes = self.get_1d_nodes(-1., 1., p+1)

        basis_tools.get_lagrange_basis_2D(quad_pts, xnodes, basis_val)

        return basis_val

    def get_grads(self, quad_pts):
        '''
        Method: get_grads
        ------------------------------
        Calculates the lagrange basis gradients for 2D quads

        INPUTS:
            x: coordinate of current node
            
        OUTPUTS: 
            gphi: evaluated gradient of basis
        '''
        dim = self.DIM
        p = self.order
        nb = self.nb
        nq = quad_pts.shape[0]

        # if basis_grad is None or basis_grad.shape != (nq,nb,dim):
        #     basis_grad = np.zeros([nq,nb,dim])
        # else: 
        #     basis_grad[:] = 0.
        basis_grad = np.zeros([nq, nb, dim])

        if p == 0:
            # basis_grad[:,:] = 0.
            return basis_grad


        xnode = self.get_1d_nodes(-1., 1., p + 1)

        basis_tools.get_lagrange_basis_2D(quad_pts, xnode, gphi=basis_grad)
        # nnode = p+1
        # xnode = basis_tools.equidistant_nodes_1D_range(-1., 1., nnode)
        # self.basis_grad = basis_grad

        return basis_grad

    def local_q1_face_nodes(self, p, face, fnodes=None):
        '''
        Method: local_q1_face_nodes
        -------------------
        Constructs the map for face nodes on 2D quads
        (For q1 elements only)

        INPUTS:
            p: order of polynomial space
            face: face value in ref space

        OUTPUTS: 
            fnodes: index of face nodes
            nfnode: number of face nodes
        '''
        nfnode = 2
        if fnodes is None: fnodes = np.zeros(nfnode, dtype=int)
        if face == 0:
            fnodes[0] = 0; fnodes[1] = p
        elif face == 1:
            fnodes[0] = p; fnodes[1] = (p+2)*p
        elif face == 2:
            fnodes[0] = (p+2)*p; fnodes[1] = (p+1)*p
        elif face == 3:
            fnodes[0] = (p+1)*p; fnodes[1] = 0
        else:
             raise IndexError

        return fnodes, nfnode

    def local_face_nodes(self, p, face, fnodes=None):
        '''
        Method: local_face_nodes
        -------------------
        Constructs the map for face nodes on 2D quads
        (For q > 1 elements)

        INPUTS:
            p: order of polynomial space
            face: face value in ref space

        OUTPUTS: 
            fnodes: index of face nodes
            nfnode: number of face nodes
        '''
        if p < 1:
            raise ValueError

        nfnode = p+1
        if fnodes is None: fnodes = np.zeros(nfnode, dtype=int)
        if face == 0:
            i0 = 0;       d =    1
        elif face == 1:
            i0 = p;       d =  p+1
        elif face == 2:
            i0 = p*(p+2); d =   -1
        elif face == 3:
            i0 = p*(p+1); d = -p-1
        else:
             raise IndexError

        fnodes[:] = i0 + np.arange(p+1, dtype=int)*d

        return fnodes, nfnode


class LagrangeTri(BasisBase, TriShape):

    basis_type = BasisType.LagrangeTri
    MODAL_OR_NODAL = ModalOrNodal.Nodal

    def __init__(self, order):
        super().__init__(order)
        self.nb = self.get_num_basis_coeff(order)
        self.calculate_normals = basis_tools.calculate_2D_normals

    def get_values(self, quad_pts):
        '''
        Method: get_values
        ------------------------------
        Calculates Lagrange basis for triangles

        INPUTS:
            x: coordinate of current node

        OUTPUTS: 
            phi: evaluated basis 
        '''
        p = self.order
        nb = self.nb
        nq = quad_pts.shape[0]

        # if basis_val is None or basis_val.shape != (nq,nb):
        #     basis_val = np.zeros([nq,nb])
        # else:
        #     basis_val[:] = 0.
        basis_val = np.zeros([nq, nb])

        if p == 0:
            basis_val[:] = 1.
            return basis_val

        xn = self.equidistant_nodes(p)

        basis_tools.get_lagrange_basis_tri(quad_pts, p, xn, basis_val)

        # self.basis_val = basis_val

        return basis_val

    def get_grads(self, quad_pts):
        '''
        Method: get_grads
        ------------------------------
        Calculates the lagrange basis gradients

        INPUTS:
            x: coordinate of current node
            
        OUTPUTS: 
            gphi: evaluated gradient of basis
        '''
        dim = self.DIM
        p = self.order
        nb = self.nb
        nq = quad_pts.shape[0]

        # if basis_grad is None or basis_grad.shape != (nq,nb,dim):
        #     basis_grad = np.zeros([nq,nb,dim])
        # else: 
        #     basis_grad[:] = 0.
        basis_grad = np.zeros([nq, nb, dim])

        if p == 0:
            # basis_grad[:,:] = 0.
            return basis_grad

        xn = self.equidistant_nodes(p)

        basis_tools.get_lagrange_grad_tri(quad_pts, p, xn, basis_grad)

        # self.basis_grad = basis_grad

        return basis_grad

    def local_q1_face_nodes(self, p, face, fnodes=None):
        '''
        Method: local_q1_face_nodes
        -------------------
        Constructs the map for face nodes on triangles
        (For q1 elements only)

        INPUTS:
            p: order of polynomial space
            face: face value in ref space

        OUTPUTS: 
            fnodes: index of face nodes
            nfnode: number of face nodes
        '''
        nfnode = 2
        if fnodes is None: fnodes = np.zeros(nfnode, dtype=int)
        if face == 0:
            fnodes[0] = p; fnodes[1] = (p+1)*(p+2)//2-1
        elif face == 1:
            fnodes[0] = (p+1)*(p+2)//2-1; fnodes[1] = 0
        elif face == 2:
            fnodes[0] = 0; fnodes[1] = p
        else:
            raise IndexError

        return fnodes, nfnode

    def local_face_nodes(self, p, face, fnodes=None):
        '''
        Method: local_face_nodes
        -------------------
        Constructs the map for face nodes on triangles
        (For q > 1 elements only)

        INPUTS:
            p: order of polynomial space
            face: face value in ref space

        OUTPUTS: 
            fnodes: index of face nodes
            nfnode: number of face nodes
        '''
        if p < 1:
            raise ValueError

        nfnode = p+1
        if fnodes is None: fnodes = np.zeros(nfnode, dtype=int)
        if face == 0:
            i0 = p; d0 = p; d1 = -1
        elif face == 1:
            i0 = (p+1)*(p+2)//2-1; d0 = -2; d1 = -1
        elif face == 2:
            i0 = 0;  d0 = 1; d1 = 0
        else:
            raise IndexError

        fnodes[0] = i0
        d = d0
        for i in range(1, p+1):
            fnodes[i] = fnodes[i-1] + d
            d += d1

        return fnodes, nfnode


class LegendreSeg(BasisBase, SegShape):

    basis_type = BasisType.LegendreSeg
    MODAL_OR_NODAL = ModalOrNodal.Modal

    def __init__(self, order):
        super().__init__(order)
        self.nb = self.get_num_basis_coeff(order)

    def get_values(self, quad_pts):
        '''
        Method: get_values
        ------------------------------
        Calculates Legendre basis for segments

        INPUTS:
            x: coordinate of current node

        OUTPUTS: 
            phi: evaluated basis 
        '''
        p = self.order
        nb = self.nb
        nq = quad_pts.shape[0]

        basis_val = np.zeros([nq, nb])

        if p == 0:
            basis_val[:] = 1.
            return basis_val

        basis_tools.get_legendre_basis_1D(quad_pts, p, basis_val)

        return basis_val

    def get_grads(self, quad_pts):
        '''
        Method: grad_tensor_lagrange
        ------------------------------
        Calculates the Legendre basis gradients

        INPUTS:
            x: coordinate of current node
            
        OUTPUTS: 
            gphi: evaluated gradient of basis
        '''
        dim = self.DIM
        p = self.order
        nb = self.nb
        nq = quad_pts.shape[0]

        basis_grad = np.zeros([nq, nb, dim])

        if p == 0:
            return basis_grad

        basis_tools.get_legendre_basis_1D(quad_pts, p, gphi=basis_grad)

        return basis_grad


class LegendreQuad(BasisBase, QuadShape):

    basis_type = BasisType.LegendreQuad
    MODAL_OR_NODAL = ModalOrNodal.Modal

    def __init__(self, order):
        super().__init__(order)
        self.nb = self.get_num_basis_coeff(order)

    def get_values(self, quad_pts):
        '''
        Method: get_values
        ------------------------------
        Calculates Legendre basis for 2D quads

        INPUTS:
            x: coordinate of current node

        OUTPUTS: 
            phi: evaluated basis 
        '''
        p = self.order
        nb = self.nb
        nq = quad_pts.shape[0]

        basis_val = np.zeros([nq, nb])

        if p == 0:
            basis_val[:] = 1.
            return basis_val

        basis_tools.get_legendre_basis_2D(quad_pts, p, basis_val)

        return basis_val

    def get_grads(self, quad_pts):
        '''
        Method: get_grads
        ------------------------------
        Calculates the Legendre basis gradients

        INPUTS:
            dim: dimension of mesh
            p: order of polynomial space
            x: coordinate of current node
            
        OUTPUTS: 
            gphi: evaluated gradient of basis
        '''

        dim = self.DIM
        p = self.order
        nb = self.nb
        nq = quad_pts.shape[0]

        basis_grad = np.zeros([nq, nb, dim])

        if p == 0:
            return basis_grad

        basis_tools.get_legendre_basis_2D(quad_pts, p, gphi=basis_grad)

        return basis_grad


class HierarchicH1Tri(BasisBase, TriShape):

    basis_type = BasisType.HierarchicH1Tri
    MODAL_OR_NODAL = ModalOrNodal.Modal

    def __init__(self, order):
        super().__init__(order)
        self.nb = self.get_num_basis_coeff(order)

    def get_values(self, quad_pts):
        '''
        Method: get_values
        ------------------------------
        Calculates Lagrange basis for triangles

        INPUTS:
            x: coordinate of current node

        OUTPUTS: 
            phi: evaluated basis 
        '''
        p = self.order
        nb = self.nb
        nq = quad_pts.shape[0]

        # if basis_val is None or basis_val.shape != (nq,nb):
        #     basis_val = np.zeros([nq,nb])
        # else:
        #     basis_val[:] = 0.
        basis_val = np.zeros([nq, nb])

        if p == 0:
            basis_val[:] = 1.
            return basis_val

        xn = self.equidistant_nodes(p)

        basis_tools.get_modal_basis_tri(quad_pts, p, xn, basis_val)

        # self.basis_val = basis_val

        return basis_val

    def get_grads(self, quad_pts):
        '''
        Method: get_grads
        ------------------------------
        Calculates the lagrange basis gradients

        INPUTS:
            x: coordinate of current node
            
        OUTPUTS: 
            gphi: evaluated gradient of basis
        '''
        dim = self.DIM
        p = self.order
        nb = self.nb
        nq = quad_pts.shape[0]

        # if basis_grad is None or basis_grad.shape != (nq,nb,dim):
        #     basis_grad = np.zeros([nq,nb,dim])
        # else: 
        #     basis_grad[:] = 0.
        basis_grad = np.zeros([nq, nb, dim])

        if p == 0:
            # basis_grad[:,:] = 0.
            return basis_grad

        xn = self.equidistant_nodes(p)

        basis_tools.get_modal_grad_tri(quad_pts, p, xn, basis_grad)

        basis_grad = 2.*basis_grad

        # self.basis_grad = basis_grad

        return basis_grad
