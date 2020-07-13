from abc import ABC, abstractmethod
import code
import numpy as np

from data import ArrayList, GenericData
from general import BasisType, ShapeType, ModalOrNodal, QuadratureType, NodeType

import meshing.gmsh as mesh_gmsh

import numerics.basis.tools as basis_tools
import numerics.basis.basis as basis_defs

from numerics.quadrature import segment, quadrilateral, triangle

RefQ1Coords = {
    BasisType.LagrangeSeg : np.array([[-1.],[1.]]),
    BasisType.LagrangeQuad : np.array([[-1.,-1.],[1.,-1.],
                                [-1.,1.],[1.,1.]]),
    BasisType.LagrangeTri : np.array([[0.,0.],[1.,0.],
                                [0.,1.]]),
    BasisType.LegendreSeg : np.array([[-1.],[1.]]),
    BasisType.LegendreQuad : np.array([[-1.,-1.],[1.,-1.],
                                [-1.,1.],[1.,1.]]),
    BasisType.HierarchicH1Tri : np.array([[0.,0.],[1.,0.],
                                [0.,1.]])
}

class ShapeBase(ABC):
    @property
    @abstractmethod
    def shape_type(self):
        pass

    @property
    @abstractmethod
    def face_shape_type(self):
        pass

    @property
    @abstractmethod
    def face_shape(self):
        pass

    @property
    @abstractmethod
    def nfaceperelem(self):
        pass

    @property
    @abstractmethod
    def dim(self):
        pass

    @property
    @abstractmethod
    def centroid(self):
        pass
    
    @abstractmethod
    def get_num_basis_coeff(self,p):
        pass

    @abstractmethod
    def equidistant_nodes(self, p, xn=None):
        pass

    def set_elem_quadrature_type(self, quadrature_name):
        self.quadrature_type = QuadratureType[quadrature_name]

    def set_face_quadrature_type(self, quadrature_name):
        self.face_shape.quadrature_type = QuadratureType[quadrature_name]

    def force_nodes_equal_quadpts(self, force_flag):
        if force_flag == True:
            self.forced_pts = self.get_num_basis_coeff(self.order)

class PointShape(ShapeBase):

    shape_type = ShapeType.Point
    face_shape_type = None
    face_shape = None
    nfaceperelem = 0
    dim = 0
    centroid = np.array([[0.]])
    
    def get_num_basis_coeff(self,p):
        return 1
    def equidistant_nodes(self, p, xn=None):
        pass
    def get_quadrature(self, mesh, order, physics=None):
        dim = self.dim
        gorder = mesh.gorder
        if physics is not None:
            qorder = physics.QuadOrder(order)
        else:
            qorder = order
        if gorder > 1:
            qorder += dim*(gorder-1)

        # qorder = 0
        return qorder

    def get_quad_data(self, order):

        quad_pts = np.zeros([1,1])
        quad_wts = np.ones([1,1])

        return quad_pts, quad_wts

class SegShape(ShapeBase):

    shape_type = ShapeType.Segment
    face_shape_type = ShapeType.Point
    face_shape = PointShape()
    nfaceperelem = 2
    dim = 1
    centroid = np.array([[0.]])

    get_face_quadrature = face_shape.get_quadrature
    get_face_quad_data = face_shape.get_quad_data

    def get_num_basis_coeff(self,p):
        return p + 1

    def equidistant_nodes(self, p, xn=None):
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

        dim = self.dim

        adim = nb,dim
        if xn is None or xn.shape != adim:
            xn = np.zeros(adim)

        if p == 0:
            xn[:] = 0.0 # 0.5
            return xn, nb

        xn[:,0] = basis_tools.equidistant_nodes_1D_range(-1., 1., nb)

        return xn, nb

    def ref_face_to_elem(self, face, nq, xface, xelem=None):
        '''
        Function: ref_face_to_elem
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
        if xelem is None: xelem = np.zeros([1,1])
        if face == 0: xelem[0] = -1.
        elif face == 1: xelem[0] = 1.
        else: raise ValueError

        return xelem

    def get_quadrature(self, mesh, order, physics=None):
        
        dim = self.dim
        gorder = mesh.gorder
        if physics is not None:
            qorder = physics.QuadOrder(order)
        else:
            qorder = order
        if gorder > 1:
            qorder += dim*(gorder-1)

        return qorder

    def get_quad_data(self, order):

        try:
            fpts = self.forced_pts
        except:
            fpts = None

        quad_pts, quad_wts = segment.get_quadrature_points_weights(order, self.quadrature_type, forced_pts=fpts)

        return quad_pts, quad_wts

class QuadShape(ShapeBase):

    shape_type = ShapeType.Quadrilateral
    face_shape_type = ShapeType.Segment
    face_shape = SegShape()
    nfaceperelem = 4
    dim = 2
    centroid = np.array([[0., 0.]])

    get_face_quadrature = face_shape.get_quadrature
    get_face_quad_data = face_shape.get_quad_data

    def get_num_basis_coeff(self,p):
        return (p + 1)**2

    def equidistant_nodes(self, p, xn=None):
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
        dim = self.dim

        adim = nb,dim
        if xn is None or xn.shape != adim:
            xn = np.zeros(adim)

        if p == 0:
            xn[:] = 0.0 # 0.5
            return xn, nb

        xseg = basis_tools.equidistant_nodes_1D_range(-1., 1., p+1)

        xn[:,0] = np.tile(xseg, (p+1,1)).reshape(-1)
        xn[:,1] = np.repeat(xseg, p+1, axis=0).reshape(-1)

        return xn, nb

    def ref_face_to_elem(self, face, nq, xface, xelem=None):
        '''
        Function: ref_face_to_elem
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
        if xelem is None: xelem = np.zeros([nq,2])

        fnodes, nfnode = self.local_q1_face_nodes(1, face)

        x0 = RefQ1Coords[BasisType.LagrangeQuad][fnodes[0]]
        x1 = RefQ1Coords[BasisType.LagrangeQuad][fnodes[1]]

        if face == 0:
            xelem[:,0] = np.reshape((xface*x1[0] - xface*x0[0])/2., nq)
            xelem[:,1] = -1.
        elif face == 1:
            xelem[:,1] = np.reshape((xface*x1[1] - xface*x0[1])/2., nq)
            xelem[:,0] = 1.
        elif face == 2:
            xelem[:,0] = np.reshape((xface*x1[0] - xface*x0[0])/2., nq)
            xelem[:,1] = 1.
        else:
            xelem[:,1] = np.reshape((xface*x1[1] - xface*x0[1])/2., nq)
            xelem[:,0] = -1.

        return xelem
    
    def get_quadrature(self, mesh, order, physics = None):
        
        dim = self.dim
        gorder = mesh.gorder
        if physics is not None:
            qorder = physics.QuadOrder(order)
        else:
            qorder = order
        if gorder > 1:
            qorder += dim*(gorder-1)
            
        qorder += 2 

        return qorder

    def get_quad_data(self, order):

        try:
            fpts = self.forced_pts
        except:
            fpts = None

        quad_pts, quad_wts = quadrilateral.get_quadrature_points_weights(order, self.quadrature_type, forced_pts=fpts)

        return quad_pts, quad_wts

class TriShape(ShapeBase):

    shape_type = ShapeType.Triangle
    face_shape_type = ShapeType.Segment
    face_shape = SegShape()
    nfaceperelem = 3
    dim = 2
    centroid = np.array([[1./3., 1./3.]])

    get_face_quadrature = face_shape.get_quadrature
    get_face_quad_data = face_shape.get_quad_data

    def get_num_basis_coeff(self,p):
        return (p + 1)*(p + 2)//2

    def equidistant_nodes(self, p, xn=None):
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
        dim = self.dim
        

        adim = nb,dim
        if xn is None or xn.shape != adim:
            xn = np.zeros(adim)

        if p == 0:
            xn[:] = 0.0 # 0.5
            return xn, nb
        n = 0
        xseg = basis_tools.equidistant_nodes_1D_range(0., 1., p+1)
        for j in range(p+1):
            xn[n:n+p+1-j,0] = xseg[:p+1-j]
            xn[n:n+p+1-j,1] = xseg[j]
            n += p+1-j

        return xn, nb

    def ref_face_to_elem(self, face, nq, xface, xelem=None):
        '''
        Function: ref_face_to_elem
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
        if xelem is None: xelem = np.zeros([nq,2])
        xf = np.zeros(nq)
        xf = xf.reshape((nq,1))
        # local q = 1 nodes on face
        fnodes, nfnode = self.local_q1_face_nodes(1, face)
        # coordinates of local q = 1 nodes on face
        x0 = RefQ1Coords[BasisType.LagrangeTri][fnodes[0]]
        x1 = RefQ1Coords[BasisType.LagrangeTri][fnodes[1]]
        # for i in range(nq):
        #     xf[i] = (xface[i] + 1.)/2.
        #     xelem[i,:] = (1. - xf[i])*x0 + xf[i]*x1
        xf = (xface + 1.)/2.
        xelem[:] = (1. - xf)*x0 + xf*x1

        return xelem

    def get_quadrature(self, mesh, order, physics = None):
        
        dim = self.dim
        gorder = mesh.gorder
        if physics is not None:
            qorder = physics.QuadOrder(order)
        else:
            qorder = order
        if gorder > 1:
            qorder += dim*(gorder-1)
                    
        return qorder

    def get_quad_data(self, order):

        quad_pts, quad_wts = triangle.get_quadrature_points_weights(order, self.quadrature_type)

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
        return '{self.__class__.__name__}(Order={self.order})'.format(self=self)

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
        dim = self.dim
        # if dim != self.dim:
            # raise Exception("Dimensions don't match")
        nb = self.nb
        basis_grad = self.basis_grad 
        if basis_grad.shape[0] == 0:
            raise Exception("basis_grad not evaluated")

        # if self.basis_pgrad is None or self.basis_pgrad.shape != (nq,nb,dim):
        #     self.basis_pgrad = np.zeros([nq,nb,dim])
        # else:
        #     self.basis_pgrad *= 0.
        basis_pgrad = np.zeros([nq, nb, dim])

        # basis_pgrad = self.basis_pgrad

        if basis_pgrad.shape != basis_grad.shape:
            raise Exception("basis_pgrad and basis_grad are different sizes")

        basis_pgrad[:] = np.transpose(np.matmul(ijac.transpose(0,2,1), basis_grad.transpose(0,2,1)), (0,2,1))

        return basis_pgrad

    def eval_basis(self, quad_pts, Get_Phi=True, Get_GPhi=False, Get_gPhi=False, ijac=None):
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

    def eval_basis_on_face(self, mesh, face, quad_pts, xelem=None, basis = None, Get_Phi=True, Get_GPhi=False, Get_gPhi=False, ijac=False):
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

        if xelem is None or xelem.shape != (nq, self.dim):
            xelem = np.zeros([nq, self.dim])
        xelem = basis.ref_face_to_elem(face, nq, quad_pts, xelem)
        self.eval_basis(xelem, Get_Phi, Get_GPhi, Get_gPhi, ijac)

        return xelem


def get_lagrange_basis_1D(x, xnodes, phi=None, gphi=None):
    '''
    Method: get_lagrange_basis_1D
    ------------------------------
    Calculates the 1D Lagrange basis functions

    INPUTS:
        x: coordinate of current node
        xnodes: coordinates of nodes in 1D ref space
        nnode: number of nodes in 1D ref space
        
    OUTPUTS: 
        phi: evaluated basis 
        gphi: evaluated physical gradient of basis
    '''
    nnodes = xnodes.shape[0]
    mask = np.ones(nnodes, bool)

    if phi is not None:
        phi[:] = 1.
        for j in range(nnodes):
            mask[j] = False
            phi[:,j] = np.prod((x - xnodes[mask])/(xnodes[j] - xnodes[mask]),axis=1)
            mask[j] = True

    if gphi is not None:
        gphi[:] = 0.

        for j in range(nnodes):
            mask[j] = False
            for i in range(nnodes):
                if i == j:
                    continue

                mask[i] = False
                if nnodes > 2: 
                    gphi[:,j,:] += np.prod((x - xnodes[mask])/(xnodes[j] - xnodes[mask]),
                        axis=1).reshape(-1,1)/(xnodes[j] - xnodes[i])
                else:
                    gphi[:,j,:] += 1./(xnodes[j] - xnodes[i])

                mask[i] = True
            mask[j] = True


def get_lagrange_basis_2D(x, xnodes, phi=None, gphi=None):
    '''
    Method: get_lagrange_basis_2D
    ------------------------------
    Calculates the 2D Lagrange basis functions

    INPUTS:
        x: coordinate of current node
        xnodes: coordinates of nodes in 1D ref space
        nnode: number of nodes in 1D ref space
        
    OUTPUTS: 
        phi: evaluated basis 
        gphi: evaluated gradient of basis
    '''
    if gphi is not None:
        gphix = np.zeros((x.shape[0], xnodes.shape[0], 1)); gphiy = np.zeros_like(gphix)
    else:
        gphix = None; gphiy = None
    # Always need phi
    phix = np.zeros((x.shape[0], xnodes.shape[0])); phiy = np.zeros_like(phix)

    nnodes_1D = xnodes.shape[0]
    lagrange_eq_seg = LagrangeSeg(nnodes_1D-1)
    get_lagrange_basis_1D(x[:, 0].reshape(-1, 1), xnodes, phix, gphix)
    get_lagrange_basis_1D(x[:, 1].reshape(-1, 1), xnodes, phiy, gphiy)

    if phi is not None:
        for i in range(x.shape[0]):
            phi[i, :] = np.reshape(np.outer(phix[i, :], phiy[i, :]), (-1, ), 'F')
    if gphi is not None:
        for i in range(x.shape[0]):
            gphi[i, :, 0] = np.reshape(np.outer(gphix[i, :, 0], phiy[i, :]), (-1, ), 'F')
            gphi[i, :, 1] = np.reshape(np.outer(phix[i, :], gphiy[i, :, 0]), (-1, ), 'F')


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

        dim = self.dim

        adim = nb,dim
        if xn is None or xn.shape != adim:
            xn = np.zeros(adim)

        if p == 0:
            xn[:] = 0.0 # 0.5
            return xn, nb

        # xn[:,0] = basis_tools.equidistant_nodes_1D_range(-1., 1., nb)
        xn[:,0] = self.get_1d_nodes(-1., 1., nb)

        return xn, nb

# <<<<<<< Updated upstream
    def get_values(self, quad_pts):
# =======
        # return xn, nb
    # def get_values(self, quad_pts, basis_val=None):
# >>>>>>> Stashed changes
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

        # if basis_val is None or basis_val.shape != (nq,nb):
        #     basis_val = np.zeros([nq,nb])
        # else:
        #     basis_val[:] = 0.
        basis_val = np.zeros([nq, nb])

        if p == 0:
            basis_val[:] = 1.
            return basis_val

# <<<<<<< Updated upstream
        # nnode = p + 1
        # xnodes = basis_tools.equidistant_nodes_1D_range(-1., 1., p + 1)
        xnodes = self.get_1d_nodes(-1., 1., p + 1)

        get_lagrange_basis_1D(quad_pts, xnodes, basis_val)
# =======
        # nnode = p+1
        # xnode = basis_tools.equidistant_nodes_1D_range(-1., 1., nnode)
        # xnode = self.get_1d_nodes(-1., 1., nnode)

        # scheme = quadpy.line_segment.gauss_lobatto(nnode)
        # xnode = scheme.points
# >>>>>>> Stashed changes

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
        dim = self.dim
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

# <<<<<<< Updated upstream
#         # nnode = p + 1
#         xnodes = basis_tools.equidistant_nodes_1D_range(-1., 1., p + 1)
# =======
        # nnode = p+1
        # xnode = basis_tools.equidistant_nodes_1D_range(-1., 1., nnode)
        xnodes = self.get_1d_nodes(-1., 1., p + 1)
        # scheme = quadpy.line_segment.gauss_lobatto(nnode)
        # xnode = scheme.points
# >>>>>>> Stashed changes

        get_lagrange_basis_1D(quad_pts, xnodes, gphi=basis_grad)

        # self.basis_grad = basis_grad

        return basis_grad

    # def get_lagrange_basis_1D(self, x, xnode, nnode, phi, gphi):
    #     '''
    #     Method: get_lagrange_basis_1D
    #     ------------------------------
    #     Calculates the 1D Lagrange basis functions

    #     INPUTS:
    #         x: coordinate of current node
    #         xnode: coordinates of nodes in 1D ref space
    #         nnode: number of nodes in 1D ref space
            
    #     OUTPUTS: 
    #         phi: evaluated basis 
    #         gphi: evaluated physical gradient of basis
    #     '''
    #     nnode = xnode.shape[0]
    #     mask = np.ones(nnode, bool)

    #     if phi is not None:
    #         phi[:] = 1.
    #         for j in range(nnode):
    #             mask[j] = False
    #             phi[:,j] = np.prod((x - xnode[mask])/(xnode[j] - xnode[mask]),axis=1)
    #             mask[j] = True

    #     if gphi is not None:
    #         gphi[:] = 0.

    #         for j in range(nnode):
    #             mask[j] = False
    #             for i in range(nnode):
    #                 if i == j:
    #                     continue

    #                 mask[i] = False
    #                 if nnode > 2: 
    #                     gphi[:,j,:] += np.prod((x - xnode[mask])/(xnode[j] - xnode[mask]),
    #                         axis=1).reshape(-1,1)/(xnode[j] - xnode[i])
    #                 else:
    #                     gphi[:,j,:] += 1./(xnode[j] - xnode[i])

    #                 mask[i] = True
    #             mask[j] = True

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
        dim = self.dim

        adim = nb,dim
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
# >>>>>>> Stashed changes
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

# <<<<<<< Updated upstream
        # nnode = p + 1
        # xnodes = basis_tools.equidistant_nodes_1D_range(-1., 1., p+1)
# =======
        # nnode = p+1
        # xnode = basis_tools.equidistant_nodes_1D_range(-1., 1., p+1)
        xnodes = self.get_1d_nodes(-1., 1., p+1)
# >>>>>>> Stashed changes

        get_lagrange_basis_2D(quad_pts, xnodes, basis_val)

        # self.basis_val = basis_val

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
        dim = self.dim
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

# <<<<<<< Updated upstream
#         # nnode = p + 1
#         xnode = basis_tools.equidistant_nodes_1D_range(-1., 1., p+1)

        get_lagrange_basis_2D(quad_pts, xnode, gphi=basis_grad)
# =======
        # nnode = p+1
        # xnode = basis_tools.equidistant_nodes_1D_range(-1., 1., nnode)
# >>>>>>> Stashed changes

        # self.basis_grad = basis_grad

        return basis_grad

    # def get_lagrange_basis_2D(self, x, xnode, nnode, phi, gphi):
    #     '''
    #     Method: get_lagrange_basis_2D
    #     ------------------------------
    #     Calculates the 2D Lagrange basis functions

    #     INPUTS:
    #         x: coordinate of current node
    #         xnode: coordinates of nodes in 1D ref space
    #         nnode: number of nodes in 1D ref space
            
    #     OUTPUTS: 
    #         phi: evaluated basis 
    #         gphi: evaluated gradient of basis
    #     '''
    #     if gphi is not None:
    #         gphix = np.zeros((x.shape[0],xnode.shape[0],1)); gphiy = np.zeros_like(gphix)
    #     else:
    #         gphix = None; gphiy = None
    #     # Always need phi
    #     phix = np.zeros((x.shape[0],xnode.shape[0])); phiy = np.zeros_like(phix)

    #     lagrange_eq_seg = LagrangeEqSeg(self.order)
    #     get_lagrange_basis_1D(x[:,0].reshape(-1,1), xnode, nnode, phix, gphix)
    #     get_lagrange_basis_1D(x[:,1].reshape(-1,1), xnode, nnode, phiy, gphiy)

    #     if phi is not None:
    #         for i in range(x.shape[0]):
    #             phi[i,:] = np.reshape(np.outer(phix[i,:], phiy[i,:]), (-1,), 'F')
    #     if gphi is not None:
    #         for i in range(x.shape[0]):
    #             gphi[i,:,0] = np.reshape(np.outer(gphix[i,:,0], phiy[i,:]), (-1,), 'F')
    #             gphi[i,:,1] = np.reshape(np.outer(phix[i,:], gphiy[i,:,0]), (-1,), 'F')

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


def get_lagrange_basis_tri(xi, p, xn, phi):

    # xn, nb = self.equidistant_nodes(p)
    nb = xn.shape[0]

    alpha = np.round(p*xn)
    alpha = np.c_[(p*np.ones(nb)-np.sum(alpha, axis=1),alpha)]
    l = np.c_[(np.ones(xi.shape[0]) - np.sum(xi, axis=1)),xi]

    if p == 0:
        phi[:] = 1.
        return 

    for i in range(nb):
        phi[:,i] = get_tri_area_coordinates(p, alpha[i], l)

    return phi


def get_tri_area_coordinates(p, alpha, l):

    N = np.ones(l.shape[0])

    N *= get_eta_function(p, alpha[0], l[:,0])
    N *= get_eta_function(p, alpha[1], l[:,1])
    N *= get_eta_function(p, alpha[2], l[:,2])

    return N


def get_eta_function(p, alpha, l, skip = -1):
    index = np.concatenate((np.arange(0, skip), np.arange(skip + 1, alpha)))

    eta = np.ones(l.shape[0])

    for i in index:
        eta *= (p * l - i) / (i + 1.)
    return eta


def get_grad_eta_function(p, alpha, l):
    geta = np.zeros_like(l)

    for i in range(int(alpha)):
        geta += ( p / (i + 1)) * get_eta_function(p, alpha, l, i)

    return geta


def get_lagrange_grad_tri(xi, p, xn, gphi):

    # xn, nb = self.equidistant_nodes(p)
    nb = xn.shape[0]
    gphi_dir = np.zeros((xi.shape[0], nb, 3))

    alpha = np.round(p*xn)
    alpha = np.c_[(p*np.ones(nb)-np.sum(alpha, axis=1),alpha)]
    l = np.c_[(np.ones(xi.shape[0]) - np.sum(xi, axis=1)),xi]

    if p == 0:
        gphi[:] = 0.
        return 
    for i in range(nb):
        gphi_dir[:,i,:] = get_tri_grad_area_coordinates(p, alpha[i], l)

    gphi[:,:,0] = gphi_dir[:,:,1] - gphi_dir[:,:,0]
    gphi[:,:,1] = gphi_dir[:,:,2] - gphi_dir[:,:,0]

    return gphi

def get_tri_grad_area_coordinates(p, alpha, l):

    dN = np.ones((l.shape[0], 3))

    N1 = get_eta_function(p, alpha[0], l[:,0])
    N2 = get_eta_function(p, alpha[1], l[:,1])
    N3 = get_eta_function(p, alpha[2], l[:,2])

    dN1 = get_grad_eta_function(p, alpha[0], l[:,0])
    dN2 = get_grad_eta_function(p, alpha[1], l[:,1])
    dN3 = get_grad_eta_function(p, alpha[2], l[:,2])

    dN[:,0] = dN1 * N2 * N3
    dN[:,1] = N1 * dN2 * N3
    dN[:,2] = N1 * N2 * dN3

    return dN


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

        xn, _ = self.equidistant_nodes(p)

        get_lagrange_basis_tri(quad_pts, p, xn, basis_val)

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
        dim = self.dim
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

        xn, _ = self.equidistant_nodes(p)

        get_lagrange_grad_tri(quad_pts, p, xn, basis_grad)

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


def get_legendre_basis_1D(x, p, phi=None, gphi=None):
    '''
    Method: get_legendre_basis_1D
    ------------------------------
    Calculates the 1D Legendre basis functions

    INPUTS:
        x: coordinate of current node
        p: order of polynomial space
        
    OUTPUTS: 
        phi: evaluated basis 
        gphi: evaluated physical gradient of basis
    '''
    leg_poly = np.polynomial.legendre.Legendre

    if phi is not None:
        phi[:, :] = 0.            
        x.shape = -1
        
        for it in range(p+1):
            phi[:, it] = leg_poly.basis(it)(x)

        x.shape = -1, 1

    if gphi is not None:
        gphi[:,:] = 0.

        for it in range(p+1):
            dleg = leg_poly.basis(it).deriv(1)
            gphi[:,it] = dleg(x)


def get_legendre_basis_2D(x, p, phi=None, gphi=None):
    '''
    Method: get_legendre_basis_2D
    ------------------------------
    Calculates the 2D Legendre basis functions

    INPUTS:
        x: coordinate of current node
        p: order of polynomial space
        
    OUTPUTS: 
        phi: evaluated basis 
        gphi: evaluated physical gradient of basis
    '''
    nq = x.shape[0]
    if gphi is not None:
        gphix = np.zeros((nq, p+1, 1)); gphiy = np.zeros_like(gphix)
    else:
        gphix = None; gphiy = None
    # Always need phi
    phix = np.zeros((nq, p+1)); phiy = np.zeros_like(phix)

    legendre_seg = LegendreSeg(p)
    get_legendre_basis_1D(x[:, 0], p, phix, gphix)
    get_legendre_basis_1D(x[:, 1], p, phiy, gphiy)

    if phi is not None:
        for i in range(nq):
            phi[i, :] = np.reshape(np.outer(phix[i, :], phiy[i, :]), (-1, ), 'F')
    if gphi is not None:
        for i in range(nq):
            gphi[i, :, 0] = np.reshape(np.outer(gphix[i, :, 0], phiy[i, :]), (-1, ), 'F')
            gphi[i, :, 1] = np.reshape(np.outer(phix[i, :], gphiy[i, :, 0]), (-1, ), 'F')


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

        # if basis_val is None or basis_val.shape != (nq,nb):
        #     basis_val = np.zeros([nq,nb])
        # else:
        #     basis_val[:] = 0.
        basis_val = np.zeros([nq, nb])

        if p == 0:
            basis_val[:] = 1.
            return basis_val

        get_legendre_basis_1D(quad_pts, p, basis_val)

        # self.basis_val = basis_val

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
        dim = self.dim
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

        get_legendre_basis_1D(quad_pts, p, gphi=basis_grad)

        # self.basis_grad = basis_grad

        return basis_grad
    
    # def get_legendre_basis_1D(self, x, phi, gphi):
    #     '''
    #     Method: get_legendre_basis_1D
    #     ------------------------------
    #     Calculates the 1D Legendre basis functions

    #     INPUTS:
    #         x: coordinate of current node
    #         p: order of polynomial space
            
    #     OUTPUTS: 
    #         phi: evaluated basis 
    #         gphi: evaluated physical gradient of basis
    #     '''
    #     leg_poly = np.polynomial.legendre.Legendre

    #     npts = x.shape[0]

    #     if phi is not None:
    #         phi[:,:] = 0.            
    #         x.shape = -1
            
    #         for it in range(npts):
    #             phi[:,it] = leg_poly.basis(it)(x)

    #         x.shape = -1,1

    #     if gphi is not None:
    #         gphi[:,:] = 0.

    #         for it in range(npts):
    #             dleg = leg_poly.basis(it).deriv(1)
    #             gphi[:,it] = dleg(x)


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

        # if basis_val is None or basis_val.shape != (nq,nb):
        #     basis_val = np.zeros([nq,nb])
        # else:
        #     basis_val[:] = 0.
        basis_val = np.zeros([nq, nb])

        if p == 0:
            basis_val[:] = 1.
            return basis_val

        get_legendre_basis_2D(quad_pts, p, basis_val)

        # self.basis_val = basis_val

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

        dim = self.dim
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

        get_legendre_basis_2D(quad_pts, p, gphi=basis_grad)

        # self.basis_grad = basis_grad

        return basis_grad

    # def get_legendre_basis_2D(self, x, p, phi, gphi):
    #     '''
    #     Method: get_legendre_basis_2D
    #     ------------------------------
    #     Calculates the 2D Legendre basis functions

    #     INPUTS:
    #         x: coordinate of current node
    #         p: order of polynomial space
            
    #     OUTPUTS: 
    #         phi: evaluated basis 
    #         gphi: evaluated physical gradient of basis
    #     '''

    #     if gphi is not None:
    #         gphix = np.zeros((x.shape[0],p+1,1)); gphiy = np.zeros_like(gphix)
    #     else:
    #         gphix = None; gphiy = None
    #     # Always need phi
    #     phix = np.zeros((x.shape[0],p+1)); phiy = np.zeros_like(phix)

    #     legendre_seg = LegendreSeg(self.order)
    #     get_legendre_basis_1D(x[:,0], phix, gphix)
    #     get_legendre_basis_1D(x[:,1], phiy, gphiy)

    #     if phi is not None:
    #         for i in range(x.shape[0]):
    #             phi[i,:] = np.reshape(np.outer(phix[i,:], phiy[i,:]), (-1,), 'F')
    #     if gphi is not None:
    #         for i in range(x.shape[0]):
    #             gphi[i,:,0] = np.reshape(np.outer(gphix[i,:,0], phiy[i,:]), (-1,), 'F')
    #             gphi[i,:,1] = np.reshape(np.outer(phix[i,:], gphiy[i,:,0]), (-1,), 'F')


def get_modal_basis_tri(xi, p, xn, phi):

    # xn, nb = self.equidistant_nodes(p)
    nb = xn.shape[0]

    phi_reorder = np.zeros_like(phi)

    # Transform to the modal basis reference element
    # [-1,-1],[1,-1],[-1,1]
    xn = 2.*xn - 1.
    xi = 2.*xi - 1.

    # Define the affine coordinates
    l = np.zeros([xi.shape[0],3])

    l[:,0] = (xi[:,1]+1.)/2.
    l[:,1] = -1.*((xi[:,1]+xi[:,0])/2.)
    l[:,2] = (xi[:,0]+1.)/2.

    if p == 0:
        phi[:] = 1.
        return 

    phi_reorder[:,[0,1,2]] = l[:,[1,2,0]]

    e1 = np.arange(3,p-1+3,1)
    e2 = np.arange(p-1+3,2*p-2+3,1)
    e3 = np.arange(2*p-2+3,3*p-3+3,1)

    phi_reorder[:,e1] = get_edge_basis(p, l[:,2], l[:,1])
    phi_reorder[:,e2] = get_edge_basis(p, l[:,0], l[:,2])
    phi_reorder[:,e3] = get_edge_basis(p, l[:,1], l[:,0])

    internal = np.arange(3*p-3+3,nb,1)

    phi_reorder[:,internal] = get_internal_basis(p, internal, l)

    index = mesh_gmsh.gmsh_node_order_tri(p)

    phi[:,:] = phi_reorder[:,index]

    return phi


def get_edge_basis(p, ll, lr):

    phi_e = np.zeros([ll.shape[0], p-1])
    for k in range(p-1):
        kernel = get_kernel_function(k, ll-lr)
        phi_e[:,k] = ll*lr*kernel

    return phi_e

def get_internal_basis(p, index, l):

    phi_i = np.zeros([l.shape[0], len(index)])

    c = 0
    for i in range(3,p+1):
        c += i-2
    
    n = np.zeros([c, 2])
    n1 = np.arange(1, p-1, 1)
    n2 = np.arange(1, p-1, 1)
    k = 0
    for i in range(len(n1)):
        for j in range(len(n2)):
            if n1[i] + n2[j] <= p-1:
                n[k,0] = n1[i]
                n[k,1] = n2[j]
                k += 1

    for m in range(c):
        phi_i[:,m] = l[:,0]*l[:,1]**n[m,0]*l[:,2]**n[m,1]

    return phi_i

def get_kernel_function(p, x):

    p+=2
    # Initialize the legendre polynomial object
    leg_poly = np.polynomial.legendre.Legendre
    x.shape = -1

    # Construct the kernel's denominator (series of Lobatto fnc's)                    
    
    # First two lobatto shape functions 
    l0 =  (1.-x)/2.
    l1 =  (1.+x)/2.

    den = l0*l1

    leg_int = leg_poly.basis(p-1).integ(m=1,lbnd=-1)
    num = np.sqrt((2.*p-1.)/2.)*leg_int(x)

    kernel = num / (1e-12 + den)

    x.shape = -1,1

    return kernel


def get_modal_grad_tri(xi, p, xn, gphi):
 
    # xn, nb = equidistant_nodes(p)
    nb = xn.shape[0]

    gphi_reorder = np.zeros_like(gphi)
    # Transform to the modal basis reference element
    # [-1,-1],[1,-1],[-1,1]

    xn = 2.*xn - 1.
    xi = 2.*xi - 1.

    gl = np.zeros([xi.shape[0],3,2])
    l = np.zeros([xi.shape[0],3])

    # Calculate the affine coordinates
    l[:,0] = (xi[:,1]+1.)/2.
    l[:,1] = -1.*((xi[:,1]+xi[:,0])/2.)
    l[:,2] = (xi[:,0]+1.)/2.

    # Calculate vertex gradients
    gl[:,0,0] = 0.
    gl[:,0,1] = 0.5 
    
    gl[:,1,0] = -0.5
    gl[:,1,1] = -0.5

    gl[:,2,0] = 0.5
    gl[:,2,1] = 0.

    if p == 0:
        phi[:] = 1.
        return 

    gphi_reorder[:,[0,1,2],:] = gl[:,[1,2,0],:]

    # Calculate edge gradients
    e1 = np.arange(3,p-1+3,1)
    e2 = np.arange(p-1+3,2*p-2+3,1)
    e3 = np.arange(2*p-2+3,3*p-3+3,1)

    dxdxi = np.zeros([3,2])
    dxdxi[0,0] = 1.   ; dxdxi[0,1] = 0.5
    dxdxi[1,0] = -0.5 ; dxdxi[1,1] = 0.5
    dxdxi[2,0] = -0.5 ; dxdxi[2,1] = -1.

    gphi_reorder[:,e1,0] = get_edge_grad(p, dxdxi[0,0], gl[:,2,0], gl[:,1,0], l[:,2], l[:,1])
    gphi_reorder[:,e1,1] = get_edge_grad(p, dxdxi[0,1], gl[:,2,1], gl[:,1,1], l[:,2], l[:,1])
    gphi_reorder[:,e2,0] = get_edge_grad(p, dxdxi[1,0], gl[:,0,0], gl[:,2,0], l[:,0], l[:,2])
    gphi_reorder[:,e2,1] = get_edge_grad(p, dxdxi[1,1], gl[:,0,1], gl[:,2,1], l[:,0], l[:,2])
    gphi_reorder[:,e3,0] = get_edge_grad(p, dxdxi[2,0], gl[:,1,0], gl[:,0,0], l[:,1], l[:,0])
    gphi_reorder[:,e3,1] = get_edge_grad(p, dxdxi[2,1], gl[:,1,1], gl[:,0,1], l[:,1], l[:,0])

    internal = np.arange(3*p-3+3,nb,1)

    gphi_reorder[:,internal,0] = get_internal_grad(p, internal, gl[:,:,0], l)
    gphi_reorder[:,internal,1] = get_internal_grad(p, internal, gl[:,:,1], l)

    index = mesh_gmsh.gmsh_node_order_tri(p)

    gphi[:,:,:] = gphi_reorder[:,index,:]

    return gphi


def get_edge_grad(p, dxdxi, gl, gr, ll, lr):

    gphi_e = np.zeros([ll.shape[0],p-1])
    for k in range(p-1):
        gkernel = get_kernel_grad(k, dxdxi, ll-lr)
        kernel = get_kernel_function(k,ll-lr)
        gphi_e[:,k] = (ll*gr+lr*gl)*kernel + ll*lr*gkernel

    return gphi_e

def get_kernel_grad(p, dxdxi, x):

    p += 2
    leg_poly = np.polynomial.legendre.Legendre
    x.shape = -1

    # First two lobatto shape functions 
    l0 =  (1.-x)/2.
    l1 =  (1.+x)/2.
    dl0 = -0.5
    dl1 = 0.5

    leg_int = leg_poly.basis(p-1).integ(m=1,lbnd=-1)
    lk = np.sqrt((2.*p-1.)/2.)*leg_int(x)

    leg = leg_poly.basis(p-1)
    dl = np.sqrt((2.*p-1.)/2.)*dxdxi*leg(x)

    num = l0*l1*dl - lk*(l1*dl0*dxdxi+l0*dl1*dxdxi)
    den = (l0*l1)**2

    kernel = num / (1.e-12+den)

    return kernel


def get_internal_grad(p, index, gl,l):

    gphi_i = np.zeros([l.shape[0],len(index)])

    c=0
    for i in range(3,p+1):
        c += i-2
    
    n = np.zeros([c,2])
    n1 = np.arange(1,p-1,1)
    n2 = np.arange(1,p-1,1)
    k = 0
    for i in range(len(n1)):
        for j in range(len(n2)):
            if n1[i] + n2[j] <= p-1:
                n[k,0] = n1[i]
                n[k,1] = n2[j]
                k+=1

    for m in range(c):
        dl2l3_1 = n[m,0]*l[:,1]**(n[m,0]-1)*l[:,2]**n[m,1]*gl[:,1]
        dl2l3_2 = n[m,1]*l[:,2]**(n[m,1]-1)*l[:,1]**n[m,0]*gl[:,2]
        gphi_i[:,m] = gl[:,0]*l[:,1]**n[m,0]*l[:,2]**n[m,1]+l[:,0]*(dl2l3_1+dl2l3_2)

    return gphi_i


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

        xn, _ = self.equidistant_nodes(p)

        get_modal_basis_tri(quad_pts, p, xn, basis_val)

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
        dim = self.dim
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

        xn, _ = self.equidistant_nodes(p)

        get_modal_grad_tri(quad_pts, p, xn, basis_grad)

        basis_grad = 2.*basis_grad

        # self.basis_grad = basis_grad

        return basis_grad
