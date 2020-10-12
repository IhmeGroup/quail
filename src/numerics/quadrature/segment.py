# ------------------------------------------------------------------------ #
#
#       File : numerics/quadrature/segment.py
#
#       Contains functions to evaluate quadrature for segment shapes
#      
# ------------------------------------------------------------------------ #
import numpy as np

import general


def get_quadrature_points_weights(order, quad_type, colocated_pts=None):
    '''
    Depending on the general.QuadratureType enum, calculate the appropriate 
    function to obtain quadrature points and weights

    Inputs:
    ------- 
        order: solution order
        quad_type: Enum that points to the appropriate quadrature calc
        colocated_pts: [OPTIONAL] number of points if forcing nodes to be 
            equal to quad_pts is turned on

    Outputs:
    --------
        qpts: quadrature point coordinates [nq, dim]
        qwts: quadrature weights [nq, 1]
    '''
    if quad_type == general.QuadratureType.GaussLegendre:
        qpts, qwts = get_quadrature_gauss_legendre(order)
    elif quad_type == general.QuadratureType.GaussLobatto:
        qpts, qwts = get_quadrature_gauss_lobatto(order, colocated_pts)
    else:
        raise NotImplementedError

    return qpts, qwts


def get_quadrature_gauss_legendre(order):
    '''
    Calculate the quadrature points and weights using Gauss-Legendre rules

    Inputs:
    ------- 
        order: solution order

    Outputs:
    --------
        qpts: quadrature point coordinates [nq, dim]
        qwts: quadrature weights [nq, 1]
    '''
    if order % 2 == 0: # if order is even, add 1
        order += 1

    npts = (order + 1)//2

    # use built-in numpy Gauss Legendre functions
    qpts, qwts = np.polynomial.legendre.leggauss(npts)

    qpts.shape = -1,1
    qwts.shape = -1,1

    return qpts, qwts # [nq, 1] and [nq, 1]


def get_quadrature_gauss_lobatto(order, colocated_pts=None):
    '''
    Calculate the quadrature points and weights using Gauss Lobatto rules

    Inputs:
    ------- 
        order: solution order 
        colocated_pts: [OPTIONAL] number of points if forcing nodes to be 
            equal to quad_pts is turned on

    Outputs:
    --------
        qpts: quadrature point coordinates [nq, dim]
        qwts: quadrature weights [nq, dim]
    '''
    if order == 1: # Gauss Lobatto not defined for order = 1
        qpts, qwts = get_quadrature_points_weights(order, 
            general.QuadratureType["GaussLegendre"])
    else:
        if colocated_pts != None:
            # get order from colocated_pts -> pass order.
            order = 2*colocated_pts - 3
            qpts, qwts = gauss_lobatto(order)
        else:
            # use order argument in function call
            qpts, qwts = gauss_lobatto(order)

        qpts=qpts.reshape(qpts.shape[0],1)
        qwts=qwts.reshape(qwts.shape[0],1)

    return qpts, qwts # [nq, 1] and [nq, 1]


def gauss_lobatto(order):
    '''
    Evaluate quadrature with Gauss Lobatto rules

    Inputs:
    ------- 
        order: solution order

    Outputs:
    --------
        qpts: quadrature point coordinates [nq,]
        qwts: quadrature weights [nq,]
    '''
    if order % 2 == 0: # if order is even, add 1
        order += 1
    npts = int((order + 3)/2)
    qpts, qwts = get_lobatto_pts_wts(npts-1, 1e-10)

    return qpts, qwts


def get_lobatto_pts_wts(n, eps):
    '''
    Computes the Lobatto nodes and weights given n-1 quadrature points. This
    method is adapted from the following reference.

    Ref: Greg von Winckel (2020). Legende-Gauss-Lobatto nodes and weights 
        (https://www.mathworks.com/matlabcentral/fileexchange/4775-legende-
        gauss-lobatto-nodes-and-weights), MATLAB Central File Exchange. 
        Retrieved October 12, 2020.

    Inputs:
    -------
        n: number of quadrature points minus one
        eps: error tolerance for iterative scheme

    Outputs:
    --------
        qpts: quadrature point coordinates [nq, dim]
        qwts: quadrature weights [nq, dim]
    '''
    leg_poly = np.polynomial.legendre.Legendre

    ind = np.arange(n+1)
    qwts = np.zeros((n+1,))
    qpts = np.zeros((n+1,))
    L = np.zeros((n+1,n+1))

    # use the Chebyshev-Gauss-Lobatto nodes as the first guess
    qpts = -np.cos(np.pi*ind / n)

    for i in range(100):

        qpts_old = qpts
        vander = np.polynomial.legendre.legvander(qpts, n)
        
        # iterative evaluation to get roots of the polynomial
        qpts = qpts_old - ( qpts*vander[:,n] - vander[:,n-1] )/( (n+1)*vander[:,n]) 

        if (max(abs(qpts - qpts_old).flatten()) < eps ):
            break     
    # construct legendre polynomials from quadrature points
    for it in range(n+1):
        L[:,it] = leg_poly.basis(it)(qpts)

    qwts = 2.0 / ( (n*(n+1))*(L[:,n]**2))

    return qpts, qwts # [nq, 1] and [nq, 1]