# ------------------------------------------------------------------------ #
#
#       File : numerics/quadrature/segment.py
#
#       Contains functions to evaluate quadrature for segment shapes
#      
# ------------------------------------------------------------------------ #
import numpy as np
import general


def get_quadrature_points_weights(order, quad_type, num_pts_colocated=0):
    '''
    Depending on the general.QuadratureType enum, calculate the appropriate 
    function to obtain quadrature points and weights

    Inputs:
    ------- 
        order: solution order
        num_pts_colocated: if greater than 0, then a colocated scheme will
            be used, i.e. the quadrature points will be the same as the 
            solution nodes; this value then determines the number of 
            quadrature points

    Outputs:
    --------
        qpts: quadrature point coordinates [nq, dim]
        qwts: quadrature weights [nq, 1]
    '''
    if quad_type == general.QuadratureType.GaussLegendre:
        qpts, qwts = get_quadrature_gauss_legendre(order)
    elif quad_type == general.QuadratureType.GaussLobatto:
        qpts, qwts = get_quadrature_gauss_lobatto(order, 
                num_pts_colocated)
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


def get_quadrature_gauss_lobatto(order, num_pts_colocated):
    '''
    Calculate the quadrature points and weights using Gauss Lobatto rules

    Inputs:
    ------- 
        order: solution order 
        num_pts_colocated: if greater than 0, then a colocated scheme will
            be used, i.e. the quadrature points will be the same as the 
            solution nodes; this value then determines the number of 
            quadrature points

    Outputs:
    --------
        qpts: quadrature point coordinates [nq, dim]
        qwts: quadrature weights [nq, dim]
    '''
    if order == 1: # Gauss-Lobatto quadrature not defined for order = 1
        qpts, qwts = get_quadrature_points_weights(order, 
            general.QuadratureType["GaussLegendre"])
    else:
        if num_pts_colocated > 0:
            # Get order from num_pts_colocated -> pass order.
            order = 2*num_pts_colocated - 3
            qpts, qwts = gauss_lobatto(order)
        else:
            # use order argument in function call
            qpts, qwts = gauss_lobatto(order)

        qpts=qpts.reshape(qpts.shape[0],1)
        qwts=qwts.reshape(qwts.shape[0],1)

    return qpts, qwts # [nq, 1] and [nq, 1]


def gauss_lobatto(order):
    '''
    Evaluate quadrature with Gauss-Lobatto rules

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
    qpts, qwts = get_lobatto_pts_wts(npts-1, general.eps)

    return qpts, qwts


def get_lobatto_pts_wts(n, tol):
    '''
    Computes the Lobatto nodes and weights given n-1 quadrature points. This
    method is adapted from the following reference.

    Ref: Greg von Winckel (2020). Legende-Gauss-Lobatto nodes and weights 
        (https://www.mathworks.com/matlabcentral/fileexchange/4775-legende-
        gauss-lobatto-nodes-and-weights), MATLAB Central File Exchange. 
        Retrieved October 12, 2020.

    Copyright (c) 2009, Greg von Winckel
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are 
    met:

    * Redistributions of source code must retain the above copyright notice, 
    this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright 
    notice, this list of conditions and the following disclaimer in the 
    documentation and/or other materials provided with the distribution

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS 
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED 
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER 
    OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR 
    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF 
    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    Inputs:
    -------
        n: number of quadrature points minus one
        tol: error tolerance for iterative scheme

    Outputs:
    --------
        qpts: quadrature point coordinates [nq, dim]
        qwts: quadrature weights [nq, dim]
    '''
    leg_poly = np.polynomial.legendre.Legendre

    ind = np.arange(n+1)
    L = np.zeros([n+1, n+1])

    # Initialize to Gauss-Lobatto Chebyshev points
    qpts = -np.cos(np.pi*ind/n)

    niter = 1000
    # Iterative evaluation to get roots of Legendre polynomial derivatives
    for i in range(niter):
        qpts_old = qpts
        vander = np.polynomial.legendre.legvander(qpts, n)
        
        qpts = qpts_old - (qpts*vander[:, n] - vander[:, n-1])/(
            (n+1)*vander[:, n]) 

        # Check if tolerance is met
        if (np.amax(np.abs(qpts - qpts_old)) < tol):
            break    

        if i == niter - 1:
            # Didn't converge
            raise ValueError

    # Evaluate Legendre polynomials
    for it in range(n+1):
        L[:, it] = leg_poly.basis(it)(qpts)

    # Quadrature weights
    qwts = 2./(n*(n+1)*L[:,n]**2.)

    return qpts, qwts # [nq, 1] and [nq, 1]