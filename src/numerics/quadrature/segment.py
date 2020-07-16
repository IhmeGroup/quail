import numpy as np
import general
import math
import code
from scipy.linalg import eigh_tridiagonal
import itertools

def get_quadrature_points_weights(order, quad_type, forced_pts=None):

	# Only Gauss-Legendre for now
	if quad_type == general.QuadratureType.GaussLegendre:
		qpts, qwts = get_quadrature_gauss_legendre(order)
	elif quad_type == general.QuadratureType.GaussLobatto:
		qpts, qwts = get_quadrature_gauss_lobatto(order, forced_pts)
	else:
		raise NotImplementedError

	return qpts, qwts

def get_quadrature_gauss_legendre(order):
	# if order is even, increment by 1
	if order % 2 == 0:
		order += 1

	npts = (order + 1)//2

	qpts, qwts = np.polynomial.legendre.leggauss(npts)

	qpts.shape = -1,1
	qwts.shape = -1,1

	return qpts, qwts

def get_quadrature_gauss_lobatto(order, forced_pts=None):

    if order == 1:
        qpts, qwts = get_quadrature_points_weights(order, general.QuadratureType["GaussLegendre"])
    else:
        if forced_pts != None:
            # Get order from forced_pts -> pass order.
            order = 2*forced_pts - 3
            qpts, qwts = gauss_lobatto(order)
        else:
            qpts, qwts = gauss_lobatto(order)

        qpts=qpts.reshape(qpts.shape[0],1)
        qwts=qwts.reshape(qwts.shape[0],1)

    return qpts, qwts

def gauss_lobatto(order):
    if order % 2 == 0:
        order += 1
    npts = int((order + 3)/2)

    alpha, beta = jacobi(npts, 0., 0.)
    qpts, qwts = get_lobatto_pts_wts(alpha, beta, -1.0, 1.0)

    return qpts, qwts


def get_lobatto_pts_wts(alpha, beta, xl1, xl2):
    """Compute the Lobatto nodes and weights with the preassigned node xl1, xl2.
    Based on the section 7 of the paper
        Some modified matrix eigenvalue problems,
        Gene Golub,
        SIAM Review Vol 15, No. 2, April 1973, pp.318--334,
    """
    from scipy.linalg import solve_banded, solve

    n = len(alpha) - 1
    en = np.zeros(n)
    en[-1] = 1
    A1 = np.vstack((np.sqrt(beta), alpha - xl1))
    J1 = np.vstack((A1[:, 0:-1], A1[0, 1:]))
    A2 = np.vstack((np.sqrt(beta), alpha - xl2))
    J2 = np.vstack((A2[:, 0:-1], A2[0, 1:]))
    g1 = solve_banded((1, 1), J1, en)
    g2 = solve_banded((1, 1), J2, en)
    C = np.array(((1, -g1[-1]), (1, -g2[-1])))
    xl = np.array((xl1, xl2))
    ab = solve(C, xl)

    alphal = alpha
    alphal[-1] = ab[0]
    betal = beta
    betal[-1] = ab[1]
    x, w = scheme_from_rc(alphal, betal)
    return x, w

def scheme_from_rc(alpha, beta):
    alpha = alpha.astype(np.float64)
    beta = beta.astype(np.float64)
    x, V = eigh_tridiagonal(alpha, np.sqrt(beta[1:]))
    w = beta[0] * V[0, :] ** 2
    return x, w

def jacobi(n, alpha, beta):
    """Generate the recurrence coefficients a_k, b_k, c_k in

    P_{k+1}(x) = (a_k x - b_k)*P_{k}(x) - c_k P_{k-1}(x)

    for the Jacobi polynomials which are orthogonal on [-1, 1] with respect to the
    weight w(x)=[(1-x)^alpha]*[(1+x)^beta]; see
    <https://en.wikipedia.org/wiki/Jacobi_polynomials#Recurrence_relations>.
    """
    iterator = Jacobi(alpha, beta)
    p0 = iterator.p0
    lst = list(itertools.islice(iterator, n))
    a = np.array([item[0] for item in lst])
    b = np.array([item[1] for item in lst])
    c = np.array([item[2] for item in lst])
    return b, c

class Jacobi:
    def __init__(self, alpha, beta):
        
        self.iterator = Monic(alpha, beta)
        self.p0 = self.iterator.p0
        return

    def __iter__(self):
        return self

    def __next__(self):
        return self.iterator.__next__()

class Monic:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.gamma = lambda x: math.gamma(float(x))

        self.frac = lambda x, y: x / y

        self.p0 = 1
        self.int_1 = (
            2 ** (alpha + beta + 1)
            * self.gamma(alpha + 1)
            * self.gamma(beta + 1)
            / self.gamma(alpha + beta + 2)
        )
        self.n = 0

    def __iter__(self):
        return self

    def __next__(self):
        frac = self.frac
        alpha = self.alpha
        beta = self.beta

        N = self.n

        a = 1

        if N == 0:
            b = frac(beta - alpha, alpha + beta + 2)
        else:
            b = frac(
                beta ** 2 - alpha ** 2,
                (2 * N + alpha + beta) * (2 * N + alpha + beta + 2),
            )

        # c[0] is not used in the actual recurrence, but is often defined as the
        # integral of the weight function of the domain, i.e.,
        # ```
        # int_{-1}^{+1} (1-x)^a * (1+x)^b dx =
        #     2^(a+b+1) * Gamma(a+1) * Gamma(b+1) / Gamma(a+b+2).
        # ```
        # Note also that we have the treat the case N==1 separately to avoid division by
        # 0 for alpha=beta=-1/2.
        if N == 0:
            c = self.int_1
        elif N == 1:
            c = frac(
                4 * (1 + alpha) * (1 + beta),
                (2 + alpha + beta) ** 2 * (3 + alpha + beta),
            )
        else:
            c = frac(
                4 * (N + alpha) * (N + beta) * N * (N + alpha + beta),
                (2 * N + alpha + beta) ** 2
                * (2 * N + alpha + beta + 1)
                * (2 * N + alpha + beta - 1),
            )
        self.n += 1
        return a, b, c