# ---------------------------------------------------------------------- # 
# This function constructs a source term for a manufactured solution to 
# the N-S equations given a specified exact solution.
#
# We are using the constants and exact solution from the following ref:
# -----------------------
# Dumbser, M. (2010). Arbitrary high order PNPM schemes on 
# unstructured meshes for the compressible Navier-Stokes equations.
# -----------------------
# Author: Brett Bornhoft
# Date: 07/28/2021
# ---------------------------------------------------------------------- # 

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

def main():

    # -------------------------------------------------------------------- #
    # Construct symbolic variables
    
    # coordinate variables
    x, y, t = sp.symbols('x1 x2 t', real=True)
    # constants for exact sol
    rhob, rho0, u0, v0, pb, p0, k, om, c = \
        sp.symbols('rhob rho0 u0 v0 pb p0 k omega c', real=True)
    # properties in NS (obtained in Quail)
    gam, R, mu, kap = sp.symbols('gamma R mu kappa', real=True)

    # -------------------------------------------------------------------- #
    # define the constants for the various params
    rhob, rho0, u0, v0, pb, p0, k, om = 1.0, 0.5, 0.25, 0.25, 1/gam, \
        0.1, 2.0*sp.pi/10.0, 2.0*sp.pi

    # -------------------------------------------------------------------- #
    # Write the exact sol to the Navier-Stokes equations (for primitives)
    rho = rhob + rho0 * sp.cos(k * (x + y)- om*t)
    u = u0 * sp.sin(k * (x + y) - om*t)
    v = v0 * sp.sin(k * (x + y) - om*t)
    p = pb + p0 * sp.sin(k * (x + y) - om*t)

    # Write functions for closing the system (Dumbser paper)
    rhoE = p/(gam-1) + sp.Rational(1,2) * rho*(u**2 + v**2) 
    T = p / (rho*R)

    c1 = sp.Rational(4, 3)
    c2 = sp.Rational(2, 3)
    tauxx = mu*(c1*sp.diff(u, x) - c2*sp.diff(v, y))
    tauxy = mu*(sp.diff(u, y) + sp.diff(v, x))
    tauyy = mu*(c1*sp.diff(v, y) - c2*sp.diff(u, x))

    # Construct forcing terms (note for dumbser paper viscous fluxes
    # should be added not subtracted)
    S_rho = sp.diff(rho, t) + sp.diff(rho*u, x) + sp.diff(rho*v, y)
    S_rhou = sp.diff(rho*u, t) + sp.diff(rho*u*u+p, x) + sp.diff(rho*u*v, y) \
            + sp.diff(tauxx, x) + sp.diff(tauxy, y)
    S_rhov = sp.diff(rho*v, t) + sp.diff(rho*u*v, x) + sp.diff(rho*v*v+p, y) \
           + sp.diff(tauxy, x) + sp.diff(tauyy, y)
    S_rhoE = sp.diff(rhoE, t) + sp.diff((rhoE+p)*u, x) + sp.diff((rhoE+p)*v, y) \
           + sp.diff(tauxx*u + tauxy*v, x) + sp.diff(tauxy*u + tauyy*v, y) \
           - sp.diff(kap*T, x, 2) - sp.diff(kap*T, y, 2)

    # -------------------------------------------------------------------- #
    # Print the latex for the sources
    print_latex('S_rho', S_rho)
    print_latex('S_rhou', S_rhou)
    print_latex('S_rhov', S_rhov)
    print_latex('S_rhoE', S_rhoE)

    # Print the pythod code for the sources
    print_pycode('S_rho', S_rho)
    print_pycode('S_rhou', S_rhou)
    print_pycode('S_rhov', S_rhov)
    print_pycode('S_rhoE', S_rhoE)

    # -------------------------------------------------------------------- #
    # Take a look at the exact solution via plotting

    # # define parameters
    gam_, Pr, cv = 1.4, 0.7, 1.0
    p_eval = p.subs(gam, gam_)

    rhop = sp.lambdify([x, y, t], rho, 'numpy')
    up = sp.lambdify([x, y, t], u, 'numpy')
    vp = sp.lambdify([x, y, t], v, 'numpy')
    pp = sp.lambdify([x, y, t], p_eval, 'numpy')

    x = np.linspace(0, 10, num=101, endpoint=True)
    X, Y = np.meshgrid(x, x)

    fig, ax = plt.subplots(1, 3)
    contour = ax[0].contourf(X, Y, rhop(X, Y, 0.0))
    contour = ax[1].contourf(X, Y, rhop(X, Y, 0.25))
    contour = ax[2].contourf(X, Y, rhop(X, Y, 0.5))
    fig.colorbar(contour)


    fig, ax = plt.subplots(1, 3)
    contour = ax[0].contourf(X, Y, up(X, Y, 0.0))
    contour = ax[1].contourf(X, Y, up(X, Y, 0.25))
    contour = ax[2].contourf(X, Y, up(X, Y, 0.5))
    fig.colorbar(contour)

    fig, ax = plt.subplots(1, 3)
    contour = ax[0].contourf(X, Y, vp(X, Y, 0.0))
    contour = ax[1].contourf(X, Y, vp(X, Y, 0.25))
    contour = ax[2].contourf(X, Y, vp(X, Y, 0.5))
    fig.colorbar(contour)

    # import code; code.interact(local=locals())
    fig, ax = plt.subplots(1, 3)
    contour = ax[0].contourf(X, Y, pp(X, Y, 0.0))
    contour = ax[1].contourf(X, Y, pp(X, Y, 0.25))
    contour = ax[2].contourf(X, Y, pp(X, Y, 0.5))
    fig.colorbar(contour)




    plt.show()

def print_pycode(name, expression):
    print(f"""

    {name}:

    """)
    print(sp.pycode(expression).replace('math', 'np'))

def print_latex(name, expression):
    print(f"""

    {name}:

    """)
    print(sp.latex(expression))

if __name__ == "__main__":
    main()
