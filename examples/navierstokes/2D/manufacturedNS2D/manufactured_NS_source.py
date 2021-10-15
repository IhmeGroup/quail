'''
Script to generate manufactured solutions for Euler and Navier-Stokes in 2D.
---------------------
Follow this reference:
Navah, F., & Nadarajah, S. (2018).
A comprehensive high-order solver verification methodology for free fluid flows.
Aerospace Science and Technology, 80, 101–126.
https://doi.org/10.1016/j.ast.2018.07.006
---------------------
Author: Kihiro Bando
Date: 2020/03/04
'''


import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from sympy.codegen.ast import Assignment


x, y = symbols('x1 x2')
rho0, u0, v0, p0 = symbols('rho0 u0 v0 p0')
gam, R, mu, kap = symbols('gamma R mu kappa')

rhox, rhoy, rhoxy = symbols('rhox rhoy rhoxy')
ux, uy, uxy = symbols('ux uy uxy')
vx, vy, vxy = symbols('vx vy vxy')
px, py, pxy = symbols('px py pxy')

arhox, arhoy, arhoxy = symbols('arhox arhoy arhoxy')
aux, auy, auxy = symbols('aux auy auxy')
avx, avy, avxy = symbols('avx avy avxy')
apx, apy, apxy = symbols('apx apy apxy')

#===============================================================================
# here you define spefic values you want for the parameters

rho0, u0, v0, p0 = 1., 2., 2., 10.

rhox, rhoy, rhoxy = 0.1, -0.2, 0.1
ux, uy, uxy = 0.3, 0.3, 0.3
vx, vy, vxy = 0.3, 0.3, 0.3
px, py, pxy = 1.0, 1.0, 0.5

arhox, arhoy, arhoxy = 1., 1., 1.
aux, auy, auxy = 3.0, 1.0, 1.0
avx, avy, avxy = 1.0, 1.0, 1.0
apx, apy, apxy = 2.0, 1.0, 1.0
#===============================================================================

rho = rho0 + rhox*sin(arhox*pi*x) + rhoy*cos(arhoy*pi*y) +\
    rhoxy*cos(arhoxy*pi*x)*cos(arhoxy*pi*y)
u = u0 + ux*sin(aux*pi*x) + uy*cos(auy*pi*y) + uxy*cos(auxy*pi*x)*cos(auxy*pi*y)
v = v0 + vx*cos(avx*pi*x) + vy*sin(avy*pi*y) + vxy*cos(avxy*pi*x)*cos(avxy*pi*y)
p = p0 + px*cos(apx*pi*x) + py*sin(apy*pi*y) + pxy*cos(apxy*pi*x)*cos(apxy*pi*y)

rhoE = p/(gam-1) + Rational(1,2) * rho*(u**2 + v**2)
T = p / (rho*R)

c1 = Rational(4, 3)
c2 = Rational(2, 3)
tauxx = mu*(c1*diff(u, x) - c2*diff(v, y))
tauxy = mu*(diff(u, y) + diff(v, x))
tauyy = mu*(c1*diff(v, y) - c2*diff(u, x))

# compute RHSs
RHS_rho = - diff(rho*u, x) - diff(rho*v, y)
RHS_rhou = - diff(rho*u*u+p, x) - diff(rho*u*v, y) \
           + diff(tauxx, x) + diff(tauxy, y)
RHS_rhov = - diff(rho*u*v, x) - diff(rho*v*v+p, y) \
           + diff(tauxy, x) + diff(tauyy, y)
RHS_rhoE = - diff((rhoE+p)*u, x) - diff((rhoE+p)*v, y) \
           + diff(tauxx*u + tauxy*v, x) + diff(tauxy*u + tauyy*v, y) \
           + diff(kap*T, x, 2) + diff(kap*T, y, 2)

# compute state gradients in case you want to specify it directly
rho_x, rho_y = diff(rho, x), diff(rho, y)
rhou_x, rhou_y = diff(rho*u, x), diff(rho*u, y)
rhov_x, rhov_y = diff(rho*v, x), diff(rho*v, y)
rhoE_x, rhoE_y = diff(rhoE, x), diff(rhoE, y)

# write conservative variables
U_rho, U_rhou, U_rhov, U_rhoE = symbols('Uq[irho], Uq[irhou], Uq[irhov], Uq[irhoE]')
code = pycode(Assignment(U_rho, rho)).replace('math', 'np') + '\n\n'
code += pycode(Assignment(U_rhou, rho*u)).replace('math', 'np') + '\n\n'
code += pycode(Assignment(U_rhov, rho*v)).replace('math', 'np') + '\n\n'
code += pycode(Assignment(U_rhoE, rhoE)).replace('math', 'np') + '\n\n'
with open('init.py', 'w') as f:
    f.write(code)

# write forcing terms (not the sign!)
S1, S2, S3, S4 = symbols('S_rho, S_rhou, S_rhov, S_rhoE')
code = pycode(Assignment(S1, -RHS_rho)).replace('math', 'np') + '\n\n'
code += pycode(Assignment(S2, -RHS_rhou)).replace('math', 'np') + '\n\n'
code += pycode(Assignment(S3, -RHS_rhov)).replace('math', 'np') + '\n\n'
code += pycode(Assignment(S4, -RHS_rhoE)).replace('math', 'np') + '\n\n'
with open('source.py', 'w') as f:
    f.write(code)

# write state gradients
#gUx_rho, gUx_rhou, gUx_rhov, gUx_rhoE = symbols('gUx[0] gUx[1] gUx[2] gUx[3]')
#gUy_rho, gUy_rhou, gUy_rhov, gUy_rhoE = symbols('gUy[0] gUy[1] gUy[2] gUy[3]')
#code = ccode(Assignment(gUx_rho, rho_x)) + '\n\n'
#code += ccode(Assignment(gUx_rhou, rhou_x)) + '\n\n'
#code += ccode(Assignment(gUx_rhov, rhov_x)) + '\n\n'
#code += ccode(Assignment(gUx_rhoE, rhoE_x)) + '\n\n'
#code += ccode(Assignment(gUy_rho, rho_y)) + '\n\n'
#code += ccode(Assignment(gUy_rhou, rhou_y)) + '\n\n'
#code += ccode(Assignment(gUy_rhov, rhov_y)) + '\n\n'
#code += ccode(Assignment(gUy_rhoE, rhoE_y)) + '\n\n'
#with open('grad.cpp', 'w') as f:
#    f.write(code)

# plot the primitives

rhonp = lambdify([x, y], rho, 'numpy')
unp = lambdify([x, y], u, 'numpy')
vnp =  lambdify([x, y], v, 'numpy')
pnp =  lambdify([x, y], p, 'numpy')

x = np.linspace(0, 1, num=101, endpoint=True)
X, Y = np.meshgrid(x, x)

fig, ax = plt.subplots(1,1)
contour = ax.contourf(X, Y, rhonp(X,Y))
fig.colorbar(contour)

fig, ax = plt.subplots(1,1)
contour = ax.contourf(X, Y, unp(X,Y))
fig.colorbar(contour)

fig, ax = plt.subplots(1,1)
contour = ax.contourf(X, Y, vnp(X,Y))
fig.colorbar(contour)

fig, ax = plt.subplots(1,1)
contour = ax.contourf(X, Y, pnp(X,Y))
fig.colorbar(contour)

plt.show()
