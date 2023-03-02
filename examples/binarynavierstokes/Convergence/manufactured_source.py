'''
Script to generate manufactured solutions for Binary Navier-Stokes in 1D.
---------------------
Follow this reference:
Navah, F., & Nadarajah, S. (2018).
A comprehensive high-order solver verification methodology for free fluid flows.
Aerospace Science and Technology, 80, 101–126.
https://doi.org/10.1016/j.ast.2018.07.006
---------------------
'''
import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from sympy.codegen.ast import Assignment


x = symbols('x1')
p0, T0, u0, Y0 = symbols('p0 T0 u0 Y0')
dp, dT, du, dY = symbols('dp dT du dY')
R0, R1, cv0, cv1, mu0, mu1, Pr, Sc = symbols('R0 R1 cv0 cv1 mu0 mu1 Pr Sc')

#===============================================================================
# here you define spefic values you want for the parameters

p0, T0, u0, Y0 = 1e5, 300., 10., 0.5
dp, dT, du, dY = 0.1, 0.3, 0.3, 0.3
#===============================================================================

p = p0 * (1. + dp * sin(2.*pi*x))
T = T0 * (1. + dT * sin(2.*pi*x))
u = u0 * (1. + du * sin(2.*pi*x))
Y = Y0 * (1. + dY * sin(2.*pi*x))

R = R0*Y + R1*(1.-Y)
cv = cv0*Y + cv1*(1.-Y)
mu = mu0*Y + mu1*(1.-Y)

rho = p/R/T

kappa = mu/rho/Pr*cv
D = mu/rho/Sc
ph = ((R0+cv0)-(R1+cv1))*T

E = cv*T + Rational(1,2)*(u**2)
rhoE = rho*E

c1 = Rational(4, 3)
tauxx = mu*(c1*diff(u, x))

q = kappa*diff(T,x)

J = rho*D*diff(Y,x)

RHS_rho = -diff(rho*u,x)
RHS_rhou = -diff(rho*u*u+p, x) + diff(tauxx, x)
RHS_rhoE = - diff((rhoE+p)*u, x) + diff(tauxx*u,x) + diff(q,x) + diff(ph*J,x)
RHS_rhoY = -diff(rho*Y*u,x) + diff(J,x)

# write forcing terms
S1, S2, S3, S4 = symbols('S_rho, S_rhou, S_rhoE, S_rhoY')
code = pycode(Assignment(S1, -RHS_rho)).replace('math', 'np') + '\n\n'
code += pycode(Assignment(S2, -RHS_rhou)).replace('math', 'np') + '\n\n'
code += pycode(Assignment(S3, -RHS_rhoE)).replace('math', 'np') + '\n\n'
code += pycode(Assignment(S4, -RHS_rhoY)).replace('math', 'np') + '\n\n'
with open('source.py', 'w') as f:
    f.write(code)