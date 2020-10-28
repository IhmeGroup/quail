import sys; sys.path.append('../../../../src'); sys.path.append('./src')
import physics.chemistry.functions as chem_fcns
import physics.chemistry.chemistry as chemistry

import numpy as np
import code
import matplotlib.pyplot as plt
from matplotlib import rc

font = {'family' : 'serif',
		'serif'  : 'Times',
        'size'   : 14}

plt.rc('font', **font)
rc('text',usetex=True)


def get_temperature_jacobian(gam, qo, R, U):
		
		# irho, irhou, irhoE, irhoY = physics.get_state_indices()
		
		dTdU = np.zeros([U.shape[0], U.shape[-1], U.shape[-1]])

		rho = U[:, 0]
		rhou = U[:, 1]
		rhoE = U[:, 2]
		rhoY = U[:, 3]

		E = rhoE/rho
		Y = rhoY/rho
		u = rhou/rho

		gamR = (gam - 1.) / R
		dTdU[:, 3, 0] = (gamR / rho) * (-1.*E + u**2 + qo*Y)
		dTdU[:, 3, 1] = (gamR / rho) * (-1.*u)
		dTdU[:, 3, 2] = gamR / rho
		dTdU[:, 3, 3] = -1.*qo * (gamR / rho)

		return dTdU # [nq, ns, ns]

nelem = 2000
fig, ax = plt.subplots()
fig2, ax2 = plt.subplots()

A = 16418.
b = 0.
Tign=25.
source = chem_fcns.Arrhenius(A=16418., b=0.0, Tign=25.)
physics = chemistry.Chemistry1D

rho = np.linspace(0.1,2.,nelem)
rhoE = 50.*np.ones(nelem)
rhou = 2.*np.ones(nelem)
rhoY = np.ones(nelem)
U = np.zeros([nelem,4])

U[:,0] = rho
U[:,1] = rhou
U[:,2] = rhoE
U[:,3] = rhoY

qo = 25.
gam = 1.4
R = 1.

p = (gam - 1.)*(rhoE - 0.5*rhou*rhou/rho - qo*rhoY)
T = p/(rho*R)

# Get numerical source Jacobian wrt density
K = A * np.exp(-Tign / T)
S = -K*rhoY

dS = np.diff(S)
drho=np.diff(rho)

dSdrho_num = dS/drho

# Get analytical source Jacobian wrt density
jac = np.zeros([U.shape[0], U.shape[-1], U.shape[-1]])

dTdU = get_temperature_jacobian(gam, qo, R, U)

dT = np.diff(T)
dTdrho_num = dT/drho
ax2.plot(rho[0:nelem-1],dTdrho_num, marker='o', label='Num. Jac')
ax2.plot(rho, dTdU[:,3,0], label='Ana Jac')
ax2.legend()


dKdrho =  (A * Tign * np.exp(-Tign / T) * dTdU[:, 3, 0] ) / T**2 
jac[:, 3, 0] =  (-1.*dKdrho * U[:, 3])

ax.plot(rho[0:nelem-1], dSdrho_num, marker='o', label='Num. Jac')
ax.plot(rho, jac[:,3,0], label='Analytic Jac')
ax.set_xlabel(r'$\rho$')
ax.set_ylabel(r'$\frac{\partial S}{\partial\left(\rho \right)}$')

ax.legend()
plt.show()
