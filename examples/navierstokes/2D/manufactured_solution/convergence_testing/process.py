# ------------------------------------------------------------------------ #
#
# Script that takes in solution files for various time integration schemes and
# plots their order of convergence (error vs time)
#
# ------------------------------------------------------------------------ #
import numpy as np
import importlib
import pickle
import matplotlib.pyplot as plt
from matplotlib import rc

font = {'family' : 'serif',
        'serif'  : 'Times',
        'size'   : 14}

plt.rc('font', **font)
rc('text',usetex=True)

def print_errors(N, errors):
    '''
    This functions takes in an array of time step sizes or 
    number of elements and errors corresponding to this info
    and prints our the order of accuracy between two stages 
    of refinement.

    Inputs:
    -------
        N: array of time step sizes or number for elements
        errors: array of errors corresponding to N
    '''
    for i in range(errors.shape[0]-1):
        err = np.log(errors[i+1]/errors[i]) / np.log(N[i+1]/N[i])
        print(err)


def read_data_file(fname):
    '''
    This function reads a data file (pickle format).

    Inputs:
    -------
        fname: file name (str)

    Outputs:
    --------
        solver: solver object
    '''
    # Open and get solver
    with open(fname, 'rb') as fo:
        solution = pickle.load(fo)

    return solution


# ------------------------------------------- #
# Set meshes and orders
meshx = np.array([2, 4, 8, 16, 32])
order = np.array([1, 2, 3])
# ------------------------------------------- #


err = np.zeros([order.shape[0], meshx.shape[0]])
ref = np.zeros([order.shape[0], 2])

Nelem2 = np.array([meshx[0], meshx[-1]])
fac = Nelem2[0]/Nelem2[1]

# Read in solution files
for idt in range(len(order)):  
    err[idt] =  read_data_file(f'P{order[idt]}.pkl')
    ref[idt] = np.array([err[idt, 0]*3, err[idt, 0]*3*fac**(order[idt]+1)])


# Plot errors
fig, ax = plt.subplots()
al1 = 0.5 # Sets opacity for reference lines
for i in range(order.shape[0]):
    ax.plot(Nelem2, ref[i], ls='--', color='k', alpha=al1)
    ax.plot(meshx, err[i], marker='o', label=f'p={order[i]}')

ax.set_ylabel('$||\\epsilon_\\rho||$')
ax.set_xlabel('$\sqrt{N_e}$')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title('Convergence of Navier-Stokes')
ax.legend()

# Print rate of convergence
print('')
for j in range(order.shape[0]):
    print('------------------------------')
    print(f'Errors for p={order[j]}')
    print_errors(meshx, err[j])
    print('------------------------------')
plt.show()
