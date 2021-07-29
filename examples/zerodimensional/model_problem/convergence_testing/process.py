'''
Script that takes in solution files for various time integration schemes and
plots their order of convergence (error vs time)

Author: Brett Bornhoft
Date: 07/29/2021
'''
import numpy as np
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

# Set dt array
dt = np.array([0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 
    0.00390625, 0.001953125, 9.765625e-4, 4.8828125e-4])

sol_bdf1  = np.zeros([dt.shape[0], 1])
sol_trap  = np.zeros([dt.shape[0], 1])
sol_rk4   = np.zeros([dt.shape[0], 1])

# Read in solution files
for idt in range(len(dt)):  
    sol_rk4[idt] =  read_data_file(f'RK4/{idt}.pkl')
    sol_bdf1[idt] = read_data_file(f'BDF1/{idt}.pkl')
    sol_trap[idt] = read_data_file(f'Trapezoidal/{idt}.pkl')

# Exact solution
ref = np.exp(-1.)

# Set absolute error
err_rk4 = np.abs(sol_rk4 - ref)
err_bdf1 = np.abs(sol_bdf1 - ref)
err_trap = np.abs(sol_trap - ref)

# Get slopes for reference order lines
dt2 = np.array([dt[0],dt[-1]])
fac = dt2[1]/dt2[0]
m1_slope = np.array([err_bdf1[0, 0]*1.5, err_bdf1[0, 0]*1.5*fac**1])
m2_slope = np.array([err_trap[0, 0]*1.5, err_trap[0, 0]*1.5*fac**2])
m4_slope = np.array([err_rk4[0, 0]*1.5, err_rk4[0, 0]*1.5*fac**4])

# Plot errors
al1 = 0.5 # Sets opacity for reference lines
for i in range(1):
    fig, ax = plt.subplots()
    ax.plot(dt2, m1_slope, ls='--', color='k', alpha=al1)
    ax.plot(dt2, m2_slope, ls='--', color='tab:pink', alpha=al1)
    ax.plot(dt2, m4_slope, ls='--', color='g', alpha=al1)

    ax.plot(dt, err_rk4[:, i], marker='o', label='RK4', color='g')
    ax.plot(dt, err_bdf1[:, i], marker='o', label='BDF1', color='k')
    ax.plot(dt, err_trap[:, i], marker='o', label='Trap', color='tab:pink')


    ax.set_ylabel('$\\epsilon$')
    ax.set_xlabel('$\\Delta t$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(1e-16, 10.)
    ax.set_title('Model Problem')
    ax.legend()

# Print order of accuracy to screen
print('errors BDF1')
print_errors(dt, err_bdf1)
print('errors Trap')
print_errors(dt, err_trap)
print('errors RK4')
print_errors(dt, err_rk4)

plt.show()