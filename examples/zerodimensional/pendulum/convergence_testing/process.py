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

# Read in reference solution
dt = np.array([0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 
    0.00390625, 0.001953125, 9.765625e-4, 4.8828125e-4])

# sol_bdf1  = np.zeros([dt.shape[0], 1])
sol_trap  = np.zeros([dt.shape[0], 1])
#sol_rk4  = np.zeros([dt.shape[0], 1])

for idt in range(len(dt)):
    # sol_rk4[idt] =     read_data_file('RK4/' +str(idt)+'.pkl')
    # sol_bdf1[idt] = read_data_file('BDF1/'+str(idt)+'.pkl')
    sol_trap[idt] = read_data_file('Trapezoidal/'+str(idt)+'.pkl')

# Exact solution
g = 9.81
l = 0.6
ref = 0.1745*np.cos(np.sqrt(g/l)*6.0)

# err_rk4 = np.abs(sol_rk4 - ref)
# err_bdf1 = np.abs(sol_bdf1 - ref)
err_trap = np.abs(sol_trap - ref)


dt2 = np.array([dt[0],dt[-1]])
fac = dt2[1]/dt2[0]
# m1_slope = np.array([err_bdf1[0, 0]*1.5, err_bdf1[0, 0]*1.5*fac**1])
m2_slope = np.array([err_trap[0, 0]*1.5, err_trap[0, 0]*1.5*fac**2])
# m4_slope = np.array([err_rk4[0, 0]*1.5, err_rk4[0, 0]*1.5*fac**4])

al1 = 0.5
for i in range(sol_trap.shape[1]):
    fig, ax = plt.subplots()
    # ax.plot(dt2, m1_slope, ls='--', color='k', alpha=al1)
    ax.plot(dt2, m2_slope, ls='--', color='r', alpha=al1)
    # ax.plot(dt2, m4_slope, ls='--', color='g', alpha=al1)

    # ax.plot(dt2, err_rk4[:, i], marker='o', label='RK4', color='tab:brown')
    # ax.plot(dt, err_bdf1[:, i], marker='o', label='BDF1', color='k')
    ax.plot(dt, err_trap[:, i], marker='o', label='Trap', color='r')

    ax.set_ylabel('$\\epsilon$')
    ax.set_xlabel('$\\Delta t$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Pendulum')
    ax.legend()

# print('errors BDF1')
# print_errors(dt, err_bdf1)
print('errors Trap')
print_errors(dt, err_trap)
# print('errors RK4')
# print_errors(dt, err_rk4)

plt.show()
