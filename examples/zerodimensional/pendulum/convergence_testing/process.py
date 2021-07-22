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
# dt_int = np.array([40, 20, 10, 5, 2, 1])
dt_ref = np.array([0, 1, 2, 3, 4, 5])

# sol_ader_p1 = np.zeros([dt.shape[0], 1])
# sol_ader_p2 = np.zeros([dt.shape[0], 1])
# sol_ader_p3 = np.zeros([dt.shape[0], 1])
# sol_ader_p4 = np.zeros([dt.shape[0], 1])
# sol_bdf1  = np.zeros([dt.shape[0], 1])
sol_trap  = np.zeros([dt.shape[0], 1])
# sol_lsoda  = np.zeros([dt.shape[0], 1])

sol_rk4  = np.zeros([dt_rk4.shape[0], 1])

    # sol_strang[idt] =  read_data_file('Strang/' +str(dt_int[idt])+'.pkl')
for idt in range(len(dt_rk4)):  
    sol_rk4[idt] =     read_data_file('RK4/' +str(idt)+'.pkl')
    # sol_ader_p1[idt] = read_data_file('ADER/p1/'+str(idt)+'.pkl')
    # sol_ader_p2[idt] = read_data_file('ADER/p2/'+str(idt)+'.pkl')
    # sol_ader_p3[idt] = read_data_file('ADER/p3/'+str(idt)+'.pkl')
    # sol_ader_p4[idt] = read_data_file('ADER/p4/'+str(idt)+'.pkl')
    # sol_bdf1[idt] = read_data_file('BDF1/'+str(idt)+'.pkl')
    sol_trap[idt] = read_data_file('Trapezoidal/'+str(idt)+'.pkl')
    # sol_lsoda[idt] = read_data_file('LSODA/'+str(idt)+'.pkl')

# Exact solution
g = 9.81
l = 0.6
ref = 0.1745*np.cos(np.sqrt(g/l)*6.0)

# import code; code.interact(local=locals())
# for idt in range(len(dt_ref)):
    # sol_ref[idt] = read_data_file('RK4/'+str(idt) + '.pkl')

# err_p1 = np.abs((sol_ader_p1 - ref))
# err_p2 = np.abs((sol_ader_p2 - ref))
# err_p3 = np.abs((sol_ader_p3 - ref))
# err_p4 = np.abs((sol_ader_p4 - ref))
# err_strang = np.abs(sol_strang - sol_ref[ref_id])/sol_ref[ref_id]
# err_rk4 = np.abs(sol_rk4 - ref)
# err_bdf1 = np.abs(sol_bdf1 - ref)
err_trap = np.abs(sol_trap - ref)
# err_lsoda = np.abs(sol_lsoda - ref)


dt2 = np.array([dt[0],dt[-1]])
fac = dt2[1]/dt2[0]
# m1_slope = np.array([err_bdf1[0, 0]*1.5, err_bdf1[0, 0]*1.5*fac**1])
m2t_slope = np.array([err_trap[0, 0]*1.5, err_trap[0, 0]*1.5*fac**2])

# m2_slope = np.array([err_p1[0, 0]*1.5, err_p1[0, 0]*1.5*fac**3])
# m3_slope = np.array([err_p2[0, 0]*1.5, err_p2[0, 0]*1.5*fac**5])
# m4_slope = np.array([err_p3[0, 0]*1.5, err_p3[0, 0]*1.5*fac**7])
# m5_slope = np.array([err_p4[0, 0]*1.5, err_p4[0, 0]*1.5*fac**9])

# title = ['$t_f=40$ s', '$t_f=80$ s', '$t_f=120$ s', '$t_f=160$ s']
al1 = 0.5

for i in range(sol_ader_p1.shape[1]):
    fig, ax = plt.subplots()
    # ax.plot(dt2, m1_slope, ls='--', color='k', alpha=al1)
    # ax.plot(dt2, m2_slope, ls='--', color='r', alpha=al1)
    ax.plot(dt2, m2t_slope, ls='--', color='tab:pink', alpha=al1)
    # ax.plot(dt2, m3_slope, ls='--', color='b', alpha=al1)
    # ax.plot(dt2, m4_slope, ls='--', color='g', alpha=al1)
    # ax.plot(dt2, m5_slope, ls='--', color='tab:orange', alpha=al1)

    # ax.plot(dt_rk4, err_p1[:, i], marker='o', label='$p=1$', color='r')
    # ax.plot(dt_rk4, err_p2[:, i], marker='o', label='$p=2$', color='b')
    # ax.plot(dt_rk4, err_p3[:, i], marker='o', label='$p=3$', color='g')
    # ax.plot(dt_rk4, err_p4[:, i], marker='o', label='$p=4$', color='tab:orange')
    # ax.plot(dt_rk4, err_rk4[:, i], marker='o', label='RK4', color='tab:brown')
    # ax.plot(dt_rk4, err_bdf1[:, i], marker='o', label='BDF1', color='k')
    ax.plot(dt_rk4, err_trap[:, i], marker='o', label='Trap', color='tab:pink')
    # ax.plot(dt_rk4, err_lsoda[:, i], marker='o', label='LSODA', color='tab:gray')


    ax.set_ylabel('$\\epsilon$')
    ax.set_xlabel('$\\Delta t$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(1e-16, 10.)
    ax.set_title('Pendulum')
    ax.legend()


# err_p1_l2 = np.sqrt(np.sum((sol_ader_p1 - sol_ref[ref_id])**2, axis=1))
# err_p2_l2 = np.sqrt(np.sum((sol_ader_p2 - sol_ref[ref_id])**2, axis=1))
# err_p3_l2 = np.sqrt(np.sum((sol_ader_p3 - sol_ref[ref_id])**2, axis=1))
# err_p4_l2 = np.sqrt(np.sum((sol_ader_p4 - sol_ref[ref_id])**2, axis=1))
# err_strang_l2 = np.sqrt(np.sum((sol_strang - sol_ref[ref_id])**2, axis=1))
# err_rk4_l2 = np.sqrt(np.sum((sol_rk4 - sol_ref[ref_id])**2, axis=1))

# print('errors p=1')
# print_errors(dt_rk4, err_p1)
# print('error p=2')
# print_errors(dt_rk4, err_p2)
# print('errors p=3')
# print_errors(dt_rk4, err_p3)
# print('errors p=4')
# print_errors(dt_rk4, err_p4)
# print('errors BDF1')
# print_errors(dt_rk4, err_bdf1)
print('errors Trap')
print_errors(dt_rk4, err_trap)
# print('errors RK4')
# print_errors(dt_rk4, err_rk4)

plt.show()