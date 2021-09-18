import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc


font = {'family' : 'serif',
        'serif'  : 'Times',
        'size'   : 14}

plt.rc('font', **font)
rc('text',usetex=True)

'''
Case: Isentropic Vortex
    EndTime = 1.0 (s)
'''
def print_errors(N, errors):
    for i in range(errors.shape[0]-1):
        err = np.log(errors[i+1]/errors[i]) / np.log(N[i+1]/N[i])
        print(err)
def array_errors(N, errors):
    err=np.zeros([N.shape[0]-1])
    for i in range(errors.shape[0]-1):
        err[i] = np.log(errors[i+1]/errors[i]) / np.log(N[i+1]/N[i])
    return err

Nelem = (np.array([16., 32., 64., 128., 256.]))

Nelem_init = (np.array([16., 32., 64., 128., 256., 512., 1024.]))

'''
Case: DG Solver
'''
err_p1 = np.array([0.029410722395493,
				   0.013241771576863,
				   0.004230881944153,
				   0.000788835919487,
				   0.000144233117381
				   ])

err_p2 = np.array([0.011100682219852,
				   0.001675823936838,
				   0.000140868606561,
				   0.000011993682687,
				   0.000001248680145
				   ])

err_p3 = np.array([0.003052986853586,
				   0.000175231208591,
				   0.000005109013096,
				   0.000000301873198,
				   0.000000020600341
				   ])

err_p4 = np.array([0.000776304447595,
				   0.000011239264323,
				   0.000000306361576,
				   0.000000009393208,
				   0.000000000289345

				   ])

'''
Case: ADERG Solver
'''
ader_err_p1 = np.array([0.029406152914930,
	0.013218062663511,
	0.004203674369099,
	0.000801303050325,
	0.000147419773299])


ader_err_p2 = np.array([0.011103674180358,
	0.001688350094041,
	0.000167159467530,
	0.000020948051834,
	0.000001903585546])

ader_err_p3 = np.array([0.003078249233384,
	0.000183218147349,
	0.000006151190609,
	0.000000306333846,
	0.000000021319655])

ader_err_p4 = np.array([0.000785168522130, # dt = 1e-3
						0.000012239318477, # dt = 1e-3
						0.000000578969011, # dt = 1e-3
						0.000000011289942, # dt = 5e-4
						0.000000000307912, # dt = 1e-4
						])

print('errors p=1')
print_errors(Nelem, err_p1)
print('error p=2')
print_errors(Nelem, err_p2)
print('errors p=3')
print_errors(Nelem, err_p3)
print('errors p=4')
print_errors(Nelem, err_p4)

print('ader errors p=1')
print_errors(Nelem, ader_err_p1)
print('ader errors p=2')
print_errors(Nelem, ader_err_p2)
print('ader errors p=3')
print_errors(Nelem, ader_err_p3)
print('ader errors p=4')
print_errors(Nelem, ader_err_p4)

rate_p1 = array_errors(Nelem, err_p1)
rate_p2 = array_errors(Nelem, err_p2)
rate_p3 = array_errors(Nelem, err_p3)
rate_p4 = array_errors(Nelem, err_p4)

Nelem2 = np.array([Nelem[0], Nelem[-1]])

fac = Nelem2[0]/Nelem2[1]
m2_slope = np.array([err_p1[0]*3, err_p1[0]*3*fac**2])
m3_slope = np.array([err_p2[0]*3, err_p2[0]*3*fac**3])
m4_slope = np.array([err_p3[0]*3, err_p3[0]*3*fac**4])
m5_slope = np.array([err_p4[0]*3, err_p4[0]*3*fac**5])
#
fig, ax = plt.subplots()

ax.plot(Nelem, err_p1, marker='o', label='$p=1$')
ax.plot(Nelem2, m2_slope, ls='--', color='k')
ax.plot(Nelem, err_p2, marker='o', label='$p=2$')
ax.plot(Nelem2, m3_slope, ls='--', color='k')
ax.plot(Nelem, err_p3, marker='o', label='$p=3$')
ax.plot(Nelem2, m4_slope, ls='--', color='k')
ax.plot(Nelem, err_p4, marker='o', label='$p=4$')
ax.plot(Nelem2, m5_slope, ls='--', color='k')
#
ax.grid(b=None, which='major', axis='both',linestyle='-.',color='lightgrey')
#
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$N_{e}$')
ax.set_ylabel(r'$||\varepsilon_u||_2$')
ax.legend()
##fig.savefig('conv_1', format='pdf', crop_level=2)
#
fig, ax = plt.subplots()
ax.set_ylabel(r'Convergence Rate')
ax.set_xlabel(r'$\sqrt{N_{e}}$')
dummy1,=plt.plot(Nelem[1:], np.abs(rate_p1), ls='-.', ms='8', marker='o', color='k', label='$p=1,\ q=1$')
# dummy2,=plt.plot(NelemQ22[1:], np.abs(rate_p1q2), ls='-.', ms='8', marker='o', color='k', fillstyle='none',label='$p=1,\ q=2$')
#
pltp1,=plt.plot(Nelem[1:], np.abs(rate_p1), ls='-.', ms='8', marker='o', color='tab:blue', label='$p=1,\ q=1$')
pltp2,=plt.plot(Nelem[1:], np.abs(rate_p2), ls='-.', ms='8', marker='o', color='tab:orange', label='$p=2,\ q=1$')
pltp3,=plt.plot(Nelem[1:], np.abs(rate_p3), ls='-.', ms='8', marker='o', color='tab:green', label='$p=3,\ q=1$')
pltp4,=plt.plot(Nelem[1:], np.abs(rate_p4), ls='-.', ms='8',  marker='o', color='tab:purple', label='$p=4,\ q=1$')

# leg1 = plt.legend([pltp1, pltp2, pltp3], ['$p=1$', '$p=2$', '$p=3$'], ncol=1, markerscale=0.02)

plt.plot(Nelem2, [2., 2.], ls='-', color='tab:blue')
plt.plot(Nelem2, [3., 3.], ls='-', color='tab:orange')
plt.plot(Nelem2, [4., 4.], ls='-', color='tab:green')
plt.plot(Nelem2, [5., 5.], ls='-', color='tab:purple')

plt.grid(b=None, which='major', axis='both',linestyle='-.',color='lightgrey')

fig, ax = plt.subplots()
ax.plot(Nelem, ader_err_p1, marker='o', label='$p=1$')
ax.plot(Nelem2, m2_slope, ls='--', color='k')
ax.plot(Nelem, ader_err_p2, marker='o', label='$p=2$')
ax.plot(Nelem2, m3_slope, ls='--', color='k')
ax.plot(Nelem, ader_err_p3, marker='o', label='$p=3$')
ax.plot(Nelem2, m4_slope, ls='--', color='k')
ax.plot(Nelem, ader_err_p4, marker='o', label='$p=4$')
ax.plot(Nelem2, m5_slope, ls='--', color='k')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$N_{e}$')
ax.set_ylabel(r'$||\varepsilon_u||_2$')
ax.legend()
# plt.gca().add_artist(leg1)
#
##fig.savefig('conv_final.pdf', format='pdf')
#plt.savefig('conv_final.pdf', bbox_inches='tight', pad_inches=0.0)
plt.show()
