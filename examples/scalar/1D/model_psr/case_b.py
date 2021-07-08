import code
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

font = {'family' : 'serif',
        'serif'  : 'Times',
        'size'   : 14}

plt.rc('font', **font)
rc('text',usetex=True)
def extract_data(filename):
    infile = open(filename, 'r')
    endLine = infile.readlines()
    #infile.readline() # skip the first line
    time = []
    pmax = []
    jac = []
    for line in endLine:
        data = line.split()
        time.append(float(data[0]))
        pmax.append(float(data[2]))
        jac.append(float(data[4]))
    infile.close()

    time = np.array(time)
    pmax = np.array(pmax)
    jac = np.array(jac)
    return time, pmax, jac

fig, ax = plt.subplots()

plt.ylabel('$T$')
plt.xlabel('$t$')
#Da = 80.
Da =1.

time, T , jac= extract_data('ana_jac.txt')
ax.plot(time/Da, T, color='k', label='Analytical')
time, T, jac = extract_data('num_jac.txt')
ax.plot(time, T, ls='-.', color='r', label='Numerical')

ax.legend()
plt.show()
