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

    time = []
    data1 = []

    for line in endLine:
        data = line.split()
        time.append(float(data[0]))
        data1.append(float(data[2]))

    infile.close()

    time = np.array(time)
    data1 = np.array(data1)

    return time, data1

# Plots the comparison of ADERDG and Strang against a
# reference solution for the extinction and ignition model
# cases.
fig, ax = plt.subplots()

plt.ylabel('$T$')
plt.xlabel('$t/Da$')

Da = 15.89

time, T = extract_data('REF_CASEA.txt')
ax.plot(time/Da, T, color='k', label='Ref')

time, T = extract_data('ADERDG_CASEA.txt')
ax.plot(time/Da, T, ls='-.', color='r', label='ADERDG P5')

time, T = extract_data('STRANG_CASEA.txt')
ax.plot(time/Da, T, ls='-.', color='b', label='Strang')

ax.legend()


fig, ax = plt.subplots()

plt.ylabel('$T$')
plt.xlabel('$t/Da$')

Da = 80.

time, T = extract_data('REF_CASEB.txt')
ax.plot(time/Da, T, color='k', label='Ref')

time, T = extract_data('ADERDG_CASEB.txt')
ax.plot(time/Da, T, ls='-.', color='r', label='ADERDG P5')

time, T = extract_data('STRANG_CASEB.txt')
ax.plot(time/Da, T, ls='-.', color='b', label='Strang')

ax.legend()
plt.show()
