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
    for line in endLine:
        data = line.split()
        time.append(float(data[0]))
        pmax.append(float(data[2]))
    infile.close()

    time = np.array(time)
    pmax = np.array(pmax)
    return time, pmax

fig, ax = plt.subplots()

#ax.plot(time, pmax, label='$p=1, ADER$', color='k')
plt.ylabel('$T$')
plt.xlabel('$t/Da$')


time, pmax = extract_data('time_hist.txt')
ax.plot(time, pmax, label='$Test$')

ax.legend()
plt.show()
