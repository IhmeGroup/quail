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

plt.ylabel('$T$')
plt.xlabel('$t$')
#Da = 80.
Da =1.

time, T = extract_data('time_hist.txt')
ax.plot(time/Da, T,color='k')

ax.legend()
plt.show()
