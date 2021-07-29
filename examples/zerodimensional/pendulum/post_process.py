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
    '''
    Extracts needed data from file

    Inputs:
    -------
        filename: string for filename

    Outputs:
    --------
        data1: first data entry
        data2: second data entry
    '''
    infile = open(filename, 'r')
    endLine = infile.readlines()

    data1 = []
    data2 = []
    for line in endLine:
        data = line.split()
        data1.append(float(data[0]))
        data2.append(float(data[2]))
    infile.close()

    data1 = np.array(data1)
    data2 = np.array(data2)
    return data1, data2

fig, ax = plt.subplots()

plt.ylabel('$\\theta$')
plt.xlabel('$t$')
time, T = extract_data('time_hist.txt')
ax.scatter(time, T,color='k')
ax.legend()
plt.show()
