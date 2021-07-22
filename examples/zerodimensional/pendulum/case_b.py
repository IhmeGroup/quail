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
plt.xlabel('$t$')
#Da = 80.
# Da =1.
#Da = 15.89
#time, T = extract_data('ref_caseb_update.txt')
#time, T= extract_data('time_refinement/time_accuracy/ref_2p5.txt')
#ax.scatter(time/Da, T, label='Ref', color = 'k')
#time, T = extract_data('strang_caseb.txt')
#ax.plot(time/Da, T, label='Strang ($h=83.3$)', color = 'b')
#time, T = extract_data('strang_caseb.txt')
#ax.plot(time/Da, T, label='Strang (Trap)')


#time, T = extract_data('simpler_caseb.txt')
#ax.plot(time/Da, T, label='Simpler', color = 'g', linestyle='-.')
#time, T = extract_data('simpler_switched_sources.txt')
#ax.plot(time/Da, T, label='Simpler (switched sources)', linestyle='-.')
#time, T = extract_data('strang_switched_sources.txt')
#ax.plot(time/Da, T, label='Strang (switched sources)')
#time, T = extract_data('ader_p2_fullimplicit.txt')
#ax.plot(time/Da, T, label='ADERP2 (ref)')
#time, T = extract_data('strang_lower_timestep.txt')
#ax.plot(time/Da, T,label='Strang ($h=16.66$)')
#time, T = extract_data('ader_16p66.txt')
#ax.plot(time/Da, T, linestyle='-.',label='ADER ($h=16.66$)')imp
#time, T = extract_data('caseb_p1.txt')
#ax.scatter(time/Da, T, color='r',label='$p=1$')
#time, T = extract_data('caseb_p2.txt')
#ax.scatter(time/Da, T, color='g', label='$p=2$')
#time, T = extract_data('caseb_p3.txt')
#ax.scatter(time/Da, T, color='b',label='$p=3$')
#time, T = extract_data('caseb_p4.txt')
#ax.scatter(time/Da, T, color='m',label='$p=4$')
#time, T = extract_data('caseb_p3_split.txt')
#ax.scatter(time/Da, T, color='g',label='$p=3$, split')

#time, T = extract_data('caseb_p5.txt')
#ax.scatter(time/Da, T, color='c',label='$p=5$')
time, T = extract_data('time_hist.txt')
ax.scatter(time, T,color='k')
#time, T = extract_data('caseb_p5_split.txt')
#ax.scatter(time/Da, T, label='ADER $p=5$')

#time, T = extract_data('caseb_p7_split.txt')
#ax.scatter(time/Da, T, label='ADER $p=7$')



#time, T = extract_data('caseb_ode_83p3.txt')
#ax.scatter(time/Da, T, color='k',label='ODE')

ax.legend()
plt.show()
