Uq[irho] = 0.1*np.sin(np.pi*x1) + 0.1*np.cos(np.pi*x1)*np.cos(np.pi*x2) - 0.2*np.cos(np.pi*x2) + 1.0

Uq[irhou] = (0.1*np.sin(np.pi*x1) + 0.1*np.cos(np.pi*x1)*np.cos(np.pi*x2) - 0.2*np.cos(np.pi*x2) + 1.0)*(0.3*np.sin(3*np.pi*x1) + 0.3*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.3*np.cos(np.pi*x2) + 2.0)

Uq[irhov] = (0.1*np.sin(np.pi*x1) + 0.1*np.cos(np.pi*x1)*np.cos(np.pi*x2) - 0.2*np.cos(np.pi*x2) + 1.0)*(0.3*np.sin(np.pi*x2) + 0.3*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.3*np.cos(np.pi*x1) + 2.0)

Uq[irhoE] = (4.0*(0.15*np.sin(3*np.pi*x1) + 0.15*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.15*np.cos(np.pi*x2) + 1)**2 + 4.0*(0.15*np.sin(np.pi*x2) + 0.15*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 0.15*np.cos(np.pi*x1) + 1)**2)*(0.05*np.sin(np.pi*x1) + 0.05*np.cos(np.pi*x1)*np.cos(np.pi*x2) - 0.1*np.cos(np.pi*x2) + 0.5) + (1.0*np.sin(np.pi*x2) + 0.5*np.cos(np.pi*x1)*np.cos(np.pi*x2) + 1.0*np.cos(2*np.pi*x1) + 10.0)/(gamma - 1)

