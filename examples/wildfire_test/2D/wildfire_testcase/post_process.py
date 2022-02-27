import processing.post as post
import processing.plot as plot
import processing.readwritedatafiles as readwritedatafiles
import numpy as np
import matplotlib.pyplot as plt

# Read data file
fname = "state_variables_time_hist_TsSource.txt"
file = open(fname)
time = np.zeros((201,1))
rho_wood = np.zeros((201,1))
rho_water = np.zeros((201,1))
Ts = np.zeros((201,1))
i = 0
for line in file:
	if i < 201: 
		UqData = line.strip().split()
		time[i,:] = UqData[0]
		rho_wood[i,:] = UqData[2]
		rho_water[i,:] = UqData[4]
		Ts[i,:] = UqData[6]
		i = i+1
		# print(i)

# x = 5
# breakpoint()

# Density of Wood Plot
plt.figure(0)
plt.plot(time,rho_wood,label=r'$\rho_{wood}$',color='black')
plt.xlabel(r'$t\:[s]$',fontsize=12)
plt.ylabel(r'$\rho\:[kg/m^3]$',fontsize=12)
# plt.xlim([0,200])
plt.legend(fontsize=10)


# Density of Water Plot
plt.figure(1)
plt.plot(time[1:],rho_water[1:],label=r'$\rho_{water}$',color='darkblue')
plt.xlabel(r'$t\:[s]$',fontsize=12)
plt.ylabel(r'$\rho\:[kg/m^3]$',fontsize=12)
# plt.xlim([0,200])
plt.legend(fontsize=10)

# Temperature 
plt.figure(2)
plt.plot(time[1:],Ts[1:],label=r'$T_s$',color='darkorange')
plt.xlabel(r'$t\:[s]$',fontsize=12)
plt.ylabel(r'$T\:[K]$',fontsize=12)
# plt.xlim([0,200])
plt.legend(fontsize=10)

plot.show_plot(0)
plot.show_plot(1)
plot.show_plot(2)

# solver = readwritedatafiles.read_data_file(fname)

# # Unpack
# mesh = solver.mesh
# physics = solver.physics

# # Compute L2 error
# post.get_error(mesh, physics, solver, "Density")

# ''' Plot '''
# # Density contour
# plot.prepare_plot(linewidth=0.5)
# plot.plot_solution(mesh, physics, solver, "Density", plot_numerical=True, 
# 		plot_exact=False, plot_IC=False, create_new_figure=True, fmt='bo', 
# 		legend_label="DG", include_mesh=True, regular_2D=True, 
# 		show_elem_IDs=True)

# ### Line probe (y = 1) ###
# plot.prepare_plot(close_all=False, linewidth=1.5)
# # Parameters
# xy1 = [-5.,1.]; xy2 = [5.,1.]
# # Initial condition
# plot.plot_line_probe(mesh, physics, solver, "Density", xy1=xy1, xy2=xy2, 
# 		plot_numerical=False, plot_exact=False, plot_IC=True, 
# 		create_new_figure=True, ylabel=None, vs_x=True, fmt="k-.", 
# 		legend_label=None)
# # Exact solution
# plot.plot_line_probe(mesh, physics, solver, "Density", xy1=xy1, xy2=xy2, 
# 		plot_numerical=False, plot_exact=True, plot_IC=False, 
# 		create_new_figure=False, fmt="k-", legend_label=None)
# # DG solution
# plot.plot_line_probe(mesh, physics, solver, "Density", xy1=xy1, xy2=xy2, 
# 		plot_numerical=True, plot_exact=False, plot_IC=False, 
# 		create_new_figure=False, fmt="bo", legend_label=None)
# # Save figure
# plot.save_figure(file_name='line', file_type='pdf', crop_level=2)

# plot.show_plot()
