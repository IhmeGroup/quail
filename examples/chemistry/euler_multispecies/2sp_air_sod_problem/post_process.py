import processing.post as post
import processing.plot as plot
import processing.readwritedatafiles as readwritedatafiles
import numpy as np
import cantera as ct

# Read data file
fname = "Data_final.pkl"
solver = readwritedatafiles.read_data_file(fname)

# Unpack
mesh = solver.mesh
physics = solver.physics

# Flag for num vs avg
num = True
avg = False

# Set gas objects
num_elems = mesh.num_elems
nq = solver.basis.equidistant_nodes(max([1, 3*solver.order])).shape[0]

solver.physics.gas_elems = np.ndarray((num_elems, nq),
	dtype=np.object)

solver.physics.gas_elems[:, :] = ct.Solution('air_test.yaml')


''' Plot '''
### Density
plot.prepare_plot()
# Exact solution
#plot.plot_solution(mesh, physics, solver, "Density", plot_numerical=False, 
#		plot_exact=True, plot_IC=False, create_new_figure=True, 
#		fmt='k-.', legend_label="Exact")
# DG solution
# plot.plot_solution(mesh, physics, solver, "Density", plot_numerical=num, 
# 		plot_exact=False, plot_IC=False, plot_average=avg,
# 		create_new_figure=False, fmt='b', legend_label="Numerical")

# plot.save_figure(file_name='Density', file_type='png')

# ### Pressure
# # Exact solution
# #plot.plot_solution(mesh, physics, solver, "Pressure", plot_numerical=False, 
# #		plot_exact=True, plot_IC=False, create_new_figure=True, 
# #		fmt='k-.', legend_label="Exact")
# # DG solution
plot.plot_solution(mesh, physics, solver, "SpecificHeatRatio", plot_numerical=num, 
		plot_exact=False, plot_IC=False, plot_average=avg,
		create_new_figure=False, fmt='b', legend_label="Numerical")

# plot.save_figure(file_name='Pressure', file_type='png')

# ### Velocity 
# # Exact solution
# #plot.plot_solution(mesh, physics, solver, "Velocity", plot_numerical=False, 
# #		plot_exact=True, plot_IC=False, create_new_figure=True, 
# #		fmt='k-.', legend_label="Exact")
# # DG solution
# plot.plot_solution(mesh, physics, solver, "Velocity", plot_numerical=num, 
# 		plot_exact=False, plot_IC=False, plot_average=avg,
# 		create_new_figure=False, fmt='b', legend_label="Numerical")

# plot.save_figure(file_name='Velocity', file_type='png')

plot.show_plot()
