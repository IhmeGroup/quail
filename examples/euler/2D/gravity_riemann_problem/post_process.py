import numpy as np

import processing.post as post
import processing.plot as plot
import processing.readwritedatafiles as readwritedatafiles

# Read data file
fname = "Data_final.pkl"
#fname = "Data_n160_P2.pkl"
solver = readwritedatafiles.read_data_file(fname)

# Unpack
mesh = solver.mesh
physics = solver.physics


''' Plot '''
levels = np.arange(0., 5., 0.5)
# Density contour
plot.prepare_plot(linewidth=0.5)
plot.plot_solution(mesh, physics, solver, "Density", plot_numerical=True, 
		plot_exact=False, plot_IC=False, create_new_figure=True, fmt='bo', 
		legend_label="DG", include_mesh=False, regular_2D=True, 
		show_elem_IDs=False, levels=levels)

plot.save_figure(file_name='contour', file_type='pdf')

### Line probe (y = 1.7875) ###
plot.prepare_plot(close_all=False, linewidth=1.5)
# Parameters
xy1 = [0., 1.7875]; xy2 = [2., 1.7875]

# DG solution
plot.plot_line_probe(mesh, physics, solver, "Density", xy1=xy1, xy2=xy2, 
		plot_numerical=True, plot_exact=False, plot_IC=False, 
		create_new_figure=True, fmt="bo", ignore_legend = True)


# Save figure
plot.save_figure(file_name='line', file_type='pdf', crop_level=2)

plot.show_plot()
