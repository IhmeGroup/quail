import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import processing.post as post
import processing.plot as plot
import processing.readwritedatafiles as readwritedatafiles

# All .pkl files in current directory
files = glob.glob(os.getcwd() + '/*.pkl')

# Loop over each file
for fname in files:
    solver = readwritedatafiles.read_data_file(fname)
    # Unpack
    mesh = solver.mesh
    physics = solver.physics
    # Compute L2 error
    post.get_error(mesh, physics, solver, "Density")
    ''' Plot '''
    # Density contour
    plot.prepare_plot(linewidth=0.5)
    plot.plot_solution(mesh, physics, solver, "Density", plot_numerical=True,
            plot_exact=False, plot_IC=False, create_new_figure=True, fmt='bo',
            legend_label="DG", include_mesh=True, regular_2D=True,
            show_elem_IDs=True, levels=np.linspace(.5, 1.2, 10))
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    # Save figure
    plot.save_figure(file_name = fname.split('/')[-1].split('.')[0],
            file_type='png', crop_level=2)

## Line probe (y = 1)
#plot.prepare_plot(close_all=False, linewidth=1.5)
## Parameters
#xy1 = [-5.,1.]; xy2 = [5.,1.]
## Initial condition
#plot.plot_line_probe(mesh, physics, solver, "Density", xy1=xy1, xy2=xy2,
#		plot_numerical=False, plot_exact=False, plot_IC=True,
#		create_new_figure=True, ylabel=None, vs_x=True, fmt="k-.",
#		legend_label=None)
## Exact solution
#plot.plot_line_probe(mesh, physics, solver, "Density", xy1=xy1, xy2=xy2,
#		plot_numerical=False, plot_exact=True, plot_IC=False,
#		create_new_figure=False, fmt="k-", legend_label=None)
## DG solution
#plot.plot_line_probe(mesh, physics, solver, "Density", xy1=xy1, xy2=xy2,
#		plot_numerical=True, plot_exact=False, plot_IC=False,
#		create_new_figure=False, fmt="bo", legend_label=None)
## Save figure
#plot.save_figure(file_name='line', file_type='svg', crop_level=2)

#plot.show_plot()

os.system('feh *.png')
