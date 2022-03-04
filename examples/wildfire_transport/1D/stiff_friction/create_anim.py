import matplotlib.animation as animation
import matplotlib.pyplot as plt

import processing.post as post
import processing.plot as plot
import processing.readwritedatafiles as readwritedatafiles


plot.prepare_plot(linewidth=0.5)
fig = plt.figure()
ax = plt.gca()

imgs_all = []
j = 0
# Loop through data files
# Note: this loop only uses the first 15 frames. This is all that should be 
# necessary to see the dynamics of the energy being damped.
for i in range(15):
	print(i)

	# Read data file
	fname = "Data_" + str(i) + ".pkl"
	solver = readwritedatafiles.read_data_file(fname)

	# Unpack
	mesh = solver.mesh
	physics = solver.physics

	# Plot solution
	plot.plot_solution(mesh, physics, solver, "Energy", plot_numerical=True, 
			plot_exact=False, plot_IC=False, create_new_figure=False, 
			ylabel=None, fmt='bo', legend_label="DG", ignore_legend=True)

	plot.plot_solution(mesh, physics, solver, "Energy", plot_IC=True, 
			plot_numerical=False, create_new_figure=False, 
			fmt='k--', ignore_legend=True)

	imgs = ax.get_lines().copy()

	# Add to imgs_all
	if j == 0:
		plt.legend(loc="best")
		imgs_all.append(imgs)
	else:
		nc = len(imgs_all[j-1])
		imgs_all.append(imgs[-nc:])

	j += 1


anim = animation.ArtistAnimation(fig, imgs_all, interval=500, blit=False,
		repeat_delay=None)

plt.show()

# Save mp4
anim.save("anim.mp4")
