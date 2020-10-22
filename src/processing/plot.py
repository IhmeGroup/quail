# ------------------------------------------------------------------------ #
#
#       File : src/processing/plot.py
#
#       Contains functions for plotting 1D and 2D solutions and meshes.
#
# ------------------------------------------------------------------------ #
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.tri as tri
import numpy as np

import meshing.meshbase as mesh_defs
import meshing.tools as mesh_tools

import numerics.helpers.helpers as helpers


def prepare_plot(reset=False, defaults=False, close_all=True, fontsize=12.,
		font={'family':'serif', 'serif': ['DejaVu Sans']}, linewidth=1.5,
		markersize=4.0, axis=None, cmap='viridis', equal_AR=False):
	'''
	This function sets parameters for plotting.

	Inputs:
	-------
	    reset: if True, will reset to default parameters before setting
	    	input arguments
	    defaults: if True, will reset to default parameters and then
	    	immediately return (input arguments not set)
    	close_all: if True, will close all current figures
    	fontsize: font size
    	font: font
    	linewidth: line width
    	markersize: size of markers
    	axis: axis limits [2*dim]
    	cmap: colormap
    	equal_AR: if True, will set equal aspect ratio (only affects 2D)
	'''
	if reset or defaults:
		mpl.rcdefaults() # return to default settings
		if defaults:
			return

	# Use tex syntax
	plt.rc('text', usetex=True)

	# Set parameters
	if close_all:
		plt.close("all")
	mpl.rcParams['font.size'] = fontsize
	plt.rc('font',**font)
	# font={'family':'serif', 'serif': ['computer modern roman']}
	mpl.rcParams['lines.linewidth'] = linewidth
	mpl.rcParams['lines.markersize'] = markersize
	mpl.rcParams['image.cmap'] = cmap

	if axis is not None:
		plt.axis(axis)
	if equal_AR:
		plt.gca().set_aspect('equal', adjustable='box')


def save_figure(file_name='fig', file_type='pdf', crop_level=1):
	'''
	This function saves the current figure to disk.

	Inputs:
	-------
	    file_name: name of file to save
	    file_type: type of file to save
	    crop_level: 0 for no cropping, 1 for some cropping, 2 for a lot
	    	of cropping
	'''
	file_name = file_name + '.' + file_type
	if crop_level == 0:
		# Don't crop
		plt.savefig(file_name)
	elif crop_level == 1:
		# Crop a little
		plt.savefig(file_name, bbox_inches='tight')
	elif crop_level == 2:
		# Crop a lot
		plt.savefig(file_name, bbox_inches='tight', pad_inches=0.0)
	else:
		raise ValueError


def show_plot(interactive=False):
	'''
	This function is a wrapper for plt.show() (displays all open figures).
	'''
	plt.show()


def plot_1D(physics, x, var_plot, ylabel, fmt, legend_label, skip=None):
	'''
	This function creates a 1D plot.

	Inputs:
	-------
		physics: physics object
		x: x-coordinates [num_pts, 1]
		var_plot: variable to plot evaluated at x [num_pts, 1]
		ylabel: y-axis label
		fmt: format string for plotting, e.g. "bo" for blue circles
		legend_label: legend label
		skip: integer value that determines the increment between each point
			to skip when plotting (None for no skipping)
	'''
	# Reshape

	x = np.reshape(x, (-1,))
	var_plot = np.reshape(var_plot, x.shape)

	# Sort
	idx = np.argsort(x)
	idx.flatten()
	x = x[idx]
	var_plot = var_plot[idx]

	# Plot
	plt.plot(x[::skip], var_plot[::skip], fmt, label=legend_label)
	plt.ylabel(ylabel)


def triangulate(physics, x, var):
	'''
	This function creates a triangulation.

	Inputs:
	-------
		physics: physics object
		x: xy-coordinates [num_pts, 2]
		var: variable to plot evaluated at x [num_pts, 1]

	Outputs:
	--------
		tris: Triangulation object
		var_tris: variable to plot evaluated at x (modified with removal
			of duplicates and flattened)
		x: xy-coordinates (duplicates removed and reshaped)
		var: variable to plot evaluated at x (duplicates remove and
			reshaped)
	'''
	if physics.DIM != 2:
		raise ValueError

	# Remove duplicates
	x.shape = -1,2
	num_pts = x.shape[0]
	x, idx = np.unique(x, axis=0, return_index=True)

	# Flatten
	X = x[:,0].flatten()
	Y = x[:,1].flatten()
	var.shape = num_pts, -1
	var = var.flatten()[idx]

	# Triangulation
	triang = tri.Triangulation(X, Y)
	tris = triang; var_tris = var

	return tris, var_tris


def plot_2D_regular(physics, x, var_plot, **kwargs):
	'''
	This function plots 2D contours via a triangulation of the coordinates
	stored in x. This is appropriate only for regular domains since the
	entire domain is triangulated at once.

	Inputs:
	-------
		physics: physics object
	    x: xy-coordinates; 3D array of shape [num_elems, num_pts, 2], where
	    	num_pts is the number of points per element
	    var_plot: variable to plot evaluated at x; 3D array of shape
	    	[num_elems, num_pts, num_state_vars], where num_pts is the
	    	number of points per element
	    kwargs: keyword arguments (see below)

	Outputs:
	--------
		tcf.levels: contour levels
		x: xy-coordinates (duplicates removed and reshaped)
		var_plot: variable to plot evaluated at x (duplicates remove and
			reshaped)

	Notes:
	------
		For coarse solutions, the resulting plot may misrepresent the actual
		solution due to bias in the triangulation.
	'''
	# Get triangulation and plot
	tris, var_tris = triangulate(physics, x, var_plot)
	if "nlevels" in kwargs:
		tcf = plt.tricontourf(tris, var_tris, kwargs["nlevels"])
	elif "levels" in kwargs:
		tcf = plt.tricontourf(tris, var_tris, levels=kwargs["levels"])
	else:
		tcf = plt.tricontourf(tris, var_tris)

	# Show triangulation if requested
	if "show_triangulation" in kwargs:
		if kwargs["show_triangulation"]:
			plt.triplot(tris, lw=0.5, color='white')

	return tcf.levels


def plot_2D_general(physics, x, var_plot, **kwargs):
	'''
	This function plots 2D contours via a triangulation of the coordinates
	stored in x. This is appropriate for general domains, but is much slower
	than plot_2D_regular.

	Inputs:
	-------
		physics: physics object
	    x: xy-coordinates; 3D array of shape [num_elems, num_pts, 2], where
	    	num_pts is the number of points per element
	    var_plot: variable to plot evaluated at x; 3D array of shape
	    	[num_elems, num_pts, num_state_vars], where num_pts is the
	    	number of points per element
	    kwargs: keyword arguments (see below)

	Notes:
	------
		For coarse solutions, the resulting plot may misrepresent the actual
		solution due to bias in the triangulation.
	'''
	''' If not specified, get default contour levels '''
	if "levels" not in kwargs:
		figtmp = plt.figure()
		# Run this for the sole purpose of getting default contour levels
		levels = plot_2D_regular(physics, np.copy(x), np.copy(var_plot),
				**kwargs)
		plt.close(figtmp)
	else:
		levels = kwargs["levels"]

	''' Loop through elements '''
	num_elems = x.shape[0]
	for elem_ID in range(num_elems):
		# Triangulate each element one-by-one
		tris, utri = triangulate(physics, x[elem_ID], var_plot[elem_ID])
		# Plot
		plt.tricontourf(tris, utri, levels=levels, extend="both")
		# Show triangulation if requested
		if "show_triangulation" in kwargs:
			if kwargs["show_triangulation"]:
				plt.triplot(triang, lw=0.5, color='white')


def plot_2D(physics, x, var_plot, ylabel, regular_2D, equal_AR=False,
		**kwargs):
	'''
	This function plots 2D contours via a triangulation of the coordinates
	stored in x. This is appropriate for general domains, but is much slower
	than plot_2D_regular.

	Inputs:
	-------
		physics: physics object
	    x: xy-coordinates; 3D array of shape [num_elems, num_pts, 2], where
	    	num_pts is the number of points per element
	    var_plot: variable to plot evaluated at x; 3D array of shape
	    	[num_elems, num_pts, num_state_vars], where num_pts is the
	    	number of points per element
	    ylabel: y-axis label
	    regular_2D: if True, then entire domain will be triangulated at once;
	    	appropriate only for regular domains
    	equal_AR: if True, will set equal aspect ratio
	    kwargs: keyword arguments (see below)

	Outputs:
	--------
		x: xy-coordinates (modified only if regular_2D is True)
		var_plot: variable to plot evaluated at x (modified only if
			regular_2D is True)

	Notes:
	------
		For coarse solutions, the resulting plot may misrepresent the actual
		solution due to bias in the triangulation.
	'''
	''' Plot solution '''
	if regular_2D:
		plot_2D_regular(physics, x, var_plot, **kwargs)
	else:
		plot_2D_general(physics, x, var_plot, **kwargs)

	''' Label plot '''
	if "ignore_colorbar" in kwargs and kwargs["ignore_colorbar"]:
		# Do nothing
		pass
	else:
		cb = plt.colorbar()
		cb.ax.set_title(ylabel)
	plt.ylabel("$y$")
	if equal_AR:
		plt.gca().set_aspect('equal', adjustable='box')


def finalize_plot(xlabel="x", **kwargs):
	'''
	This function makes final changes to the current plot.

	Inputs:
	-------
	    xlabel: x-axis label
	    kwargs: keyword arguments (see below)
	'''
	plt.xlabel("$" + xlabel + "$")
	ax = plt.gca()
	handles, labels = ax.get_legend_handles_labels()
	if "ignore_legend" in kwargs and kwargs["ignore_legend"]:
		pass
		# do nothing
	elif handles != []:
		# only create legend if handles can be found
		plt.legend(loc="best")


def interpolate_2D_soln_to_points(physics, x, var, xpoints):
	'''
	This function interpolates a variable to an arbitrary set of points.
	2D only.

	Inputs:
	-------
	    physics: physics object
	    x: xy-coordinates at which variable is already evaluated [num_x, 2]
	    var: values of variable evaluated at x [num_x, 1]
	    xpoints: xy-coordinates to interpolate variable to [num_pts, 2]

	Outputs:
	--------
		var_points: values of variable interpolated to xpoints [num_pts, 2]
		x: xy-coordinates at which variable is already evaluated (duplicates
			removed and reshaped)
	    var: values of variable evaluated at x (duplicates removed and
	    	reshaped)
	'''
	if physics.DIM != 2:
		raise ValueError
	tris, utri = triangulate(physics, x, var)
	interpolator = tri.LinearTriInterpolator(tris, utri)

	var_points = interpolator(xpoints[:,0], xpoints[:,1])

	return var_points


def plot_line_probe(mesh, physics, solver, var_name, xy1, xy2, num_pts=101,
		plot_numerical=True, plot_exact=False, plot_IC=False,
		create_new_figure=True, ylabel=None, vs_x=True, fmt="k-",
		legend_label=None, **kwargs):
	'''
	This function evaluates a given variable only a specified line segment
	and creates a 1D plot. 2D only.

	Inputs:
	-------
		mesh: mesh object
	    physics: physics object
	    solver: solver object
	    var_name: name of variable
	    xy1: xy-coordinate of 1st endpoint of line segment
	    xy2: xy-coordinate of 2nd endpoint of line segment
	    num_pts: number of points along line segment
	    plot_numerical: plot numerical solution
	    plot_exact: plot exact solution
	    plot_IC: plot initial condition
	    create_new_figure: if True, will create new figure before plotting
	    ylabel: y-axis label
	    vs_x: if True, will plot variable vs. x; if False, will plot
	    	variable vs. y
		fmt: format string for plotting, e.g. "bo" for blue circles
	    legend_label: legend label
	    kwargs: keyword arguments (see below)
	'''
	''' Compatibility checks '''
	if mesh.dim != 2:
		raise ValueError

	plot_sum = plot_numerical + plot_exact + plot_IC
	if plot_sum >= 2:
		raise ValueError("Can only plot one solution at a time")
	elif plot_sum == 0:
		raise ValueError("Need to plot a solution")

	''' Construct points on line segment '''
	x1 = xy1[0]; y1 = xy1[1]
	x2 = xy2[0]; y2 = xy2[1]
	xline = np.linspace(x1, x2, num_pts)
	yline = np.linspace(y1, y2, num_pts)

	''' Interpolation '''
	x = get_sample_points(mesh, solver, physics, solver.basis, True)
	xyline = np.array([xline, yline]).transpose()

	if plot_numerical:
		var = get_numerical_solution(physics, solver.state_coeffs, x,
				solver.basis, var_name)
		var_plot = interpolate_2D_soln_to_points(physics, x, var, xyline)
		default_label = "Numerical"
	elif plot_exact:
		var_plot = get_analytical_solution(physics, physics.exact_soln,
				xyline, solver.time, var_name)
		default_label = "Exact"
	elif plot_IC:
		var_plot = get_analytical_solution(physics, physics.IC, xyline, 0.,
				var_name)
		default_label = "Initial"

	''' Plot '''
	if create_new_figure:
		plt.figure()
	if legend_label is None:
		legend_label = default_label

	ylabel = get_ylabel(physics, var_name, ylabel)

	if vs_x:
		xlabel = "x"
		line = xline
	else:
		xlabel = "y"
		line = yline

	plot_1D(physics, line, var_plot, ylabel, fmt, legend_label)

	finalize_plot(xlabel=xlabel, **kwargs)


def get_sample_points(mesh, solver, physics, basis, equidistant=True):
	'''
	This function returns sample points at which to evaluate the solution
	for plotting.

	Inputs:
	-------
		mesh: mesh object
	    physics: physics object
	    basis: basis object
	    equidistant: if True, then sample points will be equidistant
	    	(within each element); if False, sample points will be based
	    	on quadrature points

	Outputs:
	-------
		x: sample points [num_elems, num_pts, dim], where num_pts is the
			number of sample points per element
	'''
	# Extract
	dim = mesh.dim
	U = solver.state_coeffs
	order = solver.order

	# Get sample points in reference space
	if equidistant:
		xref = basis.equidistant_nodes(max([1, 3*order]))
	else:
		quad_order = basis.get_quadrature_order(mesh, max([2, 2*order]),
				physics=physics)
		gbasis = mesh.gbasis
		xref, _ = gbasis.get_quadrature_data(quad_order)

	# Allocate
	num_pts = xref.shape[0]
	x = np.zeros([mesh.num_elems, num_pts, dim])

	# Evaluate basis at reference-space points
	basis.get_basis_val_grads(xref, True, False, False, None)

	# Convert reference-space points to physical space
	for elem_ID in range(mesh.num_elems):
		xphys = mesh_tools.ref_to_phys(mesh, elem_ID, xref)
		x[elem_ID,:,:] = xphys

	return x


def get_analytical_solution(physics, fcn_data, x, time, var_name):
	'''
	This function evaluates the analytical solution at a set of points.

	Inputs:
	-------
	    physics: physics object
	    fcn_data: function object (for either exact solution or initial
	    	condition)
	    x: coordinates at which to evaluate solution
	    time: time at which to evaluate solution
	    var_name: name of variable to get

	Outputs:
	-------
		var_plot: values of variable obtained at x [num_pts, 1]
	'''
	# U_plot = fcn_data.get_state(physics, x=np.reshape(x, (-1, physics.DIM)),
	# 		t=time)
	if x.ndim != 3:
		U_plot = fcn_data.get_state(physics, x=np.expand_dims(x,0), t=time)
	else:
		U_plot = fcn_data.get_state(physics, x=x, t=time)

	var_plot = physics.compute_variable(var_name, U_plot)

	return var_plot


def get_numerical_solution(physics, U, x, basis, var_name):
	'''
	This function evaluates the numerical solution at a set of points.

	Inputs:
	-------
	    physics: physics object
	    U: state polynomial coefficients
	    	[num_elems, num_basis_coeffs, num_state_vars]
	    x: coordinates at which to evaluate solution
	    	[num_elems, num_pts, dim], where num_pts is the number of sample
	    	points per element
	    basis: basis object
	    var_name: name of variable to get

	Outputs:
	-------
		var_numer: values of variable obtained at x
			[num_elems, num_pts, 1]
	'''
	Uq = helpers.evaluate_state(U, basis.basis_val)
	var_numer = physics.compute_variable(var_name, Uq)

	return var_numer


def get_ylabel(physics, var_name, ylabel=None):
	'''
	This function returns an appropriate y-axis label based on the variable
	name and the physics object.

	Inputs:
	-------
	    physics: physics object
	    var_name: name of variable to get
	    ylabel: if not None, then nothing will happen

	Outputs:
	-------
		ylabel: y-axis label
	'''
	if ylabel is None:
		try:
			ylabel = physics.StateVariables[var_name].value
		except KeyError:
			ylabel = physics.AdditionalVariables[var_name].value
		ylabel = "$" + ylabel + "$"

	return ylabel


def plot_mesh(mesh, equal_AR=False, **kwargs):
	'''
	This function plots the mesh. Element IDs can also be plotted (set in
	kwargs).

	Inputs:
	-------
	    mesh: mesh object
	    equal_AR: if True, will set equal aspect ratio
	    kwargs: keyword arguments (see below)

	Notes:
	------
		For curved faces, line segments are plotted in between each node.
	'''
	gbasis = mesh.gbasis
	dim = mesh.dim

	if dim == 1:
		y = plt.ylim()

	'''
	Loop through interior_faces and plot interior faces
	'''
	for interior_face in mesh.interior_faces:
		# Loop through both connected elements to account for periodic
		# boundaries
		for e in range(2):
			if e == 0:
				elem_ID = interior_face.elemL_ID
				face_ID = interior_face.faceL_ID
			else:
				elem_ID = interior_face.elemR_ID
				face_ID = interior_face.faceR_ID

			# Get local node IDs on face
			local_node_IDs = gbasis.get_local_face_node_nums(mesh.gorder,
					face_ID)

			# Get coordinates
			elem = mesh.elements[elem_ID]
			coords = elem.node_coords[local_node_IDs]
			if dim == 1:
				x = np.full(2, coords[:, 0])
			else:
				x = coords[:, 0]; y = coords[:, 1]

			# Plot face
			plt.plot(x, y, 'k-')

	'''
	Loop through boundary_groups and plot boundary faces
	'''
	for boundary_group in mesh.boundary_groups.values():
		for boundary_face in boundary_group.boundary_faces:
			# Get adjacent element info
			elem_ID = boundary_face.elem_ID
			face_ID = boundary_face.face_ID

			# Get local node IDs on face
			local_node_IDs = gbasis.get_local_face_node_nums(mesh.gorder,
					face_ID)

			# Get coordinates
			elem = mesh.elements[elem_ID]
			coords = elem.node_coords[local_node_IDs]
			if dim == 1:
				x = np.full(2, coords[:, 0])
			else:
				x = coords[:, 0]; y = coords[:, 1]

			# Plot face
			plt.plot(x, y, 'k-')

	'''
	If requested, plot element IDs at element centroids
	'''
	if "show_elem_IDs" in kwargs and kwargs["show_elem_IDs"]:
		for elem_ID in range(mesh.num_elems):
			xc = mesh_tools.get_element_centroid(mesh, elem_ID)
			if dim == 1:
				yc = np.mean(y)
			else:
				yc = xc[0,1]
			plt.text(xc[0,0], yc, str(elem_ID))

	# Labels, aspect ratio
	plt.xlabel("$x$")
	if dim == 2: plt.ylabel("$y$")
	if equal_AR:
		plt.gca().set_aspect('equal', adjustable='box')


def plot_solution(mesh, physics, solver, var_name, plot_numerical=True,
		plot_exact=False, plot_IC=False, create_new_figure=True, ylabel=None,
		fmt='k-', legend_label=None, equidistant_pts=True,
		include_mesh=False, regular_2D=False, equal_AR=False, skip=None,
		**kwargs):
	'''
	This function plots the solution. For 2D calculations, the solution will
	be plotted using a triangulation.

	Inputs:
	-------
	    mesh: mesh object
	    physics: physics object
	    solver: solver object
	    var_name: name of variable to plot
	    plot_numerical: plot numerical solution
	    plot_exact: plot exact solution
	    plot_IC: plot initial condition
	    create_new_figure: if True, will create new figure before plotting
	    ylabel: y-axis label
		fmt: format string for plotting, e.g. "bo" for blue circles
	    legend_label: legend label
	    equidistant_pts: if True, then solution will be evaluated at
	    	equidistant points (within each element); if False, then solution
	    	will be evaluated at quadrature points
	    include_mesh: if True, then the mesh will be superimposed
	    regular_2D: if True, then entire domain will be triangulated at once
	    	(appropriate only for regular domains); if False, then each
	    	element will be triangulated one-by-one (appropriate for
	    	general domains)
	    equal_AR: if True, will set equal aspect ratio
		skip: integer value that determines the increment between each point
			to skip when plotting (only matters for 1D)
	    kwargs: keyword arguments (see below)
	'''
	''' Compatibility check '''
	plot_sum = plot_numerical + plot_exact + plot_IC
	if plot_sum >= 2:
		raise ValueError("Can only plot one solution at a time")
	elif plot_sum == 0:
		raise ValueError("Need to plot a solution")

	# Extract params
	time = solver.time
	dim = mesh.dim

	# Get sample points
	x = get_sample_points(mesh, solver, physics, solver.basis,
			equidistant_pts)

	''' Evaluate desired variable at sample points '''
	if plot_numerical:
		var_plot = get_numerical_solution(physics, solver.state_coeffs, x,
				solver.basis, var_name)
		default_label = "Numerical"
	elif plot_exact:
		var_plot = get_analytical_solution(physics, physics.exact_soln, x,
				time, var_name)
		var_plot.shape = x.shape[0], x.shape[1], -1
		default_label = "Exact"
	elif plot_IC:
		var_plot = get_analytical_solution(physics, physics.IC, x, 0.,
				var_name)
		var_plot.shape = x.shape[0], x.shape[1], -1
		default_label = "Initial"

	if legend_label is None:
		legend_label = default_label

	''' Plot '''
	ylabel = get_ylabel(physics, var_name, ylabel)
	if create_new_figure:
		plt.figure()

	if dim == 1:
		plot_1D(physics, x, var_plot, ylabel, fmt, legend_label, skip)
	else:
		plot_2D(physics, x, var_plot, ylabel, regular_2D, equal_AR, **kwargs)

	if include_mesh:
		plot_mesh(mesh, **kwargs)

	# Final embellishments
	finalize_plot(**kwargs)
