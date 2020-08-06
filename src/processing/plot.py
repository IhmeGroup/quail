import code
import copy
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.tri as tri
import numpy as np

import meshing.meshbase as mesh_defs
import meshing.tools as mesh_tools

import numerics.helpers.helpers as helpers


def PreparePlot(reset=False, defaults=False, close_all=True, fontsize=12., font={'family':'serif', 'serif': ['DejaVu Sans']},
	linewidth=1.5, markersize=4.0, axis=None, cmap='viridis', EqualAR=False):
	# font={'family':'serif', 'serif': ['computer modern roman']}
	# Note: matplotlib.rcdefaults() returns to default settings

	if reset or defaults:
		mpl.rcdefaults()
		if defaults:
			return

	# For now, always use tex syntax
	plt.rc('text',usetex=True)

	if close_all:
		plt.close("all")
	mpl.rcParams['font.size']=fontsize
	plt.rc('font',**font)
	mpl.rcParams['lines.linewidth'] = linewidth
	mpl.rcParams['lines.markersize'] = markersize
	mpl.rcParams['image.cmap'] = cmap

	if axis is not None:
		plt.axis(axis)
	if EqualAR:
		plt.gca().set_aspect('equal', adjustable='box')


def SaveFigure(FileName='fig', FileType='pdf', CropLevel=1, **kwargs):
	FileName = FileName + '.' + FileType
	if CropLevel == 0:
		# Don't crop
		plt.savefig(FileName)
	elif CropLevel == 1:
		# Crop a little
		plt.savefig(FileName, bbox_inches='tight')
	elif CropLevel == 2:
		# Crop a lot
		plt.savefig(FileName, bbox_inches='tight', pad_inches=0.0)
	else:
		raise ValueError


def ShowPlot(Interactive=False):
	if Interactive:
		# doesn't work for now
		plt.ion()
		plt.show()
	else:
		plt.show()


def Plot1D(physics, x, u, SolnLabel, VariableName=None, u_exact=None, u_IC=None, 
		u_var_calculated=False, **kwargs):
	### reshape
	# uplot = u[:,:,iplot]
	# uplot = np.reshape(uplot, (-1,))
	# u_exact = np.reshape(u_exact, (-1,1))
	x = np.reshape(x, (-1,))
	nplot = x.shape[0]
	u.shape = nplot,-1
	if not u_var_calculated:
		uplot = physics.ComputeScalars(VariableName, u)
	else:
		# assume desired variable already calculated
		uplot = u

	### sort
	idx = np.argsort(x)
	idx.flatten()
	uplot = uplot[idx]
	# u_exact = u_exact[idx]
	x = x[idx]

	if u_exact is not None: 
		# u_ex = u_exact[:,:,iplot]
		# u_ex.shape = -1,
		u_exact.shape = nplot,-1
		u_ex = physics.ComputeScalars(VariableName, u_exact)
		plt.plot(x,u_ex,'k-',label="Exact")

	if u_IC is not None: 
		# u_ex = u_exact[:,:,iplot]
		# u_ex.shape = -1,
		u_IC.shape = nplot,-1
		u_i = physics.ComputeScalars(VariableName, u_IC)
		plt.plot(x,u_i,'k--',label="Initial")

	if "legend_label" in kwargs:
		legend_label = kwargs["legend_label"]
	else:
		legend_label = "DG"
	plt.plot(x,uplot,'bo',label=legend_label) 
	plt.ylabel(SolnLabel)


def plot_1D(physics, x, var_plot, ylabel, fmt, legend_label, skip=0, **kwargs):
	### reshape
	# uplot = u[:,:,iplot]
	# uplot = np.reshape(uplot, (-1,))
	# u_exact = np.reshape(u_exact, (-1,1))
	# x = np.reshape(x, (-1,))
	# nplot = x.shape[0]
	# u.shape = nplot,-1
	# if not u_var_calculated:
	# 	uplot = physics.ComputeScalars(VariableName, u)
	# else:
	# 	# assume desired variable already calculated
	# 	uplot = u

	# Reshape
	x = np.reshape(x, (-1,))
	var_plot = np.reshape(var_plot, x.shape)

	### sort
	idx = np.argsort(x)
	idx.flatten()
	x = x[idx]
	var_plot = var_plot[idx]
	# u_exact = u_exact[idx]

	# if u_exact is not None: 
	# 	# u_ex = u_exact[:,:,iplot]
	# 	# u_ex.shape = -1,
	# 	u_exact.shape = nplot,-1
	# 	u_ex = physics.ComputeScalars(VariableName, u_exact)
	# 	plt.plot(x,u_ex,'k-',label="Exact")

	# if u_IC is not None: 
	# 	# u_ex = u_exact[:,:,iplot]
	# 	# u_ex.shape = -1,
	# 	u_IC.shape = nplot,-1
	# 	u_i = physics.ComputeScalars(VariableName, u_IC)
	# 	plt.plot(x,u_i,'k--',label="Initial")

	# if "legend_label" in kwargs:
	# 	legend_label = kwargs["legend_label"]
	# else:
	# 	legend_label = "DG"
	if skip !=0:
		plt.plot(x[idx[0]:idx[-1]:skip], var_plot[idx[0]:idx[-1]:skip], fmt, label=legend_label) 
	else:
		plt.plot(x, var_plot, fmt, label=legend_label) 
	plt.ylabel(ylabel)


def triangulate(physics, x, var):
	### Remove duplicates
	x.shape = -1,2
	nold = x.shape[0]
	x, idx = np.unique(x, axis=0, return_index=True)

	### Flatten
	X = x[:,0].flatten()
	Y = x[:,1].flatten()
	var.shape = nold,-1
	# U = physics.ComputeScalars(variable_name, u).flatten()
	# U = u[:,:,iplot].flatten()
	var = var.flatten()[idx]

	### Triangulation
	triang = tri.Triangulation(X, Y)
	# refiner = tri.UniformTriRefiner(triang)
	# tris, utri = refiner.refine_field(U, subdiv=0)
	tris = triang; var_tris = var

	return tris, var_tris


def Plot2D_Regular(physics, x, var_plot, **kwargs):
	'''
	Function: Plot2D
	-------------------
	This function plots 2D contours via a triangulation

	NOTES:
			For coarse solutions, the resulting plot may misrepresent
			the actual solution due to bias in the triangulation

	INPUTS:
	    x: (x,y) coordinates; 3D array of size [num_elems_tot x nPoint x 2], where
	    	nPoint is the number of points per element
	    u: values of solution at the points in x; 3D array of size
	    	[num_elems_tot x nPoint x NUM_STATE_VARS]

	OUTPUTS:
	    Plot is created
	'''

	### Remove duplicates
	# x.shape = -1,2
	# nold = x.shape[0]
	# x, idx = np.unique(x, axis=0, return_index=True)

	# ### Flatten
	# X = x[:,0].flatten()
	# Y = x[:,1].flatten()
	# u.shape = nold,-1
	# U = physics.ComputeScalars(VariableName, u).flatten()
	# # U = u[:,:,iplot].flatten()
	# U = U[idx]

	# ### Triangulation
	# triang = tri.Triangulation(X, Y)
	# # refiner = tri.UniformTriRefiner(triang)
	# # tris, utri = refiner.refine_field(U, subdiv=0)
	# tris = triang; utri = U

	tris, var_tris = triangulate(physics, x, var_plot)
	if "nlevels" in kwargs:
		TCF = plt.tricontourf(tris, var_tris, kwargs["nlevels"])
	elif "levels" in kwargs:
		TCF = plt.tricontourf(tris, var_tris, levels=kwargs["levels"])
	else:
		TCF = plt.tricontourf(tris, var_tris)

	### Plot contours 
	# cb = plt.colorbar()
	# cb.ax.set_title(SolnLabel)
	# plt.ylabel("$y$")
	if "show_triangulation" in kwargs:
		if kwargs["show_triangulation"]: 
			plt.triplot(tris, lw=0.5, color='white')
	# plt.axis("equal")

	return TCF.levels


def Plot2D_General(physics, x, var_plot, **kwargs):
	'''
	Function: Plot2D
	-------------------
	This function plots 2D contours via a triangulation

	NOTES:
			For coarse solutions, the resulting plot may misrepresent
			the actual solution due to bias in the triangulation

	INPUTS:
	    x: (x,y) coordinates; 3D array of size [num_elems_tot x nPoint x 2], where
	    	nPoint is the number of points per element
	    u: values of solution at the points in x; 3D array of size
	    	[num_elems_tot x nPoint x NUM_STATE_VARS]

	OUTPUTS:
	    Plot is created
	'''

	''' If not specified, get default contour levels '''
	if "levels" not in kwargs:
		# u1 = np.copy(u)
		# u1.shape = -1,u.shape[-1]
		# u2 = physics.ComputeScalars(VariableName, u1, u1.shape[0]).flatten()
		# u2.shape = u.shape[0:2]
		# figtmp = plt.figure()
		# if "nlevels" in kwargs:
		# 	CS = plt.contourf(u2, kwargs["nlevels"])
		# else:
		# 	CS = plt.contourf(u2)
		# levels = CS.levels
		# plt.close(figtmp)

		figtmp = plt.figure()
		levels = Plot2D_Regular(physics, np.copy(x), np.copy(var_plot), **kwargs)
		# plt.colorbar()
		# ShowPlot()
		plt.close(figtmp)
	else:
		levels = kwargs["levels"]

	num_elems_tot = x.shape[0]
	nPoint = x.shape[1]
	''' Loop through elements '''
	for elem in range(num_elems_tot):
		# Extract x and y
		# X = x[elem,:,0].flatten()
		# Y = x[elem,:,1].flatten()
		# # Compute requested scalar
		# U = physics.ComputeScalars(VariableName, u[elem,:,:]).flatten()
		# # Triangulation
		# triang = tri.Triangulation(X, Y)
		# tris = triang; utri = U

		tris, utri = triangulate(physics, x[elem], var_plot[elem])
		# Plot
		plt.tricontourf(tris, utri, levels=levels, extend="both")
		# if "nlevels" in kwargs:
		# 	plt.tricontourf(tris, utri, kwargs["nlevels"])
		# elif "levels" in kwargs:
		# 	plt.tricontourf(tris, utri, levels=kwargs["levels"], extend="both")
		# else:
		# 	plt.tricontourf(tris, utri)
		if "show_triangulation" in kwargs:
			if kwargs["show_triangulation"]: 
				plt.triplot(triang, lw=0.5, color='white')


# def Plot2D(physics, x, u, VariableName, SolnLabel, Regular2D, EqualAR=False, **kwargs):
# 	if Regular2D:
# 		Plot2D_Regular(physics, x, u, VariableName, SolnLabel, EqualAR, **kwargs)
# 	else:
# 		Plot2D_General(physics, x, u, VariableName, SolnLabel, EqualAR, **kwargs)

# 	''' Label plot '''
# 	if "ignore_colorbar" in kwargs and kwargs["ignore_colorbar"]:
# 		pass 
# 		# do nothing
# 	else:
# 		cb = plt.colorbar()
# 		cb.ax.set_title(SolnLabel)
# 	plt.ylabel("$y$")
# 	if EqualAR:
# 		plt.gca().set_aspect('equal', adjustable='box')
# 	# plt.axis("equal")


def plot_2D(physics, x, var_plot, ylabel, regular_2D, equal_AR=False, **kwargs):
	if regular_2D:
		Plot2D_Regular(physics, x, var_plot, **kwargs)
	else:
		Plot2D_General(physics, x, var_plot, **kwargs)

	''' Label plot '''
	if "ignore_colorbar" in kwargs and kwargs["ignore_colorbar"]:
		pass 
		# do nothing
	else:
		cb = plt.colorbar()
		cb.ax.set_title(ylabel)
	plt.ylabel("$y$")
	if equal_AR:
		plt.gca().set_aspect('equal', adjustable='box')
	# plt.axis("equal")


def finalize_plot(xlabel="x", **kwargs):
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
	tris, utri = triangulate(physics, x, var)
	interpolator = tri.LinearTriInterpolator(tris, utri)

	var_points = interpolator(xpoints[:,0], xpoints[:,1])

	return var_points


def plot_line_probe(mesh, physics, solver, var_name, xy1, xy2, nPoint=101, plot_numerical=True, plot_exact=False,
		plot_IC=False, create_new_figure=True, ylabel=None, vs_x=True, fmt="k-", legend_label=None, **kwargs):

	plot_sum = plot_numerical + plot_exact + plot_IC
	if plot_sum >= 2:
		raise ValueError("Can only plot one solution at a time")
	elif plot_sum == 0:
		raise ValueError("Need to plot a solution")

	# Construct points on line segment
	x1 = xy1[0]; y1 = xy1[1]
	x2 = xy2[0]; y2 = xy2[1]
	xline = np.linspace(x1, x2, nPoint)
	yline = np.linspace(y1, y2, nPoint)

	# Interpolation
	x = get_sample_points(mesh, physics, solver.basis, True)
	xyline = np.array([xline,yline]).transpose()
	# U_line = interpolate_2D_soln_to_points(physics, x, u, xyline, var_name)

	if create_new_figure:
		plt.figure()

	# Analytical?
	# u_exact, u_IC = get_analytical_solution(physics, xyline, solver.Time, PlotExact, PlotIC)
	# if u_exact is not None:
	# 	u_exact = physics.ComputeScalars(VariableName, u_exact)
	# 	# u_exact = interpolate_2D_soln_to_points(physics, x, u_exact, xyline, variable_name)
	# if u_IC is not None:
	# 	u_IC = physics.ComputeScalars(VariableName, u_IC)
		# u_IC = interpolate_2D_soln_to_points(physics, x, u_IC, xyline, variable_name)

	if plot_numerical:
		var = get_numerical_solution(physics, physics.U, x, solver.basis, var_name)
		var_plot = interpolate_2D_soln_to_points(physics, x, var, xyline)
		default_label = "Numerical"
	elif plot_exact:
		var_plot = get_analytical_solution(physics, physics.ExactSoln, xyline, solver.Time, var_name)
		default_label = "Exact"
	elif plot_IC:
		var_plot = get_analytical_solution(physics, physics.IC, xyline, 0., var_name)
		default_label = "Initial"

	if legend_label is None:
		legend_label = default_label

	# SolnLabel = get_solution_label(physics, variable_name, Label)
	ylabel = get_ylabel(physics, var_name, ylabel)

	if vs_x:
		xlabel = "x"
		line = xline
	else:
		xlabel = "y"
		line = yline
	plot_1D(physics, line, var_plot, ylabel, fmt, legend_label, 0, **kwargs)
	# Plot1D(physics, line, uline, SolnLabel, var_name, u_exact, u_IC, u_var_calculated=True, **kwargs)

	# code.interact(local=locals())

	### Finalize plot
	finalize_plot(xlabel=xlabel, **kwargs)
	# plt.xlabel("$x$")
	# ax = plt.gca()
	# handles, labels = ax.get_legend_handles_labels()
	# if handles != []:
	# 	# only create legend if handles can be found
	# 	plt.legend(loc="best")



def get_sample_points(mesh, physics, basis, equidistant):
	## Extract data
	dim = mesh.dim
	U = physics.U
	order = physics.order
	sr = physics.NUM_STATE_VARS

	# Get points to plot at
	# Note: assumes uniform element type
	if equidistant:
		xpoint = basis.equidistant_nodes(max([1, 3*order]))
	else:
		quad_order = basis.get_quadrature_order(mesh, max([2,2*order]), physics=physics)
		gbasis = mesh.gbasis
		xpoint, _ = gbasis.get_quadrature_data(quad_order)

		# QuadOrder,_ = get_gaussian_quadrature_elem(mesh, basis, max([2,2*Order]), physics)
		# quadData = QuadData(mesh, mesh.gbasis, EntityType.Element, QuadOrder)
		# xpoint = gbasis.quad_pts
	npoint = xpoint.shape[0]

	# u = np.zeros([mesh.num_elems,npoint,sr])
	# u_exact = np.copy(u)
	x = np.zeros([mesh.num_elems,npoint,dim])
	# PhiData = Basis.BasisData(physics.Basis,Order,mesh)
	basis.get_basis_val_grads(xpoint, True, False, False, None)
	el = 0
	for elem in range(mesh.num_elems):
		U_ = U[elem]

		xphys = mesh_defs.ref_to_phys(mesh, elem, xpoint)
		x[el,:,:] = xphys
		# u[el,:,:] = np.matmul(basis.basis_val, U_)

		el += 1

	return x


def get_analytical_solution(physics, fcn_data, x, time, var_name):

	Uplot = physics.CallFunction(fcn_data, x=np.reshape(x, (-1, physics.dim)), t=time)
	var_plot = physics.ComputeScalars(var_name, Uplot)
	# var_plot.shape = x.shape[0], x.shape[1], -1
	# if u is not None: u_IC.shape = u.shape

	# # Exact solution?
	# if get_exact:
	# 	u_exact = physics.CallFunction(physics.ExactSoln, x=np.reshape(x, (-1,physics.dim)), t=time)
	# 	if u is not None: u_exact.shape = u.shape
	# else:
	# 	u_exact = None
	# # IC ?
	# if get_IC:
	# 	u_IC = physics.CallFunction(physics.IC, x=np.reshape(x,(-1,physics.dim)), t=0.)
	# 	if u is not None: u_IC.shape = u.shape
	# else:
	# 	u_IC = None

	return var_plot


def get_numerical_solution(physics, U, x, basis, var_name, already_interpolated=False):
	GeomPhiData = None

	if already_interpolated:
		var_numer = physics.ComputeScalars(var_name, U)
	else:
		var_numer = np.zeros([x.shape[0], x.shape[1], 1])
		for elem in range(x.shape[0]):
			# Up = np.matmul(basis.basis_val, U[elem])
			Up = helpers.evaluate_state(U[elem], basis.basis_val)
			var_numer[elem,:,:] = physics.ComputeScalars(var_name, Up)

	return var_numer


def get_ylabel(physics, variable_name, ylabel=None):
	if ylabel is None:
		try:
			ylabel = physics.StateVariables[variable_name].value
		except KeyError:
			ylabel = physics.AdditionalVariables[variable_name].value
		ylabel = "$" + ylabel + "$"

	return ylabel


# def PlotSolution(mesh, physics, solver, VariableName, create_new_figure=True, PlotExact=False, PlotIC=False, Label=None, Equidistant=True,
# 	include_mesh=False, Regular2D=False, EqualAR=False, **kwargs):

# 	# iplot_sr = physics.VariableType[VariableName]
# 	if PlotExact:
# 		# if not physics.ExactSoln.Function:
# 		if physics.ExactSoln is None:
# 			raise Exception("No exact solution provided")

# 	## Extract params
# 	EndTime = solver.Time
# 	dim = mesh.dim

# 	# Get sample points
# 	x, u = get_sample_points(mesh, physics, solver.basis, Equidistant)

# 	# # Exact solution?
# 	# if PlotExact:
# 	# 	u_exact = physics.CallFunction(physics.ExactSoln, x=np.reshape(x, (-1,dim)), Time=EndTime)
# 	# 	u_exact.shape = u.shape
# 	# else:
# 	# 	u_exact = None
# 	# # IC ?
# 	# if PlotIC:
# 	# 	u_IC = physics.CallFunction(physics.IC, x=np.reshape(x,(-1,dim)),Time=0.)
# 	# 	u_IC.shape = u.shape
# 	# else:
# 	# 	u_IC = None
# 	# Solution label

# 	u_exact, u_IC = get_analytical_solution(physics, x, EndTime, PlotExact, PlotIC, u)

# 	# if Label is None:
# 	# 	try:
# 	# 		Label = physics.StateVariables[VariableName].value
# 	# 	except KeyError:
# 	# 		Label = physics.AdditionalVariables[VariableName].value
# 	# SolnLabel = "$" + Label + "$"

# 	SolnLabel = get_solution_label(physics, VariableName, Label)

# 	# Plot solution
# 	if create_new_figure:
# 		plt.figure()

# 	if dim == 1:
# 		Plot1D(physics, x, u, SolnLabel, VariableName, u_exact, u_IC, **kwargs)
# 	else:
# 		if PlotExact: u = u_exact # plot either only numerical or only exact
# 		Plot2D(physics, x, u, VariableName, SolnLabel, Regular2D, EqualAR, **kwargs)

# 	### Finalize plot
# 	finalize_plot(**kwargs)

# 	# plt.xlabel("$x$")
# 	# ax = plt.gca()
# 	# handles, labels = ax.get_legend_handles_labels()
# 	# if handles != []:
# 	# 	# only create legend if handles can be found
# 	# 	plt.legend(loc="best")

# 	# if dim == 1 and include_mesh:
# 	# 	plot_mesh_1D(mesh, **kwargs)
# 	# elif dim == 2 and include_mesh:
# 	# 	PlotMesh2D(mesh, **kwargs)

# 	if include_mesh:
# 		plot_mesh(mesh, **kwargs)


def plot_solution(mesh, physics, solver, var_name, plot_numerical=True, plot_exact=False, plot_IC=False, create_new_figure=True, 
			ylabel=None, fmt='k-', legend_label=None, equidistant_pts=True, 
			include_mesh=False, regular_2D=False, equal_AR=False, skip=0, **kwargs):

	plot_sum = plot_numerical + plot_exact + plot_IC
	if plot_sum >= 2:
		raise ValueError("Can only plot one solution at a time")
	elif plot_sum == 0:
		raise ValueError("Need to plot a solution")

	## Extract params
	time = solver.Time
	dim = mesh.dim

	# Get sample points
	x = get_sample_points(mesh, physics, solver.basis, equidistant_pts)

	if plot_numerical:
		var_plot = get_numerical_solution(physics, physics.U, x, solver.basis, var_name)
		default_label = "Numerical"
	elif plot_exact:
		var_plot = get_analytical_solution(physics, physics.ExactSoln, x, time, var_name)
		var_plot.shape = x.shape[0], x.shape[1], -1
		default_label = "Exact"
	elif plot_IC:
		var_plot = get_analytical_solution(physics, physics.IC, x, 0., var_name)
		var_plot.shape = x.shape[0], x.shape[1], -1
		default_label = "Initial"

	if legend_label is None:
		legend_label = default_label

	ylabel = get_ylabel(physics, var_name, ylabel)

	# Plot solution
	if create_new_figure:
		plt.figure()

	if dim == 1:
		plot_1D(physics, x, var_plot, ylabel, fmt, legend_label, skip, **kwargs)
	else:
		# if PlotExact: u = u_exact # plot either only numerical or only exact
		plot_2D(physics, x, var_plot, ylabel, regular_2D, equal_AR, **kwargs)

	# plt.xlabel("$x$")
	# ax = plt.gca()
	# handles, labels = ax.get_legend_handles_labels()
	# if handles != []:
	# 	# only create legend if handles can be found
	# 	plt.legend(loc="best")

	# if dim == 1 and include_mesh:
	# 	plot_mesh_1D(mesh, **kwargs)
	# elif dim == 2 and include_mesh:
	# 	PlotMesh2D(mesh, **kwargs)

	if include_mesh:
		plot_mesh(mesh, **kwargs)

	### Finalize plot
	finalize_plot(**kwargs)


def plot_mesh(mesh, EqualAR=False, **kwargs):

	gbasis = mesh.gbasis
	dim = mesh.dim 

	if dim == 1:
		y = plt.ylim()

	'''
	Loop through IFaces and plot interior faces
	'''
	for IFace in mesh.IFaces:
		# Loop through both connected elements to account for periodic 
		# boundaries
		for e in range(2):
			if e == 0:
				elem_id = IFace.elemL_id; face = IFace.faceL_id
			else:
				elem_id = IFace.elemR_id; face = IFace.faceR_id

			# Get local nodes on face
			lfnodes = gbasis.get_local_face_node_nums(mesh.gorder, face)

			# Convert to global node numbering
			# fnodes = mesh.Elem2Nodes[elem][fnodes]

			# # Physical coordinates of global nodes
			# coords = mesh.node_coords[fnodes]

			elem = mesh.elements[elem_id]
			coords = elem.node_coords[lfnodes]
			if dim == 1:
				x = np.full(2, coords[:,0])
			else:
				x = coords[:,0]; y = coords[:,1]

			# Plot face
			plt.plot(x, y, 'k-')

	'''
	Loop through BFaces and plot boundary faces
	'''
	# for BFG in mesh.boundary_groups:
	for BFG in mesh.boundary_groups.values():
		for BFace in BFG.BFaces:
			# Get adjacent element info
			elem_id = BFace.elem_id; face = BFace.face_id

			# Get local nodes on face
			lfnodes = gbasis.get_local_face_node_nums( 
				mesh.gorder, face)

			# Convert to global node numbering
			# fnodes[:] = mesh.Elem2Nodes[elem][fnodes[:]]

			# Physical coordinates of global nodes
			# coords = mesh.node_coords[fnodes]

			elem = mesh.elements[elem_id]
			coords = elem.node_coords[lfnodes]
			if dim == 1:
				x = np.full(2, coords[:,0])
			else:
				x = coords[:,0]; y = coords[:,1]

			# Plot face
			plt.plot(x, y, 'k-')

	'''
	If requested, plot element IDs at element centroids
	'''
	if "show_elem_IDs" in kwargs and kwargs["show_elem_IDs"]:
		for elem_id in range(mesh.num_elems):
			xc = mesh_tools.get_element_centroid(mesh, elem_id)
			if dim == 1:
				yc = np.mean(y)
			else:
				yc = xc[0,1]
			plt.text(xc[0,0], yc, str(elem_id))


	plt.xlabel("$x$")
	if dim == 2: plt.ylabel("$y$")
	if EqualAR:
		plt.gca().set_aspect('equal', adjustable='box')
	# plt.axis("equal")


# def PlotMesh2D(mesh, EqualAR=False, **kwargs):
	
# 	gbasis = mesh.gbasis
# 	# Sanity check
# 	if mesh.dim != 2:
# 		raise ValueError

# 	'''
# 	Loop through IFaces and plot interior faces
# 	'''
# 	for IFace in mesh.IFaces:
# 		# Choose left element
# 		elem = IFace.elemL_id; face = IFace.faceL_id

# 		# Get local nodes on face
# 		fnodes, nfnode = gbasis.local_face_nodes( 
# 			mesh.gorder, face)

# 		# Convert to global node numbering
# 		fnodes[:] = mesh.Elem2Nodes[elem][fnodes[:]]

# 		# Physical coordinates of global nodes
# 		coords = mesh.node_coords[fnodes]
# 		x = coords[:,0]; y = coords[:,1]

# 		# Plot face
# 		plt.plot(x, y, 'k-')

# 	'''
# 	Loop through BFaces and plot boundary faces
# 	'''
# 	for BFG in mesh.boundary_groups:
# 		for BFace in BFG.BFaces:
# 			# Get adjacent element info
# 			elem = BFace.elem_id; face = BFace.face_id

# 			# Get local nodes on face
# 			fnodes, nfnode = gbasis.local_face_nodes( 
# 				mesh.gorder, face)

# 			# Convert to global node numbering
# 			fnodes[:] = mesh.Elem2Nodes[elem][fnodes[:]]

# 			# Physical coordinates of global nodes
# 			coords = mesh.node_coords[fnodes]
# 			x = coords[:,0]; y = coords[:,1]

# 			# Plot face
# 			plt.plot(x, y, 'k-')

# 	'''
# 	If requested, plot element IDs at element centroids
# 	'''
# 	if "show_elem_IDs" in kwargs and kwargs["show_elem_IDs"]:
# 		for elem in range(mesh.num_elems):
# 			xc = mesh_tools.get_element_centroid(mesh, elem)
# 			plt.text(xc[0,0], xc[0,1], str(elem))



# 	plt.xlabel("$x$")
# 	plt.ylabel("$y$")
# 	if EqualAR:
# 		plt.gca().set_aspect('equal', adjustable='box')
# 	# plt.axis("equal")
