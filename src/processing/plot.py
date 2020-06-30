import code
import copy
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.tri as tri
import numpy as np

import meshing.meshbase as mesh_defs
import meshing.tools as mesh_tools

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


def Plot1D(EqnSet, x, u, SolnLabel, VariableName=None, u_exact=None, u_IC=None, 
		u_var_calculated=False, **kwargs):
	### reshape
	# uplot = u[:,:,iplot]
	# uplot = np.reshape(uplot, (-1,))
	# u_exact = np.reshape(u_exact, (-1,1))
	x = np.reshape(x, (-1,))
	nplot = x.shape[0]
	u.shape = nplot,-1
	if not u_var_calculated:
		uplot = EqnSet.ComputeScalars(VariableName, u)
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
		u_ex = EqnSet.ComputeScalars(VariableName, u_exact)
		plt.plot(x,u_ex,'k-',label="Exact")

	if u_IC is not None: 
		# u_ex = u_exact[:,:,iplot]
		# u_ex.shape = -1,
		u_IC.shape = nplot,-1
		u_i = EqnSet.ComputeScalars(VariableName, u_IC)
		plt.plot(x,u_i,'k--',label="Initial")
	plt.plot(x,uplot,'bo',label="DG") 
	plt.ylabel(SolnLabel)


def triangulate(EqnSet, x, u, variable_name):
	### Remove duplicates
	x.shape = -1,2
	nold = x.shape[0]
	x, idx = np.unique(x, axis=0, return_index=True)

	### Flatten
	X = x[:,0].flatten()
	Y = x[:,1].flatten()
	u.shape = nold,-1
	U = EqnSet.ComputeScalars(variable_name, u).flatten()
	# U = u[:,:,iplot].flatten()
	U = U[idx]

	### Triangulation
	triang = tri.Triangulation(X, Y)
	# refiner = tri.UniformTriRefiner(triang)
	# tris, utri = refiner.refine_field(U, subdiv=0)
	tris = triang; utri = U

	return tris, utri


def Plot2D_Regular(EqnSet, x, u, VariableName, SolnLabel, EqualAR=False, **kwargs):
	'''
	Function: Plot2D
	-------------------
	This function plots 2D contours via a triangulation

	NOTES:
			For coarse solutions, the resulting plot may misrepresent
			the actual solution due to bias in the triangulation

	INPUTS:
	    x: (x,y) coordinates; 3D array of size [nElemTot x nPoint x 2], where
	    	nPoint is the number of points per element
	    u: values of solution at the points in x; 3D array of size
	    	[nElemTot x nPoint x NUM_STATE_VARS]

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
	# U = EqnSet.ComputeScalars(VariableName, u).flatten()
	# # U = u[:,:,iplot].flatten()
	# U = U[idx]

	# ### Triangulation
	# triang = tri.Triangulation(X, Y)
	# # refiner = tri.UniformTriRefiner(triang)
	# # tris, utri = refiner.refine_field(U, subdiv=0)
	# tris = triang; utri = U

	tris, utri = triangulate(EqnSet, x, u, VariableName)
	if "nlevels" in kwargs:
		TCF = plt.tricontourf(tris, utri, kwargs["nlevels"])
	elif "levels" in kwargs:
		TCF = plt.tricontourf(tris, utri, levels=kwargs["levels"])
	else:
		TCF = plt.tricontourf(tris, utri)

	### Plot contours 
	# cb = plt.colorbar()
	# cb.ax.set_title(SolnLabel)
	# plt.ylabel("$y$")
	if "ShowTriangulation" in kwargs:
		if kwargs["ShowTriangulation"]: 
			plt.triplot(triang, lw=0.5, color='white')
	# plt.axis("equal")

	return TCF.levels


def Plot2D_General(EqnSet, x, u, VariableName, SolnLabel, EqualAR=False, **kwargs):
	'''
	Function: Plot2D
	-------------------
	This function plots 2D contours via a triangulation

	NOTES:
			For coarse solutions, the resulting plot may misrepresent
			the actual solution due to bias in the triangulation

	INPUTS:
	    x: (x,y) coordinates; 3D array of size [nElemTot x nPoint x 2], where
	    	nPoint is the number of points per element
	    u: values of solution at the points in x; 3D array of size
	    	[nElemTot x nPoint x NUM_STATE_VARS]

	OUTPUTS:
	    Plot is created
	'''

	''' If not specified, get default contour levels '''
	if "levels" not in kwargs:
		# u1 = np.copy(u)
		# u1.shape = -1,u.shape[-1]
		# u2 = EqnSet.ComputeScalars(VariableName, u1, u1.shape[0]).flatten()
		# u2.shape = u.shape[0:2]
		# figtmp = plt.figure()
		# if "nlevels" in kwargs:
		# 	CS = plt.contourf(u2, kwargs["nlevels"])
		# else:
		# 	CS = plt.contourf(u2)
		# levels = CS.levels
		# plt.close(figtmp)

		figtmp = plt.figure()
		levels = Plot2D_Regular(EqnSet, np.copy(x), np.copy(u), VariableName, SolnLabel, **kwargs)
		# plt.colorbar()
		# ShowPlot()
		plt.close(figtmp)
	else:
		levels = kwargs["levels"]

	nElemTot = x.shape[0]
	nPoint = x.shape[1]
	''' Loop through elements '''
	for elem in range(nElemTot):
		# Extract x and y
		# X = x[elem,:,0].flatten()
		# Y = x[elem,:,1].flatten()
		# # Compute requested scalar
		# U = EqnSet.ComputeScalars(VariableName, u[elem,:,:]).flatten()
		# # Triangulation
		# triang = tri.Triangulation(X, Y)
		# tris = triang; utri = U

		tris, utri = triangulate(EqnSet, x[elem], u[elem], VariableName)
		# Plot
		plt.tricontourf(tris, utri, levels=levels, extend="both")
		# if "nlevels" in kwargs:
		# 	plt.tricontourf(tris, utri, kwargs["nlevels"])
		# elif "levels" in kwargs:
		# 	plt.tricontourf(tris, utri, levels=kwargs["levels"], extend="both")
		# else:
		# 	plt.tricontourf(tris, utri)
		if "ShowTriangulation" in kwargs:
			if kwargs["ShowTriangulation"]: 
				plt.triplot(triang, lw=0.5, color='white')


def Plot2D(EqnSet, x, u, VariableName, SolnLabel, Regular2D, EqualAR=False, **kwargs):
	if Regular2D:
		Plot2D_Regular(EqnSet, x, u, VariableName, SolnLabel, EqualAR, **kwargs)
	else:
		Plot2D_General(EqnSet, x, u, VariableName, SolnLabel, EqualAR, **kwargs)

	''' Label plot '''
	if "ignore_colorbar" in kwargs and kwargs["ignore_colorbar"]:
		pass 
		# do nothing
	else:
		cb = plt.colorbar()
		cb.ax.set_title(SolnLabel)
	plt.ylabel("$y$")
	if EqualAR:
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


def interpolate_2D_soln_to_points(EqnSet, x, u, xpoints, variable_name):
	tris, utri = triangulate(EqnSet, x, u, variable_name)
	interpolator = tri.LinearTriInterpolator(tris, utri)

	upoints = interpolator(xpoints[:,0], xpoints[:,1])

	return upoints


def plot_line_probe(mesh, EqnSet, solver, variable_name, xy1, xy2, nPoint=101, PlotExact=False, PlotIC=False, 
		Label=None, vs_x=True, **kwargs):

	# Construct points on line segment
	x1 = xy1[0]; y1 = xy1[1]
	x2 = xy2[0]; y2 = xy2[1]
	xline = np.linspace(x1, x2, nPoint)
	yline = np.linspace(y1, y2, nPoint)

	# Interpolation
	x, u = get_sample_points(mesh, EqnSet, solver.basis, True)
	xyline = np.array([xline,yline]).transpose()
	uline = interpolate_2D_soln_to_points(EqnSet, x, u, xyline, variable_name)

	# Analytical?
	u_exact, u_IC = get_analytical_solution(EqnSet, xyline, solver.Time, PlotExact, PlotIC)
	# if u_exact is not None:
	# 	u_exact = EqnSet.ComputeScalars(VariableName, u_exact)
	# 	# u_exact = interpolate_2D_soln_to_points(EqnSet, x, u_exact, xyline, variable_name)
	# if u_IC is not None:
	# 	u_IC = EqnSet.ComputeScalars(VariableName, u_IC)
		# u_IC = interpolate_2D_soln_to_points(EqnSet, x, u_IC, xyline, variable_name)

	SolnLabel = get_solution_label(EqnSet, variable_name, Label)

	plt.figure()
	if vs_x:
		xlabel = "x"
		line = xline
	else:
		xlabel = "y"
		line = yline
	Plot1D(EqnSet, line, uline, SolnLabel, variable_name, u_exact, u_IC, u_var_calculated=True, **kwargs)

	# code.interact(local=locals())

	### Finalize plot
	finalize_plot(xlabel=xlabel, **kwargs)
	# plt.xlabel("$x$")
	# ax = plt.gca()
	# handles, labels = ax.get_legend_handles_labels()
	# if handles != []:
	# 	# only create legend if handles can be found
	# 	plt.legend(loc="best")



def get_sample_points(mesh, EqnSet, basis, equidistant):
	## Extract data
	dim = mesh.Dim
	U = EqnSet.U
	order = EqnSet.order
	sr = EqnSet.NUM_STATE_VARS

	# Get points to plot at
	# Note: assumes uniform element type
	if equidistant:
		xpoint, npoint = basis.equidistant_nodes(max([1,3*order]))
	else:
		quad_order = basis.get_quadrature(mesh, max([2,2*order]), physics=EqnSet)
		gbasis = mesh.gbasis
		xpoint,_ = gbasis.get_quad_data(quad_order)

		# QuadOrder,_ = get_gaussian_quadrature_elem(mesh, basis, max([2,2*Order]), EqnSet)
		# quadData = QuadData(mesh, mesh.gbasis, EntityType.Element, QuadOrder)
		# xpoint = gbasis.quad_pts
		npoint = xpoint.shape[0]

	u = np.zeros([mesh.nElem,npoint,sr])
	# u_exact = np.copy(u)
	x = np.zeros([mesh.nElem,npoint,dim])
	# PhiData = Basis.BasisData(EqnSet.Basis,Order,mesh)
	basis.eval_basis(xpoint, True, False, False, None)
	GeomPhiData = None
	el = 0
	for elem in range(mesh.nElem):
		U_ = U[elem]

		xphys, GeomPhiData = mesh_defs.ref_to_phys(mesh, elem, GeomPhiData, xpoint)
		x[el,:,:] = xphys
		u[el,:,:] = np.matmul(basis.basis_val, U_)

		el += 1

	return x, u


def get_analytical_solution(EqnSet, x, time, get_exact, get_IC, u=None):
	# Exact solution?
	if get_exact:
		u_exact = EqnSet.CallFunction(EqnSet.ExactSoln, x=np.reshape(x, (-1,EqnSet.dim)), t=time)
		if u is not None: u_exact.shape = u.shape
	else:
		u_exact = None
	# IC ?
	if get_IC:
		u_IC = EqnSet.CallFunction(EqnSet.IC, x=np.reshape(x,(-1,EqnSet.dim)), t=0.)
		if u is not None: u_IC.shape = u.shape
	else:
		u_IC = None

	return u_exact, u_IC


def get_solution_label(EqnSet, variable_name, label=None):
	if label is None:
		try:
			label = EqnSet.StateVariables[variable_name].value
		except KeyError:
			label = EqnSet.AdditionalVariables[variable_name].value
	soln_label = "$" + label + "$"

	return soln_label


def PlotSolution(mesh, EqnSet, solver, VariableName, create_new_figure=True, PlotExact=False, PlotIC=False, Label=None, Equidistant=True,
	include_mesh=False, Regular2D=False, EqualAR=False, **kwargs):

	# iplot_sr = EqnSet.VariableType[VariableName]
	if PlotExact:
		# if not EqnSet.ExactSoln.Function:
		if EqnSet.ExactSoln is None:
			raise Exception("No exact solution provided")

	## Extract params
	EndTime = solver.Time
	dim = mesh.Dim

	# Get sample points
	x, u = get_sample_points(mesh, EqnSet, solver.basis, Equidistant)

	# # Exact solution?
	# if PlotExact:
	# 	u_exact = EqnSet.CallFunction(EqnSet.ExactSoln, x=np.reshape(x, (-1,dim)), Time=EndTime)
	# 	u_exact.shape = u.shape
	# else:
	# 	u_exact = None
	# # IC ?
	# if PlotIC:
	# 	u_IC = EqnSet.CallFunction(EqnSet.IC, x=np.reshape(x,(-1,dim)),Time=0.)
	# 	u_IC.shape = u.shape
	# else:
	# 	u_IC = None
	# Solution label

	u_exact, u_IC = get_analytical_solution(EqnSet, x, EndTime, PlotExact, PlotIC, u)

	# if Label is None:
	# 	try:
	# 		Label = EqnSet.StateVariables[VariableName].value
	# 	except KeyError:
	# 		Label = EqnSet.AdditionalVariables[VariableName].value
	# SolnLabel = "$" + Label + "$"

	SolnLabel = get_solution_label(EqnSet, VariableName, Label)

	# Plot solution
	if create_new_figure:
		plt.figure()

	if dim == 1:
		Plot1D(EqnSet, x, u, SolnLabel, VariableName, u_exact, u_IC, **kwargs)
	else:
		if PlotExact: u = u_exact # plot either only numerical or only exact
		Plot2D(EqnSet, x, u, VariableName, SolnLabel, Regular2D, EqualAR, **kwargs)

	### Finalize plot
	finalize_plot(**kwargs)

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


def plot_mesh(mesh, EqualAR=False, **kwargs):

	gbasis = mesh.gbasis
	dim = mesh.Dim 

	if dim == 1:
		y = plt.ylim()

	'''
	Loop through IFaces and plot interior faces
	'''
	for IFace in mesh.IFaces:
		# Loop through both connected elements to account for periodic boundaries
		for e in range(2):
			if e == 0:
				elem = IFace.ElemL; face = IFace.faceL
			else:
				elem = IFace.ElemR; face = IFace.faceR

			# Get local nodes on face
			fnodes, nfnode = gbasis.local_face_nodes( 
				mesh.gorder, face)

			# Convert to global node numbering
			fnodes[:] = mesh.Elem2Nodes[elem][fnodes[:]]

			# Physical coordinates of global nodes
			coords = mesh.Coords[fnodes]
			if dim == 1:
				x = np.full(2, coords[:,0])
			else:
				x = coords[:,0]; y = coords[:,1]

			# Plot face
			plt.plot(x, y, 'k-')

	'''
	Loop through BFaces and plot boundary faces
	'''
	# for BFG in mesh.BFaceGroups:
	for BFG in mesh.BFaceGroups.values():
		for BFace in BFG.BFaces:
			# Get adjacent element info
			elem = BFace.Elem; face = BFace.face

			# Get local nodes on face
			fnodes, nfnode = gbasis.local_face_nodes( 
				mesh.gorder, face)

			# Convert to global node numbering
			fnodes[:] = mesh.Elem2Nodes[elem][fnodes[:]]

			# Physical coordinates of global nodes
			coords = mesh.Coords[fnodes]
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
		for elem in range(mesh.nElem):
			xc = mesh_tools.get_element_centroid(mesh, elem)
			if dim == 1:
				yc = np.mean(y)
			else:
				yc = xc[0,1]
			plt.text(xc[0,0], yc, str(elem))


	plt.xlabel("$x$")
	if dim == 2: plt.ylabel("$y$")
	if EqualAR:
		plt.gca().set_aspect('equal', adjustable='box')
	# plt.axis("equal")


# def PlotMesh2D(mesh, EqualAR=False, **kwargs):
	
# 	gbasis = mesh.gbasis
# 	# Sanity check
# 	if mesh.Dim != 2:
# 		raise ValueError

# 	'''
# 	Loop through IFaces and plot interior faces
# 	'''
# 	for IFace in mesh.IFaces:
# 		# Choose left element
# 		elem = IFace.ElemL; face = IFace.faceL

# 		# Get local nodes on face
# 		fnodes, nfnode = gbasis.local_face_nodes( 
# 			mesh.gorder, face)

# 		# Convert to global node numbering
# 		fnodes[:] = mesh.Elem2Nodes[elem][fnodes[:]]

# 		# Physical coordinates of global nodes
# 		coords = mesh.Coords[fnodes]
# 		x = coords[:,0]; y = coords[:,1]

# 		# Plot face
# 		plt.plot(x, y, 'k-')

# 	'''
# 	Loop through BFaces and plot boundary faces
# 	'''
# 	for BFG in mesh.BFaceGroups:
# 		for BFace in BFG.BFaces:
# 			# Get adjacent element info
# 			elem = BFace.Elem; face = BFace.face

# 			# Get local nodes on face
# 			fnodes, nfnode = gbasis.local_face_nodes( 
# 				mesh.gorder, face)

# 			# Convert to global node numbering
# 			fnodes[:] = mesh.Elem2Nodes[elem][fnodes[:]]

# 			# Physical coordinates of global nodes
# 			coords = mesh.Coords[fnodes]
# 			x = coords[:,0]; y = coords[:,1]

# 			# Plot face
# 			plt.plot(x, y, 'k-')

# 	'''
# 	If requested, plot element IDs at element centroids
# 	'''
# 	if "show_elem_IDs" in kwargs and kwargs["show_elem_IDs"]:
# 		for elem in range(mesh.nElem):
# 			xc = mesh_tools.get_element_centroid(mesh, elem)
# 			plt.text(xc[0,0], xc[0,1], str(elem))



# 	plt.xlabel("$x$")
# 	plt.ylabel("$y$")
# 	if EqualAR:
# 		plt.gca().set_aspect('equal', adjustable='box')
# 	# plt.axis("equal")
