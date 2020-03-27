import numpy as np
import copy
import code
import Quadrature
import Basis
import Mesh
from General import *
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.tri as tri


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


def Plot1D(EqnSet, x, u, VariableName, SolnLabel, u_exact, u_IC, **kwargs):
	### reshape
	# uplot = u[:,:,iplot]
	# uplot = np.reshape(uplot, (-1,))
	# u_exact = np.reshape(u_exact, (-1,1))
	x = np.reshape(x, (-1,))
	nplot = x.shape[0]
	u.shape = nplot,-1
	uplot = EqnSet.ComputeScalars(VariableName, u, nplot)

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
		u_ex = EqnSet.ComputeScalars(VariableName, u_exact, nplot)
		plt.plot(x,u_ex,'k-',label="Exact")

	if u_IC is not None: 
		# u_ex = u_exact[:,:,iplot]
		# u_ex.shape = -1,
		u_IC.shape = nplot,-1
		u_i = EqnSet.ComputeScalars(VariableName, u_IC, nplot)
		plt.plot(x,u_i,'k--',label="Initial")
	plt.plot(x,uplot,'bo',label="DG") 
	plt.ylabel(SolnLabel)


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
	    	[nElemTot x nPoint x StateRank]

	OUTPUTS:
	    Plot is created
	'''

	### Remove duplicates
	x.shape = -1,2
	nold = x.shape[0]
	x, idx = np.unique(x, axis=0, return_index=True)

	### Flatten
	X = x[:,0].flatten()
	Y = x[:,1].flatten()
	u.shape = nold,-1
	U = EqnSet.ComputeScalars(VariableName, u, nold).flatten()
	# U = u[:,:,iplot].flatten()
	U = U[idx]

	### Triangulation
	triang = tri.Triangulation(X, Y)
	# refiner = tri.UniformTriRefiner(triang)
	# tris, utri = refiner.refine_field(U, subdiv=0)
	tris = triang; utri = U
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
	    	[nElemTot x nPoint x StateRank]

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
		X = x[elem,:,0].flatten()
		Y = x[elem,:,1].flatten()
		# Compute requested scalar
		U = EqnSet.ComputeScalars(VariableName, u[elem,:,:], nPoint).flatten()
		# Triangulation
		triang = tri.Triangulation(X, Y)
		tris = triang; utri = U
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
	cb = plt.colorbar()
	cb.ax.set_title(SolnLabel)
	plt.ylabel("$y$")
	if EqualAR:
		plt.gca().set_aspect('equal', adjustable='box')
	# plt.axis("equal")


def PlotSolution(mesh, EqnSet, EndTime, VariableName, PlotExact=False, PlotIC=False, Label=None, Equidistant=True,
	IncludeMesh2D=False, Regular2D=False, EqualAR=False, **kwargs):

	# iplot_sr = EqnSet.VariableType[VariableName]

	if PlotExact:
		if not EqnSet.ExactSoln.Function:
			raise Exception("No exact solution provided")

	## Extract data
	dim = mesh.Dim
	U = EqnSet.U.Arrays
	Order = np.amax(EqnSet.Orders)
	egrp = np.where(Order == EqnSet.Orders)[0][0]
	sr = EqnSet.StateRank
	# Get points to plot at
	# Note: assumes uniform element type
	if Equidistant:
		xpoint, npoint = Basis.EquidistantNodes(EqnSet.Bases[egrp], max([1,3*Order]))
	else:
		QuadOrder,_ = Quadrature.GetQuadOrderElem(mesh, egrp, EqnSet.Bases[egrp], max([2,2*Order]), EqnSet)
		quadData = Quadrature.QuadData(mesh, mesh.ElemGroups[egrp].QBasis, EntityType.Element, QuadOrder)
		npoint = quadData.nquad
		xpoint = quadData.xquad
	
	u = np.zeros([mesh.nElemTot,npoint,sr])
	# u_exact = np.copy(u)
	x = np.zeros([mesh.nElemTot,npoint,dim])
	PhiData = Basis.BasisData(EqnSet.Bases[egrp],Order,npoint,mesh)
	PhiData.EvalBasis(xpoint, True, False, False, None)
	GeomPhiData = None
	el = 0
	for egrp in range(mesh.nElemGroup):
		for elem in range(mesh.nElems[egrp]):
			U_ = U[egrp][elem]

			JData = Basis.JacobianData(mesh)
			JData.ElemJacobian(egrp,elem,npoint,xpoint,mesh,Get_detJ=True)

			xphys, GeomPhiData = Mesh.Ref2Phys(mesh, egrp, elem, GeomPhiData, npoint, xpoint)
			x[el,:,:] = xphys
			# u_exact[el,:,:] = f_exact(xphys, EndTime)

			# interpolate state at quad points
			for ir in range(sr):
				u[el,:,ir] = np.matmul(PhiData.Phi, U_[:,ir])

			el += 1

	# Exact solution?
	if PlotExact:
		u_exact = EqnSet.CallFunction(EqnSet.ExactSoln, x=np.reshape(x, (-1,dim)), Time=EndTime)
		u_exact.shape = u.shape
	else:
		u_exact = None
	# IC ?
	if PlotIC:
		u_IC = EqnSet.CallFunction(EqnSet.IC, x=np.reshape(x,(-1,dim)),Time=0.)
		u_IC.shape = u.shape
	else:
		u_IC = None
	# Solution label
	if Label is None:
		try:
			Label = EqnSet.StateVariables[VariableName].value
		except KeyError:
			Label = EqnSet.AdditionalVariables[VariableName].value
	SolnLabel = "$" + Label + "$"

	# Plot solution
	plt.figure()
	if dim == 1:
		Plot1D(EqnSet, x, u, VariableName, SolnLabel, u_exact, u_IC, **kwargs)
	else:
		if PlotExact: u = u_exact # plot either only numerical or only exact
		Plot2D(EqnSet, x, u, VariableName, SolnLabel, Regular2D, EqualAR, **kwargs)

	### Finalize plot
	plt.xlabel("$x$")
	ax = plt.gca()
	handles, labels = ax.get_legend_handles_labels()
	if handles != []:
		# only create legend if handles can be found
		plt.legend(loc="best")

	if dim == 2 and IncludeMesh2D:
		PlotMesh2D(mesh, **kwargs)


def PlotMesh2D(mesh, EqualAR=False, **kwargs):
	# Sanity check
	if mesh.Dim != 2:
		raise ValueError

	'''
	Loop through IFaces and plot interior faces
	'''
	for IFace in mesh.IFaces:
		# Choose adjacent element with higher QOrder
		QOrderL = mesh.ElemGroups[IFace.ElemGroupL].QOrder
		QOrderR = mesh.ElemGroups[IFace.ElemGroupR].QOrder
		if QOrderL >= QOrderR:
			egrp = IFace.ElemGroupL; elem = IFace.ElemL; face = IFace.faceL
		else:
			egrp = IFace.ElemGroupR; elem = IFace.ElemR; face = IFace.faceR

		EG = mesh.ElemGroups[egrp]

		# Get local nodes on face
		fnodes, nfnode = Basis.LocalFaceNodes(EG.QBasis, 
			EG.QOrder, face)

		# Convert to global node numbering
		fnodes[:] = EG.Elem2Nodes[elem][fnodes[:]]

		# Physical coordinates of global nodes
		coords = mesh.Coords[fnodes]
		x = coords[:,0]; y = coords[:,1]

		# Plot face
		plt.plot(x, y, 'k-')

	'''
	Loop through BFaces and plot boundary faces
	'''
	for BFG in mesh.BFaceGroups:
		for BFace in BFG.BFaces:
			# Get adjacent element info
			egrp = BFace.ElemGroup; elem = BFace.Elem; face = BFace.face

			EG = mesh.ElemGroups[egrp]

			# Get local nodes on face
			fnodes, nfnode = Basis.LocalFaceNodes(EG.QBasis, 
				EG.QOrder, face)

			# Convert to global node numbering
			fnodes[:] = EG.Elem2Nodes[elem][fnodes[:]]

			# Physical coordinates of global nodes
			coords = mesh.Coords[fnodes]
			x = coords[:,0]; y = coords[:,1]

			# Plot face
			plt.plot(x, y, 'k-')


	plt.xlabel("$x$")
	plt.ylabel("$y$")
	if EqualAR:
		plt.gca().set_aspect('equal', adjustable='box')
	# plt.axis("equal")




