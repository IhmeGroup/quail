import numpy as np
import copy
import code
from Quadrature import GetQuadOrderElem, QuadData
from Basis import BasisData, JacobianData
from Mesh import Mesh, Ref2Phys
from General import *
import matplotlib as mpl
from matplotlib import pyplot as plt


def PreparePlot(close_all=True, fontsize=12., font={'family':'serif', 'serif': ['computer modern roman']},
	linewidth=1.5, markersize=4.0):
	# Note: matplotlib.rcdefaults() returns to default settings

	# For now, always use tex syntax
	plt.rc('text',usetex=True)

	if close_all:
		plt.close("all")
	mpl.rcParams['font.size']=fontsize
	plt.rc('font',**font)
	mpl.rcParams['lines.linewidth'] = linewidth
	mpl.rcParams['lines.markersize'] = markersize


def Plot1D(mesh, EqnSet, EndTime, VariableName, PlotExact=True, Label=None):
	# plot setup
	# fontsize=12
	# plt.close("all")
	# fig, ax = plt.subplots()
	# plt.tick_params(axis="both",direction="in")
	# fig = plt.figure()
	# ax = fig.gca()
	# ax.plot(x, y)
	# font = {'family':'serif', 'serif': ['computer modern roman']}
	# plt.rc('font',**font)
	# mpl.rcParams['font.size']=fontsize
	# plt.rc('text',usetex=True)

	# iplot_sr = 2 # which state variable to plot
	# if EqnSet.StateRank == 1: iplot_sr = 0

	iplot_sr = EqnSet.VariableType[VariableName]

	if PlotExact:
		if not EqnSet.ExactSoln.Function:
			raise Exception("No exact solution provided")

	## Extract data
	dim = mesh.Dim
	U = EqnSet.U.Arrays
	Order = np.amax(EqnSet.Orders)
	egrp = np.where(Order == EqnSet.Orders)[0][0]
	QuadOrder,_ = GetQuadOrderElem(mesh, egrp, EqnSet.Bases[egrp], 2*Order, EqnSet)
	quadData = QuadData(mesh, egrp, EntityType.Element, QuadOrder)
	nq = quadData.nquad
	xq = quadData.xquad
	wq = quadData.wquad
	sr = EqnSet.StateRank
	u = np.zeros([mesh.nElemTot,nq,sr])
	# u_exact = np.copy(u)
	x = np.zeros([mesh.nElemTot,nq,dim])
	PhiData = BasisData(EqnSet.Bases[egrp],Order,nq,mesh)
	PhiData.EvalBasis(xq, True, False, False, None)
	el = 0
	for egrp in range(mesh.nElemGroup):
		for elem in range(mesh.nElems[egrp]):
			U_ = U[egrp][elem]

			JData = JacobianData(mesh)
			JData.ElemJacobian(egrp,elem,nq,xq,mesh,Get_detJ=True)

			xphys = Ref2Phys(mesh, egrp, elem, PhiData, nq, xq)
			x[el,:,:] = xphys
			# u_exact[el,:,:] = f_exact(xphys, EndTime)

			# interpolate state at quad points
			for ir in range(sr):
				u[el,:,ir] = np.matmul(PhiData.Phi, U_[:,ir])

			el += 1

	### reshape
	uplot = u[:,:,iplot_sr]
	uplot = np.reshape(uplot, (-1,))
	# u_exact = np.reshape(u_exact, (-1,1))
	x = np.reshape(x, (-1,))

	### sort
	idx = np.argsort(x)
	idx.shape = -1,
	uplot = uplot[idx]
	# u_exact = u_exact[idx]
	x = x[idx]

	# if sr == 1: 
	# 	u_exact = f_exact(x,EndTime)
	# else: 
	# 	u_exact_all = SmoothIsentropic1D(x=x,t=EndTime,gam=EqnSet.Params["SpecificHeatRatio"])
	# 	u_exact = u_exact_all[:,iplot_sr]

	if PlotExact:
		u_exact = EqnSet.CallFunction(EqnSet.ExactSoln, x=np.reshape(x, (-1,1)), Time=EndTime)
		u_exact = u_exact[:,iplot_sr]

	# Solution label
	if Label is None:
		try:
			Label = EqnSet.VarLabelType[VariableName].value
		except KeyError:
			Label = "u"
	ylabel = "$" + Label + "$"

	### plot
	ms = 4
	if PlotExact: plt.plot(x,u_exact,'k-',label="Exact") #,markersize=ms)
	plt.plot(x,uplot,'bo',label="DG") #markersize=ms)
	# plot_local_poly(u,basis,N,P,x,n,tfinal,'g--')
	plt.xlabel("$x$")
	plt.ylabel(ylabel)
	# plt.ylabel("$\\rho$")
	# plt.ylabel("$\\rho E$")
	plt.legend(loc="best")
	# plt.legend(("Exact","DG"))
	plt.show()

