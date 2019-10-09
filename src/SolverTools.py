import numpy as np
import copy
import code
from Quadrature import *
from Basis import *
from Mesh import *
from General import *
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy.optimize import fsolve
import MeshTools


def SmoothIsentropic1D(x,t,gam):
	a = 0.9 # hard-coded
	rho0 = lambda x,a: 1 + a*np.sin(np.pi*x)
	pressure = lambda rho,gam: rho**gam
	rho = lambda x1,x2,a: 0.5*(rho0(x1,a) + rho0(x2,a))
	vel = lambda x1,x2,a: np.sqrt(3)*(rho(x1,x2,a) - rho0(x1,a))

	f1 = lambda x1,x,t,a: x + np.sqrt(3)*rho0(x1,a)*t - x1
	f2 = lambda x2,x,t,a: x - np.sqrt(3)*rho0(x2,a)*t - x2

	x_ = x.reshape(-1)
	x1 = fsolve(f1, 0.*x_, (x_,t,a))
	if np.abs(x1.any()) > 1.: raise Exception("x1 = %g out of range" % (x1))
	x2 = fsolve(f2, 0.*x_, (x_,t,a))
	if np.abs(x2.any()) > 1.: raise Exception("x2 = %g out of range" % (x2))

	r = rho(x1,x2,a)
	u = vel(x1,x2,a)
	p = pressure(r,gam)

	rE = p/(gam-1.) + 0.5*r*u*u

	U = np.array([r, r*u, rE]).transpose() # [nq,sr]

	return U


def MultInvMassMatrix(mesh, solver, dt, R, U):
	EqnSet = solver.EqnSet
	DataSet = solver.DataSet
	# if MMinv is None:
	# 	MMinv = GetInvMassMatrix(mesh, 0, 0, EqnSet.Orders[0])

	try:
		MMinv_all = DataSet.MMinv_all
	except AttributeError:
		# not found; need to compute
		MMinv_all = ComputeInvMassMatrices(mesh, EqnSet, solver=solver)


	if dt is None:
		c = 1.
	else:
		c = dt
	# if not U:
	# 	U = copy.deepcopy(R)

	for egrp in range(mesh.nElemGroup):
		for elem in range(mesh.nElems[egrp]):
			U_ = U.Arrays[egrp][elem]
			U_[:,:] = c*np.matmul(MMinv_all.Arrays[egrp][elem], R.Arrays[egrp][elem])
			# code.interact(local=locals())


def L2_error(mesh,EqnSet,EndTime,VariableName):
	# [err,ele_err] = l2_error(x,t,u,P,basis,f_exact,N)
	# Computes the L2 error of an enriched function compared to f_exact
	# x - the locations of the ends of the elements
	# t - time
	# u - coefficients
	# P - polynomial order of basis
	# basis - element basis polynomials
	# f_exact - exact function
	# N - number of elements
	# quad_ints - quadrature intervals

	U = EqnSet.U.Arrays

	# Check for exact solution
	if not EqnSet.ExactSoln.Function:
		raise Exception("No exact solution provided")

	# Get elem volumes 
	TotVol,_ = MeshTools.ElementVolumes(mesh)
	# # ElemVol = copy.deepcopy(U)
	# ElemVol = ArrayList(SimilarArray=EqnSet.U).Arrays
	# TotVol = 0.
	# quadData = None
	# JData = JacobianData(mesh)
	# for egrp in range(mesh.nElemGroup):
	# 	ElemVol[egrp][:] = 0.

	# 	Order = mesh.ElemGroups[egrp].QOrder

	# 	QuadOrder,QuadChanged = GetQuadOrderElem(egrp, Order, EqnSet.Bases[egrp], mesh, quadData=quadData)
	# 	if QuadChanged:
	# 		quadData = QuadData(QuadOrder, EntityType.Element, egrp, mesh)

	# 	nq = quadData.nquad
	# 	xq = quadData.xquad
	# 	wq = quadData.wquad

	# 	# PhiData = BasisData(egrp,Order,entity,nq,xq,mesh,True,True)
	# 	# PhiData = BasisData(ShapeType.Segment,Order,nq,mesh)
	# 	# PhiData.EvalBasis(xq, True, False, False, None)

	# 	for elem in range(mesh.nElems[egrp]):
	# 		JData.ElemJacobian(egrp,elem,nq,xq,mesh,Get_detJ=True)

	# 		for iq in range(nq):
	# 			ElemVol[egrp][elem] += wq[iq] * JData.detJ[iq*(JData.nq != 1)]

	# 		TotVol += ElemVol[egrp][elem,0,0]


	# Get error
	# ElemErr = copy.deepcopy(U)
	# ElemErr = ArrayList(SimilarArray=EqnSet.U).Arrays
	ElemErr = ArrayList(nArray=mesh.nElemGroup,ArrayDims=[mesh.nElems])
	TotErr = 0.
	sr = EqnSet.StateRank
	quadData = None
	JData = JacobianData(mesh)
	ier = EqnSet.VariableType[VariableName]
	for egrp in range(mesh.nElemGroup):
		ElemErr.Arrays[egrp][:] = 0.

		Order = EqnSet.Orders[egrp]
		Basis = EqnSet.Bases[egrp]

		for elem in range(mesh.nElems[egrp]):
			U_ = U[egrp][elem]

			QuadOrder,QuadChanged = GetQuadOrderElem(egrp, 2*np.amax([Order,1]), Basis, mesh, EqnSet, quadData)
			if QuadChanged:
				quadData = QuadData(QuadOrder, EntityType.Element, egrp, mesh)

			nq = quadData.nquad
			xq = quadData.xquad
			wq = quadData.wquad

			if QuadChanged:
				# PhiData = BasisData(egrp,Order,entity,nq,xq,mesh,True,True)
				PhiData = BasisData(Basis,Order,nq,mesh)
				PhiData.EvalBasis(xq, True, False, False, None)
				xphys = np.zeros([nq, mesh.Dim])

			JData.ElemJacobian(egrp,elem,nq,xq,mesh,Get_detJ=True)

			xphys = Ref2Phys(mesh, egrp, elem, PhiData, nq, xq, xphys)
			# if sr == 1: u_exact = f_exact(xphys, EndTime)
			# else: u_exact = SmoothIsentropic1D(x=xphys,t=EndTime,gam=EqnSet.Params["SpecificHeatRatio"])
			u_exact = EqnSet.CallFunction(EqnSet.ExactSoln, x=xphys, Time=EndTime)

			# interpolate state at quad points
			u = np.zeros([nq, sr])
			for ir in range(sr):
				u[:,ir] = np.matmul(PhiData.Phi, U_[:,ir])

			err = 0.
			for iq in range(nq):
				err += (u[iq,ier] - u_exact[iq,ier])**2.*wq[iq] * JData.detJ[iq*(JData.nq != 1)]
			ElemErr.Arrays[egrp][elem] = err
			TotErr += ElemErr.Arrays[egrp][elem]

	# TotErr /= TotVol
	TotErr = np.sqrt(TotErr/TotVol)

	# print("Total volume = %g" % (TotVol))
	print("Total error = %g" % (TotErr))

	return TotErr, ElemErr




    # for i in range(N):
    #     xl = x[i]
    #     xr = x[i+1]
    #     jac = (xr-xl)/2.
        
    #     ele_err[i] = np.sum(jac*g_w*(eval_basis(ref2phy(g_pt,xl,xr),xl,xr,u[:,i],P,basis)-f_exact(ref2phy(g_pt,xl,xr),t))**2)
    #     err += np.sum(jac*g_w*(eval_basis(ref2phy(g_pt,xl,xr),xl,xr,u[:,i],P,basis)-f_exact(ref2phy(g_pt,xl,xr),t))**2)
    
    # err = np.sqrt(err/(x[-1]-x[0]))
    # #err = (err/(x(end)-x(1)))

    # return err,ele_err

def Plot1D(mesh, EqnSet, EndTime, VariableName, PlotExact=True, Label=None):
	# plot setup
	fontsize=12
	plt.close("all")
	# fig, ax = plt.subplots()
	# plt.tick_params(axis="both",direction="in")
	# fig = plt.figure()
	# ax = fig.gca()
	# ax.plot(x, y)
	font = {'family':'serif', 'serif': ['computer modern roman']}
	plt.rc('font',**font)
	mpl.rcParams['font.size']=fontsize
	plt.rc('text',usetex=True)

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
	QuadOrder,_ = GetQuadOrderElem(egrp, 2*Order, EqnSet.Bases[egrp], mesh, EqnSet)
	quadData = QuadData(QuadOrder, EntityType.Element, egrp, mesh)
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
	if PlotExact: plt.plot(x,u_exact,'k-',label="Exact",markersize=ms)
	plt.plot(x,uplot,'bo',label="DG",markersize=ms)
	# plot_local_poly(u,basis,N,P,x,n,tfinal,'g--')
	plt.xlabel("$x$")
	plt.ylabel(ylabel)
	# plt.ylabel("$\\rho$")
	# plt.ylabel("$\\rho E$")
	plt.legend(loc="best")
	# plt.legend(("Exact","DG"))
	plt.show()


