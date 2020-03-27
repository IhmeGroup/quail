import numpy as np
import code
import Basis
import Data

def MultInvMassMatrix(mesh, solver, dt, R, U):
	EqnSet = solver.EqnSet
	DataSet = solver.DataSet
	# if MMinv is None:
	# 	MMinv = GetInvMassMatrix(mesh, 0, 0, EqnSet.Orders[0])

	try:
		MMinv_all = DataSet.MMinv_all
	except AttributeError:
		# not found; need to compute
		MMinv_all = Basis.ComputeInvMassMatrices(mesh, EqnSet, solver=solver)


	if dt is None:
		c = 1.
	else:
		c = dt
	# if not U:
	# 	U = copy.deepcopy(R)

	for elem in range(mesh.nElem):
		U_ = U[elem]
		U_[:,:] = c*np.matmul(MMinv_all[elem], R[elem])

def ProjectStateToNewBasis(solver, EqnSet, mesh, basis_old, Order_old):
	''' Old state '''
	U = EqnSet.U

	''' Allocate new state '''
	# New basis, order information stored in EqnSet
	# ArrayDims = [[mesh.nElems[egrp],Basis.Order2nNode(EqnSet.Bases[egrp], EqnSet.Orders[egrp]), EqnSet.StateRank] \
	# 				for egrp in range(mesh.nElemGroup)]
	# U_new = Data.ArrayList(nArray=mesh.nElemGroup, ArrayDims=ArrayDims)
	U_new = np.zeros([mesh.nElem, Basis.Order2nNode(EqnSet.Basis, EqnSet.Order), EqnSet.StateRank])

	''' Loop through elements '''
	basis = EqnSet.Basis
	Order = EqnSet.Order
	## New mass matrix inverse (in reference space)
	MMinv,_ = Basis.GetElemInvMassMatrix(mesh, basis, Order)
	## Projection matrix
	PM = Basis.GetProjectionMatrix(mesh, basis_old, Order_old, basis, Order, MMinv)
	for elem in range(mesh.nElem):
		Uc = U[elem]
		Uc_new = U_new[elem]

		# New coefficients
		Uc_new[:] = np.matmul(PM, Uc)

	''' Store in EqnSet '''
	delattr(EqnSet, "U")
	EqnSet.U = U_new
			

def MultInvADER(mesh, solver, dt, W, U):
	EqnSet = solver.EqnSet
	DataSet = solver.DataSet
	# if MMinv is None:
	# 	MMinv = GetInvMassMatrix(mesh, 0, 0, EqnSet.Orders[0])
	try:
		ADERinv_all = DataSet.ADERinv_all
	except AttributeError:
		# not found; need to compute
		ADERinv_all = Basis.ComputeInvADERMatrices(mesh, EqnSet, dt, solver=solver)

	for elem in range(mesh.nElem):
		U_ = U[elem]
		U_[:,:] = np.matmul(ADERinv_all[elem], W[elem])