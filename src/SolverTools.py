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

	for egrp in range(mesh.nElemGroup):
		for elem in range(mesh.nElems[egrp]):
			U_ = U.Arrays[egrp][elem]
			U_[:,:] = c*np.matmul(MMinv_all.Arrays[egrp][elem], R.Arrays[egrp][elem])

def ProjectStateToNewBasis(solver, EqnSet, mesh, basis_old, Order_old):
	''' Old state '''
	U = EqnSet.U

	''' Allocate new state '''
	# New basis, order information stored in EqnSet
	ArrayDims = [[mesh.nElems[egrp],Basis. Order2nNode(EqnSet.Bases[egrp], EqnSet.Orders[egrp]), EqnSet.StateRank] \
					for egrp in range(mesh.nElemGroup)]
	U_new = Data.ArrayList(nArray=mesh.nElemGroup, ArrayDims=ArrayDims)

	''' Loop through elements '''
	for EG in mesh.ElemGroups:
		basis = EqnSet.Bases[egrp]
		Order = EqnSet.Orders[egrp]
		## New mass matrix inverse (in reference space)
		MMinv,_ = Basis.GetElemInvMassMatrix(mesh, basis, Order)
		## Projection matrix
		PM = Basis.GetProjectionMatrix(mesh, basis_old, Order_old, basis, Order, MMinv)
		for elem in range(EG.nElem):
			Uc = U.Arrays[egrp][elem]
			Uc_new = U_new.Arrays[egrp][elem]

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

	for egrp in range(mesh.nElemGroup):
		for elem in range(mesh.nElems[egrp]):
			U_ = U.Arrays[egrp][elem]
			U_[:,:] = np.matmul(ADERinv_all.Arrays[egrp][elem], W.Arrays[egrp][elem])