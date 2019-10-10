import numpy as np
import code
from Basis import ComputeInvMassMatrices


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
			# code.interact(local=locals())
			