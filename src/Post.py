import numpy as np
import copy
import code
from Quadrature import get_gaussian_quadrature_elem, QuadData
import Basis
from Mesh import Mesh, ref_to_phys
from General import *
import MeshTools
from Data import ArrayList


def L2_error(mesh,EqnSet,solver,VariableName,PrintError=True,NormalizeByVolume=True):

	Time = solver.Time
	U = EqnSet.U
	basis = solver.basis
	# Check for exact solution
	if not EqnSet.ExactSoln.Function:
		raise Exception("No exact solution provided")

	# Get elem volumes 
	if NormalizeByVolume:
		TotVol,_ = MeshTools.element_volumes(mesh)
	else:
		TotVol = 1.

	# Get error
	# ElemErr = copy.deepcopy(U)
	# ElemErr = ArrayList(SimilarArray=EqnSet.U).Arrays
	# ElemErr = ArrayList(nArray=mesh.nElemGroup,ArrayDims=[mesh.nElems])
	ElemErr = np.zeros([mesh.nElem])
	TotErr = 0.
	sr = EqnSet.StateRank
	quadData = None
	# JData = JacobianData(mesh)
	# ier = EqnSet.VariableType[VariableName]
	GeomPhiData = None

	# ElemErr.Arrays[egrp][:] = 0.

	Order = EqnSet.Order
	# basis = EqnSet.Basis

	for elem in range(mesh.nElem):
		U_ = U[elem]

		QuadOrder,QuadChanged = get_gaussian_quadrature_elem(mesh, basis, 2*np.amax([Order,1]), EqnSet, quadData)
		if QuadChanged:
			quadData = QuadData(mesh, mesh.gbasis, EntityType.Element, QuadOrder)

		xq = quadData.quad_pts
		wq = quadData.quad_wts
		nq = xq.shape[0]
		
		if QuadChanged:
			# PhiData = BasisData(basis,Order,mesh)
			basis.eval_basis(xq, True, False, False, None)
			xphys = np.zeros([nq, mesh.Dim])

		djac,_,_ = Basis.element_jacobian(mesh,elem,xq,get_djac=True)

		xphys, GeomPhiData = ref_to_phys(mesh, elem, GeomPhiData, xq, xphys, QuadChanged)
		u_exact = EqnSet.CallFunction(EqnSet.ExactSoln, x=xphys, Time=Time)

		# interpolate state at quad points
		u = np.zeros([nq, sr])
		for ir in range(sr):
			u[:,ir] = np.matmul(basis.basis_val, U_[:,ir])
		u[:] = np.matmul(basis.basis_val, U_)

		# Computed requested quantity
		s = EqnSet.ComputeScalars(VariableName, u)
		s_exact = EqnSet.ComputeScalars(VariableName, u_exact)

		# err = 0.
		# for iq in range(nq):
		# 	err += (s[iq] - s_exact[iq])**2.*wq[iq] * JData.djac[iq*(JData.nq != 1)]
		err = np.sum((s - s_exact)**2.*wq*djac)
		ElemErr[elem] = err
		TotErr += ElemErr[elem]

	# TotErr /= TotVol
	TotErr = np.sqrt(TotErr/TotVol)

	# print("Total volume = %g" % (TotVol))
	if PrintError:
		print("Total error = %.15f" % (TotErr))

	return TotErr, ElemErr