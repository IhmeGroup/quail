import numpy as np
import General
from Basis import GetInvMassMatrix, GetStiffnessMatrix, BasisData
from Quadrature import GetQuadOrderElem, QuadData


### Function to calculate mass matrix, stiffness matrix, basis polynomials
def CalculateBasisAndMatrices(mesh, basis, Order):
	## Mass, stiffness matrix
	MMinv,_ = GetInvMassMatrix(mesh, egrp=0, elem=0, basis=basis, Order=Order)
	SM,_ = GetStiffnessMatrix(mesh, egrp=0, elem=0, basis=basis, Order=Order)

	## Evaluate basis polynomials
	# Quadrature
	QuadOrder,_ = GetQuadOrderElem(mesh, egrp=0, basis=basis, Order=Order)
	quadData = QuadData(mesh=mesh, egrp=0, entity=General.EntityType["IFace"], Order=QuadOrder)
	xq = quadData.xquad; nq = quadData.nquad; 
	# Basis on left face
	PhiDataLeft = BasisData(basis,Order,nq,mesh)
	PhiDataLeft.EvalBasisOnFace(mesh, egrp=0, face=0, xq=xq, xelem=None, Get_Phi=True)
	PhiLeft = PhiDataLeft.Phi.transpose() # [nn,1]
	# Basis on right face
	PhiDataRight = BasisData(basis,Order,nq,mesh)
	PhiDataRight.EvalBasisOnFace(mesh, egrp=0, face=1, xq=xq, xelem=None, Get_Phi=True)
	PhiRight = PhiDataRight.Phi.transpose() # [nn,1]
	nn = PhiDataLeft.nn

	return MMinv, SM, PhiLeft, PhiRight, nn


### Function to calculate eigenvalues
def GetEigValues(MMinv, SM, PhiLeft, PhiRight, L, p, h):
	A = np.matmul(-MMinv, SM + np.matmul(PhiLeft, PhiRight.transpose())*np.exp(-1.j*L*(p+1)) \
			- np.matmul(PhiRight, PhiRight.transpose()))

	# Eigenvalues
	alpha,_ = np.linalg.eig(A) 
	Omega = h*alpha/1.j

	Omega_r = np.real(Omega)
	Omega_i = np.imag(Omega)

	return Omega_r, Omega_i