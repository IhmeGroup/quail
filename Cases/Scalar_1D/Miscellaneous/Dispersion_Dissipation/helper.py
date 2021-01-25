import numpy as np
import general
from Basis import GetInvMassMatrix, get_stiffness_matrix, BasisData
from Quadrature import get_gaussian_quadrature_elem, QuadData
import code


### Function to calculate mass matrix, stiffness matrix, basis polynomials
def CalculateBasisAndMatrices(mesh, basis, Order):
	## Mass, stiffness matrix
	MMinv,_ = get_elem_inv_mass_matrix(mesh, basis=basis, Order=Order, elem=0, physical_space=True)
	SM = get_stiffness_matrix(mesh, basis=basis, Order=Order, elem=0)

	## Evaluate basis polynomials
	# Quadrature
	QuadOrder,_ = get_gaussian_quadrature_elem(mesh, egrp=0, basis=basis, Order=Order)
	quadData = QuadData(mesh=mesh, basis=mesh.ElemGroups[0].QBasis, entity=General.EntityType["IFace"], Order=QuadOrder)
	xq = quadData.quad_pts; nq = quad_pts.shape[0]; 
	# Basis on left face
	PhiDataLeft = BasisData(basis,Order,mesh)
	PhiDataLeft.get_basis_face_val_grads(mesh, egrp=0, face=0, xq=xq, xelem=None, get_val=True)
	PhiLeft = PhiDataLeft.Phi.transpose() # [nn,1]
	# Basis on right face
	PhiDataRight = BasisData(basis,Order,mesh)
	PhiDataRight.get_basis_face_val_grads(mesh, egrp=0, face=1, xq=xq, xelem=None, get_val=True)
	PhiRight = PhiDataRight.Phi.transpose() # [nn,1]
	nn = PhiDataLeft.Phi.shape[1]

	return MMinv, SM, PhiLeft, PhiRight, nn


### Function to calculate eigenvalues
def GetEigValues(MMinv, SM, PhiLeft, PhiRight, L, p, h, alpha, solver=None):
	# A = np.matmul(-MMinv, SM + np.matmul(PhiLeft, PhiRight.transpose())*np.exp(-1.j*L*(p+1)) \
	# 		- np.matmul(PhiRight, PhiRight.transpose()))

	# # Eigenvalues
	# eigs,_ = np.linalg.eig(A) 
	# Omega = h*eigs/1.j

	A = np.matmul(-h/1.j*MMinv, SM + np.matmul(0.5*PhiLeft, alpha*PhiLeft.transpose() + \
		(2. - alpha)*PhiRight.transpose()*np.exp(-1.j*L*(p+1))) \
		- np.matmul(0.5*PhiRight, (2.-alpha)*PhiRight.transpose() + \
		alpha*np.exp(1.j*L*(p+1))*PhiLeft.transpose()))

	# Eigenvalues
	Omega,v = np.linalg.eig(A) 

	Omega_r = np.real(Omega)
	Omega_i = np.imag(Omega)

	# if solver is not None:
	# 	physics = solver.physics
	# 	physics.IC.Set(theta = 1.j*L*(p+1))
	# 	solver.init_state()
	# 	U = physics.U.Arrays
	# 	Unorm = U/np.linalg.norm(U)

	return Omega_r, Omega_i