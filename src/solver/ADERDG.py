import code
import numpy as np 

from numerics.quadrature.quadrature import get_gaussian_quadrature_elem, get_gaussian_quadrature_face, QuadData
from numerics.basis.basis import *
import numerics.timestepping.stepper as stepper
import numerics.limiter as Limiter

from meshing.meshbase import *
import meshing.tools as MeshTools

from solver.tools import project_state_to_new_basis
import processing.post as Post
import processing.readwritedatafiles as ReadWriteDataFiles

from data import ArrayList, GenericData
import errors

global echeck
echeck = -1

from solver.DG import *


class ElemOperatorsADER(ElemOperators):

	def get_basis_and_geom_data(self, mesh, basis, order):
		# separate these later

		# Unpack
		quad_pts = self.quad_pts 
		basis.eval_basis(quad_pts, Get_Phi=True, Get_GPhi=False)

		self.basis_val = basis.basis_val

class IFaceOperatorsADER(IFaceOperators):

	def get_gaussian_quadrature(self, mesh, EqnSet, basis, order):

		QuadOrder, _ = get_gaussian_quadrature_face(mesh, None, mesh.gbasis, order, EqnSet, None)
		quadData = QuadData(mesh, basis, EntityType.IFace, QuadOrder)
		self.quad_pts = quadData.quad_pts
		self.quad_wts = quadData.quad_wts

	def get_basis_and_geom_data(self, mesh, basis, order):
		# separate these later

		# Unpack
		dim = mesh.Dim
		quad_pts = self.quad_pts 
		nq = quad_pts.shape[0]
		nb = basis.get_num_basis_coeff(order)
		nFacePerElem = mesh.nFacePerElem + 2

		# Allocate
		self.faces_to_basisL = np.zeros([nFacePerElem,nq,nb])
		self.faces_to_basisR = np.zeros([nFacePerElem,nq,nb])
		self.normals_ifaces = np.zeros([mesh.nIFace,nq,dim])

		for f in range(nFacePerElem):
			# Left
			#eval_basis_on_face_ader(mesh, basis_st, face_stL, quad_pts_st, xelemLPhi, Get_Phi=True)
			_ = basis.eval_basis_on_face_ader(mesh, basis, f, quad_pts, None, Get_Phi=True)
			self.faces_to_basisL[f] = basis.basis_val
			# Right
			_ = basis.eval_basis_on_face_ader(mesh, basis, f, quad_pts[::-1], None, Get_Phi=True)
			self.faces_to_basisR[f] = basis.basis_val

		i = 0
		for IFace in mesh.IFaces:
			# Normals
			nvec = iface_normal(mesh, IFace, quad_pts)
			self.normals_ifaces[i] = nvec
			i += 1

class BFaceOperatorsADER(IFaceOperatorsADER):

	def get_basis_and_geom_data(self, mesh, basis, order):
		# separate these later

		# Unpack
		dim = mesh.Dim
		quad_pts = self.quad_pts 
		nq = quad_pts.shape[0]
		nb = basis.get_num_basis_coeff(order)
		nFacePerElem = mesh.nFacePerElem + 2

		# Allocate
		self.faces_to_basis = np.zeros([nFacePerElem,nq,nb])
		self.faces_to_xref = np.zeros([nFacePerElem,nq,dim+1])
		self.normals_bfgroups = []
		self.x_bfgroups = []

		GeomPhiData = None

		for f in range(nFacePerElem):
			# Left
			self.faces_to_xref[f] = xref = basis.eval_basis_on_face_ader(mesh, basis, f, quad_pts, None, Get_Phi=True)
			self.faces_to_basis[f] = basis.basis_val

		i = 0
		for BFG in mesh.BFaceGroups:
			self.normals_bfgroups.append(np.zeros([BFG.nBFace,nq,dim]))
			self.x_bfgroups.append(np.zeros([BFG.nBFace,nq,dim]))
			normal_bfgroup = self.normals_bfgroups[i]
			x_bfgroup = self.x_bfgroups[i]
			j = 0
			for BFace in BFG.BFaces:
				# Normals
				nvec = bface_normal(mesh, BFace, quad_pts)
				normal_bfgroup[j] = nvec

				# Physical coordinates of quadrature points
				x, GeomPhiData = ref_to_phys(mesh, BFace.Elem, GeomPhiData, self.faces_to_xref[BFace.face], None, True)
				# Store
				x_bfgroup[j] = x

				# Increment
				j += 1
			i += 1

	def alloc_other_arrays(self, EqnSet, basis, order):
		quad_pts = self.quad_pts 
		nq = quad_pts.shape[0]
		ns = EqnSet.StateRank

		self.UqI = np.zeros([nq, ns])
		self.UqB = np.zeros([nq, ns])
		self.Fq = np.zeros([nq, ns])


class ADEROperators(object):
	def __init__(self):
		self.MM = None
		self.iMM = None
		self.K = None
		self.iK = None 
		self.FTL = None 
		self.FTR = None
		self.SMT = None
		self.SMS_elems = None		

	def calc_ader_matrices(self, mesh, basis, basis_st, order, dt):

		nb = basis_st.nb
		dim = mesh.Dim

		SMS_elems = np.zeros([mesh.nElem, nb, nb, dim])
		iMM_elems = np.zeros([mesh.nElem, nb, nb])

		# Get flux matrices in time
		FTL = get_temporal_flux_ader(mesh, basis_st, basis_st, order, elem=0, PhysicalSpace=False)
		FTR = get_temporal_flux_ader(mesh, basis_st, basis, order, elem=0, PhysicalSpace=False)

		# Get stiffness matrix in time
		SMT = get_stiffness_matrix_ader(mesh, basis, basis_st, order, dt, elem=0, gradDir=1, PhysicalSpace = False)

		# Get stiffness matrix in space
		for elem in range(mesh.nElem):
			SMS = get_stiffness_matrix_ader(mesh, basis, basis_st, order, dt, elem, gradDir=0, PhysicalSpace = True)
			SMS_elems[elem,:,:,0] = SMS.transpose()
			# iMM =  get_elem_inv_mass_matrix_ader(mesh, basis_st, order, elem, PhysicalSpace=True)
			# iMM_elems[elem,:,:] = iMM

		# Get mass matrix in space-time
		MM =  get_elem_mass_matrix_ader(mesh, basis_st, order, elem=-1, PhysicalSpace=False)
		iMM =  get_elem_inv_mass_matrix_ader(mesh, basis_st, order, elem=-1, PhysicalSpace=True)

		self.FTL = FTL
		self.FTR = FTR
		self.SMT = SMT
		self.SMS_elems = SMS_elems
		self.MM = MM
		self.iMM = iMM
		self.K = FTL - SMT
		self.iK = np.linalg.inv(self.K) 

	def get_geom_data(self, mesh, basis, order):

		# Unpack
		dim = mesh.Dim 
		nElem = mesh.nElem 
		nb = basis.nb
		xnode = None
		gbasis=mesh.gbasis
		xnode, nnode = gbasis.equidistant_nodes(order,xnode)

		# Allocate
		self.jac_elems = np.zeros([nElem,nb,dim,dim])
		self.ijac_elems = np.zeros([nElem,nb,dim,dim])
		self.djac_elems = np.zeros([nElem,nb,1])
		self.x_elems = np.zeros([nElem,nb,dim])

		GeomPhiData = None

		for elem in range(mesh.nElem):
			# Jacobian
			djac, jac, ijac = element_jacobian(mesh, elem, xnode, get_djac=True, get_jac=True, get_ijac=True)


			self.jac_elems[elem] = np.tile(jac,(nnode,1,1))
			self.ijac_elems[elem] = np.tile(ijac,(nnode,1,1))
			self.djac_elems[elem] = np.tile(djac,(nnode,1))

			# Physical coordinates of nodal points
			x, GeomPhiData = ref_to_phys(mesh, elem, GeomPhiData, xnode)
			# Store
			self.x_elems[elem] = np.tile(x,(nnode,1))


	def compute_operators(self, mesh, EqnSet, basis, basis_st, order, dt):

		self.calc_ader_matrices(mesh, basis, basis_st, order, dt)
		self.get_geom_data(mesh, basis_st, order)

class ADERDG_Solver(DG_Solver):
	'''
	Class: ADERDG_Solver
	--------------------------------------------------------------------------
	Use the ADER-DG method to solve a given set of PDEs
	'''
	
	def check_solver_params(self):
		'''
		Method: check_solver_params
		--------------------------------------------------------------------------
		Checks the validity of the solver parameters

		'''
		Params = self.Params
		mesh = self.mesh
		EqnSet = self.EqnSet
		### Check interp basis validity
		if BasisType[Params["InterpBasis"]] == BasisType.LagrangeEqSeg or BasisType[Params["InterpBasis"]] == BasisType.LegendreSeg:
		    if mesh.Dim != 1:
		        raise Errors.IncompatibleError
		else:
		    if mesh.Dim != 2:
		        raise Errors.IncompatibleError

		### Check limiter ###
		if Params["ApplyLimiter"] is 'ScalarPositivityPreserving' \
			and EqnSet.StateRank > 1:
				raise IncompatibleError
		if Params["ApplyLimiter"] is 'PositivityPreserving' \
			and EqnSet.StateRank == 1:
				raise IncompatibleError

		### Check time integration scheme ###
		TimeScheme = Params["TimeScheme"]
		if TimeScheme is not "ADER":
			raise Errors.IncompatibleError

		### Check flux/source coefficient interpolation compatability with basis functions.
		if Params["InterpolateFlux"] is True and BasisType[Params["InterpBasis"]] == BasisType.LegendreSeg:
			raise Errors.IncompatibleError


	def precompute_matrix_operators(self):
		mesh = self.mesh 
		EqnSet = self.EqnSet

		basis = self.basis
		basis_st = self.basis_st

		dt = self.Params['EndTime']/self.Params['nTimeStep']

		self.elem_operators = ElemOperators()
		self.elem_operators.compute_operators(mesh, EqnSet, basis, EqnSet.Order)
		self.iface_operators = IFaceOperators()
		self.iface_operators.compute_operators(mesh, EqnSet, basis, EqnSet.Order)
		self.bface_operators = BFaceOperators()
		self.bface_operators.compute_operators(mesh, EqnSet, basis, EqnSet.Order)

		# Calculate ADER specific space-time operators
		self.elem_operators_st = ElemOperatorsADER()
		self.elem_operators_st.compute_operators(mesh, EqnSet, basis_st, EqnSet.Order)
		self.iface_operators_st = IFaceOperatorsADER()
		self.iface_operators_st.compute_operators(mesh, EqnSet, basis_st, EqnSet.Order)
		self.bface_operators_st = BFaceOperatorsADER()
		self.bface_operators_st.compute_operators(mesh, EqnSet, basis_st, EqnSet.Order)

		self.ader_operators = ADEROperators()
		self.ader_operators.compute_operators(mesh, EqnSet, basis, basis_st, EqnSet.Order, dt)

	def calculate_predictor_step(self, dt, W, Up):
		'''
		Method: calculate_predictor_step
		-------------------------------------------
		Calls the predictor step for each element

		INPUTS:
			dt: time step 
			W: previous time step solution in space only

		OUTPUTS:
			Up: predicted solution in space-time
		'''
		mesh = self.mesh
		EqnSet = self.EqnSet

		for elem in range(mesh.nElem):
			Up[elem] = self.calculate_predictor_elem(elem, dt, W[elem], Up[elem])

		return Up

	def calculate_predictor_elem(self, elem, dt, Wp, Up):
		'''
		Method: calculate_predictor_elem
		-------------------------------------------
		Calculates the predicted solution state for the ADER-DG method using a nonlinear solve of the
		weak form of the DG discretization in time.

		INPUTS:
			elem: element index
			dt: time step 
			W: previous time step solution in space only

		OUTPUTS:
			Up: predicted solution in space-time
		'''
		EqnSet = self.EqnSet
		ns = EqnSet.StateRank
		mesh = self.mesh

		basis = self.basis #basis2
		basis_st = self.basis_st #basis1

		order = EqnSet.Order
		
		elem_ops = self.elem_operators
		ader_ops = self.ader_operators
		
		quad_wts = elem_ops.quad_wts
		basis_val = elem_ops.basis_val 
		djac_elems = elem_ops.djac_elems 
		
		djac = djac_elems[elem]
		# _, ElemVols = MeshTools.element_volumes(mesh, self)

		FTR = ader_ops.FTR
		MM = ader_ops.MM
		SMS = ader_ops.SMS_elems[elem]

		# K = ader_ops.K
		iK = ader_ops.iK
		# W_bar = np.zeros([1,ns])
		# Wq = np.matmul(basis_val, Wp)
		# vol = ElemVols[elem]

		# W_bar[:] = np.matmul(Wq.transpose(),quad_wts*djac).T/vol

		# Wh = np.average(W)

		# def F(u):
		# 	S = 0.
		# 	S = EqnSet.SourceState(1, 0., 0., u, S)
		# 	F = u - S[0,0] - W_bar[0,0]
		# 	return F

		#U_bar = fsolve(F, W_bar)
		# nu= -100000.
		# dsdu = nu
		# Up[:] = U_bar
		#code.interact(local=locals())
		#dsdu = (1./nu)*(2.*U_bar-3.*U_bar**2 - 0.5 +2.*U_bar*0.5)
		#dsdu = (1./nu)*(2.*Up-3.*Up**2 - 0.5 +2.*Up*0.5)
		#code.interact(local=locals())

		### Hacky implementation of implicit source term
		# Kp = K-MM*dt*dsdu

		# iK = np.linalg.inv(Kp)

		srcpoly = self.source_coefficients(elem, dt, order, basis_st, Up)
		flux = self.flux_coefficients(elem, dt, order, basis_st, Up)

		ntest = 10
		for i in range(ntest):

			# Up_new = np.matmul(iK,(np.matmul(MM,srcpoly)-np.matmul(SMS,fluxpoly)+np.matmul(FTR,Wp)-np.matmul(MM,dt*dsdu*Up)))
			Up_new = np.matmul(iK,(np.matmul(MM,srcpoly)-np.einsum('ijk,jlk->il',SMS,flux)+np.matmul(FTR,Wp)))
			err = Up_new - Up

			if np.amax(np.abs(err))<1e-10:
				Up = Up_new
				break

			Up = Up_new
			
			srcpoly = self.source_coefficients(elem, dt, order, basis_st, Up)
			fluxpoly = self.flux_coefficients(elem, dt, order, basis_st, Up)


		return Up

	def calculate_residual_elem(self, elem, Up, ER):
		'''
		Method: calculate_residual_elem
		-------------------------------------------
		Calculates the residual from the volume integral for each element

		INPUTS:
			elem: element index
			U: solution state

		OUTPUTS:
		ER: calculated residiual array (for volume integral of specified element)
		'''
		EqnSet = self.EqnSet
		mesh = self.mesh
		ns = EqnSet.StateRank
		dim = EqnSet.Dim

		elem_ops = self.elem_operators
		elem_ops_st = self.elem_operators_st
		quad_wts = elem_ops.quad_wts
		quad_wts_st = elem_ops_st.quad_wts
		quad_pts_st = elem_ops_st.quad_pts
		basis_val = elem_ops.basis_val 
		basis_val_st = elem_ops_st.basis_val
		basis_pgrad_elems = elem_ops.basis_pgrad_elems
		djac_elems = elem_ops.djac_elems 
		x_elems = elem_ops.x_elems
		Sq = elem_ops_st.Sq

		TimePhiData = None

		# Unpack
		basis_pgrad = basis_pgrad_elems[elem]
		djac = djac_elems[elem]
		nq = quad_wts.shape[0]
		nq_st = quad_wts_st.shape[0]
		nb = basis_val.shape[1]
		x = x_elems[elem]

		# interpolate state and gradient at quad points
		Uq = np.matmul(basis_val_st, Up)

		Fq = EqnSet.ConvFluxInterior(Uq, F=None) # [nq,sr,dim]

		# for ir in range(sr):
		# 	for k in range(nn): # Loop over basis function in space
		# 		for i in range(nq): # Loop over time
		# 			for j in range(nq): # Loop over space
		# 				gPsi = PsiData.gPhi[j,k]
		# 				ER[k,ir] += wq[i]*wq[j]*JData.djac[j*(JData.nq!=1)]*F[i,j,ir]*gPsi

		# F = np.reshape(F,(nqST,sr,dim))
		
		ER += np.tensordot(np.tile(basis_pgrad,(nb,1,1)), Fq*(quad_wts_st.reshape(nq,nq)*djac).reshape(nq_st,1,1), axes=([0,2],[0,2])) # [nb, ns]
		
		t = np.zeros([nq_st,dim])
		t, TimePhiData = ref_to_phys_time(mesh, elem, self.Time, self.Stepper.dt, TimePhiData, quad_pts_st, t, None)
		
		Sq[:] = 0.
		Sq = EqnSet.SourceState(nq_st, x, t, Uq, Sq) # [nq,sr,dim]

		# s = np.reshape(s,(nq,nq,sr))
		# #Calculate source term integral
		# for ir in range(sr):
		# 	for k in range(nn):
		# 		for i in range(nq): # Loop over time
		# 			for j in range(nq): # Loop over space
		# 				Psi = PsiData.Phi[j,k]
		# 				ER[k,ir] += wq[i]*wq[j]*s[i,j,ir]*JData.djac[j*(JData.nq!=1)]*Psi
		# s = np.reshape(s,(nqST,sr))

		ER += np.matmul(np.tile(basis_val,(nb,1)).transpose(),Sq*(quad_wts_st.reshape(nq,nq)*djac).reshape(nq_st,1)) # [nb, ns]

		if elem == echeck:
			code.interact(local=locals())

		return ER

	def calculate_residual_iface(self, iiface, UpL, UpR, RL, RR):
		'''
		Method: calculate_residual_iface
		-------------------------------------------
		Calculates the residual from the boundary integral for each internal face

		INPUTS:
			iiface: internal face index
			UL: solution array from left neighboring element
			UR: solution array from right neighboring element
			
		OUTPUTS:
			RL: calculated residual array (left neighboring element contribution)
			RR: calculated residual array (right neighboring element contribution)
		'''
		mesh = self.mesh
		EqnSet = self.EqnSet
		IFace = mesh.IFaces[iiface]
		elemL = IFace.ElemL
		elemR = IFace.ElemR
		faceL = IFace.faceL
		faceR = IFace.faceR

		if faceL == 0:
			faceL_st = 3
		elif faceL == 1:
			faceL_st = 1
		else:
			return IncompatibleError
		if faceR == 0:
			faceR_st = 3
		elif faceR == 1:
			faceR_st = 1
		else:
			return IncompatibleError

		iface_ops = self.iface_operators
		iface_ops_st = self.iface_operators_st
		quad_pts = iface_ops.quad_pts
		quad_wts = iface_ops.quad_wts
		quad_wts_st = iface_ops_st.quad_wts
		faces_to_basisL = iface_ops.faces_to_basisL
		faces_to_basisR = iface_ops.faces_to_basisR
		faces_to_basisL_st = iface_ops_st.faces_to_basisL
		faces_to_basisR_st = iface_ops_st.faces_to_basisR

		normals_ifaces = iface_ops.normals_ifaces
		UqL = iface_ops.UqL
		UqR = iface_ops.UqR
		Fq = iface_ops.Fq

		nq = quad_wts.shape[0]
		nq_st = quad_wts_st.shape[0]

		basis_valL = faces_to_basisL[faceL]
		basis_valR = faces_to_basisR[faceR]

		basis_valL_st = faces_to_basisL_st[faceL_st]
		basis_valR_st = faces_to_basisR_st[faceR_st]

		nbL = basis_valL.shape[1]
		nbR = basis_valR.shape[1]
		nb = np.amax([nbL,nbR])

		UqL = np.matmul(basis_valL_st, UpL)
		UqR = np.matmul(basis_valR_st, UpR)

		normals = normals_ifaces[iiface]

		Fq = EqnSet.ConvFluxNumerical(UqL, UqR, normals, nq_st, GenericData()) # [nq_st,ns]

		# F = np.reshape(F,(nq,nqST,sr))
		
		# for ir in range(sr):
		# 	#for k in range(nn): # Loop over basis function in space
		# 	for i in range(nqST): # Loop over time
		# 		for j in range(nq): # Loop over space
		# 			PsiL = PsiDataL.Phi[j,:]
		# 			PsiR = PsiDataR.Phi[j,:]
		# 			RL[:,ir] -= wqST[i]*wq[j]*F[j,i,ir]*PsiL
		# 			RR[:,ir] += wqST[i]*wq[j]*F[j,i,ir]*PsiR

		# F = np.reshape(F,(nqST,sr))


		RL -= np.matmul(np.tile(basis_valL,(nb,1)).transpose(), Fq*quad_wts_st*quad_wts) # [nb, ns]
		RR += np.matmul(np.tile(basis_valR,(nb,1)).transpose(), Fq*quad_wts_st*quad_wts) # [nb, ns]

		if elemL == echeck or elemR == echeck:
			if elemL == echeck: print("Left!")
			else: print("Right!")
			code.interact(local=locals())

		return RL, RR

	def calculate_residual_bface(self, ibfgrp, ibface, U, R):
		'''
		Method: calculate_residual_bface
		-------------------------------------------
		Calculates the residual from the boundary integral for each boundary face

		INPUTS:
			ibfgrp: index of BC group
			ibface: index of boundary face
			U: solution array from internal element
			
		OUTPUTS:
			R: calculated residual array (from boundary face)
		'''
		mesh = self.mesh
		dim = mesh.Dim
		EqnSet = self.EqnSet
		BFG = mesh.BFaceGroups[ibfgrp]
		BFace = BFG.BFaces[ibface]
		elem = BFace.Elem
		face = BFace.face

		bface_ops = self.bface_operators
		bface_ops_st = self.bface_operators_st
		quad_pts = bface_ops.quad_pts
		quad_wts = bface_ops.quad_wts
		quad_pts_st = bface_ops_st.quad_pts
		quad_wts_st = bface_ops_st.quad_wts
		faces_to_xref_st = bface_ops_st.faces_to_xref

		faces_to_basis = bface_ops.faces_to_basis
		faces_to_basis_st = bface_ops_st.faces_to_basis
		normals_bfgroups = bface_ops.normals_bfgroups
		x_bfgroups = bface_ops.x_bfgroups
		UqI = bface_ops_st.UqI
		UqB = bface_ops_st.UqB
		Fq = bface_ops.Fq


		if face == 0:
			face_st = 3
		elif face == 1:
			face_st = 1
		else:
			return IncompatibleError
	
		basis_val = faces_to_basis[face]
		basis_val_st = faces_to_basis_st[face_st]
		xref_st = faces_to_xref_st[face_st]

		nq_st = quad_wts_st.shape[0]
		nb = basis_val.shape[1]

		TimePhiData = None

		t = np.zeros([nq_st,dim])
		t, TimePhiData = ref_to_phys_time(mesh, elem, self.Time, self.Stepper.dt, TimePhiData, xref_st, t, None)

		# interpolate state and gradient at quad points
		UqI = np.matmul(basis_val_st, U)

		normals = normals_bfgroups[ibfgrp][ibface]
		x = x_bfgroups[ibfgrp][ibface]

		# Get boundary state
		BC = EqnSet.BCs[ibfgrp]
		UqB = EqnSet.BoundaryState(BC, nq_st, x, t, normals, UqI, UqB)

		Fq = EqnSet.ConvFluxBoundary(BC, UqI, UqB, normals, nq_st, GenericData()) # [nq_st,ns]

		# F = np.reshape(F,(nq,nqST,sr))

		# for ir in range(sr):
		# 	for i in range(nqST): # Loop over time
		# 		for j in range(nq): # Loop over space
		# 			Psi = PsiData.Phi[j,:]
		# 			R[:,ir] -= wqST[i]*wq[j]*F[j,i,ir]*Psi
	
		# F = np.reshape(F,(nqST,sr))

		R -= np.matmul(np.tile(basis_val,(nb,1)).transpose(),Fq*quad_wts_st*quad_wts)

		if elem == echeck:
			code.interact(local=locals())

		return R

	def flux_coefficients(self, elem, dt, order, basis, Up):
		'''
		Method: flux_coefficients
		-------------------------------------------
		Calculates the polynomial coefficients for the flux functions in ADER-DG

		INPUTS:
			elem: element index
			dt: time step size
			order: solution order
			U: solution array
			
		OUTPUTS:
			F: polynomical coefficients of the flux function
		'''

		EqnSet = self.EqnSet
		mesh = self.mesh
		ns = EqnSet.StateRank
		dim = EqnSet.Dim
		Params = self.Params

		InterpolateFlux = Params["InterpolateFlux"]

		elem_ops = self.elem_operators
		djac_elems = elem_ops.djac_elems 
		djac = djac_elems[elem]


		rhs = np.zeros([basis.get_num_basis_coeff(order),ns,dim],dtype=Up.dtype)
		F = np.zeros_like(rhs)

		if not InterpolateFlux:
			
			ader_ops = self.ader_operators
			elem_ops_st = self.elem_operators_st
			basis_val_st = elem_ops_st.basis_val
			
			quad_wts_st = elem_ops_st.quad_wts
			nq_st = quad_wts_st.shape[0]
			quad_pts = elem_ops.quad_pts
			nq = quad_pts.shape[0]
			
			iMM = ader_ops.iMM_elems[elem]

			Uq = np.matmul(basis_val_st, Up)
			Fq = EqnSet.ConvFluxInterior(Uq, F=None) # [nq_st,ns,dim]

			
			rhs *=0.

			# for ir in range(ns):
			# 	for k in range(nb): # Loop over basis function in space
			# 		for i in range(nq): # Loop over time
			# 			for j in range(nq): # Loop over space
			# 				#Phi = PhiData.Phi[j,k]
			# 				rhs[k,ir] += wq[i]*wq[j]*JData.djac[j*(JData.nq!=1)]*f[i,j,ir]*Phi[i,j,k]

			rhs = np.matmul(basis_val_st.transpose(), Fq*(quad_wts_st*(np.tile(djac,(nq,1))))) # [nb, ns]

			F = np.dot(iMM,rhs)*dt/2.0

		else:

			Fq = EqnSet.ConvFluxInterior(Up,F=None)
			F = Fq*dt/2.0

		return F

	def source_coefficients(self, elem, dt, order, basis, Up):
		'''
		Method: source_coefficients
		-------------------------------------------
		Calculates the polynomial coefficients for the source functions in ADER-DG

		INPUTS:
			elem: element index
			dt: time step size
			order: solution order
			U: solution array
			
		OUTPUTS:
			S: polynomical coefficients of the flux function
		'''
		mesh = self.mesh
		dim = mesh.Dim
		EqnSet = self.EqnSet
		ns = EqnSet.StateRank
		Params = self.Params
		entity = EntityType.Element
		InterpolateFlux = Params["InterpolateFlux"]

		elem_ops = self.elem_operators
		elem_ops_st = self.elem_operators_st
		djac_elems = elem_ops.djac_elems 
		djac = djac_elems[elem]
		Sq = elem_ops_st.Sq

		x_elems = elem_ops.x_elems
		x = x_elems[elem]

		TimePhiData = None

		if not InterpolateFlux:

			rhs = np.zeros([basis.get_num_basis_coeff(order),ns,dim],dtype=Up.dtype)

			ader_ops = self.ader_operators
			basis_val_st = elem_ops_st.basis_val

			quad_wts_st = elem_ops_st.quad_wts
			nq_st = quad_wts_st.shape[0]
			quad_pts_st = elem_ops_st.quad_pts
			quad_pts = elem_ops.quad_pts
			nq = quad_pts.shape[0]
			
			Uq = np.matmul(basis_val_st, Up)
			Fq = EqnSet.ConvFluxInterior(Uq, F=None) # [nq_st,ns,dim]

			iMM = ader_ops.iMM_elems[elem]


			t = np.zeros([nq_st,dim])
			t, TimePhiData = ref_to_phys_time(mesh, elem, self.Time, self.Stepper.dt, TimePhiData, quad_pts_st, t, None)

			Uq = np.matmul(basis_val_st,Up)
		
			Sq[:] = 0.
			Sq = EqnSet.SourceState(nq_st, x, t, Uq, Sq) # [nq,sr,dim]

			rhs *=0.

			rhs[:] = np.matmul(basis_val_st.transpose(),Sq*quad_wts_st*(np.tile(djac,(nq,1)))) # [nb, ns]
			S = np.dot(iMM,rhs)*dt/2.0

		else:

			ader_ops = self.ader_operators
			x_elems_ader = ader_ops.x_elems
			x_ader = x_elems_ader[elem]

			xnode, nb = basis.equidistant_nodes(order)

			t = np.zeros([nb,dim])
			t, TimePhiData = ref_to_phys_time(mesh, elem, self.Time, self.Stepper.dt, TimePhiData, xnode, t, None)
			Sq = np.zeros([t.shape[0],ns])

			Sq = EqnSet.SourceState(nb, x_ader, t, Up, Sq)
			S = Sq*dt/2.0

		return S