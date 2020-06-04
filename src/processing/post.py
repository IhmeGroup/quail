import code
import copy
import numpy as np
import sys
from matplotlib import pyplot as plt

from numerics.quadrature.quadrature import get_gaussian_quadrature_elem, QuadData
import numerics.basis.basis as Basis

from meshing.meshbase import Mesh, ref_to_phys
import meshing.tools as MeshTools

from general import *
from data import ArrayList
import errors

import processing.plot as Plot



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

	Order = EqnSet.order
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


def get_boundary_info(mesh, EqnSet, solver, bname, variable_name, integrate=True, 
		vec=0., dot_normal_with_vec=False, plot_vs_x=False, plot_vs_y=False, Label=None):

	if mesh.Dim != 2:
		raise Errors.IncompatibleError
	# Find boundary face group
	found = False
	ibfgrp = 0
	for BFG in mesh.BFaceGroups:
		if BFG.Name == bname:
			found = True
			break
		ibfgrp += 1

	if not found:
		raise Errors.DoesNotExistError

	bface_ops = solver.bface_operators
	quad_pts = bface_ops.quad_pts
	quad_wts = bface_ops.quad_wts
	faces_to_basis = bface_ops.faces_to_basis
	normals_bfgroups = bface_ops.normals_bfgroups
	x_bfgroups = bface_ops.x_bfgroups

	nq = quad_wts.shape[0]

	# For plotting
	plot = True
	if plot_vs_x:
		xlabel = "x"
		d = 0
	elif plot_vs_y:
		xlabel = "y"
		d = 1
	else:
		plot = False
	if plot:
		bvalues = np.zeros([BFG.nBFace, nq]) # [nBFace, nq]
		bpoints = x_bfgroups[ibfgrp][:,:,d].flatten() # [nBFace, nq, dim]

	integ_val = 0.

	if dot_normal_with_vec:
		# convert to numpy array
		vec = np.array(vec)
		vec.shape = 1,2

	for ibface in range(BFG.nBFace):
		BFace = BFG.BFaces[ibface]
		elem = BFace.Elem
		face = BFace.face

		basis_val = faces_to_basis[face]

		# interpolate state and gradient at quad points
		Uq = np.matmul(basis_val, EqnSet.U[elem])

		# Get requested variable
		varq = EqnSet.ComputeScalars(variable_name, Uq) # [nq, 1]

		normals = normals_bfgroups[ibfgrp][ibface] # [nq, dim]
		jac = np.linalg.norm(normals, axis=1, keepdims=True) # [nq, 1]

		# If requested, account for normal and dot with input dir
		if dot_normal_with_vec:
			varq *= np.sum(normals/jac*vec, axis=1, keepdims=True)

		# Integrate and add to running sum
		if integrate:
			integ_val += np.sum(varq*jac*quad_wts)

		if plot:
			bvalues[ibface,:] = varq.reshape(-1)

	if integrate:
		print("Boundary integral = %g" % (integ_val))

	if plot:
		plt.figure()
		bvalues = bvalues.flatten()
		SolnLabel = Plot.get_solution_label(EqnSet, variable_name, Label)
		Plot.Plot1D(EqnSet, bpoints, bvalues, SolnLabel, u_var_calculated=True)
		Plot.finalize_plot(xlabel=xlabel)







