import code
import copy
from matplotlib import pyplot as plt
import numpy as np
import sys

from data import ArrayList
import errors

import meshing.meshbase as mesh_defs
import meshing.tools as MeshTools

import numerics.basis.tools as basis_tools

import processing.plot as plot_defs



def L2_error(mesh, EqnSet, solver, VariableName, PrintError=True, NormalizeByVolume=True):

	Time = solver.Time
	U = EqnSet.U
	basis = solver.basis
	# Check for exact solution
	# if not EqnSet.ExactSoln.Function:
	if EqnSet.ExactSoln is None:
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
	sr = EqnSet.NUM_STATE_VARS
	quadData = None
	# JData = JacobianData(mesh)
	# ier = EqnSet.VariableType[VariableName]
	GeomPhiData = None

	# ElemErr.Arrays[egrp][:] = 0.

	Order = EqnSet.order
	# basis = EqnSet.Basis

	for elem in range(mesh.nElem):
		U_ = U[elem]

		quad_order = basis.get_quadrature_order(mesh, 2*np.amax([Order,1]), physics=EqnSet)
		gbasis = mesh.gbasis
		xq, wq = gbasis.get_quadrature_data(quad_order)
		nq = xq.shape[0]
		
		basis.eval_basis(xq, True, False, False, None)
		xphys = np.zeros([nq, mesh.Dim])

		djac,_,_ = basis_tools.element_jacobian(mesh,elem,xq,get_djac=True)

		xphys, GeomPhiData = mesh_defs.ref_to_phys(mesh, elem, GeomPhiData, xq, xphys)
		u_exact = EqnSet.CallFunction(EqnSet.ExactSoln, x=xphys, t=Time)

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


def get_boundary_info(mesh, physics, solver, bname, var_name, integrate=True, vec=0., dot_normal_with_vec=False, 
		plot_vs_x=False, plot_vs_y=False, ylabel=None, fmt='k-', legend_label=None, **kwargs):

	if mesh.Dim != 2:
		raise errors.IncompatibleError
	# Find boundary face group
	# found = False
	# ibfgrp = 0
	# for BFG in mesh.BFaceGroups:
	# 	if BFG.Name == bname:
	# 		found = True
	# 		break
	# 	ibfgrp += 1

	# if not found:
	# 	raise errors.DoesNotExistError

	BFG = mesh.BFaceGroups[bname]
	ibfgrp = BFG.number

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
		Uq = np.matmul(basis_val, physics.U[elem])

		# Get requested variable
		varq = physics.ComputeScalars(var_name, Uq) # [nq, 1]

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
		# SolnLabel = plot_defs.get_solution_label(EqnSet, variable_name, ylabel)
		# plot_defs.Plot1D(EqnSet, bpoints, bvalues, SolnLabel, u_var_calculated=True, **kwargs)
		ylabel = plot_defs.get_ylabel(physics, var_name, ylabel)
		plot_defs.plot_1D(physics, bpoints, bvalues, ylabel, fmt, legend_label, 0)
		plot_defs.finalize_plot(xlabel=xlabel, **kwargs)







