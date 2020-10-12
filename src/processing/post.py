# ------------------------------------------------------------------------ #
#
#       File : src/processing/post.py
#
#       Contains functions for computing error and integrating boundary
#		data.
#      
# ------------------------------------------------------------------------ #
import copy
from matplotlib import pyplot as plt
import numpy as np
import sys

import errors

import meshing.meshbase as mesh_defs
import meshing.tools as mesh_tools

import numerics.basis.tools as basis_tools
import numerics.helpers.helpers as helpers

import processing.plot as plot_defs


def get_error(mesh, physics, solver, var_name, ord=2, print_error=True, 
		normalize_by_volume=True):
	'''
	This function computes the Lp-error, where p is the "ord" input argument.

	Inputs:
	-------
	    mesh: mesh object
	    physics: physics object
	    solver: solver object
	    var_name: name of variable to compute error of
	    ord: order of the error
	    print_error: if True, will print total error
	    normalize_by_volume: if True, will normalize the error by the
	    	volume of the domain

	Outputs:
	--------
	    tot_err: total error
	    err_elems: error over each element [num_elems]
	'''
	# Extract info
	time = solver.time
	U = physics.U
	basis = solver.basis
	order = physics.order
	if physics.exact_soln is None:
		raise ValueError("No exact solution provided")

	# Get element volumes 
	if normalize_by_volume:
		_, tot_vol = mesh_tools.element_volumes(mesh, solver)
	else:
		tot_vol = 1.

	# Allocate, initialize
	err_elems = np.zeros([mesh.num_elems])
	tot_err = 0.

	# Loop through elements
	for elem_ID in range(mesh.num_elems):
		Uc = U[elem_ID]

		# Get quadrature data
		quad_order = basis.get_quadrature_order(mesh, 2*np.amax([order, 1]), 
				physics=physics)
		gbasis = mesh.gbasis
		quad_pts, quad_wts = gbasis.get_quadrature_data(quad_order)
		
		# Evaluate exact solution at quadrature points
		xphys = mesh_tools.ref_to_phys(mesh, elem_ID, quad_pts)
		u_exact = physics.exact_soln.get_state(physics, x=xphys, t=time)
		# u_exact = physics.call_function(physics.exact_soln, x=xphys, t=time)

		# Interpolate state to quadrature points
		basis.get_basis_val_grads(quad_pts, True)
		u = helpers.evaluate_state(Uc, basis.basis_val)

		# Computed requested quantity
		s = physics.compute_variable(var_name, u)
		s_exact = physics.compute_variable(var_name, u_exact)

		# Calculate element-local error
		djac, _, _ = basis_tools.element_jacobian(mesh, elem_ID, quad_pts, 
				get_djac=True)
		err = np.sum((s - s_exact)**ord*quad_wts*djac)
		err_elems[elem_ID] = err
		tot_err += err_elems[elem_ID]

	tot_err = (tot_err/tot_vol)**(1./ord)

	# Print if requested
	if print_error:
		print("Total error = %.15f" % (tot_err))

	return tot_err, err_elems


def get_boundary_info(solver, mesh, physics, bname, var_name, 
		dot_normal_with_vec=False, vec=0., integrate=True, plot_vs_x=False, 
		plot_vs_y=False, ylabel=None, fmt='k-', legend_label=None, **kwargs):
	'''
	This function integrates and/or plots a given quantity over a specific
	boundary.

	Inputs:
	-------
	    mesh: mesh object
	    physics: physics object
	    solver: solver object
	    bname: name of boundary
	    var_name: name of variable to compute
	    dot_normal_with_vec: if dot_normal_with_vec is True, will multiply 
	    	var by the dot product between the outward-pointing unit normal 
	    	vector and vec 
	    vec: vector to dot with normal (see above) [dim]
	    integrate: if True, will integrate variable over boundary
	    plot_vs_x: if True, will plot variable vs. x
	    plot_vs_y: if True, will plot variable vs. y (lower priority than
	    	plot_vs_x)
		ylabel: y-axis label for figure
		fmt: format string for plotting, e.g. "bo" for blue circles
		legend_label: legend label
		kwargs: keyword arguments (see plot_defs.finalize_plot)
	'''
	if mesh.dim != 2:
		raise errors.IncompatibleError

	# Extract boundary group
	boundary_group = mesh.boundary_groups[bname]
	boundary_num = boundary_group.number
	# Extract helpers
	bface_helpers = solver.bface_helpers
	quad_pts = bface_helpers.quad_pts
	quad_wts = bface_helpers.quad_wts
	faces_to_basis = bface_helpers.faces_to_basis
	normals_bfgroups = bface_helpers.normals_bfgroups
	x_bfgroups = bface_helpers.x_bfgroups
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
		bvalues = np.zeros([boundary_group.num_boundary_faces, nq]) 
			# [num_boundary_faces, nq]
		bpoints = x_bfgroups[boundary_num][:,:,d].flatten() 
			# [num_boundary_faces, nq, dim]

	integ_val = 0.

	if dot_normal_with_vec:
		# Convert to numpy array
		vec = np.array(vec)
		vec.shape = 1, 2

	# Loop through boundary faces
	for bface_ID in range(boundary_group.num_boundary_faces):
		# Extract
		boundary_face = boundary_group.boundary_faces[bface_ID]
		elem_ID = boundary_face.elem_ID
		face_ID = boundary_face.face_ID
		basis_val = faces_to_basis[face_ID]

		# Interpolate state and gradient at quad points
		Uq = helpers.evaluate_state(physics.U[elem_ID], basis_val)

		# Get requested variable
		varq = physics.compute_variable(var_name, Uq) # [nq, 1]

		# Normals
		normals = normals_bfgroups[boundary_num][bface_ID] # [nq, dim]
		jac = np.linalg.norm(normals, axis=1, keepdims=True) # [nq, 1]

		# If requested, account for normal and dot with input dir
		if dot_normal_with_vec:
			varq *= np.sum(normals/jac*vec, axis=1, keepdims=True)

		# Integrate and add to running sum
		if integrate:
			integ_val += np.sum(varq*jac*quad_wts)

		if plot:
			bvalues[bface_ID,:] = varq.reshape(-1)

	if integrate:
		print("Boundary integral = %g" % (integ_val))

	# Plot if requested
	if plot:
		plt.figure()
		bvalues = bvalues.flatten()
		ylabel = plot_defs.get_ylabel(physics, var_name, ylabel)
		plot_defs.plot_1D(physics, bpoints, bvalues, ylabel, fmt, legend_label)
		plot_defs.finalize_plot(xlabel=xlabel, **kwargs)







