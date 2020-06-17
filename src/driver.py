#!/usr/bin/env python
import code
import getopt
import importlib
import numpy as np
import os
import sys

import defaultparams as default_input
import errors
from general import SolverType, PhysicsType

import meshing.common as mesh_common
import meshing.gmsh as mesh_gmsh
import meshing.tools as mesh_tools

import physics.euler.euler as euler
import physics.scalar.scalar as scalar

import solver.DG as DG
import solver.ADERDG as ADERDG


def set_physics(order, basis_type, mesh, physics_type):
    dim = mesh.Dim 

    if PhysicsType[physics_type] == PhysicsType.ConstAdvScalar and dim == 1:
        physics_ref = scalar.ConstAdvScalar1D
    elif PhysicsType[physics_type] == PhysicsType.ConstAdvScalar and dim == 2:
        physics_ref = scalar.ConstAdvScalar2D
    elif PhysicsType[physics_type] == PhysicsType.Burgers and dim == 1:
        physics_ref = scalar.Burgers1D
    elif PhysicsType[physics_type] == PhysicsType.Euler and dim == 1:
        physics_ref = euler.Euler1D
    elif PhysicsType[physics_type] == PhysicsType.Euler and dim == 2:
        physics_ref = euler.Euler2D
    else:
        raise NotImplementedError

    physics = physics_ref(order, basis_type, mesh)

    return physics


def overwrite_params(params, params_new, allow_new_keys=False):
	if params_new is None:
		return params

	for key in params_new:
		if not allow_new_keys and key not in params.keys(): 
			raise KeyError
		params[key] = params_new[key]

	return params


def read_inputs(TimeStepping, Numerics, Output, Mesh,
		Physics, InitialCondition, BoundaryConditions, SourceTerms):

	# defaults
	stepper_params = default_input.TimeStepping
	numerics_params = default_input.Numerics
	output_params = default_input.Output
	mesh_params = default_input.Mesh
	physics_params = default_input.Physics
	IC_params = default_input.InitialCondition
	BC_params = default_input.BoundaryConditions
	source_params = default_input.SourceTerms

	stepper_params = overwrite_params(stepper_params, TimeStepping)
	numerics_params = overwrite_params(numerics_params, Numerics)
	output_params = overwrite_params(output_params, Output)
	mesh_params = overwrite_params(mesh_params, Mesh)
	physics_params = overwrite_params(physics_params, Physics, True)
	IC_params = overwrite_params(IC_params, InitialCondition, True)
	BC_params = overwrite_params(BC_params, BoundaryConditions, True)
	source_params = overwrite_params(source_params, SourceTerms, True)

	# TODO: process enums

	return stepper_params, numerics_params, output_params, mesh_params, \
			physics_params, IC_params, BC_params, source_params


def driver(TimeStepping=None, Numerics=None, Output=None, Mesh=None, Physics=None, 
		InitialCondition=None, BoundaryConditions=None, SourceTerms=None):


	'''
	Input deck
	'''
	stepper_params, numerics_params, output_params, mesh_params, physics_params, \
			IC_params, BC_params, source_params = read_inputs(TimeStepping, Numerics, 
			Output, Mesh, Physics, InitialCondition, BoundaryConditions, SourceTerms)


	'''
	Mesh
	'''
	if mesh_params["File"] is not None:
		mesh = mesh_gmsh.ReadGmshFile(mesh_params["File"])
	else:
		# Unpack
		shape = mesh_params["ElementShape"]
		xmin = mesh_params["xmin"]
		xmax = mesh_params["xmax"]
		nElem_x = mesh_params["nElem_x"]
		nElem_y = mesh_params["nElem_y"]
		ymin = mesh_params["ymin"]
		ymax = mesh_params["ymax"]
		if shape is "Segment":
			if mesh_params["PeriodicBoundariesX"] != [] and mesh_params["PeriodicBoundariesY"] == []:
				periodic = True
				mesh = mesh_common.mesh_1D(Uniform=True, nElem=nElem_x, xmin=xmin, xmax=xmax, Periodic=periodic)
			else:
				periodic = False
				mesh = mesh_common.mesh_1D(Uniform=True, nElem=nElem_x, xmin=xmin, xmax=xmax, Periodic=periodic)
		else:
			# 2D - quads or tris
			mesh = mesh_common.mesh_2D(nElem_x=nElem_x, nElem_y=nElem_y, Uniform=True, xmin=xmin, xmax=xmax, 
					ymin=ymin, ymax=ymax)
			if shape is "Triangle":
				mesh = mesh_common.split_quadrils_into_tris(mesh)

	pb = [None]*4
	# Store periodic boundaries in pb
	i = 0
	for b in mesh_params["PeriodicBoundariesX"]:
		pb[i] = b
		i += 1
	if mesh.Dim == 2:
		i = 2
		for b in mesh_params["PeriodicBoundariesY"]:
			pb[i] = b
			i += 1
	if pb != [None]*4 and periodic is False:
		# need to check 1D first
		if mesh.Dim == 1:
			raise Exception
		mesh_tools.MakePeriodicTranslational(mesh, x1=pb[0], x2=pb[1], y1=pb[2], y2=pb[3])

	'''
	Physics
	'''
	# Create physics object
	order = numerics_params["InterpOrder"]
	basis_type = numerics_params["InterpBasis"]
	# if physics_params["Type"] is "ConstAdvScalar":
	# 	if mesh.Dim == 1:
	# 		physics = scalar.ConstAdvScalar1D(order, basis, mesh)
	# 	else:
	# 		physics = scalar.ConstAdvScalar2D(order, basis, mesh)
	# elif physics_params["Type"] is "Burgers":
	# 	if mesh.Dim == 1:
	# 		physics = scalar.Burgers1D(order, basis, mesh)
	# 	else:
	# 		raise NotImplementedError
	# elif physics_params["Type"] is "Euler":
	# 	if mesh.Dim == 1:
	# 		physics = euler.Euler1D(order, basis, mesh)
	# 	else:
	# 		physics = euler.Euler2D(order, basis, mesh)

	physics = set_physics(order, basis_type, mesh, physics_params["Type"])

	# Set parameters
	pparams = physics_params.copy()
	pparams.pop("Type") # don't pass this key
	# physics.SetParams(**pparams)
	conv_flux_type = pparams.pop("ConvFlux")
	physics.set_conv_num_flux(conv_flux_type)
	physics.set_physical_params(**pparams)

	# temporary
	# if IC_params["Function"] is "Gaussian":
	# 	fcn = physics.FcnGaussian
	# 	IC_params["Function"] = fcn
	# else:
	# 	raise Exception

	# Exact Solution
	set_exact = IC_params["SetAsExact"]
	iparams = IC_params.copy()
	iparams.pop("SetAsExact") # don't pass this key
	iparams.pop("Function")
	physics.set_exact(exact_type=IC_params["Function"],**iparams)

	# Initial conditions
	IC_params["Function"] = physics.set_IC(IC_type=IC_params["Function"])


	# Boundary conditions
	for bname in BC_params:
		bparams = BC_params[bname].copy()
		# bparams.pop("Function")

		# btype = physics.BCType[bparams["BCType"]]
		# bparams.pop("BCType")

		# # EqnSet.set_BC(BC_type="StateAll", fcn_type="DampingSine", omega = 2*np.pi, nu=nu)
		# # bparams["BCType"] = btype
		# # bparams["Function"] = set_function(physics, bparams["Function"])
		# # code.interact(local=locals())

		# EqnSet.set_BC(BC_type="StateAll", fcn_type="SmoothIsentropicFlow", a=0.9)

		###
		# code.interact(local=locals())
		# physics.SetBC(bname, **bparams)
		
		BC_type = bparams.pop("BCType")

		try:
			fcn_type = bparams.pop("Function")
			physics.set_BC(BC_type, fcn_type, **bparams)
		except KeyError:
			physics.set_BC(BC_type, **bparams)

	# Source terms
	for sparams in source_params.values():
		sname = sparams["Function"]
		sparams.pop("Function")
		physics.set_source(source_type=sname, **sparams)

		# physics.SetSource(**sparams)

	'''
	Solver
	'''
	# Merge params
	solver_params = {**stepper_params, **numerics_params, **output_params}
	solver_type = solver_params.pop("Solver")
	if SolverType[solver_type] is SolverType.DG:
		solver = DG.DG(solver_params, physics, mesh)
	elif SolverType[solver_type] is SolverType.ADERDG:
		solver = ADERDG.ADERDG(solver_params, physics, mesh)
	else: 
		raise NotImplementedError

	'''
	Run
	'''
	solver.solve()


	return solver, physics, mesh

def main(argv):
	inputfile = ''
	postfile = ''
	try:
		opts, args = getopt.getopt(argv,"hi:p:",["ifile=","pfile="])
	except getopt.GetoptError:
		print('test.py -i <inputfile> -p <postfile>')
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print('test.py -i <inputfile> -p <postfile>')
			sys.exit()
		elif opt in ("-i", "--ifile"):
			inputfile = arg
		elif opt in ("-p", "--pfile"):
			postfile = arg

	inputfile=inputfile.replace('.py','')
	postfile=postfile.replace('.py','')


	CurrentDir = os.path.dirname(os.path.abspath(inputfile)) + "/"
	sys.path.append(CurrentDir)
	deck = importlib.import_module(inputfile)

	solver, EqnSet, mesh = driver(deck.TimeStepping, deck.Numerics, deck.Output, deck.Mesh,
		deck.Physics, deck.InitialCondition, deck.BoundaryConditions, deck.SourceTerms)

	auto_process = solver.Params["AutoProcess"]
	if auto_process is True:
		if postfile is not '':
			postprocess = importlib.import_module(postfile)
		else:
			postfile = 'post_process'
			postprocess = importlib.import_module(postfile)

if __name__ == "__main__":
	main(sys.argv[1:])






