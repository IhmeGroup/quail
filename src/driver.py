#!/usr/bin/env python
import code
import getopt
import importlib
import numpy as np
import os
import sys

import defaultparams as DefaultInput
import errors
import general

import meshing.common as MeshCommon
import meshing.gmsh as MeshGmsh
import meshing.tools as MeshTools

import physics.euler.euler as Euler
import physics.scalar.scalar as Scalar

import processing.plot as Plot
import processing.post as Post

import solver.DG as Solver

def main(argv):
   inputfile = ''
   outputfile = ''
   try:
      opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
   except getopt.GetoptError:
      print('test.py -i <inputfile> -o <outputfile>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('test.py -i <inputfile> -o <outputfile>')
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg

   inputfile=inputfile.replace('.py','')

   CurrentDir = os.path.dirname(os.path.abspath(inputfile)) + "/"
   sys.path.append(CurrentDir)

   importlib.import_module(inputfile)

if __name__ == "__main__":
   main(sys.argv[1:])


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
	stepper_params = DefaultInput.TimeStepping
	numerics_params = DefaultInput.Numerics
	output_params = DefaultInput.Output
	mesh_params = DefaultInput.Mesh
	physics_params = DefaultInput.Physics
	IC_params = DefaultInput.InitialCondition
	BC_params = DefaultInput.BoundaryConditions
	source_params = DefaultInput.SourceTerms

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
		mesh = MeshGmsh.ReadGmshFile(mesh_params["File"])
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
				mesh = MeshCommon.mesh_1D(Uniform=True, nElem=nElem_x, xmin=xmin, xmax=xmax, Periodic=periodic)
			else:
				periodic = False
				mesh = MeshCommon.mesh_1D(Uniform=True, nElem=nElem_x, xmin=xmin, xmax=xmax, Periodic=periodic)
		else:
			# 2D - quads or tris
			mesh = MeshCommon.mesh_2D(nElem_x=nElem_x, nElem_y=nElem_y, Uniform=True, xmin=xmin, xmax=xmax, 
					ymin=ymin, ymax=ymax)
			if shape is "Triangle":
				mesh = MeshCommon.split_quadrils_into_tris(mesh)

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
		MeshTools.MakePeriodicTranslational(mesh, x1=pb[0], x2=pb[1], y1=pb[2], y2=pb[3])

	'''
	Physics
	'''
	# Create physics object
	order = numerics_params["InterpOrder"]
	basis = numerics_params["InterpBasis"]
	if physics_params["Type"] is "ConstAdvScalar":
		if mesh.Dim == 1:
			physics = Scalar.ConstAdvScalar1D(order, basis, mesh)
		else:
			physics = Scalar.ConstAdvScalar2D(order, basis, mesh)
	elif physics_params["Type"] is "Burgers":
		if mesh.Dim == 1:
			physics = Scalar.Burgers1D(order, basis, mesh)
		else:
			raise NotImplementedError
	elif physics_params["Type"] is "Euler":
		if mesh.Dim == 1:
			physics = Euler.Euler1D(order, basis, mesh)
		else:
			physics = Euler.Euler2D(order, basis, mesh)

	# Set parameters
	pparams = physics_params.copy()
	pparams.pop("Type") # don't pass this key
	physics.SetParams(**pparams)

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
		bparams.pop("Function")

		btype = physics.BCType[bparams["BCType"]]
		bparams.pop("BCType")

		# EqnSet.set_BC(BC_type="FullState", fcn_type="DampingSine", omega = 2*np.pi, nu=nu)
		physics.set_BC(BC_type=btype.name, fcn_type=BC_params[bname]["Function"], **bparams)
		# bparams["BCType"] = btype
		# bparams["Function"] = set_function(physics, bparams["Function"])
		# code.interact(local=locals())

		###
		# code.interact(local=locals())
		# physics.SetBC(bname, **bparams)

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
	if solver_type is "DG":
		solver = Solver.DG_Solver(solver_params, physics, mesh)
	elif solver_type is "ADERDG":
		solver = Solver.ADERDG_Solver(solver_params, physics, mesh)
	else: 
		raise NotImplementedError

	'''
	Run
	'''
	solver.solve()


	return solver, physics, mesh







