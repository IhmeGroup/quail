import numpy as np
import os
import pytest
import sys
sys.path.append('../src')

import meshing.gmsh as mesh_gmsh
import meshing.tools as mesh_tools

rtol = 1e-15
atol = 1e-15

def test_ref_to_phys_gives_physical_nodes(filled_mesh):
	'''
	Make sure that going from reference geometric nodes to physical space
	returns the physical nodes.
	'''
	# Reference geometric nodes
	xref = np.array([ [0, 0], [1, 0], [0, 1] ])
	# Get nodes in physical space
	xphys = mesh_tools.ref_to_phys(filled_mesh, 0, xref)
	# These should be double the reference nodes
	np.testing.assert_allclose(xphys, xref * 2, rtol, atol)

def test_element_volumes_for_2x2_triangles(filled_mesh):
	'''
	Make sure that two triangles in a 2x2 square each have an area of 2.
	'''
	# Get element volumes
	vol_elems, domain_vol = mesh_tools.element_volumes(filled_mesh, solver=None)
	# These are two 2x2 triangles, so should have an area of 2, totalling to 4
	np.testing.assert_allclose(vol_elems, 2, rtol, atol)
	np.testing.assert_allclose(domain_vol, 4, rtol, atol)


def test_element_centroid_for_2x2_triangles(filled_mesh):
	'''
	Make sure that going from reference geometric nodes to physical space
	returns the physical nodes.
	'''
	# Get centroid
	centroid = mesh_tools.get_element_centroid(filled_mesh, 0)
	# Make sure it is double the original centroid
	np.testing.assert_allclose(centroid, 2 * filled_mesh.gbasis.CENTROID, rtol,
			atol)

def test_periodic_translational_for_2x2_triangles():
	'''
	Make sure that a mesh with two triangles can be make periodic on all sides.
	'''
	# Path to mesh with two triangles
	file_path = (os.path.dirname(os.path.realpath(__file__)) +
			'/test_data/two_triangles_v2.msh')
	# Import mesh
	mesh = mesh_gmsh.import_gmsh_mesh(file_path)
	# Make periodic
	mesh_tools.make_periodic_translational(mesh, x1='x1', x2='x2',
			y1='y1', y2='y2')
	# Make sure that no boundary groups remain
	assert(len(mesh.boundary_groups) == 0)
	assert(mesh.num_boundary_groups == 0)
	# Make sure that now there are three interior faces
	assert(len(mesh.interior_faces) == 3)
	assert(mesh.num_interior_faces == 3)
	# Element 0 should neighbor 1 on all sides, and element 1 should neighbor 0
	# on all sides
	np.testing.assert_array_equal(mesh.elements[0].face_to_neighbors,
			np.ones(3))
	np.testing.assert_array_equal(mesh.elements[1].face_to_neighbors,
			np.zeros(3))
