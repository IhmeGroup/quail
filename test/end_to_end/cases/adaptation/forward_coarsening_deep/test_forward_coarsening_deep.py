import os
import pickle
import subprocess
import sys
sys.path.append('../src')

import numpy as np
import pytest

import list_of_cases
import physics.euler.euler as euler
import physics.scalar.scalar as scalar
import physics.chemistry.chemistry as chemistry
import solver.DG as DG
import solver.ADERDG as ADERDG

rtol = 1e-15
atol = 1e-15


# Markers distinguish tests into different categories
@pytest.mark.adapt
@pytest.mark.dg
def test_case():

    # Get Quail directory
    quail_dir = os.path.dirname(os.getcwd())

    # Enter case directory
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # Call the Quail executable
    result = subprocess.check_output([
            f'{quail_dir}/src/quail', 'input_file.py',
            ], stderr=subprocess.STDOUT)
    # Print results of run
    print(result.decode('utf-8'))

    # Open initial condition data file
    with open('Data_0.pkl', 'rb') as f:
        # Read final solution from file
        solver_init = pickle.load(f)

    # Open resulting data file
    with open('Data_final.pkl', 'rb') as f:
        # Read final solution from file
        solver = pickle.load(f)

        # Loop over interior faces
        for face, face_init in zip(solver.mesh.interior_faces,
                solver_init.mesh.interior_faces):
            # Ensure that all faces have the same neighbors, orientation, and Q1
            # nodes
            np.testing.assert_equal(face.elemL_ID, face_init.elemL_ID)
            np.testing.assert_equal(face.elemR_ID, face_init.elemR_ID)
            np.testing.assert_equal(face.faceL_ID, face_init.faceL_ID)
            np.testing.assert_equal(face.faceR_ID, face_init.faceR_ID)
            np.testing.assert_allclose(face.refQ1nodes_L,
                    face_init.refQ1nodes_L)
            np.testing.assert_allclose(face.refQ1nodes_R,
                    face_init.refQ1nodes_R)
        # Loop over boundary faces
        for bgroup, bgroup_init in zip(solver.mesh.boundary_groups.values(),
                solver_init.mesh.boundary_groups.values()):
            for face, face_init in zip(bgroup.boundary_faces,
                    bgroup_init.boundary_faces):
                # Ensure that all faces have the same neighbors, orientation,
                # and Q1 nodes
                np.testing.assert_equal(face.elem_ID, face_init.elem_ID)
                np.testing.assert_equal(face.face_ID, face_init.face_ID)
                np.testing.assert_allclose(face.refQ1nodes,
                        face_init.refQ1nodes)

    # Return to test directory
    os.chdir(f'{quail_dir}/test')
