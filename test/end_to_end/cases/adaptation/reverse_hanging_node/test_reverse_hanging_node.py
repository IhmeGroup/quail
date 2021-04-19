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

    # Open resulting data file
    with open('Data_final.pkl', 'rb') as f:
        # Read final solution from file
        solver = pickle.load(f)

        # Get newly created faces
        face0 = solver.mesh.elements[0].faces[0].children[0]
        face1 = solver.mesh.elements[0].faces[0].children[1]
        # Verify neighbors of new faces
        np.testing.assert_equal(face0.elemL_ID, 0)
        np.testing.assert_equal(face0.elemR_ID, 1)
        np.testing.assert_equal(face1.elemL_ID, 0)
        np.testing.assert_equal(face1.elemR_ID, 2)
        # Verify face IDs of new faces
        np.testing.assert_equal(face0.faceL_ID, 0)
        np.testing.assert_equal(face0.faceR_ID, 0)
        np.testing.assert_equal(face1.faceL_ID, 0)
        np.testing.assert_equal(face1.faceR_ID, 0)
        # Verify reference Q1 nodes of new faces
        np.testing.assert_allclose(face0.refQ1nodes_L,
                np.array([ [.5, .5], [0, 1] ]))
        np.testing.assert_allclose(face1.refQ1nodes_L,
                np.array([ [1, 0], [.5, .5] ]))
        breakpoint()
        np.testing.assert_allclose(face0.refQ1nodes_R,
                np.array([ [0, 1], [1, 0] ]))
        np.testing.assert_allclose(face1.refQ1nodes_R,
                np.array([ [0, 1], [1, 0] ]))

    # Return to test directory
    os.chdir(f'{quail_dir}/test')
