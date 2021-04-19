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
        face00 = solver.mesh.elements[0].faces[0].children[0].children[0]
        face01 = solver.mesh.elements[0].faces[0].children[0].children[1]
        face1 = solver.mesh.elements[0].faces[0].children[1]
        # Verify neighbors of new faces
        np.testing.assert_equal(face00.elemL_ID, 0)
        np.testing.assert_equal(face00.elemR_ID, 1)
        np.testing.assert_equal(face01.elemL_ID, 0)
        np.testing.assert_equal(face01.elemR_ID, 3)
        np.testing.assert_equal(face1.elemL_ID, 0)
        np.testing.assert_equal(face1.elemR_ID, 2)
        # Verify face IDs of new faces
        np.testing.assert_equal(face00.faceL_ID, 0)
        np.testing.assert_equal(face00.faceR_ID, 0)
        np.testing.assert_equal(face01.faceL_ID, 0)
        np.testing.assert_equal(face01.faceR_ID, 0)
        np.testing.assert_equal(face1.faceL_ID, 0)
        np.testing.assert_equal(face1.faceR_ID, 0)
        # Verify reference Q1 nodes of new faces
        np.testing.assert_allclose(face00.refQ1nodes_L,
                np.array([ [.25, .75], [0, 1] ]))
        np.testing.assert_allclose(face01.refQ1nodes_L,
                np.array([ [.5, .5], [.25, .75] ]))
        np.testing.assert_allclose(face1.refQ1nodes_L,
                np.array([ [1, 0], [.5, .5] ]))
        np.testing.assert_allclose(face00.refQ1nodes_R,
                np.array([ [0, 1], [1, 0] ]))
        np.testing.assert_allclose(face01.refQ1nodes_R,
                np.array([ [0, 1], [1, 0] ]))
        np.testing.assert_allclose(face1.refQ1nodes_R,
                np.array([ [0, 1], [1, 0] ]))
        # Check that boundary faces on the refined side have the right neighbor
        np.testing.assert_equal(solver.mesh.elements[2].faces[1].elem_ID, 2)
        np.testing.assert_equal(solver.mesh.elements[1].faces[2].elem_ID, 1)

    # Return to test directory
    os.chdir(f'{quail_dir}/test')
