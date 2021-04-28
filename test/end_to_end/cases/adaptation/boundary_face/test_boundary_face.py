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
        face0 = solver.mesh.elements[0].faces[1]
        face1 = solver.mesh.elements[2].faces[1]
        # Verify neighbors of new faces
        np.testing.assert_equal(face0.elem_ID, 0)
        np.testing.assert_equal(face1.elem_ID, 2)
        # Verify face IDs of new faces
        np.testing.assert_equal(face0.face_ID, 1)
        np.testing.assert_equal(face1.face_ID, 1)
        # Verify reference Q1 nodes of new faces
        np.testing.assert_allclose(face0.refQ1nodes,
                np.array([ [0, 1], [0, 0] ]))
        np.testing.assert_allclose(face1.refQ1nodes,
                np.array([ [0, 1], [0, 0] ]))

    # Return to test directory
    os.chdir(f'{quail_dir}/test')
