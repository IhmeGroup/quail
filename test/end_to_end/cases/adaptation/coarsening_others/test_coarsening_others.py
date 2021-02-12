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

        # Correct final neighbors of affected elements
        affected_elems = np.array([4, 13, 12, 10, 5, 7], dtype=int)
        neighbors_expected = np.array([
                [13, 12, 10],
                [4, 5, 7],
                [3, 4, 6],
                [1, 2, 4],
                [14, 13, 11],
                [16, 15, 13],
                ])
        # Assert
        neighbors = np.array([solver.mesh.elements[i].face_to_neighbors for i in
                affected_elems])
        #breakpoint()
        #np.testing.assert_array_equal(neighbors, neighbors_expected)

    # Return to test directory
    os.chdir(f'{quail_dir}/test')
