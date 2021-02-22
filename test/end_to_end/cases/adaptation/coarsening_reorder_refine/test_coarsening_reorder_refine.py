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
        breakpoint()

        # Correct final neighbors of affected elements
        affected_elems = np.array(
                [5, 17, 6, 18, 24, 25, 16, 13, 14, 7, 10, 9], dtype=int)
        neighbors_expected = np.array([
                [17, 16, 13],
                [5, 6, 9],
                [25, 17, 24],
                [24, 7, 25],
                [18, 6, 14],
                [6, 18, 10],
                [4, 5, 8],
                [1, 2, 5],
                [2, 3, 24],
                [19, 18, 15],
                [22, 21, 25],
                [21, 20, 17],
                ])
        # Assert
        neighbors = np.array([solver.mesh.elements[i].face_to_neighbors for i in
                affected_elems])
        np.testing.assert_array_equal(neighbors, neighbors_expected)

    # Return to test directory
    os.chdir(f'{quail_dir}/test')
