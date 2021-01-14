import os
import pickle
import subprocess
import sys
sys.path.append('src')

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
def test_case(test_data):

    # Unpack test data
    Uc_expected_list, quail_dir, datafile_name = test_data

    # Get test case name
    test_name = sys.path[0].split('_end/')[-1]
    # Get expected solution for this test case
    Uc_expected = Uc_expected_list[test_name]

    # Enter case directory
    os.chdir(sys.path[0])

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
        Uc = solver.state_coeffs
        # Assert
        np.testing.assert_allclose(Uc, Uc_expected, rtol, atol)

    # Return to Quail directory
    os.chdir(quail_dir)
