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

@pytest.mark.parametrize('case_dir', list_of_cases.case_dirs)
def test_case(test_data, case_dir):

    # Unpack test data
    Uc_expected, quail_dir, datafile_name = test_data

    # Move to the test case directory
    os.chdir(f'{quail_dir}/test/end_to_end/{case_dir}')

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

    # Move back to top-level directory
    os.chdir(quail_dir)
