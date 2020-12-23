import os
import pickle
import subprocess
import sys
sys.path.append('../../src')

import numpy as np

import physics.euler.euler as euler
import physics.scalar.scalar as scalar
import physics.chemistry.chemistry as chemistry
import solver.DG as DG
import solver.ADERDG as ADERDG


def generate_regression_test_data():

    # Get script directory
    script_dir = os.getcwd()

    # Names of case directories
    case_dirs = [
            'scalar/1D/constant_advection',
            'scalar/1D/inviscid_burgers',
            ]
    # Add full path
    case_dirs = [f'{script_dir}/{case_dir}' for case_dir in case_dirs]
    n_cases = len(case_dirs)

    # Name and path of data file which stores the regression test data
    datafile_name = f'{script_dir}/regression_data.npy'
    # If this already exists, delete it
    os.system(f'rm -f {datafile_name}')

    # Loop over all case directories
    results = []
    for case_dir in case_dirs:
        # Move to the test case directory
        os.chdir(case_dir)
        # Call the Quail executable
        result = subprocess.check_output([
                f'{script_dir}/../../src/quail', 'input_file.py',
                ], stderr=subprocess.STDOUT)
        # Print results of run
        print(result.decode('utf-8'))

        # Open resulting solution file
        with open('Data_final.pkl', 'rb') as solutionfile:
            # Load data
            solver = pickle.load(solutionfile)
            # Save final solution
            results.append(solver.state_coeffs)

    # Save results to the regression test datafile
    with open(datafile_name, 'wb') as datafile:
        for data in results: np.save(datafile, data)

    # Read file
    output = []
    with open(datafile_name, 'rb') as datafile:
        for i in range(n_cases):
            results.append(np.load(datafile))


if __name__ == "__main__":
    generate_regression_test_data()
