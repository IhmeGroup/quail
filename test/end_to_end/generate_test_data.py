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

    datafile_name = 'regression_data.npy'

    # Get script directory
    script_dir = os.getcwd()
    # Move to the test case directory
    os.chdir('scalar/1D/constant_advection')
    # Call the Quail executable
    result = subprocess.check_output([
            f'{script_dir}/../../src/quail', 'constant_advection.py',
            ], stderr=subprocess.STDOUT)
    # Print results of run
    print(result.decode('utf-8'))

    # Open resulting data file
    with open('Data_final.pkl', 'rb') as f:
        # Load data
        solver = pickle.load(f)
        # Save final solution to file
        append_to_file(datafile_name, solver.state_coeffs)

    # Read file
    with open(datafile_name, 'rb') as datafile:
        a = np.load(datafile)
        #b = np.load(datafile)
        breakpoint()


def append_to_file(datafile_name, data):
    with open(datafile_name, 'wb') as datafile:
        np.save(datafile, data)




if __name__ == "__main__":
    generate_regression_test_data()
