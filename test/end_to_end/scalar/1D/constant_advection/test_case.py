import os
import pickle
import subprocess
import sys
sys.path.append('src')

import numpy as np
import pytest

import physics.euler.euler as euler
import physics.scalar.scalar as scalar
import physics.chemistry.chemistry as chemistry
import solver.DG as DG
import solver.ADERDG as ADERDG

rtol = 1e-15
atol = 1e-15

def test_case():
    '''
    '''
    # Get current working directory
    # TODO: This assumes that you call Pytest from the Quail directory
    quail_dir = os.getcwd()
    # Move to the test case directory
    os.chdir(sys.path[0])
    # Call the Quail executable
    result = subprocess.check_output([
            f'{quail_dir}/src/quail', 'constant_advection.py',
            ], stderr=subprocess.STDOUT)
    # Print results of run
    print(result.decode('utf-8'))

    # Open resulting data file
    with open('Data_final.pkl', 'rb') as f:
        solver = pickle.load(f)
        Uc = solver.state_coeffs
        breakpoint()

    ## Evaluate Lagrange basis at nodes
    #phi = basis.get_values(basis.get_nodes(order))
    ## This matrix should be identity
    #expected = np.identity(phi.shape[0])
    ## Assert
    #np.testing.assert_allclose(phi, expected, rtol, atol)


def generate_regression_test_data():

    datafile_name = 'regression_data.npy'

    # Get current working directory
    # TODO: This assumes that you call Pytest from the Quail directory
    quail_dir = os.getcwd()
    # Move to the test case directory
    os.chdir(sys.path[0])
    # Call the Quail executable
    result = subprocess.check_output([
            f'{quail_dir}/src/quail', 'constant_advection.py',
            ], stderr=subprocess.STDOUT)
    # Print results of run
    print(result.decode('utf-8'))

    # Open resulting data file
    with open('Data_final.pkl', 'rb') as f:
        solver = pickle.load(f)
        Uc = solver.state_coeffs
        breakpoint()


def append_to_file(datafile_name, data):
    with open(datafile_name, 'wb') as datafile:
        np.save(datafile, data)


    with open(datafile_name, 'rb') as datafile:
        a = np.load(datafile)
        #b = np.load(datafile)
    print(a)#, b)
