import os

import numpy as np
import pytest

import list_of_cases

@pytest.fixture
def test_data():
    # Get current working directory
    # TODO: This assumes that you call Pytest from the Quail directory
    quail_dir = os.getcwd()

    # Name and path of data file which stores the regression test data
    datafile_name = f'{quail_dir}/test/end_to_end/regression_data.npy'

    # List of case directories
    case_dirs = list(list_of_cases.case_dirs.keys())

    # Read file storing regression test data
    output = {}
    with open(datafile_name, 'rb') as datafile:
        for i in range(len(case_dirs)):
            output[case_dirs[i]] = np.load(datafile)
    yield output, quail_dir, datafile_name
