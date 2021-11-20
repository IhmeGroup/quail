import os
import numpy as np
import pytest
import sys

import list_of_cases


@pytest.fixture
def test_data():
	'''
	This fixture reads in the regression test data and yields it, along with
	the Quail directory. After the test ends, it returns to the test directory
	to continue with the next tests.

	Outputs:
	--------
		Uc_expected_list: list of solution vectors for each test case
		quail_dir: the absolute path of the Quail directory
	'''
	# Get Quail directory
	quail_dir = os.path.dirname(os.getcwd())

	# Name and path of data file which stores the regression test data
	datafile_name = f'{quail_dir}/test/end_to_end/regression_data.npz'

	# List of case directories
	case_dirs = list(list_of_cases.case_dirs.keys())

	# Read file storing regression test data
	Uc_expected_list = {}
	with open(datafile_name, 'rb') as datafile:
		npzfile = np.load(datafile)
		for i, array in enumerate(npzfile.files):
			Uc_expected_list[case_dirs[i]] = npzfile[array];
	yield Uc_expected_list, quail_dir

	# Return to test directory
	os.chdir(f'{quail_dir}/test')
