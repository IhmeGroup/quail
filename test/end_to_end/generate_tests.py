import numpy as np
import os
import subprocess
import sys
import itertools

sys.path.append('../../src')

import list_of_cases


def generate_tests():
	'''
	This function is used to generate the Pytest scripts for regression tests.
	The scripts are autogenerated for the purpose of assigning each script a
	different set of markers, so that tests can be differentiated from one
	another. This function works by reading in a base testing script and
	adding the markers in for each case individually, then writing the script
	to a file.
	'''

	# Get script directory
	script_dir = sys.path[0]

	# Add full path to case directories
	case_dirs = [f'{script_dir}/cases/{case_dir}' for case_dir in
			list_of_cases.case_dirs.keys()]
	markers = list(list_of_cases.case_dirs.values())
	n_cases = len(case_dirs)

	# Read in base test script
	with open ("base_test_script.py", "r") as base_test_file:
		base_test = base_test_file.readlines()

	# Find which line to add tolerances to
	line_of_tol = 0
	for i, line in enumerate(base_test):
		if line.startswith('# Tolerances'):
			line_of_tol = i + 1
			break

	# Find which line to add markers to
	line_of_markers = 0
	for i, line in enumerate(base_test):
		if line.startswith('# Markers'):
			line_of_markers = i + 1
			break
	# Loop over all case directories
	for i, (case_dir, marker_list) in enumerate(zip(case_dirs, markers)):
		# Move to the test case directory
		os.chdir(case_dir)

		# Add tolerances to test script
		test_script = base_test.copy()

		test_script.insert(line_of_tol, f'rtol = {marker_list[1][0]}\n')
		test_script.insert(line_of_tol+1, f'atol = {marker_list[1][1]}\n')
		marker_list.pop()

		for marker in itertools.islice(marker_list[0] , 0, None):
		# for marker in marker_list:
			test_script.insert(line_of_markers+2, f'@pytest.mark.{marker}\n')

		# Create test script
		with open(f'test_case_{i}.py', 'w') as test_case_file:
			for line in test_script:
				test_case_file.write(line)


if __name__ == "__main__":
	generate_tests()