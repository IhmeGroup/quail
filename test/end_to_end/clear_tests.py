import os


def clear_tests():
	'''
	This function is used to clear the generated Pytest scripts for regression
	tests.  This is useful for getting rid of old tests (for example, when
	switching branches).
	'''
	# Search for all test case files
	case_files = os.popen('grep -lir --include \*.py "test_case_*" cases/'
			).readlines()

	# Loop over all test cases
	for case in case_files:
		# Remove test script, if it exists
		try:
			os.remove(case.strip())
		except OSError:
			pass


if __name__ == "__main__":
	clear_tests()
