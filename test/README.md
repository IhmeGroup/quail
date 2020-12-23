# Automated Testing
This directory contains the automated testing framework for Quail. The testing
library used is Pytest.

## Organization
The test directory is split into two directories, `component` and `end_to_end`, which
represent component-level and end-to-end tests respectively.
 - Component-level tests will test only a small component of Quail. These will
   generally include things like unit tests and integration tests.
 - End-to-end tests will test a full run through the code. This means calling
   the Quail executable with an input file, performing iterations, and comparing
   the result stored in the output file.

## Conventions
Some conventions are established to keep the testing framework uniform and easy
to update when changes to the code are made.

### Component-Level Tests
Conventions for component-level tests are as follows.
 - All test files begin with `test_`, followed by the rest of the file
   name. For example, the test file for `basis.py` would be `test_basis.py`.
 - The test files in directory `test/component/` should look identical to the `src/`
   directory. For example, to test something in `src/numerics/basis/basis.py`,
   this test should be placed in `test/component/numerics/basis/test_basis.py`.
