import numpy as np
import pytest
import sys
sys.path.append('../src')

@pytest.fixture
def basis(Basis, order):
	'''
	This fixture yields a Basis object created with the given order.
	'''
	yield Basis(order)
