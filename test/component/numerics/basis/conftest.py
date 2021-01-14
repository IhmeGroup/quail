import sys
sys.path.append('../src')

import numpy as np
import pytest

@pytest.fixture
def basis(Basis, order):
    yield Basis(order)
