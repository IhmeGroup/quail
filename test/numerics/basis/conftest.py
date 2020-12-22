import sys
sys.path.append('src')

import numpy as np
import pytest

import numerics.basis.basis as basis_defs

@pytest.fixture
def lagrange_basis(order):
    yield basis_defs.LagrangeTri(order)
