import sys
sys.path.append('src')

import numpy as np
import pytest

import numerics.adaptation.adapter as adapter_defs
import solver.DG as DG

rtol = 1e-15
atol = 1e-15


def test_adapt_should_work_for_Q1_triangles():
    pass
    #solver = DG.DG()
    ## Assert
    #np.testing.assert_allclose(phi, expected, rtol, atol)
