"""Testing case for Cloude decomposition.
Compare the algorithm to a published nearest coherent matrix."""
from .cloude import cloude_decomposition
import numpy as np
from numpy.testing import assert_array_almost_equal

# Example matrix from Ossikovski, Optics Letters 37, 578 (2012)
MM = np.array([[[1, 0.0707, 0.0348, -0.0060],
                [0.0480, 0.4099, 0.0077, 0.0650],
                [0.0162, -0.0184, 0.2243, -0.3580],
                [0.0021, -0.0465, 0.3571, 0.1783]]])

# Cloude decomposition for above matrix
MMc = np.array([[[0.5614, 0.0789, 0.0310, -0.0076],
                [0.0799, 0.5569, 0.0420, 0.0498],
                [0.0254, 0.0258, 0.2763, -0.4812],
                [0.0146, -0.0584, 0.4804, 0.2719]]])

def test_cloude_decomposition():
    """Check whether the algorithm is accurate to the published result up until the 4. decimal"""
    assert_array_almost_equal(cloude_decomposition(MM, cut_off=3), MMc, decimal=4)
