"""Testing case for Cloude decomposition.
Compare the algorithm to a published nearest coherent matrix."""
from cloude import cloude_decomposition
import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest

# Example matrix from Ossikovski, Optics Letters 37, 578 (2012) and a second arbitrary matrix
MM = np.array([[[1, 0.0707, 0.0348, -0.0060],
                [0.0480, 0.4099, 0.0077, 0.0650],
                [0.0162, -0.0184, 0.2243, -0.3580],
                [0.0021, -0.0465, 0.3571, 0.1783]],
                [[1, -0.6125, 0.3377, -0.2466],
                [-0.5766, 0.7387, 0.0096, 0.4076],
                [0.3778, -0.1007, 0.6254, -0.3359],
                [0.2536, -0.4551, 0.3205, 0.3956]]])

# Cloude decomposition for above matrix
MMc = np.array([[[0.5614, 0.0789, 0.0310, -0.0076],
                [0.0799, 0.5569, 0.0420, 0.0498],
                [0.0254, 0.0258, 0.2763, -0.4812],
                [0.0146, -0.0584, 0.4804, 0.2719]],
                [[ 0.97205775, -0.61319432,  0.33421084, -0.26181083],
                [-0.57935978,  0.74430871,  0.00381699,  0.41266805],
                [ 0.37723109, -0.10568332,  0.63313559, -0.34484902],
                [ 0.27979999, -0.44664309,  0.31538681,  0.40984917]]])

def test_cloude_decomposition():
    """Is accurate to the published result up until the 4. decimal"""
    assert_array_almost_equal(cloude_decomposition(np.array([MM[0]])),
                              np.array([MMc[0]]),
                              decimal=4)

def test_cloude_decomposition_vectorized():
    """Is accurate to the published result up until the 4. decimal
    when vectorizing over to matrices"""
    assert_array_almost_equal(cloude_decomposition(MM),
                              MMc,
                              decimal=4)

def test_decomposition_of_2d_array():
    """Executes a 2D array without spectral dimension correctly"""
    assert_array_almost_equal(cloude_decomposition(MM[0]), np.array([MMc[0]]), decimal=4)

def test_error_on_non_numpy_input():
    """Raises an error if no numpy array is provided"""
    with pytest.raises(ValueError):
        cloude_decomposition(1)

def test_error_on_non_numpy_input_for_ev_mask():
    """Raises an error if no numpy array is provided"""
    with pytest.raises(ValueError):
        cloude_decomposition(MM, 0)
