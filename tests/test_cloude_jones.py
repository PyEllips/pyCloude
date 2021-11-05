"""Testing case for Cloude decomposition into jones matrix.
Compare the algorithm to a published nearest coherent matrix."""
from cloude import cloude_decomp_jones
import numpy as np
from numpy.testing import assert_array_almost_equal

mm = np.array([[[1, -0.6125, 0.3377, -0.2466],
               [-0.5766, 0.7387, 0.0096, 0.4076],
               [0.3778, -0.1007, 0.6254, -0.3359],
               [0.2536, -0.4551, 0.3205, 0.3956]],
               [[1, -0.6125, 0.3377, -0.2466],
               [-0.5766, 0.7387, 0.0096, 0.4076],
               [0.3778, -0.1007, 0.6254, -0.3359],
               [0.2536, -0.4551, 0.3205, 0.3956]]])

jm = np.array([[[-0.465 + 0.1959 * 1j, -0.2437 + 1j*0.2603],
               [-0.1786 + 0.2497 *1j, -1.173 - 0.1959 * 1j]],
               [[-0.465 + 0.1959 * 1j, -0.2437 + 1j*0.2603],
               [-0.1786 + 0.2497 *1j, -1.173 - 0.1959 * 1j]]], dtype=np.complex128)

def test_cloude_jones_decomp():
    """The Jones decomposition matches the literature value up to the 3. decimal"""
    assert_array_almost_equal(cloude_decomp_jones(mm[0])[0], jm[0], decimal=3)

def test_cloude_jones_decomp_vectorized():
    """The Jones decomposition matches the literature value up to the 3. decimal
    when executed vectorized"""
    assert_array_almost_equal(cloude_decomp_jones(mm), jm, decimal=3)
