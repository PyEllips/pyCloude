"""Transformations for Mueller matrices"""
import numpy as np

def mueller_to_hermitian(m_m):
    """Transform a mueller matrix into a hermitian covariance matrix
        using the relation 1/4 ∑ m_ij * (σ_i ⊗ σ_j),
        where σ_i are the Pauli spin matrices.

    Args:
        m_m ((N, 4, 4) numpy.ndarray): Numpy array containing N mueller matrices.

    Returns:
        (N, 4, 4) numpy.array: Numpy array containing the respective covariance matrix for each
                            mueller matrix.
    """
    cov_matrix = np.zeros(m_m.shape, dtype='complex64')
    cov_matrix[:, 0, 0] = m_m[:, 0, 0] + m_m[:, 0, 1]    + m_m[:, 1, 0]    + m_m[:, 1, 1]
    cov_matrix[:, 0, 1] = m_m[:, 0, 2] + 1j*m_m[:, 0, 3] + m_m[:, 1, 2]    + 1j*m_m[:, 1, 3]
    cov_matrix[:, 0, 2] = m_m[:, 2, 0] + m_m[:, 2, 1]    - 1j*m_m[:, 3, 0] - 1j*m_m[:, 3, 1]
    cov_matrix[:, 0, 3] = m_m[:, 2, 2] + 1j*m_m[:, 2, 3] - 1j*m_m[:, 3, 2] + m_m[:, 3, 3]

    cov_matrix[:, 1, 1] = m_m[:, 0, 0] - m_m[:, 0, 1]    + m_m[:, 1, 0]    - m_m[:, 1, 1]
    cov_matrix[:, 1, 2] = m_m[:, 2, 2] - 1j*m_m[:, 2, 3] - 1j*m_m[:, 3, 2] - m_m[:, 3, 3]
    cov_matrix[:, 1, 3] = m_m[:, 2, 0] - m_m[:, 2, 1]    - 1j*m_m[:, 3, 0] + 1j*m_m[:, 3, 1]

    cov_matrix[:, 2, 2] = m_m[:, 0, 0] + m_m[:, 0, 1]    - m_m[:, 1, 0]    - m_m[:, 1, 1]
    cov_matrix[:, 2, 3] = m_m[:, 0, 2] - m_m[:, 1, 2]    + 1j*m_m[:, 0, 3] - 1j*m_m[:, 1, 3]
    cov_matrix[:, 3, 3] = m_m[:, 0, 0] - m_m[:, 0, 1]    - m_m[:, 1, 0]    + m_m[:, 1, 1]

    cov_matrix[:, 1, 0] = np.conjugate(cov_matrix[:, 0, 1])
    cov_matrix[:, 2, 0] = np.conjugate(cov_matrix[:, 0, 2])
    cov_matrix[:, 3, 0] = np.conjugate(cov_matrix[:, 0, 3])
    cov_matrix[:, 2, 1] = np.conjugate(cov_matrix[:, 1, 2])
    cov_matrix[:, 3, 1] = np.conjugate(cov_matrix[:, 1, 3])
    cov_matrix[:, 3, 2] = np.conjugate(cov_matrix[:, 2, 3])

    cov_matrix = np.divide(cov_matrix, 4.0)

    return cov_matrix

def hermitian_to_mueller(c_m):
    """Transform a hermitian covariance matrix back into a mueller matrix
    using the relation m_ij = tr[(σ_i ⊗ σ_j) H],
    where σ_i are the Pauli spin matrices and H is the covariance matrix.

    Args:
        c_m ((N, 4, 4) numpy.ndarray): Numpy array containing N covariance matrices.

    Returns:
        (N, 4, 4) numpy.ndarray: Numpy array containing the respective mueller matrix for each
                            covariance matrix.
    """
    mueller_matrix = np.zeros(c_m.shape, dtype='float64')
    mueller_matrix[:, 0, 0] = np.real(c_m[:, 0, 0] + c_m[:, 1, 1] + c_m[:, 2, 2] + c_m[:, 3, 3])
    mueller_matrix[:, 0, 1] = np.real(c_m[:, 0, 0] - c_m[:, 1, 1] + c_m[:, 2, 2] - c_m[:, 3, 3])
    mueller_matrix[:, 0, 2] = np.real(c_m[:, 0, 1] + c_m[:, 1, 0] + c_m[:, 2, 3] + c_m[:, 3, 2])
    mueller_matrix[:, 0, 3] = np.imag(c_m[:, 0, 1] - c_m[:, 1, 0] + c_m[:, 2, 3] - c_m[:, 3, 2])
    mueller_matrix[:, 1, 0] = np.real(c_m[:, 0, 0] + c_m[:, 1, 1] - c_m[:, 2, 2] - c_m[:, 3, 3])
    mueller_matrix[:, 1, 1] = np.real(c_m[:, 0, 0] - c_m[:, 1, 1] - c_m[:, 2, 2] + c_m[:, 3, 3])
    mueller_matrix[:, 1, 2] = np.real(c_m[:, 0, 1] + c_m[:, 1, 0] - c_m[:, 2, 3] - c_m[:, 3, 2])
    mueller_matrix[:, 1, 3] = np.imag(c_m[:, 0, 1] - c_m[:, 1, 0] - c_m[:, 2, 3] + c_m[:, 3, 2])
    mueller_matrix[:, 2, 0] = np.real(c_m[:, 0, 2] + c_m[:, 2, 0] + c_m[:, 1, 3] + c_m[:, 3, 1])
    mueller_matrix[:, 2, 1] = np.real(c_m[:, 0, 2] + c_m[:, 2, 0] - c_m[:, 1, 3] - c_m[:, 3, 1])
    mueller_matrix[:, 2, 2] = np.real(c_m[:, 0, 3] + c_m[:, 3, 0] + c_m[:, 1, 2] + c_m[:, 2, 1])
    mueller_matrix[:, 2, 3] = np.imag(c_m[:, 0, 3] - c_m[:, 3, 0] - c_m[:, 1, 2] + c_m[:, 2, 1])
    mueller_matrix[:, 3, 0] = np.imag(c_m[:, 2, 0] - c_m[:, 0, 2] - c_m[:, 1, 3] + c_m[:, 3, 1])
    mueller_matrix[:, 3, 1] = np.imag(c_m[:, 2, 0] - c_m[:, 0, 2] + c_m[:, 1, 3] - c_m[:, 3, 1])
    mueller_matrix[:, 3, 2] = np.imag(c_m[:, 3, 0] - c_m[:, 0, 3] + c_m[:, 2, 1] - c_m[:, 1, 2])
    mueller_matrix[:, 3, 3] = np.real(c_m[:, 0, 3] + c_m[:, 3, 0] - c_m[:, 1, 2] - c_m[:, 2, 1])

    return mueller_matrix

def coherency_matrix(m_m):
    """Transform a mueller matrix into it's corresponding coherency matrix

    Args:
        m_m ((N, 4, 4) numpy.ndarray): Numpy array containing N mueller matrices.

    Returns:
        (N, 4, 4) numpy.array: Numpy array containing the respective coherency matrix for each
                            mueller matrix.
    """
    coh_matrix = np.zeros(m_m.shape, dtype=np.complex128)
    coh_matrix[:, 0, 0] = m_m[:, 0, 0] + m_m[:, 1, 1] + m_m[:, 2, 2] + m_m[:, 3, 3]
    coh_matrix[:, 0, 1] = m_m[:, 0, 1] + m_m[:, 1, 0] - 1j*m_m[:, 2, 3] + 1j*m_m[:, 3, 2]
    coh_matrix[:, 0, 2] = m_m[:, 0, 2] + m_m[:, 2, 0] + 1j*m_m[:, 1, 3] - 1j*m_m[:, 3, 1]
    coh_matrix[:, 0, 3] = m_m[:, 0, 3] - 1j*m_m[:, 1, 2] + 1j*m_m[:, 2, 1] + m_m[:, 3, 0]

    coh_matrix[:, 1, 0] = m_m[:, 0, 1] + m_m[:, 1, 0] + 1j*m_m[:, 2, 3] - 1j*m_m[:, 3, 2]
    coh_matrix[:, 1, 1] = m_m[:, 0, 0] + m_m[:, 1, 1] - m_m[:, 2, 2] - m_m[:, 3, 3]
    coh_matrix[:, 1, 2] = 1j*m_m[:, 0, 3] + m_m[:, 1, 2] + m_m[:, 2, 1] - 1j*m_m[:, 3, 0]
    coh_matrix[:, 1, 3] = -1j*m_m[:, 0, 2] + 1j*m_m[:, 2, 0] + m_m[:, 1, 3] + m_m[:, 3, 1]

    coh_matrix[:, 2, 0] = m_m[:, 0, 2] + m_m[:, 2, 0] - 1j*m_m[:, 1, 3] + 1j*m_m[:, 3, 1]
    coh_matrix[:, 2, 1] = -1j*m_m[:, 0, 3] + m_m[:, 1, 2] + m_m[:, 2, 1] + 1j*m_m[:, 3, 0]
    coh_matrix[:, 2, 2] = m_m[:, 0, 0] - m_m[:, 1, 1] + m_m[:, 2, 2] - m_m[:, 3, 3]
    coh_matrix[:, 2, 3] = 1j*m_m[:, 0, 1] - 1j*m_m[:, 1, 0] + m_m[:, 2, 3] + m_m[:, 3, 2]

    coh_matrix[:, 3, 0] = m_m[:, 0, 3] + 1j*m_m[:, 1, 2] - 1j*m_m[:, 2, 1] + m_m[:, 3, 0]
    coh_matrix[:, 3, 1] = 1j*m_m[:, 0, 2] - 1j*m_m[:, 2, 0] + m_m[:, 1, 3] + m_m[:, 3, 1]
    coh_matrix[:, 3, 2] = -1j*m_m[:, 0, 1] + 1j*m_m[:, 1, 0] + m_m[:, 2, 3] + m_m[:, 3, 2]
    coh_matrix[:, 3, 3] = m_m[:, 0, 0] - m_m[:, 1, 1] - m_m[:, 2, 2] + m_m[:, 3, 3]

    coh_matrix = np.divide(coh_matrix, 4.0)
    return coh_matrix
