"""An implementation of the Cloude decomposition algorithm"""
import numpy as np
import pandas as pd

def read_mueller_matrix_from_file(fname):
    """Read a Mueller matrix from a Sentech ASCII file.
    Save the file in SpectraRay under Save As -> Ascii (.txt)"""
    mueller_matrix = pd.read_csv(fname, sep=r'\s+', index_col=0).iloc[:, 36:-1]
    mueller_matrix.index.name = 'Wavelength'
    mueller_matrix.columns = ['M11', 'M12', 'M13', 'M14',
                              'M21', 'M22', 'M23', 'M24',
                              'M31', 'M32', 'M33', 'M34',
                              'M41', 'M42', 'M43', 'M44']

    return mueller_matrix

def reshape_matrix_into_dataframe(exp_df, mueller_matrix):
    """Reshape a numpy 4x4 array containing mueller matrix elements
    to a dataframe with columns Mxy. The index labels for each column
    are taken from the provided exp_df."""
    mueller_df = pd.DataFrame(index=exp_df.index, columns=exp_df.columns)
    mueller_df.values[:] = mueller_matrix.reshape(-1, 16)

    return mueller_df

def mueller_to_hermitian(m_m):
    """Transform a mueller matrix into a hermitian covariance matrix
    using the relation 1/4 ∑ m_ij * (σ_i ⊗ σ_j),
    where σ_i are the Pauli spin matrices."""
    cov_matrix = np.zeros(m_m.shape, dtype='complex64')
    cov_matrix[:, 0, 0] = m_m[:, 0, 0]    + m_m[:, 0, 1]    + m_m[:, 1, 0]    + m_m[:, 1, 1]
    cov_matrix[:, 0, 1] = m_m[:, 0, 2]    + 1j*m_m[:, 0, 3] + m_m[:, 1, 2]    + 1j*m_m[:, 1, 3]
    cov_matrix[:, 0, 2] = m_m[:, 2, 0]    + m_m[:, 2, 1]    - 1j*m_m[:, 3, 0] - 1j*m_m[:, 3, 1]
    cov_matrix[:, 0, 3] = m_m[:, 2, 2]    + 1j*m_m[:, 2, 3] - 1j*m_m[:, 3, 2] + m_m[:, 3, 3]

    cov_matrix[:, 1, 1] = m_m[:, 0, 0]    - m_m[:, 0, 1]    + m_m[:, 1, 0]    - m_m[:, 1, 1]
    cov_matrix[:, 1, 2] = m_m[:, 2, 2]    - 1j*m_m[:, 2, 3] - 1j*m_m[:, 3, 2] - m_m[:, 3, 3]
    cov_matrix[:, 1, 3] = m_m[:, 2, 0]    - m_m[:, 2, 1]    - 1j*m_m[:, 3, 0] + 1j*m_m[:, 3, 1]

    cov_matrix[:, 2, 2] = m_m[:, 0, 0]    + m_m[:, 0, 1]    - m_m[:, 1, 0]    - m_m[:, 1, 1]
    cov_matrix[:, 2, 3] = m_m[:, 0, 2]    - m_m[:, 1, 2]    + 1j*m_m[:, 0, 3] - 1j*m_m[:, 1, 3]
    cov_matrix[:, 3, 3] = m_m[:, 0, 0]    - m_m[:, 0, 1]    - m_m[:, 1, 0]    + m_m[:, 1, 1]

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
    where σ_i are the Pauli spin matrices and H is the covariance matrix."""
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

def cloude_decomposition(exp_matrix,
                         ev_mask=np.array([True, False, False, False]),
                         output_eigenvector=False):
    """Cloude decomposition of a Mueller matrix MM"""
    if not isinstance(exp_matrix, np.ndarray):
        raise ValueError(f'exp_matrix has to be a numpy array, not {type(exp_matrix)}')

    if exp_matrix.ndim == 2 and exp_matrix.shape == (4, 4):
        exp_matrix = np.array([exp_matrix])

    if exp_matrix.ndim != 3 or exp_matrix.shape[1:] != (4, 4):
        raise ValueError(f'Malformed Mueller matrix (with dimension {exp_matrix.shape}), '
                          'has to be of dimension (N, 4, 4)')

    if not isinstance(ev_mask, np.ndarray):
        raise ValueError(f'ev_mask has to be a numpy array of type bool, not {type(exp_matrix)}')

    if not ev_mask.ndim == 1 and ev_mask.shape == (4,) and ev_mask.dtype == np.bool:
        raise ValueError(f'ev_mask has to be of shape (4,) not {ev_mask.shape}')

    cov_matrix = mueller_to_hermitian(exp_matrix)
    eig_val, eig_vec = np.linalg.eigh(cov_matrix)

    # Sort eigenvalues and -vectors descending by eigenvalue
    idx = (-eig_val).argsort(axis=1)
    idx_vec = np.repeat(idx, 4, axis=1).reshape((*idx.shape, 4))

    eig_val_sorted = np.take_along_axis(eig_val, idx, axis=1)
    eig_vec_sorted = np.take_along_axis(np.transpose(eig_vec, (0, 2, 1)), idx_vec, axis=1)

    eig_val_diag = np.apply_along_axis(lambda ev: ev * np.eye(4), 1, eig_val_sorted * ev_mask)
    cov_matrix_j = np.transpose(eig_vec_sorted, (0, 2, 1)) @\
                    eig_val_diag @\
                    np.conjugate(eig_vec_sorted)
    mueller_matrix = hermitian_to_mueller(cov_matrix_j)

    if output_eigenvector:
        return mueller_matrix, eig_val_sorted
    return mueller_matrix
