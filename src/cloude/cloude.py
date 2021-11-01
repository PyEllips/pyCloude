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
    mueller_df = pd.DataFrame(index=exp_df.index, columns=exp_df.columns, dtype='float64')
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

def depolarization_index(mueller_matrix):
    """Calculate the depolarization index of a mueller matrix"""
    if isinstance(mueller_matrix, np.ndarray)\
            and mueller_matrix.ndim == 3\
            and mueller_matrix.shape[1:] == (4, 4):
        return np.sqrt(np.sum(mueller_matrix**2, axis=(1, 2)) - mueller_matrix[:, 0, 0]**2)\
            / np.sqrt(3) / mueller_matrix[:, 0, 0]

    if isinstance(mueller_matrix, np.ndarray)\
            and mueller_matrix.ndim == 2\
            and mueller_matrix.shape == (4, 4):
        return np.sqrt(np.sum(mueller_matrix**2) - mueller_matrix[0, 0]**2)\
            / np.sqrt(3) / mueller_matrix[0, 0]

    if isinstance(mueller_matrix, pd.DataFrame)\
            and mueller_matrix.shape[-1] == (16):
        return np.sqrt((mueller_matrix**2).sum(axis=1) - mueller_matrix.loc[:, 'M11']**2)\
            / np.sqrt(3) / mueller_matrix.loc[:, 'M11']

    raise ValueError(f"Mueller matrix of type {type(mueller_matrix)} not supported. "
                      "Please use either a pandas Dataframe of shape (N, 16) "
                      "or a numpy ndarray of shape (N, 4, 4)")

def cloude_decomposition(exp_matrix,
                         ev_mask=np.array([True, False, False, False]),
                         output_eigenvector=False):
    """Cloude decomposition of a Mueller matrix MM"""

    # Check input values
    if not isinstance(exp_matrix, np.ndarray):
        raise ValueError(f'exp_matrix has to be a numpy array, not {type(exp_matrix)}')

    if exp_matrix.ndim == 2 and exp_matrix.shape == (4, 4):
        exp_matrix = np.array([exp_matrix])

    if exp_matrix.ndim != 3 or exp_matrix.shape[1:] != (4, 4):
        raise ValueError(f'Malformed Mueller matrix (with dimension {exp_matrix.shape}), '
                          'has to be of dimension (N, 4, 4)')

    if isinstance(ev_mask, list) and len(ev_mask) == 4:
        ev_mask = np.array(ev_mask, dtype=np.bool)

    if not isinstance(ev_mask, np.ndarray):
        raise ValueError(f'ev_mask has to be a numpy array of type bool, not {type(exp_matrix)}')

    if not ev_mask.ndim == 1 and ev_mask.shape == (4,) and ev_mask.dtype == np.bool:
        raise ValueError(f'ev_mask has to be of shape (4,) not {ev_mask.shape}')

    # Convert mueller matrix into covariance matrix and calculate eigenvalues/-vectors
    cov_matrix = mueller_to_hermitian(exp_matrix)
    eig_val, eig_vec = np.linalg.eigh(cov_matrix)

    # Sort eigenvalues and -vectors descending by eigenvalue
    idx = (-eig_val).argsort(axis=1)
    idx_vec = np.transpose(np.repeat(idx, 4, axis=1).reshape((*idx.shape, 4)),
                           (0, 2, 1))

    eig_val_sorted = np.take_along_axis(eig_val, idx, axis=1)
    eig_vec_sorted = np.take_along_axis(eig_vec, idx_vec, axis=2)

    # Calculate 4 covariance matrices for each set of eigenvalue and -vector
    eig_val_diag = np.apply_along_axis(lambda ev: ev * np.eye(4),
                                       1,
                                       eig_val_sorted * ev_mask)
    cov_matrix_j = eig_vec_sorted @\
                    eig_val_diag @\
                    np.conjugate(np.transpose(eig_vec_sorted, (0, 2, 1)))
    mueller_matrix = hermitian_to_mueller(cov_matrix_j)

    if output_eigenvector:
        return mueller_matrix, eig_val_sorted
    return mueller_matrix
