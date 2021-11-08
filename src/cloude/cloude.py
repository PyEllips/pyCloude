"""An implementation of the Cloude decomposition algorithm"""
import numpy as np
import pandas as pd
from .transformations import mueller_to_hermitian, hermitian_to_mueller, coherency_matrix

def mmatrix_from_file(fname):
    """Read a Mueller matrix from a Sentech ASCII file.
    Save the file in SpectraRay under Save As -> Ascii (.txt)"""
    mueller_matrix = pd.read_csv(fname, sep=r'\s+', index_col=0).iloc[:, 36:-1]
    mueller_matrix.index.name = 'Wavelength'
    mueller_matrix.columns = ['M11', 'M12', 'M13', 'M14',
                              'M21', 'M22', 'M23', 'M24',
                              'M31', 'M32', 'M33', 'M34',
                              'M41', 'M42', 'M43', 'M44']

    return mueller_matrix

def mmatrix_to_dataframe(exp_df, mueller_matrix):
    """Reshape a numpy 4x4 array containing mueller matrix elements
    to a dataframe with columns Mxy. The index labels for each column
    are taken from the provided exp_df."""
    mueller_df = pd.DataFrame(index=exp_df.index, columns=exp_df.columns, dtype='float64')
    mueller_df.values[:] = mueller_matrix.reshape(-1, 16)

    return mueller_df

def dataframe_to_mmatrix(mueller_df):
    """Reshape a dataframe with mueller matrix elements with columns Mxy
    to a numpy 4x4 array."""
    return mueller_df.values.reshape(-1, 4, 4)

def dataframe_to_psi_delta(mueller_df):
    """Convert a mueller matrix dataframe with columns Mxy
    to a dataframe containing Ψ and Δ values.
    This is only perfectly reasonable for isotropic materials."""
    N = -mueller_df.loc[:,['M12', 'M21']].mean(axis=1)
    C = mueller_df.loc[:,'M33']
    S = (mueller_df.loc[:,'M34'] - mueller_df.loc[:,'M43'])/2

    Ψ = (C + 1j * S / (1 + N)).apply(lambda x: np.arctan(np.abs(x)))
    Δ = (C + 1j * S / (1 + N)).apply(lambda x: np.angle(x))

    return pd.DataFrame({'Ψ': Ψ,
                         'Δ': Δ}, index=mueller_df.index)

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

def check_input_values(matrix, ev_mask):
    if not isinstance(matrix, np.ndarray):
        raise ValueError(f'exp_matrix has to be a numpy array, not {type(matrix)}')

    if matrix.ndim == 2 and matrix.shape == (4, 4):
        matrix = np.array([matrix])

    if matrix.ndim != 3 or matrix.shape[1:] != (4, 4):
        raise ValueError(f'Malformed Mueller matrix (with dimension {matrix.shape}), '
                          'has to be of dimension (N, 4, 4)')

    if isinstance(ev_mask, list) and len(ev_mask) == 4:
        ev_mask = np.array(ev_mask, dtype=np.bool)

    if not isinstance(ev_mask, np.ndarray):
        raise ValueError(f'ev_mask has to be a numpy array of type bool, not {type(matrix)}')

    if not ev_mask.ndim == 1 and ev_mask.shape == (4,) and ev_mask.dtype == np.bool:
        raise ValueError(f'ev_mask has to be of shape (4,) not {ev_mask.shape}')

    return matrix, ev_mask

def sorted_eigh(matrix):
    """Calculate the sorted eigenvalues and eigenvectors
    of a hermitian matrix"""
    eig_val, eig_vec = np.linalg.eigh(matrix)

    # Sort eigenvalues and -vectors descending by eigenvalue
    idx = (-eig_val).argsort(axis=1)
    idx_vec = np.transpose(np.repeat(idx, 4, axis=1).reshape((*idx.shape, 4)),
                           (0, 2, 1))

    eig_val_sorted = np.take_along_axis(eig_val, idx, axis=1)
    eig_vec_sorted = np.take_along_axis(eig_vec, idx_vec, axis=2)

    return eig_val_sorted, eig_vec_sorted

def cloude_decomp(exp_matrix,
                         ev_mask=np.array([True, False, False, False]),
                         output_eigenvector=False):
    """Cloude decomposition of a Mueller matrix MM"""

    exp_matrix, ev_mask = check_input_values(exp_matrix, ev_mask)

    # Convert mueller matrix into covariance matrix and calculate eigenvalues/-vectors
    cov_matrix = mueller_to_hermitian(exp_matrix)
    eig_val_sorted, eig_vec_sorted = sorted_eigh(cov_matrix)

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

def cloude_decomp_jones(mueller_matrix, 
                        ev_mask=np.array([True, False, False, False]), 
                        output_eigenvector=False):
    """Decompose a mueller matrix into four jones matrices"""
    mueller_matrix, ev_mask = check_input_values(mueller_matrix, ev_mask)
    coh_matrix = coherency_matrix(mueller_matrix)
    ev, evec = sorted_eigh(coh_matrix)

    J = np.zeros((*evec.shape[:-1], 2, 2), dtype=np.complex128)
    J[:, :, 0, 0] = evec[:, 0, :] + evec[:, 1, :]
    J[:, :, 0, 1] = evec[:, 2, :] - 1j * evec[:, 3, :]
    J[:, :, 1, 0] = evec[:, 2, :] + 1j * evec[:, 3, :]
    J[:, :, 1, 1] = evec[:, 0, :] - evec[:, 1, :]

    jsum = np.apply_along_axis(lambda jm: jm * ev * ev_mask,
                               1,
                               J).sum(axis=2)[:, 0]

    if output_eigenvector:
        return jsum, ev
    return jsum
