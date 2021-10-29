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

def mueller_to_hermitian(M):
    H = np.zeros(M.shape, dtype='complex64')
    H[:, 0, 0] = M[:, 0, 0]    + M[:, 0, 1]    + M[:, 1, 0]    + M[:, 1, 1]
    H[:, 0, 1] = M[:, 0, 2]    + 1j*M[:, 0, 3] + M[:, 1, 2]    + 1j*M[:, 1, 3]
    H[:, 0, 2] = M[:, 2, 0]    + M[:, 2, 1]    - 1j*M[:, 3, 0] - 1j*M[:, 3, 1]
    H[:, 0, 3] = M[:, 2, 2]    + 1j*M[:, 2, 3] - 1j*M[:, 3, 2] + M[:, 3, 3]

    H[:, 1, 1] = M[:, 0, 0]    - M[:, 0, 1]    + M[:, 1, 0]    - M[:, 1, 1]
    H[:, 1, 2] = M[:, 2, 2]    - 1j*M[:, 2, 3] - 1j*M[:, 3, 2] - M[:, 3, 3]
    H[:, 1, 3] = M[:, 2, 0]    - M[:, 2, 1]    - 1j*M[:, 3, 0] + 1j*M[:, 3, 1]

    H[:, 2, 2] = M[:, 0, 0]    + M[:, 0, 1]    - M[:, 1, 0]    - M[:, 1, 1]
    H[:, 2, 3] = M[:, 0, 2]    - M[:, 1, 2]    + 1j*M[:, 0, 3] - 1j*M[:, 1, 3]
    H[:, 3, 3] = M[:, 0, 0]    - M[:, 0, 1]    - M[:, 1, 0]    + M[:, 1, 1]

    H[:, 1, 0] = np.conjugate(H[:, 0, 1])
    H[:, 2, 0] = np.conjugate(H[:, 0, 2])
    H[:, 3, 0] = np.conjugate(H[:, 0, 3])
    H[:, 2, 1] = np.conjugate(H[:, 1, 2])
    H[:, 3, 1] = np.conjugate(H[:, 1, 3])
    H[:, 3, 2] = np.conjugate(H[:, 2, 3])

    H = np.divide(H, 4.0)

    return H

# Vectorized case - keep for further recoding
# def hermitian_to_mueller(H):
#     M = np.zeros(H.shape, dtype='float64')
#     M[:, 0, 0] = np.real(H[:, 0, 0] + H[:, 1, 1] + H[:, 2, 2] + H[:, 3, 3])
#     M[:, 0, 1] = np.real(H[:, 0, 0] - H[:, 1, 1] + H[:, 2, 2] - H[:, 3, 3])
#     M[:, 0, 2] = np.real(H[:, 0, 1] + H[:, 1, 0] + H[:, 2, 3] + H[:, 3, 2])
#     M[:, 0, 3] = np.imag(H[:, 0, 1] - H[:, 1, 0] + H[:, 2, 3] - H[:, 3, 2])
#     M[:, 1, 0] = np.real(H[:, 0, 0] + H[:, 1, 1] - H[:, 2, 2] - H[:, 3, 3])
#     M[:, 1, 1] = np.real(H[:, 0, 0] - H[:, 1, 1] - H[:, 2, 2] + H[:, 3, 3])
#     M[:, 1, 2] = np.real(H[:, 0, 1] + H[:, 1, 0] - H[:, 2, 3] - H[:, 3, 2])
#     M[:, 1, 3] = np.imag(H[:, 0, 1] - H[:, 1, 0] - H[:, 2, 3] + H[:, 3, 2])
#     M[:, 2, 0] = np.real(H[:, 0, 2] + H[:, 2, 0] + H[:, 1, 3] + H[:, 3, 1])
#     M[:, 2, 1] = np.real(H[:, 0, 2] + H[:, 2, 0] - H[:, 1, 3] - H[:, 3, 1])
#     M[:, 2, 2] = np.real(H[:, 0, 3] + H[:, 3, 0] + H[:, 1, 2] + H[:, 2, 1])
#     M[:, 2, 3] = np.imag(H[:, 0, 3] - H[:, 3, 0] - H[:, 1, 2] + H[:, 2, 1])
#     M[:, 3, 0] = np.imag(H[:, 2, 0] - H[:, 0, 2] - H[:, 1, 3] + H[:, 3, 1])
#     M[:, 3, 1] = np.imag(H[:, 2, 0] - H[:, 0, 2] + H[:, 1, 3] - H[:, 3, 1])
#     M[:, 3, 2] = np.imag(H[:, 3, 0] - H[:, 0, 3] + H[:, 2, 1] - H[:, 1, 2])
#     M[:, 3, 3] = np.real(H[:, 0, 3] + H[:, 3, 0] - H[:, 1, 2] - H[:, 2, 1])

#     M = np.divide(M, 2.0)
#     return M

def hermitian_to_mueller(H):
    M = np.zeros(H.shape, dtype='float64')
    M[0, 0] = np.real(H[0, 0] + H[1, 1] + H[2, 2] + H[3, 3])
    M[0, 1] = np.real(H[0, 0] - H[1, 1] + H[2, 2] - H[3, 3])
    M[0, 2] = np.real(H[0, 1] + H[1, 0] + H[2, 3] + H[3, 2])
    M[0, 3] = np.imag(H[0, 1] - H[1, 0] + H[2, 3] - H[3, 2])
    M[1, 0] = np.real(H[0, 0] + H[1, 1] - H[2, 2] - H[3, 3])
    M[1, 1] = np.real(H[0, 0] - H[1, 1] - H[2, 2] + H[3, 3])
    M[1, 2] = np.real(H[0, 1] + H[1, 0] - H[2, 3] - H[3, 2])
    M[1, 3] = np.imag(H[0, 1] - H[1, 0] - H[2, 3] + H[3, 2])
    M[2, 0] = np.real(H[0, 2] + H[2, 0] + H[1, 3] + H[3, 1])
    M[2, 1] = np.real(H[0, 2] + H[2, 0] - H[1, 3] - H[3, 1])
    M[2, 2] = np.real(H[0, 3] + H[3, 0] + H[1, 2] + H[2, 1])
    M[2, 3] = np.imag(H[0, 3] - H[3, 0] - H[1, 2] + H[2, 1])
    M[3, 0] = np.imag(H[2, 0] - H[0, 2] - H[1, 3] + H[3, 1])
    M[3, 1] = np.imag(H[2, 0] - H[0, 2] + H[1, 3] - H[3, 1])
    M[3, 2] = np.imag(H[3, 0] - H[0, 3] + H[2, 1] - H[1, 2])
    M[3, 3] = np.real(H[0, 3] + H[3, 0] - H[1, 2] - H[2, 1])

    return M

def cloude_decomposition(MM, cut_off=2):
    """Cloude decomposition of a Mueller matrix MM"""
    if not isinstance(MM, np.ndarray) or MM.ndim != 3 or MM.shape[1:] != (4, 4):
        raise ValueError(f'Malformed Mueller matrix (with dimension {MM.shape}),'
                          'has to be of dimension (N, 4, 4)')
    if cut_off > 3 or cut_off < 0:
        raise ValueError('Cutoff must be in a range between (including) 0 and 3')

    cov_matrix = mueller_to_hermitian(MM)
    eig_val, eig_vec = np.linalg.eigh(cov_matrix)

    # Sort eigenvalues and -vectors descending by eigenvalue
    idx = (-eig_val).argsort(axis=1)
    idx_vec = np.repeat(idx, 4, axis=1).reshape((*idx.shape, 4))

    eig_val_sorted = np.take_along_axis(eig_val, idx, axis=1)
    eig_vec_sorted = np.take_along_axis(np.transpose(eig_vec, (0, 2, 1)), idx_vec, axis=1)

    # Calculate Mueller matrices from eigenvalues
    mueller_matrix = np.zeros((eig_val_sorted.shape[0], 4, 4))
    for i in range(eig_val_sorted.shape[0]):
        mueller_j = np.zeros((4, 4))
        for j in range(eig_val_sorted.shape[1] - cut_off):
            eigv = np.array([eig_vec_sorted[i, j]], dtype='complex64')
            cov_matrix_j = eigv.T @ np.conjugate(eigv)
            mueller_j += hermitian_to_mueller(eig_val_sorted[i, j] * cov_matrix_j)

        mueller_matrix[i] = mueller_j

    return mueller_matrix
