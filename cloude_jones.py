"""An implementation of the Cloude decomposition algorithm
using an alternative description by O. Arteaga with an intermediate Jones matrix."""
import numpy as np
import pandas as pd

def read_mueller_matrix_from_file(fname):
    MM = pd.read_csv(fname, sep=r'\s+', index_col=0).iloc[:, 36:-1]
    MM.index.name = 'Wavelength'
    MM.columns = ['M11', 'M12', 'M13', 'M14',
                  'M21', 'M22', 'M23', 'M24',
                  'M31', 'M32', 'M33', 'M34',
                  'M41', 'M42', 'M43', 'M44']
    
    return MM

def reshape_matrix_into_dataframe(exp_df, matrix):
    df = pd.DataFrame(index=exp_df.index, columns=exp_df.columns)
    df.values[:] = matrix.reshape(-1, 16)

    return df

def calc_coherency_matrix(M):
    H = np.zeros(M.shape, dtype='complex64')
    H[:, 0, 0] = M[:, 0, 0]    + M[:, 1, 1]    + M[:, 2, 2]    + M[:, 3, 3]
    H[:, 0, 1] = M[:, 0, 1]    + M[:, 1, 0]    - 1j*M[:, 2, 3] + 1j*M[:, 3, 2]
    H[:, 0, 2] = M[:, 0, 2]    + M[:, 2, 0]    + 1j*M[:, 1, 3] - 1j*M[:, 3, 1]
    H[:, 0, 3] = M[:, 0, 3]    - 1j*M[:, 1, 2] + 1j*M[:, 2, 1] + M[:, 3, 0]

    H[:, 1, 0] = M[:, 0, 1]    + M[:, 1, 0]    + 1j*M[:, 2, 3] - 1j*M[:, 3, 2]
    H[:, 1, 1] = M[:, 0, 0]    + M[:, 1, 1]    - M[:, 2, 2]    - M[:, 3, 3]
    H[:, 1, 2] = 1j*M[:, 0, 3] + M[:, 1, 2]    + M[:, 2, 1]    - 1j*M[:, 3, 0]
    H[:, 1, 3] = 1j*M[:, 2, 0] - 1j*M[:, 0, 2] + M[:, 1, 3]    + M[:, 3, 1]

    H[:, 2, 0] = M[:, 0, 2]    + M[:, 2, 0]    - 1j*M[:, 1, 3] + 1j*M[:, 3, 1]
    H[:, 2, 1] = M[:, 1, 2]    - 1j*M[:, 0, 3] + M[:, 2, 1]    + 1j*M[:, 3, 0]
    H[:, 2, 2] = M[:, 0, 0]    - M[:, 1, 1]    + M[:, 2, 2]    -  M[:, 3, 3]
    H[:, 2, 3] = 1j*M[:, 0, 1] - 1j*M[:, 1, 0] + M[:, 2, 3]    + M[:, 3, 2]

    H[:, 3, 0] = M[:, 0, 3]    + 1j*M[:, 1, 2] - 1j*M[:, 2, 1] + M[:, 3, 0]
    H[:, 3, 1] = 1j*M[:, 0, 2] - 1j*M[:, 2, 0] + M[:, 1, 3]    + M[:, 3, 1]
    H[:, 3, 2] = 1j*M[:, 1, 0] - 1j*M[:, 0, 1] + M[:, 2, 3]    + M[:, 3, 2]
    H[:, 3, 3] = M[:, 0, 0]    - M[:, 1, 1]    - M[:, 2, 2]    + M[:, 3, 3]

    H = np.divide(H, 4.0)
    
    return H

def get_T_matrices():
    T = np.array([[1.0, 0.0, 0.0,  1.0],
                  [1.0, 0.0, 0.0, -1.0],
                  [0.0, 1.0, 1.0,  0.0],
                  [0.0, 1j , -1j,  0.0]], dtype='complex64')
    Tinv = np.linalg.inv(T)

    return T, Tinv

def cloude_decomposition(MM, cut_off=2):
    """Cloude decomposition of a Mueller matrix MM"""
    if not isinstance(MM, np.ndarray) or MM.ndim != 3 or MM.shape[1:] != (4, 4):
        raise ValueError(f'Malformed Mueller matrix (with dimension {MM.shape}), has to be of dimension (N, 4, 4)')
    if cut_off > 3 or cut_off < 0:
        raise ValueError('Cutoff must be in a range between (including) 0 and 3')

    H = calc_coherency_matrix(MM)
    eig_val, eig_vec = np.linalg.eigh(H)
    
    # Sort eigenvalues and -vectors descending by eigenvalue
    idx = (-eig_val).argsort(axis=1)
    idx_vec = np.repeat(idx, 4, axis=1).reshape((*idx.shape, 4))

    eig_val_sorted = np.take_along_axis(eig_val, idx, axis=1)
    eig_vec_sorted = np.take_along_axis(np.transpose(eig_vec, (0, 2, 1)), idx_vec, axis=1)

    # Calculate Jones matrices
    J = np.zeros((*eig_val_sorted.shape, 2, 2), dtype='complex64')
    J[:, :, 0, 0] = eig_vec_sorted[:, :, 0] + eig_vec_sorted[:, :, 1]
    J[:, :, 0, 1] = eig_vec_sorted[:, :, 2] - 1j * eig_vec_sorted[:, :, 3]
    J[:, :, 1, 0] = eig_vec_sorted[:, :, 2] + 1j * eig_vec_sorted[:, :, 3]
    J[:, :, 1, 1] = eig_vec_sorted[:, :, 0] - eig_vec_sorted[:, :, 1]

    T, Tinv = get_T_matrices()

    # Calculate Mueller matrices from Jones matrices
    M = np.zeros((J.shape[0], 4, 4))
    for i in range(J.shape[0]):
        Mj = np.zeros((4, 4))
        for j in range(J.shape[1] - cut_off):
            if eig_val_sorted[i, j] > 0:
                mm = eig_val_sorted[i, j] * np.real(T @ np.kron(J[i, j], np.conjugate(J[i, j])) @ Tinv)
                Mj += mm #/ mm[0, 0] / 4
        
        M[i] = Mj

    return M
