# pyCloude
A cloude decomposition implementation

## Installation
Download package and execute either

    python setup.py install
to install the package or

    python setup.py develop
to install it in development mode.

## Example usage
    from cloude import cloude_decomp
    import numpy as np
    
    mueller_matrix = np.array([[1, 0.0707, 0.0348, -0.0060],
                               [0.0480, 0.4099, 0.0077, 0.0650],
                               [0.0162, -0.0184, 0.2243, -0.3580],
                               [0.0021, -0.0465, 0.3571, 0.1783]])
    
    decomp_matrix = cloude_decomp(mueller_matrix)
The matrices are scaled by their respective eigenvalues. If you want to use them to fit to an experiment, normalize them by the M_00 matrix element (`decomp_matrix[:, 0, 0]`).
To get the corresponding Jones matrix estimate you may use

    from cloude import cloude_decomp_jones
    
    decomp_matrix = cloude_decomp_jones(mueller_matrix)

### Parameters
Parameter usage is the same for `cloude_decomp` and `cloude_decomp_jones` functions.
For an output of the eigenvalues use the `output_eigenvalue` parameter

    decomp_matrix, ev = cloude_decomp(mueller_matrix, output_eigenvalue=True)
By default the matrix with the highest eigenvalue is returned, which is equivalent to
    
    decomp_matrix = cloude_decomp(mueller_matrix, ev_mask=[1, 0, 0, 0])
You may tune the output as an arbitrary sum of decompositions by the eigenvalue mask

    decomp_matrix = cloude_decomp(mueller_matrix, ev_mask=[1, 1, 1, 1])
returns the full sum of all coherent matrices, hence `decomp_matrix == mueller_matrix`.
