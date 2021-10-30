# pyCloude
A cloude decomposition implementation

## Example usage
    from cloude import cloude_decomposition
    import numpy as np
    
    mueller_matrix = np.array([[1, 0.0707, 0.0348, -0.0060],
                               [0.0480, 0.4099, 0.0077, 0.0650],
                               [0.0162, -0.0184, 0.2243, -0.3580],
                               [0.0021, -0.0465, 0.3571, 0.1783]])
    
    cloude_decomposition(mueller_matrix)
