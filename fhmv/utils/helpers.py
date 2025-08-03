# fhmv/utils/helpers.py

"""
Shared helper functions for the FHMV project.
"""

import numpy as np

def log_sum_exp(log_terms: np.ndarray, axis=None, keepdims=False) -> np.ndarray:
    """
    Numerically stable computation of log(sum(exp(log_terms))).

    Args:
        log_terms (np.ndarray): Array of terms in log space.
        axis (int, optional): Axis along which to compute. Defaults to None (all elements).
        keepdims (bool, optional): If True, keeps the reduced dimensions. Defaults to False.

    Returns:
        np.ndarray: log(sum(exp(log_terms)))
    """
    max_log = np.max(log_terms, axis=axis, keepdims=True) # keepdims for broadcasting
    # Handle cases where all log_terms are -inf
    is_all_inf = np.isinf(max_log)
    
    # Subtract max_log for numerical stability
    terms_shifted = log_terms - max_log
    
    log_sum_exp_val = max_log + np.log(np.sum(np.exp(terms_shifted), axis=axis, keepdims=keepdims) + 1e-30) # Add epsilon for log(0)
    
    # If max_log was -inf, result should be -inf
    if isinstance(is_all_inf, np.ndarray):
        log_sum_exp_val[is_all_inf] = -np.inf
    elif is_all_inf:
        return -np.inf
        
    return log_sum_exp_val


# Example of how modules might use the constants:
# from fhmv.utils import constants
# if state_name == constants.FHMV_REGIME_AMB:
#     # do something

# if __name__ == '__main__': # Example usage of log_sum_exp
#     a = np.array([-1000, -1001, -999])
#     print(f"LogSumExp of {a}: {log_sum_exp(a)}") # Should be close to -999

#     b = np.array([[-1000, -1001], [-999, -1002]])
#     print(f"LogSumExp of {b} along axis 1: {log_sum_exp(b, axis=1)}") 
#     # Expected: [log(exp(-1000)+exp(-1001)), log(exp(-999)+exp(-1002))] ~ [-1000, -999]

#     c = np.array([-np.inf, -np.inf])
#     print(f"LogSumExp of {c}: {log_sum_exp(c)}")