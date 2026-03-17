import numpy as np

def best_lowbody_tensor_for_CP_intract(Q, safe=False, epsilon=1.0e-16):
    """
    Update rule for nonnegative tensor Q with shape (I1, I2, ..., ID, K) to
    minimizes the KL-divergence for the CP intract.

    Parameters
    ----------
    Q : ndarray
        Nonnegative tensor of shape (I1, I2, ..., ID, K).

    Returns
    -------
    Q_new : ndarray
        Updated tensor with the same shape as Q.
    """
    D = Q.ndim - 1  # number of tensor modes (excluding K)

    denom1 = Q.sum()                   # sum over all indices including k
    denom2 = Q.sum(axis=tuple(range(D)))  # sum over i1..iD, keep k
    denom = denom1 * (denom2 ** (D - 1))

    numerator = np.ones_like(Q)
    for d in range(D):
        axes = tuple(ax for ax in range(D) if ax != d)
        marg = Q.sum(axis=axes, keepdims=True)  # shape: (1,...,Id,...,1,K)
        numerator *= marg

    best_tensor = numerator / denom
    if safe:
        best_tensor_safe = np.where(best_tensor == 0, epsilon, best_tensor)
        best_tensor_safe /= np.sum(best_tensor_safe)
        return best_tensor_safe
        
    else:
        return best_tensor