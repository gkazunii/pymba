import numpy as np
import utils_alg as ua
from chi_logexp import chi_log
from chi_logexp import chi_exp
from divergence import chi_escort

def prob_from_theta(theta, chi=1, **kwargs):
    """
    Compute the χ-probability tensor from natural parameters (theta).

    Parameters
    ----------
    theta : ndarray
        Tensor of χ-natural parameters.
    chi : int or str, optional
        Type of χ-exponential. Default is 1 (standard exponential).
    **kwargs : dict
        Additional arguments for `chi_exp`, e.g., `q` for Tsallis.

    Returns
    -------
    ndarray
        Probability tensor obtained by applying cumulative summations
        and χ-exponential.
    """
    P = theta.copy()
    for axis in range(theta.ndim):
        P = np.cumsum(P, axis=axis)
    return chi_exp(P, chi=chi, **kwargs)

def theta_from_prob(P, chi=1, avoid_nan=False, add_val=1.0e-8, **kwargs):
    """
    Recover natural parameters (theta) from a χ-probability tensor.
    Inverse function of prob_from_theta.

    Parameters
    ----------
    P : ndarray
        Probability tensor.
    chi : int or str, optional
        Type of χ-logarithm. Default is 1 (standard log).
    avoid_nan : bool, optional
        If True, add a small positive constant to P to avoid log(0).
    add_val : float, optional
        Value to add if `avoid_nan=True`. Default is 1.0e-8.
    **kwargs : dict
        Additional arguments for `chi_log`, e.g., `q` for Tsallis.

    Returns
    -------
    ndarray
        Tensor of natural parameters (theta).
    """
    
    P = P.copy()  # avoid modifying original
    if avoid_nan:
        P += add_val
    logP = chi_log(P, chi=chi, **kwargs)
    theta = decumsum(logP, axis=0)
    for d in range(1, P.ndim):
        theta = decumsum(theta, axis=d)
    return theta

def eta_from_prob(P, chi=1, **kwargs):
    """
    Compute η-representation of a probability tensor.

    Parameters
    ----------
    P : ndarray
        Input probability tensor.
    chi : int or str, optional
        Type of escort transformation. Default is 1 (standard).
    **kwargs : dict
        Additional arguments for `chi_escort`.

    Returns
    -------
    ndarray
        η-representation tensor.
    """
    eta = chi_escort( P.copy(), chi=chi, **kwargs)
    for axis in range(P.ndim):
        eta = np.flip(np.cumsum(np.flip(eta, axis=axis), axis=axis), axis=axis)
    return eta

def escort_from_eta(eta, chi=1, **kwargs):
    """
    Recover the escort distribution from η-representation.

    Note: This is the inverse of `eta_from_prob` only if chi=1.
    For general χ, the inverse may not correspond to the escort
    of a valid probability tensor.

    Parameters
    ----------
    eta : ndarray
        η-representation tensor.
    chi : int or str, optional
        Type of escort transformation. Default is 1.
    **kwargs : dict
        Additional arguments for `chi_escort`.

    Returns
    -------
    ndarray
        Reconstructed escort tensor.

    Examples
    --------
    >>> p = np.random.rand(2, 3, 3)
    >>> p = p / np.sum(p)
    >>> escp = chi_escort(p, chi="Tsallis", q=0.3)
    >>> eta = eta_from_prob(p, chi="Tsallis", q=0.3)
    >>> reconst_escp = escort_from_eta(eta, chi="Tsallis", q=0.3)
    >>> np.allclose(escp, reconst_escp, atol=1e-5)
    True
    """
    etap = reverse_all_axes(eta)
    escort = decumsum(etap, axis=0)
    for d in range(1, eta.ndim):
        escort = decumsum(escort, axis=d)
    return reverse_all_axes(escort)

def decumsum(Y, axis):
    Y_shifted = np.roll(Y, shift=1, axis=axis)
    index = [slice(None)] * Y.ndim
    index[axis] = 0
    Y_shifted[tuple(index)] = 0
    return Y - Y_shifted

def reverse_all_axes(arr):
    return np.flip(arr)
