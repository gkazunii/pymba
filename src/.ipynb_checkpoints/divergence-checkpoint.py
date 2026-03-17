import numpy as np
import utils_alg as ua
from chi_logexp import chi_log as chi_log

def chi_divergence(P, Q, chi=1, avoid_nan=False, **kwargs):
    """
    Generalized chi-divergence from tensor P to Q.

    This function computes divergence using a parameterized family:
    - chi = 1: Reverse Kullback-Leibler divergence
    - chi = "Tsallis": Tsallis divergence (requires `q` in kwargs)
    - chi = "Kaniadakis" or "Kani": Kaniadakis divergence (requires `k` in kwargs)

    Parameters
    ----------
    P : ndarray
        Source tensor (non-negative, normalized).
    Q : ndarray
        Target tensor (non-negative, normalized).
    chi : int or str, optional
        Divergence type. One of {1, "Tsallis", "Kaniadakis", "Kani"}.
    avoid_nan : bool, optional
        If True, avoids computing log(0) in KL divergence.
    **kwargs : dict
        Additional parameters such as `q` (for Tsallis) or `k` (for Kaniadakis).

    Returns
    -------
    float
        The computed divergence.

    """
    assert ua.is_valid_probability_tensor(P), "P need to be non-negative and normalized"
    assert ua.is_valid_probability_tensor(Q), "Q need to be non-negative and normalized"
    ## chi_divergence(P,Q,chi=1) is same as inv_KL_divergence(P,Q)
    ## chi_divergence(P,Q,chi="Tsallis") is same as Tsallis_divergence(P, Q, avoid_nan=avoid_nan, q=kwargs["q"])
    if chi == 1:
        return Tsallis_divergence(P, Q, avoid_nan=avoid_nan, q=1)
    elif chi == "Tsallis":
        return Tsallis_divergence(P, Q, avoid_nan=avoid_nan, **kwargs)
    elif chi == "Kani" or chi == "Kaniadakis":
        eps = 1e-12
        P_safe = np.clip(P, eps, None)
        P_safe /= np.sum(P_safe)
        
        chi_div = np.sum( chi_escort(Q, chi=chi, **kwargs) * (chi_log(Q, chi=chi, **kwargs) - chi_log(P_safe, chi=chi, **kwargs)))
        return chi_div
    else:
        chi_div = np.sum( chi_escort(Q, chi=chi, **kwargs) * (chi_log(Q, chi=chi, **kwargs) - chi_log(P, chi=chi, **kwargs)))
        return chi_div
        
def chi_function(P, chi=1, **kwargs):
    """
    Any postive function s: R_+ -> R_+.
    Based on this defined s, we can define chi-log as
    chi_log(t) = int_{0 -> t} 1 / s(s) dx
    and its inverse
    chi_exp(t) s.t. t = chi_exp(chi_log(t))
    """
    if chi == 1:
        return P
    elif chi == "Tsallis":
        q = kwargs["q"]
        #return P**q
        P = np.asarray(P, dtype=float)
        out = np.zeros_like(P)
        mask = P > 0.0
        out[mask] = P[mask] ** q
        return out
    elif chi == "Kaniadakis" or chi == "Kani":
        k = kwargs["k"]
        return 2*P / (P**k + P**(-k))
    else:
        raise NameError(f"{chi} is not defined")   

def grad_chi(P, chi=1, **kwargs):
    """
    The first derivative chi-function x'(t) at t=P
    """
    if chi == 1 or chi == 1.0:
        diff_chi =  (P ** 0)
    elif chi == "Tsallis":
        q = kwargs["q"]
        diff_chi = q * (P**(q-1))
    elif chi == "Kaniadakis" or chi == "Kani":
        k = kwargs["k"]
        #diff_chi = ( 1 - k * np.tanh( k * np.log(P) ) ) / ( np.cosh( k * np.log(P)  ) )
        diff_chi = 2 * ( P ** k ) * (1+k - (-1+k)* (P ** (2*k)) ) / ( 1 + P ** (2*k) )**2
    elif chi == "exp":
        b = kwargs["b"]
        diff_chi = b * np.exp(b*P)
    elif chi == "stretch":
        s = kwargs["s"]
        diff_chi = np.log(P)*(-1/s) * (-1+s+s*np.log(P))
    else:
        raise NameError(f"diff function for {chi} is not defined")   

    return diff_chi

def grad_grad_chi(P, chi=1, **kwargs):
    """
    The 2nd derivative chi-function x''(t) at t=P
    """ 
    def sech(y):
        return 1.0/np.cosh(y)
    
    if chi == 1 or chi == 1.0:
        diff_diff_chi =  0 * P
    elif chi == "Tsallis":
        q = kwargs["q"]
        diff_diff_chi = q * (q-1) * (P**(q-2))
    elif chi == "Kaniadakis" or chi == "Kani":
        k = kwargs["k"]
        q = 2*k*np.log(P)
        diff_diff_chi = k * (sech(q))**3 * (k * ( -3 + np.cosh(q)) - np.sinh(q)) / (2*P)
    elif chi == "exp":
        b = kwargs["b"]
        diff_diff_chi = b * b * np.exp(b*P)
    elif chi == "stretch":
        s = kwargs["s"]
        diff_diff_chi = (-1+s) * np.log(P)**(-(1+s)/s) * (-1 + s*np.log(P)) / ( P*s )
    else:
        raise NameError(f"diff diff function for {chi} is not defined")   

    return diff_diff_chi

def chi_escort(P, chi=1, **kwargs):
    """
    Compute the escort distribution under a given chi-family.

    Parameters
    ----------
    P : ndarray
        Input tensor (non-negative).
    chi : int or str, optional
        Type of divergence. One of {1, "Tsallis", "Kaniadakis", "Kani"}.
    **kwargs : dict
        Additional parameters like `q` or `k` for Tsallis/Kaniadakis.

    Returns
    -------
    ndarray
        Escort distribution normalized to sum to 1.

    """
    chiP = chi_function(P, chi=chi, **kwargs)
    return chiP / np.sum(chiP)

def Tsallis_divergence(P, Q, q=1, avoid_nan=False):
    """ Tsallis divergence from tensor P to Q
    Both P and Q need to be non-negative and normalized.
    q == 1 is for KL-divergence

    Tsallis_divergence(P,Q,q=q) == chi_divergence(P, Q, chi="Tsallis", q=q)
    """
    assert ua.is_valid_probability_tensor(P), "P need to be non-negative and normalized"
    assert ua.is_valid_probability_tensor(Q), "Q need to be non-negative and normalized"
    
    if q == 1:
        return inv_KL_divergence(P, Q, avoid_nan=avoid_nan)
    else:
        t_div = ( 1 - np.sum( Q**q * P**(1-q) ) )
        t_div = t_div / ( (1-q) * np.sum( Q ** q) )
        return t_div
    #else:
    #    raise ValueError(f"q need to be positive")

def KL_divergence(P, T, avoid_nan=False):
    """ KL divergence from tensor P to T
    Both P and T need to be postive.
    Their total sum can be larger than 1.
    """
    if avoid_nan:
        """
        If P has zero value, KL might be nan.
        Thus, we avoid this case
        """
        Parr = P[ P != 0 ]
        Tarr = T[ P != 0 ]
        #return np.sum(Parr * np.log(Parr / Tarr)) - np.sum(P) + np.sum(T)
        return np.sum(Parr * np.log(Parr) - Parr * np.log(Tarr)) - np.sum(P) + np.sum(T)
    else:
        return np.sum(P * np.log(P / T)) - np.sum(P) + np.sum(T)

def inv_KL_divergence(P, T, avoid_nan=False):
    return KL_divergence(T, P, avoid_nan=avoid_nan)

def alpha_divergence(T,P,α, avoid_nan=False):
    """ Alpha divergence from tensor T to P
    Both P and T need to be postive.
    Their total sum can be larger than 1.
    """
    if α == 1.0:
        return KL_divergence(T,P, avoid_nan=avoid_nan)
        
    elif α == 0.0:
        return inv_KL_divergence(T,P, avoid_nan=avoid_nan)
        
    else:
        term1 = α * np.sum( T )
        term2 = (1-α) * np.sum( P )
        term3 = np.sum( T**α * P**(1-α) )
        return 1.0 / ( α*(1-α) ) * (term1 + term2 - term3)

def renyi_divergence(T,P,α, avoid_nan=False):
    assert ua.is_valid_probability_tensor(P), "P need to be non-negative and normalized"
    assert ua.is_valid_probability_tensor(Q), "Q need to be non-negative and normalized"
    
    α = α * 1.0
    if α == 0.0:
        return -np.log( np.sum(P) )
    elif α == 1.0:
        return KL_divergence(T,P,avoid_nan=avoid_nan)
    else:
        return 1.0/(α-1.0) * np.log( np.sum( T**α * P**(1-α) ) )