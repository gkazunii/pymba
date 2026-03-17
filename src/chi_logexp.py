import numpy as np

def chi_exp(x, chi=1, **kwargs):
    """
    Generalized exponential function (χ-exponential).

    Parameters
    ----------
    x : array_like
        Input value(s).
    chi : {1, "Tsallis", "Kaniadakis", "Kani", "stretch"}, optional
        Type of deformation.
    **kwargs : dict
        Additional keyword arguments for each type:
        - If chi="Tsallis", requires `q`.
        - If chi="Kaniadakis" or "Kani", requires `k`.
        - If chi="stretch", requires `s`.

    Returns
    -------
    ndarray
        χ-exponential of `x`.

    Raises
    ------
    NameError
        If unknown `chi` type is specified.

    Examples
    --------
    >>> x = np.array([2.3,3.2,2.1,2.1,4.0,-1.2,-1.2]);
    >>> chi = "Tsallis"; q = 1.1;
    >>> m = chi_log(chi_exp(x, chi=chi, q=q), chi=chi, q=q);
    >>> m == x
    array([ True,  True,  True,  True,  True,  True,  True])
    """
    if isinstance(x, list):
        x = np.array(x)
    
    if chi == 1:
        return np.exp(x)
    elif chi == "Tsallis":
        return q_exp(x, q=kwargs["q"]) 
    elif chi == "Kaniadakis" or chi == "Kani":
        return k_exp(x, k=kwargs["k"]) 
    elif chi == "stretch":
        return s_exp(x, s=kwargs["s"]) 
    else:
        raise NameError(f"{chi} is not defined")
    
def chi_log(x, chi=1, **kwargs):
    """
    Generalized logarithm function (χ-logarithm).

    Parameters
    ----------
    x : array_like
        Input value(s), must be positive.
    chi : {1, "Tsallis", "Kaniadakis", "Kani", "stretch"}, optional
        Type of deformation.
    **kwargs : dict
        Additional keyword arguments for each type:
        - If chi="Tsallis", requires `q`.
        - If chi="Kaniadakis" or "Kani", requires `k`.
        - If chi="stretch", requires `s`.

    Returns
    -------
    ndarray
        χ-logarithm of `x`.

    Raises
    ------
    AssertionError
        If `x` contains non-positive values.
    NameError
        If unknown `chi` type is specified.
    """
    if isinstance(x, list):
        x = np.array(x)
    #assert np.all(x >= 0), "x include non-positive value"
    if chi == 1:
        return np.log(x) 
    elif chi == "Tsallis":
        return q_log(x, q=kwargs["q"]) 
    elif chi == "Kaniadakis" or chi == "Kani":
        return k_log(x, k=kwargs["k"]) 
    elif chi == "stretch":
        return s_log(x, s=kwargs["s"]) 
    else:
        raise NameError(f"{chi} is not defined")


def q_exp(x, q=1):
    """
    Tsallis-exp function
    """
    if q == 1:
        return np.exp(x)
    else:
        x = np.asarray(x, dtype=float)
        base = 1.0 + (1.0 - q) * x
        out  = np.zeros_like(base)
        mask = base > 0.0
        out[mask] = base[mask] ** (1.0/(1.0-q))
        return out
        #return np.maximum(1 + (1-q) * x, 0)**(1/(1-q))
    
def q_log(x, q=1):
    """
    Tsallis-logarithm function
    """
    if q == 1:
        return np.log(x)
    else:
        return (x**(1-q) - 1) / (1-q)

def k_exp(x, k):
    """
    Kaniadakis-exp function.
    """
    x = np.asarray(x, dtype=float)
    k = float(k)
    if k == 0.0:
        return np.exp(x)
    k = abs(k)  # 念のため（exp_{-k}=exp_k なので）
    return np.exp(np.arcsinh(k * x) / k)

def k_log(x, k):
    """
    Kaniadakis-logarithm function.
    """
    x = np.asarray(x, dtype=float)
    k = float(k)
    if k == 0.0:
        return np.log(x)
    k = abs(k)
    # x>0 が必要（0確率を許すなら 0 は -inf に拡張するのが筋）
    return np.sinh(k * np.log(x)) / k

"""
def k_exp(x, k):
    #assert -1 < k < +1, "|k| < 1 is required" 
    if k == 0:
        return np.exp(x)
    else:
        return (k*x + np.sqrt( 1 + k*k * x *x))**(1/k)

def k_log(x, k):
    #assert -1 < k < +1, "|k| < 1 is required" 
    if k == 0:
        return np.log(x)
    else:
        return ( x**k - x**(-k) ) / (2*k)
"""

def s_log(x, s=1):
    """
    scratched-log function.
    """
    ## NOTE: s can be negative
    ## Please check the paper about Stretched exp.
    if s > 0:
        return ( np.log(x) )**(1/s)
    else:
        raise ValueError("s must be positive")
    
def s_exp(x, s=1):
    """
    scratched-exp function.
    """
    return ( np.exp(x**s) )
