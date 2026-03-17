import numpy as np
from divergence import grad_chi
from divergence import chi_escort

def chi_FIM(P, msk, eta, chi=1, **kwargs):
    """
    Compute the χ-Fisher Information Matrix (χ-FIM) for a given probability distribution tensor `P`.

    Depending on the parameter `chi`, this function calculates:
    - The standard Fisher Information Matrix (FIM) if `chi == 1` or if `chi == "Tsallis"` with `q == 1`.
    - The q-Fisher Information Matrix (q-FIM) associated with the Tsallis divergence when `chi == "Tsallis"` and `q != 1`.
    - The generalized χ-FIM for other χ-divergences (e.g., "Kaniadakis") otherwise.

    Parameters
    ----------
    P : np.ndarray
        Probability distribution tensor of arbitrary dimension.
    msk : array-like of shape (n_params, D)
        Index mask array selecting the natural parameters θ used in the computation.
    eta : np.ndarray
        Expectation parameters tensor, with the same shape as `P`.
    chi : {1, "Tsallis", "Kaniadakis", ...}, default=1
        Specifies the divergence type:
        - 1 for the standard KL divergence (FIM).
        - "Tsallis" for Tsallis divergence (q-FIM).
        - Other strings for general χ-divergences.
    **kwargs
        Additional parameters for specific divergences, e.g., `q` for Tsallis.

    Returns
    -------
    G : np.ndarray of shape (n_params, n_params)
        The computed chi-Fisher information matrix
        
    Notes
    -----
    - The standard FIM corresponds to the second derivative of the KL divergence.
    - The q-FIM corresponds to the second derivative of the Tsallis divergence.

    """
    if chi == 1 or (chi == "Tsallis" and kwargs["q"]==1):
        return FIM(msk, eta)
    elif chi == "Tsallis":
        q = kwargs["q"]
        return q_FIM(P, msk, eta, q)
    else:
        # Compute tilde_chi_chi = chi_escort * grad_chi
        tilde_chi_chi = chi_escort(P, chi=chi, **kwargs) * grad_chi(P, chi=chi, **kwargs)
        
        # Precompute total sum
        total_tilde_chi_chi = np.sum(tilde_chi_chi)
        
        # Compute cumulative sums along all axes (reverse direction)
        for axis in range(P.ndim):
            tilde_chi_chi = np.flip(np.cumsum(np.flip(tilde_chi_chi, axis=axis), axis=axis), axis=axis)
        
        # Shape info
        msk = np.array(msk)
        n_params, D = msk.shape
    
        # Broadcast max index over all (i,j) pairs
        max_idx = np.maximum(msk[:, None, :], msk[None, :, :])
        
        # Evaluate tilde_chi_chi at max index
        tilde_vals = tilde_chi_chi[tuple(max_idx.transpose(2,0,1))]
    
        # Evaluate tilde_chi_chi at msk points
        tilde_msk_vals = tilde_chi_chi[tuple(msk.T)]
    
        # Outer product terms
        outer_eta_tilde = np.outer(eta[tuple(msk.T)], tilde_msk_vals)
        outer_tilde_eta = np.outer(tilde_msk_vals, eta[tuple(msk.T)])
        outer_eta_eta = np.outer(eta[tuple(msk.T)], eta[tuple(msk.T)]) * total_tilde_chi_chi
    
        # Final G matrix
        G = tilde_vals - outer_eta_tilde - outer_tilde_eta + outer_eta_eta

    return G

def q_FIM(P, msk, eta, q):
    """
    Compute the q-Fisher Information Matrix (q-FIM).

    Parameters
    ----------
    P : np.ndarray
        Probability distribution tensor.
    msk : array-like of shape (n_params, D)
        Index mask for natural parameters.
    eta : np.ndarray
        Expectation parameters tensor.
    q : float
        Tsallis parameter.

    Returns
    -------
    G : np.ndarray of shape (n_params, n_params)
        q-Fisher information matrix.

    """
    
    #Pq = P ** (2*q - 1)
    Pq = np.zeros_like(P)
    mask = P > 0
    Pq[mask] = P[mask] ** (2*q - 1)
    total_Pq  = np.sum(Pq)
    
    #total_Pq1 = np.sum(P**q)
    Pq1 = np.zeros_like(P)
    Pq1[mask] = P[mask] ** q
    total_Pq1 = np.sum(Pq1)
    
    for axis in range(P.ndim):
        Pq = np.flip(np.cumsum(np.flip(Pq, axis=axis), axis=axis), axis=axis)

    msk = np.array(msk)
    n_params, D = msk.shape

    max_idx = np.maximum(msk[:, None, :], msk[None, :, :])

    Pq_max = Pq[tuple(max_idx.transpose(2,0,1))]  # shape (n_params, n_params)

    eta_vals = eta[tuple(msk.T)]  # shape (n_params,)

    G = Pq_max - eta_vals[:, None] * Pq[tuple(msk.T)][None, :] - eta_vals[None, :] * Pq[tuple(msk.T)][:, None] + eta_vals[:, None] * eta_vals[None, :] * total_Pq
    G *= q 
    G /= total_Pq1

    return G


def FIM(msk, eta):
    """
    Compute the standard Fisher Information Matrix (FIM) based on expectation parameters.

    Parameters
    ----------
    msk : array-like of shape (n_params, D)
        Index mask for natural parameters.
    eta : np.ndarray
        Expectation parameters tensor.

    Returns
    -------
    G : np.ndarray of shape (n_params, n_params)
        Fisher information matrix.

    """
    msk = np.array(msk)
    n_params, D = msk.shape
    
    max_idx = np.maximum(msk[:, None, :], msk[None, :, :])
    eta_max = eta[tuple(max_idx.transpose(2,0,1))]

    eta_vals = eta[tuple(msk.T)]
    eta_outer = np.outer(eta_vals, eta_vals)

    G = eta_max - eta_outer
    return G