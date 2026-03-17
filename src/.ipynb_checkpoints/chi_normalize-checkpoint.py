from scipy.optimize import root_scalar
import warnings
import time
import numpy as np

from chi_logexp import chi_log
from chi_logexp import chi_exp
from divergence import chi_function
from divergence import grad_chi
from transform import prob_from_theta
from transform import theta_from_prob

def get_energy_from_theta(theta):
    """
    Once you obtain theta, the Energy function can be obtained
    by just cumsum regardless of the choice of chi.
    """
    energy = theta.copy()
    if theta.flat[0] > 0.0:
        warnings.warn(
            "theta[0,...,0] is assumed to be 0 when constructing the energy function"
            ,RuntimeWarning)
        theta.flat[0] = 0.0
        
    for axis in range(theta.ndim):
        energy = np.cumsum(energy, axis=axis)
    assert np.isclose(energy.flat[0], 0), "The empty sum in energy function did not becomes 0"
    
    return energy

def get_normalized_prob_from_theta(theta, psi, chi=1, **kwargs):
    '''
    based on the defination of deformed low-body models, 
    we obtain the normalized model by energy function and given
    free energy psi. The energy function will be obtained by theta
    '''
    energy = get_energy_from_theta(theta)
    normalized_prob = chi_exp(energy - psi, chi=chi, **kwargs)
    return normalized_prob
    
def get_normalized_prob_from_energy_and_psi(energy, psi, chi=1, **kwargs):
    normalized_prob = chi_exp(energy - psi, chi=chi, **kwargs)
    return normalized_prob

def diff_diff_target_function(energy, psi, chi=1, **kwargs):
    normalized_prob = get_normalized_prob_from_energy_and_psi(energy, psi, chi=chi, **kwargs)
    dd_value = np.sum(
        chi_function(normalized_prob, chi=chi, **kwargs) 
        * grad_chi(normalized_prob, chi=chi, **kwargs)
    )
    return dd_value

def diff_target_function(energy, psi, chi=1, **kwargs):
    normalized_prob = get_normalized_prob_from_energy_and_psi(energy, psi, chi=chi, **kwargs)
    return -np.sum(chi_function(normalized_prob, chi=chi, **kwargs))

def target_function(energy, psi, chi=1, **kwargs):
    prob = get_normalized_prob_from_energy_and_psi(energy, psi, chi=chi, **kwargs)
    return np.sum(prob) - 1

"""
def get_psi_chi(energy, method="toms748", tol=1.0e-8, auto_bracket=True, chi=1, **kwargs):
    failing_flag = False

    loss = lambda psi: target_function(energy, psi, chi=chi, **kwargs)
    diff_loss = lambda psi: diff_target_function(energy, psi, chi=chi, **kwargs)
    diff_diff_loss = lambda psi: diff_diff_target_function(energy, psi, chi=chi, **kwargs)

    if method in ["bisect", "brentq", "toms748"]:
        if auto_bracket:
            #if chi == "Tsallis":
            #    max_energy = np.max(energy)
            #    n = np.size(energy)
            #    lo = +max_energy 
            #    hi = +max_energy - chi_log(1/n, chi=chi, **kwargs)
            #    bracket = (lo, hi)
            #    
            #    vlo = loss(lo)
            #    vhi = loss(hi)
            #    
            #    assert np.isfinite(max_energy), f"max_energy is not finite: {max_energy}"
                #assert loss(lo) > 0.0, f"{loss(lo)} should be positive"
                #assert loss(hi) < 0.0, f"{loss(hi)} should be negative"
            #else:
                #bracket = get_auto_bracket(loss)

            ## Original
            max_energy = np.max(energy)
            n = np.size(energy)
            lo = +max_energy 
            hi = +max_energy - chi_log(1/n, chi=chi, **kwargs)
            bracket = (lo, hi)
            vlo = loss(lo)
            vhi = loss(hi)

            print("spacing(maxE)=", np.spacing(max_energy))
            print("delta=", -chi_log(1.0/n, chi=chi, **kwargs))
            print("hi==lo?", (max_energy - chi_log(1.0/n, chi=chi, **kwargs)) == max_energy)

            assert np.isfinite(max_energy), f"max_energy is not finite: {max_energy}"
        else:
            bracket = get_fixed_bracket(chi, **kwargs)

        res = root_scalar(loss, method=method, xtol=tol, bracket=bracket)
               
    if method == "newton":
        res = root_scalar(loss, fprime=diff_loss, method=method, xtol=tol, x0=0.0)
        if not(res.converged):
            failing_flag = True
            method = "halley"

    if method == "halley":
        res = root_scalar(loss, 
                          fprime=diff_loss, 
                          fprime2=diff_diff_loss, 
                          method=method, xtol=tol, x0=0.0)

    sol_psi = res.root
    converged = res.converged
    if not(converged):
        print(f"{method} did not converge after {res.iterations} iterations")
        print(f"status:{res.flag}")
        failing_flag = True

    return sol_psi, converged, failing_flag
"""
def get_psi_chi(energy, method="toms748", tol=1e-8, auto_bracket=True, chi=1, **kwargs):
    energy = np.asarray(energy, dtype=float)
    finite = np.isfinite(energy)
    if not np.any(finite):
        raise ValueError("energy has no finite entries")

    M = np.max(energy[finite])
    E0 = energy - M  # max(E0)=0

    n = energy.size
    loss_t = lambda t: np.sum(chi_exp(E0 - t, chi=chi, **kwargs)) - 1.0

    # t の bracket（psi ではなく t）
    lo = 0.0
    hi = -chi_log(1.0/n, chi=chi, **kwargs)   # Kaniなら正になるはず

    # もし hi が不正なら最低限の幅を作る
    if not np.isfinite(hi) or hi <= lo:
        hi = 1.0

    vlo, vhi = loss_t(lo), loss_t(hi)

    # bracket が成立しないときは hi を増やしていく（t↑で各項↓を期待）
    if np.isfinite(vlo) and np.isfinite(vhi) and np.sign(vlo) * np.sign(vhi) > 0:
        step = max(1.0, abs(hi - lo))
        for _ in range(80):
            hi += step
            vhi = loss_t(hi)
            if np.sign(vlo) * np.sign(vhi) <= 0:
                break
            step *= 2.0
        else:
            raise ValueError(f"Failed to bracket root: vlo={vlo}, vhi={vhi}, lo={lo}, hi={hi}")

    res = root_scalar(loss_t, method=method, xtol=tol, bracket=(lo, hi))
    psi = M + res.root
    return psi, res.converged, (not res.converged)
    

def chi_normalized_prob_from_theta(theta, method="toms748", tol=1.0e-16, standard=True,
                                   check_normalization=False, 
                                   auto_bracket=True,
                                   chi=1, **kwargs):

    """
    Compute the normalized deformed distribution Q from the given theta.
    This function ensures that the resulting distribution sums to 1, 
    either analytically (for KL-divergence / chi=1) or numerically 
    (for Tsallis/Kani deformations).

    Parameters
    ----------
    theta : np.ndarray
        The input tensor of parameters. theta.flat[0] is set to 0 to remove
        an arbitrary constant.

    method : str, optional
        Root-finding method for numerically estimating the normalizer (psi) 
        when chi != 1. Choices are:
        - 'toms748'
        - 'bisect'
        - 'brentq'
        - 'newton'
        - 'halley'
        Default is 'toms748'.

    tol : float, optional
        Tolerance for the root-finding procedure. Default is 1.0e-10.

    check_normalization : bool, optional
        If True, verifies that the resulting distribution sums to 1 and 
        forcibly normalizes it using the standard method if necessary.
        Default is False.

    chi : int or str, optional
        Type of divergence or deformation:
        - 1 : standard KL-divergence
        - "Tsallis" : Tsallis divergence (requires optional q in kwargs)
        - "Kani"    : Kaniadakis divergence (requires optional k in kwargs)
        Default is 1.

    **kwargs : dict, optional
        Additional parameters depending on chi:
        - q : float, Tsallis q parameter (required if chi="Tsallis")
        - k : float, Kani k parameter (required if chi="Kani")

    Returns
    -------
    Q : np.ndarray
        Normalized probability distribution corresponding to theta.

    elapsed_time_for_normalization : float
        Time (in seconds) spent on normalization.

    Notes
    -----
    - For chi=1 or Tsallis(q=1) or Kani(k=0), the normalization is done analytically.
    - For deformed distributions, the normalizer psi is estimated numerically
      using the specified root-finding method.
    - If numerical normalization fails or check_normalization=True, 
      the distribution is forcibly normalized using the standard method.
    """
    
    ## The updated theta has no-meaning full value in theta[0,...,0].
    ## But we need normalized distribution to obtain eta in the next step.
    ## We obtain the value theta[0,...,] s.t. the disribution is normalized
    theta.flat[0] = 0.0

    ## Parameters for deform products
    q = kwargs.get("q", None)
    k = kwargs.get("k", None)
    
    start_time_for_normalization = time.time()
    if standard or chi == 1 or (chi == "Tsallis" and q == 1.0) or (chi == "Kani" and k == 0.0):
        ## For KL-divergence, 
        ## we do not need numerical normalization
        Q = prob_from_theta(theta, chi=chi, **kwargs)
        Q /= np.sum(Q)
        flag = False
            
    else:
        # Estimate normalizer (Free energy) numercially
        energy = get_energy_from_theta(theta)
        
        optimal_psi, converged, flag = get_psi_chi(energy, method=method, tol=tol, auto_bracket=auto_bracket, chi=chi, **kwargs)
        Q = get_normalized_prob_from_energy_and_psi(energy, optimal_psi, chi=chi, **kwargs)

    if flag or check_normalization:
        total_sum = np.sum(Q)
        print("The total sum of updated Q:", total_sum)
        if ( np.abs(total_sum - 1.0) > 1.0e-3 ) or ( np.isnan(total_sum) ):
            Q = prob_from_theta(theta)
            Q /= np.sum(Q)
            print("\033[31mDeformed normalization failed!\033[0m")
            print("The total sum is not 1.0, thus, forced to normalized by standard way")
            
    elapsed_time_for_normalization = time.time() - start_time_for_normalization
    return Q, elapsed_time_for_normalization



