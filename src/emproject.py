import divergence 
import time
import mproject
from verbose import print_initial_info_em
from verbose import print_initial_info
from verbose import print_header
from verbose import show_verbose
from verbose import show_terminate
from mproject import MBA_LBFGS
from mproject import MBA

import manage_intract
import numpy as np
import mask

def LRA(T, rank, q, max_iter_outer=100, 
        max_iter_inner=100,
        init="random", epsilon_auto=False,
        method="LBFGS",
        lr=0.1, 
        safe_estep=True,
        rel_epsilon=1e-6, rcond=1.0e-8, 
        norm_method="toms748", norm_standard=False,
        get_cost_dual=True, get_history=True, gtol_inner=1.0e-12,
        tol_inner=1.0e-8, tol_outer=1.0e-8, epsilon=1.0e-8,
        seed=None, init_inner="uniform", beta=1.0, Newton=True,
        verbose = True, lr_search=False, Newton_solver="solve",
        verbose_inner = True, maxls = 10,
        verbose_interval_inner = 10,
        verbose_interval_outer = 1,
        confirm_inequality=True, confirm_tightness=True, confirm_optimal_mstep=True,
        confirm_normalization=True,
        use_prev_theta=True, delta=1.0e-10,
        need_factor=False,
       ):

    history = {"iter":[], "time":[], "loss":[]} 

    # Original tensor's dim
    D = np.ndim(T)

    # Shape of low-body tensor Q
    tensor_shape = np.shape(T) + (rank,)
    
    # structure of Interaction in Q
    intract_CP = manage_intract.get_intract_CP(D+1)

    # Obtain the set of learned indices
    # For CP, we need all one-body and 
    # two-body between hidden and visible modes
    msk = mask.get_learn_indices(tensor_shape, intract_CP)
    n_params = len(msk)
    
    chi = "Tsallis"
    if method == "Newton":
        Newton = True
    elif method != "LBFGS":
        Newton = False
    else:
        Newton = "LBFGS"
        lr_search = True
        
    if verbose:
        print_initial_info_em(rank, q, init, init_inner, use_prev_theta, safe_estep)
        print_initial_info(init_inner, Newton, Newton_solver, lr, beta, epsilon, epsilon_auto, tensor_shape, n_params, intract_CP, lr_search, maxls, norm_method, norm_standard, chi, q=q)
        print_header(chi)
    
    ## Randomized initialization 
    ## LRA is a non-convex optimization problem and 
    ## solution depends on R_init.
    R = initial_R(tensor_shape, init)
    R_prev = np.copy(R) # Just for debuging
    
    ## Marginalization along hidden variable k
    ## P is the deformed low-rank tensor
    P = np.sum(R, axis=-1)
    
    ## R is a low-body tensor. R has one- and two-body interactions
    ## Q is on the data manifold. After marginalization, it becomes data T
    n_iter = 0

    cost_prev = +np.inf
    cost_dual_prev = +np.inf
    
    start_time = time.time()
    elapsed_time = 0.0
    
    theta_prev = None
    converge = False
    P_prev = P
    while(n_iter < max_iter_outer + 1):
        total_sum_P = np.sum(P)
        
        ## E-step: projection onto data manifold
        ## Optimal update to minimize D_f[R||Q] for R
        Q = estep(T, R, P, safe_estep, delta)

        ## For debug
        ## Confirm normalization of P
        ## If P is not normalized, we provide previous P
        if confirm_normalization:
            total_sum = np.sum(Q)
            if abs( total_sum - 1.0 ) > 1.0e-3:
                print(f"The tensor Q is not normalized: Total sum was {total_sum}")
                print(f"This error may come from instability of inv(G).") 
                ## Return previous step's distribution if the Newton solver is 
                if method == "Newton" and Newton_solver == "solve":
                    ## We replace obtained error P by previous steps P
                    ## and change the solver to more stalbe "pinv"
                    P = P_prev
                    Newton_solver = "pinv"
                    ## Conduct e-step again
                    Q = estep(T, R, P, safe_estep, delta)
                    print("Retry e-step with previous P")
                    print(f"\033[93mChange solver to {Newton_solver}\033[0m")
                else:
                    print("Please use large epsilon, and/or change Newton solover")
                    return P_prev, history
            else:
                P_prev = P
        
        ## For debug
        ## Confirm if the E-step make it tightness D_f[T, P] = D_f[Q, R]
        if confirm_tightness:
            df_TP = divergence.alpha_divergence(T, P, q, avoid_nan=True)
            df_QR = divergence.alpha_divergence(Q, R, q, avoid_nan=True)
            if abs(df_TP - df_QR) > 1.0e-3:
                print(f"No tightness df_TP:{df_TP:10.5f} df_QR:{df_QR:10.5f}")
        
        ## M-step: projection onto model manifold
        ## Optimal update to minimize D_f[R||Q] for Q
        ## This is equivalent to
        ## Optimal update to minimize D_χ[Q||R] for Q 
        if method == "LBFGS":
            R, theta, eta, _ = MBA_LBFGS(Q, intract_CP, max_iter=max_iter_inner,
                             init=init_inner, get_history=False, 
                             get_cost_dual=get_cost_dual, m_step=True,
                             verbose_interval=verbose_interval_inner,
                             norm_method=norm_method, norm_standard=norm_standard,
                             theta_0 = theta_prev, maxls=maxls, tol=tol_inner, gtol=gtol_inner,
                             chi=chi, q=q, verbose=verbose_inner)
        
        else:
            if method == "Newton":
                Newton = True
            else:
                Newton = False
            R, theta, eta, _ = MBA(Q, intract_CP, max_iter=max_iter_inner,
                             init=init_inner, get_history=False, 
                             rel_epsilon=rel_epsilon, rcond=rcond, 
                             lr=lr, beta=beta, epsilon=epsilon, Newton_solver=Newton_solver,
                             verbose_interval=verbose_interval_inner,
                             epsilon_auto=epsilon_auto,
                             norm_method=norm_method, norm_standard=norm_standard,
                             theta_0 = theta_prev, lr_search=lr_search, maxls=maxls,
                             get_cost_dual=get_cost_dual, m_step=True,
                             chi=chi, q=q, Newton=Newton, verbose=verbose_inner)
        
        ## Marginalization along hidden variable k
        P = np.sum(R, axis=-1)

        ## For debug
        ## Confirm if the M-step makes the objective smaller 
        ## i.e., D_f[Q, R] < D_f[Q, Rprev]
        if confirm_optimal_mstep:
            df_QR = divergence.alpha_divergence(Q, R, q, avoid_nan=True)
            df_QRprev = divergence.alpha_divergence(Q, R_prev, q, avoid_nan=True)
            if df_QR > df_QRprev:
                print(f"    df_QR > df_QRprev")
                print(f"     ├─── df_QR {df_QR:10.5f} df_QRprev {df_QRprev:10.5f}")
                print(f"     └─── Please set `max_iter_inner` larger or `use_prev_theta` True")

        ## For debug
        ## Confirm the ineqality D_f[T, P] < D_f[Q, R]
        if confirm_inequality:
            df_TP = divergence.alpha_divergence(T, P, q, avoid_nan=True)
            df_QR = divergence.alpha_divergence(Q, R, q, avoid_nan=True)
            if df_TP > df_QR:
                print(f"No monotonicity: df_TP {df_TP:10.5f} df_QR {df_QR:10.5f}")

        # The original objective is Tsallis divergence from P(model) to T(data)
        # This optimization is equilavent with optimize 
        # the alpha-divergence from T(data) to P(model).
        cost = divergence.alpha_divergence(T, P, q, avoid_nan=True)
        # the avobe cost can be replace as
        # cost = divergence.Tsallis_divergence(P, T, q=q, avoid_nan=True)
        cost_dual = divergence.alpha_divergence(P, T, q, avoid_nan=True)
        # the avobe dual cost can be replace as
        # cost_dual = divergence.Tsallis_divergence(T, P, q=q, avoid_nan=True)
        elapsed_time = time.time() - start_time
        if verbose and (n_iter % verbose_interval_outer == 0):
            show_verbose(cost, cost_prev, n_iter, elapsed_time, 
                     cost_dual=cost_dual, cost_dual_prev=cost_dual_prev,
                     em_step=True)

        if get_history:
            history["iter"].append( n_iter + 1 )
            history["time"].append( elapsed_time )
            history["loss"].append( float(cost) )
        
        if mproject.check_convergence(cost_prev, cost, tol_outer):
            converge = True
            break
        else:
            cost_prev = cost
            cost_dual_prev = cost_dual
            n_iter += 1
            R_prev = R
            if use_prev_theta:
                theta_prev = theta

    show_terminate(converge, n_iter, elapsed_time)

    if need_factor:
        return P, R, history
    else:
        return P, history

def estep(T, R, P, safe_estep, delta):
    if safe_estep:
        ## To avoid 0-division, we add small value `delta`
        P_safe = np.where(P == 0, delta, P)
        P_safe /= np.sum(P_safe)
        Q = R * (T[..., np.newaxis] / P_safe[..., np.newaxis])

        # In this update, Q can be 0.
        # Thus, we enforce positivity then, normalize
        # otherwise, we receive error in M-step
        # (MBA can hundle only positive input)
        Q = np.where(Q <= 0, delta*0.001, Q)
        Q_sum = np.sum(Q)
        Q /= Q_sum
    else:
        Q = R * (T[..., np.newaxis] / P[..., np.newaxis])

    return Q

def initial_R(tensor_shape, init):
    if init == "random":
        R = np.random.rand( *tensor_shape )
        R = R / np.sum(R)
        
    elif init == "uniform":
        print(tensor_shape)
        R = np.ones( tensor_shape )
        R = R / np.sum(R)
        
    elif init == "normal":
        R = np.abs(np.random.randn( *tensor_shape ))
        R = R / np.sum(R)

    else:
        raise 

    return R