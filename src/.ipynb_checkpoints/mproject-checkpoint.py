import numpy as np
import time
import itertools
import importlib
from scipy.optimize import minimize
from scipy.optimize import line_search
from scipy.optimize._linesearch import scalar_search_armijo

import mask
import manage_intract
import hessian
import closed_forms
from transform import prob_from_theta
from transform import theta_from_prob
from transform import eta_from_prob
from verbose import print_initial_info
from verbose import show_verbose
from verbose import show_terminate
from divergence import chi_divergence
from divergence import alpha_divergence
from chi_normalize import chi_normalized_prob_from_theta

def MBA(P, intract, max_iter=10, lr=0.1, seed=None,
        norm_method="toms748", norm_standard=False,
        init="uniform", theta_0=None,
        get_history=True, get_cost_alpha=True, get_cost_dual=True,
        verbose=True, verbose_interval=10, 
        lr_search=True, maxls=10, beta=1.0,
        Newton=False,
        Newton_solver="solve", epsilon=1e-8, epsilon_auto=False, rel_epsilon=1e-6, rcond=1.0e-8, 
        tol=1.0e-9, m_step=False,
        chi=1, **kwargs):

    chi, kwargs = check_input_setting(chi, kwargs)
    
    history = {"iter":[], "time":[], "loss":[], "alpha_div":[]}
    
    tensor_shape = np.shape(P)

    """
    If the input 'intract' is an integer, 
    then we do m(=intract)-body approximation.
    """
    if isinstance(intract, int):
        intract = manage_intract.get_m_body_intract(intract, len(tensor_shape))

    msk = mask.get_learn_indices(tensor_shape, intract)
    
    n_params = len(msk)
    if verbose and not(m_step):
        print_initial_info(init, Newton, Newton_solver, lr, beta, epsilon, epsilon_auto, tensor_shape, n_params, intract, 
                           lr_search, maxls, norm_method, norm_standard, chi, **kwargs)
    
    theta_B = np.zeros(n_params)
    eta_B = np.zeros(n_params)
    eta_hat_B = np.zeros(n_params)
    
    ## Random initialization
    ## Randomly select a point in the model space
    Q, theta, eta = init_values(P, intract, msk, theta_0=theta_0, seed=seed, init=init, chi=chi, **kwargs)

    # Vectorize
    theta_B = theta[tuple(msk.T)]
    eta_B = eta[tuple(msk.T)]

    eta_hat = eta_from_prob(P, chi=chi, **kwargs)
    eta_hat_B = eta_hat[tuple(msk.T)] # Fixed in learning

    n_iter = 0
    cost_prev = +np.inf
    cost_dual_prev = np.inf
    cost_alpha = None

    start_time = time.time()
    elapsed_time = 0.0
    converge = False
    while(n_iter < max_iter+1):
        if Newton:
            # Get and Scale Hessian matrix 
            #beta = (n_iter+1)
            #G = beta * hessian.chi_FIM(P, msk, eta, chi=chi, **kwargs)
            G = beta * hessian.chi_FIM(Q, msk, eta, chi=chi, **kwargs)

            if epsilon_auto:
                epsilon = update_epsilon(G, rel_epsilon)
            
            # Add values on diagonal elements for stablization
            G += epsilon * np.eye(G.shape[0])
            # If you wanna see eigenvalues...
            # print(np.linalg.eig(G)[0])

            # gradient
            grad = (eta_B - eta_hat_B) 

            # Newton direction
            if Newton_solver == "solve":
                ## Faster solver "solve"
                ## It may return NaN or a huge value without providing error
                try:
                    p = -np.linalg.solve(G, grad)
                except np.linalg.LinAlgError:
                    print("Get singular error. ")
                    print("The solver has been changed to pinv")
                    print("It may slow down")
                    Newton_solver = "pinv"
                    p = -np.linalg.pinv(G, rcond=rcond) @ grad
            elif Newton_solver == "pinv":
                ## Safer but slower solver "pinv"
                p = -np.linalg.pinv(G, rcond=rcond) @ grad
            else:
                raise ValueError(f"Unknown Newton solver: '{Newton_solver}'. Please use 'solve' or 'pinv'.")


            tuned_lr = beta
            if lr_search:
                def func_to_minimize(theta_vec):
                    theta_full = theta.copy()
                    theta_full[tuple(msk.T)] = theta_vec
                    Q, normalize_time = chi_normalized_prob_from_theta(theta_full, method=norm_method, standard=norm_standard, chi=chi, **kwargs)
                    return chi_divergence(Q, P, avoid_nan=True, chi=chi, **kwargs)
    
                def grad_func(theta_vec):
                    theta_full = theta.copy()
                    theta_full[tuple(msk.T)] = theta_vec
                    Q, normalize_time = chi_normalized_prob_from_theta(theta_full, method=norm_method, standard=norm_standard, chi=chi, **kwargs)
                    eta_val = eta_from_prob(Q, chi=chi, **kwargs)
                    eta_B_val = eta_val[tuple(msk.T)]
                    return eta_B_val - eta_hat_B
    
                tuned_lr, _, _, _, _, _ = line_search(func_to_minimize, grad_func, theta_B, p, maxiter=maxls)
    
                # if line_search is failed (lr = None), then use dafult value
                if tuned_lr is None:
                    tuned_lr = beta
                    
            # update by Newton method
            theta_B = theta_B + tuned_lr * p

        else:
            # update by gradient descent
            theta_B -= lr * (eta_B - eta_hat_B)

        # recover full theta from obtained theta
        theta[tuple(msk.T)] = theta_B 
        #print(theta)

        # update low-body distribution Q by theta
        # since theta[0,...,0] is empty, 
        # we need to normalize Q after the update
        Q, normalize_time = chi_normalized_prob_from_theta(theta, method=norm_method, standard=norm_standard, chi=chi, **kwargs)
        #Q = prob_from_theta(theta, chi=chi, **kwargs)
        #Q /= np.sum(Q)

        # update eta value by obtained low-body Q
        eta = eta_from_prob(Q, chi=chi, **kwargs)
        eta_B = eta[tuple(msk.T)]

        cost = chi_divergence(Q, P, avoid_nan=True, chi=chi, **kwargs) ## it is cost function
        if get_cost_dual:
            cost_dual = chi_divergence(P, Q, avoid_nan=True, chi=chi, **kwargs)
        if get_cost_alpha and chi == "Tsallis":
            cost_alpha = alpha_divergence(P, Q, kwargs["q"], avoid_nan=True)
            
        elapsed_time = time.time() - start_time
        if verbose and (n_iter % verbose_interval == 0):
            show_verbose( float(cost), float(cost_prev), n_iter, elapsed_time, 
                              cost_dual=cost_dual, cost_dual_prev=cost_dual_prev, 
                          cost_alpha = cost_alpha, m_step=m_step)

        if get_history:
            history["iter"].append( n_iter + 1 )
            history["time"].append( elapsed_time )
            history["loss"].append( float(cost) )
            if chi == "Tsallis":
                q = kwargs["q"]
                alpha_div = alpha_divergence(P, Q, q, avoid_nan=True)
                history["alpha_div"].append( float(alpha_div) )
            
        if check_convergence(cost_prev, cost, tol):
            if verbose and (n_iter % verbose_interval != 0):
                show_verbose( float(cost), float(cost_prev), n_iter, elapsed_time, final_step=True, cost_alpha=cost_alpha,
                              cost_dual=cost_dual, cost_dual_prev=cost_dual_prev, m_step=m_step)
            converge = True
            break
        else:
            cost_prev = cost
            cost_dual_prev = cost_dual
            n_iter += 1

    if verbose and not(m_step):
        show_terminate(converge, n_iter, elapsed_time)

    return Q, theta, eta, history

def check_input_setting(chi, kwargs):
    if chi == "Kani" and kwargs["k"] == 0:
        chi = 1
        kwargs = {"q":1.0}
        print(f"For Kaniadakis-divergence, the parameter k=0 is invalied.")
        print(f"Thus, it optimizes the KL-divergence")
    return chi, kwargs

    
def MBA_LBFGS(P, intract, max_iter=100, 
             seed=None, init="uniform", norm_method="toms748", norm_standard=False, 
             verbose=False, verbose_interval=10, 
             tol=1.0e-7, gtol=1.0e-12, get_history=False, get_cost_alpha=True,
             get_cost_dual=True, m_step=False, maxls=200, 
             theta_0=None, chi=1, **kwargs):

    chi, kwargs = check_input_setting(chi, kwargs)
    history = {"iter":[], "time":[], "loss":[], "alpha_div":[]}
    
    tensor_shape = np.shape(P)
    """
    If the input 'intract' is an integer, 
    then we do m(=intract)-body approximation.
    """
    if isinstance(intract, int):
        intract = manage_intract.get_m_body_intract(intract, len(tensor_shape))
        
    msk = mask.get_learn_indices(tensor_shape, intract)
    
    n_params = len(msk)
    #if verbose and not(m_step):
    #    print_initial_info(init, "LBFGS", 0.0, 0.0, 0.0, tensor_shape, n_params, intract, True, maxls, chi, **kwargs)

    if verbose and not(m_step):
        print_initial_info(init, "LBFGS", False, 0.0, 0.0, 0.0, False, tensor_shape, n_params, intract, True, maxls, norm_method, norm_standard, chi, **kwargs)
    
    
    theta_B = np.zeros(n_params)
    eta_B = np.zeros(n_params)
    eta_hat_B = np.zeros(n_params)
    
    ## Random initialization
    ## Randomly select a point in the model space
    Q, theta, eta = init_values(P, intract, msk, theta_0=theta_0, seed=seed, init=init, chi=chi, **kwargs)

    # Vectorize
    theta_B = theta[tuple(msk.T)]
    eta_B = eta[tuple(msk.T)]

    eta_hat = eta_from_prob(P, chi=chi, **kwargs)
    eta_hat_B = eta_hat[tuple(msk.T)] # Fixed in learning

    cost = 0.0
    cost_prev = np.inf
    cost_dual = 0.0
    cost_dual_prev = +np.inf
    
    n_iter = 0
    start_time = time.time()
    
    cost_alpha = None
    def obj_and_grad(theta_B):
        # Update low-body theta
        theta[tuple(msk.T)] = theta_B

        # update low-body tensor Q by theta
        Q, normalize_time = chi_normalized_prob_from_theta(theta, method=norm_method, 
                                                           standard=norm_standard, chi=chi, **kwargs)
        #Q = prob_from_theta(theta, chi=chi, **kwargs)
        #Q /= np.sum(Q)

        # update eta value by obtained low-body Q
        eta = eta_from_prob(Q, chi=chi, **kwargs)
        eta_B = eta[tuple(msk.T)]

        nonlocal cost
        nonlocal cost_dual
        nonlocal cost_alpha
        cost = chi_divergence(Q, P, avoid_nan=True, chi=chi, **kwargs)
        if get_cost_dual:
            cost_dual = chi_divergence(P, Q, avoid_nan=True, chi=chi, **kwargs)
        if get_cost_alpha and chi == "Tsallis":
            q = kwargs["q"]
            cost_alpha = alpha_divergence(P, Q, q, avoid_nan=True)
        
        return cost, eta_B - eta_hat_B

    def callback(theta_B):
        nonlocal n_iter
        nonlocal cost
        nonlocal cost_prev
        nonlocal cost_alpha
        nonlocal cost_dual
        nonlocal cost_dual_prev
        
        elapsed_time = time.time() - start_time
        if verbose and (n_iter % verbose_interval == 0):
            theta[tuple(msk.T)] = theta_B

            Q, _ = chi_normalized_prob_from_theta(theta, method=norm_method, 
                                                standard=norm_standard, chi=chi, **kwargs)
            show_verbose( float(cost), float(cost_prev), n_iter, elapsed_time,
                          cost_dual=cost_dual, cost_dual_prev=cost_dual_prev,
                          m_step=m_step, cost_alpha=cost_alpha
                        )

        if get_history:
            history["iter"].append( n_iter + 1 )
            history["time"].append( elapsed_time )
            history["loss"].append( float(cost) )
            if chi == "Tsallis" and get_cost_alpha:
                #q = kwargs["q"]
                #Q = prob_from_theta(theta, chi=chi, **kwargs)
                #Q /= np.sum(Q)
                #alpha_div = alpha_divergence(P, Q, q, avoid_nan=True)
                #history["alpha_div"].append( float(cost_alpha) )
                history["alpha_div"].append( float(cost_alpha) )

        cost_prev = cost
        cost_dual_prev = cost_dual
        n_iter +=1

    results = minimize(obj_and_grad, theta_B, jac=True, method="L-BFGS-B", callback=callback, 
                       options={"ftol":tol, "maxiter":max_iter, "maxls":maxls, "gtol":gtol, "maxfun":1e16})
    print(np.linalg.norm(results.jac))

    elapsed_time = time.time() - start_time
    if verbose and (n_iter % verbose_interval != 0):
        show_verbose( float(cost), float(cost_prev), n_iter, elapsed_time, final_step=True, cost_alpha=cost_alpha,
                              cost_dual=cost_dual, cost_dual_prev=cost_dual_prev, m_step=m_step)

    if verbose and not(m_step):
        show_terminate(results.success, n_iter, elapsed_time)
        print("Scipy minimizer Success?", results.success)
        print("Scipy minimizer message:", results.message)
    
    theta_B = results.x
    theta[tuple(msk.T)] = theta_B
    Q, _ = chi_normalized_prob_from_theta(theta, method=norm_method, 
                                        standard=norm_standard, chi=chi, **kwargs)
    #Q = prob_from_theta(theta, chi=chi, **kwargs)
    #Q /= np.sum(Q)
    eta = eta_from_prob(Q, chi=chi, **kwargs)
    
    return Q, theta, eta, history

def check_convergence(cost_prev, cost, tol):
    if abs(cost_prev - cost) < tol:
        return True
    else:
        return False

def init_values(P, intract, msk, init="random", seed=None, chi=1, theta_0=None, **kwargs):
    # shape of the tensor
    shape = np.shape(P)
    # input tensor dim
    dim = len(shape)
    
    init_theta = np.zeros(shape, dtype=np.float64)
    n_params = len(msk)
    if seed is not None:
        np.random.seed(seed)

    if theta_0 is None:
        if init == "random":
            # Random initialization of low-body theta
            theta_values_B = np.random.normal(0, 0.01, size=n_params)
            init_theta[tuple(msk.T)] = theta_values_B
            
        elif init == "uniform":
            # Initialization by uniform tensor
            theta_values_B = np.zeros(n_params)
            init_theta[tuple(msk.T)] = theta_values_B
            
        elif init == "KLMBA":
            ## This is the best-CP-rank approximation optimizing the KL divergence.
            ## Thus, chi needs to be 1.
            if intract == manage_intract.get_intract_CP( dim ):
                print("KLMBA is running for initialization with closed-form")
                init_prob = closed_forms.best_lowbody_tensor_for_CP_intract(P, safe=True)
                init_theta = theta_from_prob(init_prob, chi=1)
                init_eta = eta_from_prob(init_prob, chi=1)
                init_theta.flat[0] = 0
                
            else:
                print("KLMBA is running for initialization with LBFGS")
                init_prob, init_theta, init_eta, _ = MBA_LBFGS(P, intract, max_iter=100, 
                 seed=None, init="uniform", lr_search=False,
                 verbose=False,  
                 tol=1.0e-12, gtol=1.0e-12, get_history=False, get_cost_alpha=False,
                 get_cost_dual=False, m_step=False, maxls=200, 
                 theta_0=None, chi=1)
                
            print("KLMBA has been finished")
            return init_prob, init_theta, init_eta
            
        else:
            raise NameError(f"Initalize method {init} is not defined") 
        
    else:
        init_theta = theta_0
        
    init_prob, _ = chi_normalized_prob_from_theta(init_theta, standard=False, chi=chi, **kwargs)
    init_eta = eta_from_prob(init_prob, chi=chi, **kwargs)

    return init_prob, init_theta, init_eta

def estimate_largest_eigenvalue(G, power_iters):
    n = G.shape[0]
    v = np.random.randn(n)
    v /= np.linalg.norm(v)
    for _ in range(power_iters):
        w = G @ v
        v = w / np.linalg.norm(w)
    return float(v @ (G @ v))

def update_epsilon(G, rel_epsilon=1e-6, power_iters=10):
    ## Estimate largest eigenvalue
    s_max = estimate_largest_eigenvalue(G, power_iters)
    epsilon = max(1e-12, rel_epsilon * s_max)
    return epsilon
