import numpy as np
import manage_intract
from divergence import chi_divergence

def show_verbose_no_color(P, Q, n_iter, sec, chi, **kwargs):
    if n_iter == 0:
        print(f"{'Iter':>6} | {'Time':>8} | {'chi-div(P||Q)':>15} | {'chi-div(Q||P)':>15}")
        print("-" * 56)
    costPQ = chi_divergence(P,Q,avoid_nan=True, chi=chi, **kwargs)
    costQP = chi_divergence(Q,P,avoid_nan=True, chi=chi, **kwargs)
    print(f"{n_iter:6d} | {sec:8.3f} | {costPQ:15.6e} | {costQP:15.6e}")

def show_terminate(converge, n_iter, elapsed_time):
    print("-" * 56)
    if converge:
        print(f"Converged after {n_iter:6d} iterations and {elapsed_time:9.5f} sec.")
    else:
        print(f"Not converged after {n_iter:6d} iterations and {elapsed_time:9.5f} sec.")
    print("-" * 56)


def show_verbose(cost, cost_prev, n_iter, elapsed_time, final_step=False,
                 cost_dual=None, cost_dual_prev=None, 
                 cost_alpha=None,
                 m_step=False, em_step=False):
    #Tsa_divPQ = Tsallis_divergence(P, Q, **kwargs)
    #Tsa_divQP = Tsallis_divergence(Q, P, **kwargs)
    if n_iter == 0 and (not(m_step)) and (not(em_step)):
        if cost_alpha is None:
            print(f"{'Iter':>6} | {'Time':>7} | {'chi-div(P||Q)':>15} | {'chi-div(Q||P)':>15}")
        else:
            print(f"{'Iter':>6} | {'Time':>7} | {'chi-div(P||Q)':>15} | {'chi-div(Q||P)':>15} | {'alpha-div(P||Q)':>15}")
        print("-" * 56)

    # ANSI escape codes for red text
    RED = "\033[31m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    BOLD_UNDERLINE = "\033[1;4m"

    if cost > cost_prev:
        cost_str = f"{RED}{cost:15.6e}{RESET}"
    else:
        cost_str = f"{cost:15.6e}"

    if cost_dual is not None and cost_dual_prev is not None:
        if cost_dual > cost_dual_prev:
            cost_dual_str = f"{RED}{cost_dual:15.6e}{RESET}"
        else:
            cost_dual_str = f"{cost_dual:15.6e}"
    elif cost_dual is not None:
        cost_dual_str = f"{cost_dual:15.6e}"
    else:
        cost_dual_str = " " * 15

    if m_step:
        if cost_alpha is None:
            if not(final_step):
                print(f"  ├─ [M-step {n_iter:5d} ] | {elapsed_time:6.3f}s | {cost_dual_str} | {cost_str}")
            else:
                print(f"  └─ [M-step {n_iter:5d} ] | {elapsed_time:6.3f}s | {cost_dual_str} | {cost_str}")
        else:
            if not(final_step):
                print(f"  ├─ [M-step {n_iter:5d} ] | {elapsed_time:6.3f}s | {cost_dual_str} | {cost_str} | {cost_alpha:15.6e}")
            else:
                print(f"  └─ [M-step {n_iter:5d} ] | {elapsed_time:6.3f}s | {cost_dual_str} | {cost_str} | {cost_alpha:15.6e}")
                
    elif em_step == False:
        if cost_alpha is None:
            print(f"{n_iter:6d} | {elapsed_time:6.3f}s | {cost_dual_str} | {cost_str}")
        else:
            print(f"{n_iter:6d} | {elapsed_time:6.3f}s | {cost_dual_str} | {cost_str} | {cost_alpha:15.6e}")
        
    if em_step:
        print(f"[Iter {n_iter:3d}] Time: {elapsed_time:6.3f}s    Dual Cost:{cost_dual_str}    Prime Cost: {BOLD}{cost_str}{RESET}")

def print_header(chi):
    prefix = "├─ [M-step ]   "
    indent = " " * len(prefix)
    if chi != "Tsallis":
        header = f"{indent}{'Iter':>5} |  {'Time':>6} | {'chi-div(P||Q)':>15} | {'chi-div(Q||P)':>15} |"
    else:
        header = f"{indent}{'Iter':>5} |  {'Time':>6} | {'chi-div(P||Q)':>15} | {'chi-div(Q||P)':>15} | {'alpha-div(P||Q)':>15}"
    print(header)

def print_initial_info_em(rank, q, init, init_inner, use_prev_theta, safe_estep):
    LINE = "=" * 56
    print(LINE)
    print("EM-deformed low-rank approximation")
    print(f"{'  Deformed rank':<25}: {rank}")
    print("Initializations")
    
    print(f"{'  EM-procedure':<25}: {init}")
    if use_prev_theta:
        init_mstep = "Previous θ"
        print(f"{'  M-step':<25}: {init_mstep}")
    else:
        init_mstep = init_inner
        if init_inner == "KLMBA":
            print(f"{'  M-step':<25}: KL-MBA (LBFGS)")
        else:
            print(f"{'  M-step':<25}: {init_inner}")
    print(f"{'Safe E-step':<25}: {safe_estep}")

def print_initial_info_LBTC(init, init_inner, use_prev_theta, mask_nan):
    LINE = "=" * 56
    print(LINE)
    print("EM-based low-body tensor completion")
    print(f"{'  # Missing values':<25}: {np.sum(mask_nan)}")
    print("Initializations")
    
    if use_prev_theta:
        init_mstep = "Previous θ"
    else:
        init_mstep = init_inner
    print(f"{'  Missing values':<25}: {init}")
    print(f"{'  M-step':<25}: {init_mstep}")

def print_initial_info(init, method, Newton_solver, lr, beta, epsilon, epsilon_auto, tensor_shape, n_params, intract, tunelr, maxls, norm_method, norm_standard, chi, **kwargs):
    LINE = "=" * 56
    print(LINE)
    print("Algorithm settings:")
    
    if method == True:
        method = "Newton's method"
    elif method == False:
        method = "Gradient descent"

    if epsilon_auto == True:
        epsilon = "Auto (damping)"
    
    if tunelr == True:
        beta = "Auto (line search)"
        
    print(f"{'  Optimization method':<25}: {method}")
    if method == "Newton's method":
        print(f"{'  β (scaling factor)':<25}: {beta}")
        print(f"{'  ε (stabilizer)':<25}: {epsilon}")
        print(f"{'  solver':<25}: {Newton_solver}")
    if method == "Gradient descent":
        print(f"{'  lr (learning rate)':<25}: {lr}")
    print(f"{'  Cost function':<25}: {chi} divergence")
    if chi == "Kani" or chi == "Kaniadakis":
        print(f"{'    k (Kani param)':<25}: {kwargs.get('k', 'N/A')}")
    if chi == "Tsallis":
        print(f"{'    q (Tsallis param)':<25}: {kwargs.get('q', 'N/A')}")
    print(f"{'  Line search':<25}: {tunelr}")
    if tunelr:
        print(f"{'  Max line search':<25}: {maxls}")
    if init == "uniform":
        print(f"{'  Initialization':<25}: 0-body approx.")
    elif init == "KLMBA":
        print(f"{'  Initialization':<25}: KL-MBA (LBFGS)")
    else:
        print(f"{'  Initialization':<25}: {init}")
    if norm_standard == False:
        print(f"{'  Normalization':<25}: Deformed")
        print(f"{'    Numerical method':<25}: {norm_method}")
    elif norm_standard == True:
        print(f"{'  Normalization':<25}: Classic")

    print(LINE)
    size_of_input = np.prod(tensor_shape)
    print("Input Tensor Info:")
    print(f"{'  Shape of input tensor':<25}: {tensor_shape}")
    print(f"{'  Size of input tensor':<25}: {size_of_input}")
    print(f"{'  Number of params':<25}: {n_params}")
    print(f"{'  Size of FIM':<25}: {n_params} x {n_params} = {n_params*n_params}")
    print(LINE)

    activated_interacts = manage_intract.get_list_of_activated_intracts(intract)
    print("Active Interactions:")
    for d, active in enumerate(activated_interacts, start=1):
        print(f"{d}-body".rjust(7) + ": " + str(active))
    print(LINE)