import numpy as np
import itertools
import math

def get_m_body_intract(m, D):
    intract = [ [] for d in range(D)]
    for d in range(1,D+1):
        if m >= d:
            intract[d-1] = [1 for dck in range(math.comb(D,d))]
        else:
            intract[d-1] = [0 for dck in range(math.comb(D,d))]

    return intract

def check_intract(intract, D):
    if len(intract) != D:
        raise IndexError(f"Order of interaction should be {D}") 
        
    for d in range(D):
        if len(intract[d]) != math.comb(D,d+1):
            raise IndexError(f"{d} body interaction needs {math.comb(D,d)} elements")

    # Check Trivial Case
    total_num_intract = np.sum(np.concatenate(intract))
    if total_num_intract == 2**D-1:
        print("All interactions are activated.")
    if total_num_intract == 0:
        print("No interactions is activated.")

def get_list_of_activated_intracts(intract):
    D = len(intract)
    intract_for_display = [None] * D  # D個のリストを用意

    for d in range(1, D + 1):
        possible_d_body_combs = list(itertools.combinations(range(D), d))  # 0-indexed
        D_C_d = math.comb(D, d)
        active_indices = [i for i, val in enumerate(intract[d - 1]) if val == 1]

        if not active_indices:
            intract_for_display[d - 1] = []
            continue

        tmp_vec = [list(possible_d_body_combs[i]) for i in active_indices]
        intract_for_display[d - 1] = tmp_vec

    """
    # if you want show the interacts, run the following code
    >>> result = get_list_of_activated_intracts(intract)
    >>> for d, active in enumerate(result, start=1):
    >>>    print(f"{d}-body: {active}")
    """
    return intract_for_display

import math
import itertools

def get_intract_CP(D):
    """
    Generate an interaction specification (intract) that includes:
    - All 1-body interactions
    - Only 2-body interactions between the last axis (D-1) and all others
    - No higher-order interactions

    Parameters
    ----------
    D : int
        Number of dimensions (tensor order)

    Returns
    -------
    intract : list of list of int
        Interaction specification as a list of binary vectors for 1-body to D-body interactions.
        Each entry in the outer list corresponds to d-body interactions (d = 1 to D).
    """
    intract = []

    # 1-body: all active
    intract.append([1] * D)

    # 2-body: activate only those involving the last axis (D-1)
    combs_2 = list(itertools.combinations(range(D), 2))
    intract_2 = [1 if (D - 1) in comb else 0 for comb in combs_2]
    intract.append(intract_2)

    # Higher-order: all inactive
    for d in range(3, D + 1):
        intract.append([0] * math.comb(D, d))

    return intract