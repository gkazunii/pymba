import numpy as np
import itertools

def get_learn_indices(shape, intract):
    """
    Generate indices corresponding to the activated interactions (learnable positions)
    based on the interaction specification.

    Parameters
    ----------
    shape : tuple of int
        Shape of the D-dimensional tensor. Each entry corresponds to the size along that axis.
    intract : list of lists of int
        A D-element list where each element is a list of binary flags (0 or 1) indicating 
        which combinations of axes of that interaction order are active. For example,
        intract[1] controls all 2-body interactions, with length comb(D, 2).

    Returns
    -------
    learn_indices : np.ndarray of shape (N, D)
        Array of indices to be learned. Each row is a D-dimensional index vector.
        All returned indices have 0 in the inactive dimensions and values from [1, shape[i]-1]
        in the active dimensions. If no interactions are active, returns an empty array.

    Notes
    -----
    The index [0, ..., 0] is not included in the output.
    """
    D = len(shape)
    learn_indices = []
    
    for d, d_intract in enumerate(intract, start=1):
        if not any(d_intract):
            continue

        combs = list(itertools.combinations(range(D), d))
        for active, comb in zip(d_intract, combs):
            if not active:
                continue

            ranges = [range(1, shape[i]) if i in comb else [0] for i in range(D)]

            for index in itertools.product(*ranges):
                learn_indices.append(index)

    if learn_indices:
        return np.array(learn_indices, dtype=int)
    else:
        return np.empty((0, D), dtype=int)

def get_non_learn_indices(shape, learn_indices):
    """
    Compute the complement of the learnable index set — i.e., the indices that 
    are not targeted for learning, excluding the trivial [0, ..., 0] index.

    Parameters
    ----------
    shape : tuple of int
        Shape of the D-dimensional tensor.
    learn_indices : np.ndarray of shape (N, D)
        Array of learnable indices previously computed, to be excluded from the result.

    Returns
    -------
    non_learn_indices : np.ndarray of shape (M, D)
        Indices not included in learn_indices and not equal to [0, ..., 0].
    """
    D = len(shape)

    learn_set = set(map(tuple, learn_indices))

    # Generator expression to avoid memory bloat
    filtered = (
        idx for idx in itertools.product(*[range(s) for s in shape])
        if any(idx) and idx not in learn_set
    )

    # Use list comprehension for fast conversion
    return np.fromiter((i for t in filtered for i in t), dtype=int).reshape(-1, D)
