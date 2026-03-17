import numpy as np

def generate_low_rank_tensor(shape, rank, seed=None, normalize=False):
    """
    Generate a low-rank tensor of specified order and rank using CP decomposition.

    Parameters:
        shape (list of int): The shape of the tensor (e.g., [5, 6, 7]).
        rank (int): The CP rank of the tensor.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        numpy.ndarray: A float64 tensor with the specified shape and rank.
    """
    if seed is not None:
        np.random.seed(seed)

    D = len(shape)  # Order of the tensor
    # Generate factor matrices for each mode
    factors = [np.random.rand(n, rank).astype(np.float64) for n in shape]

    # Initialize the tensor
    tensor = np.zeros(shape, dtype=np.float64)

    # Construct the tensor as a sum of rank-1 outer products
    for r in range(rank):
        outer_prod = factors[0][:, r]
        for d in range(1, D):
            outer_prod = np.multiply.outer(outer_prod, factors[d][:, r])
        tensor += outer_prod

    if normalize:
        tensor = tensor / np.sum(tensor)

    return tensor