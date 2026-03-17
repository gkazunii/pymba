import numpy as np
def is_valid_probability_tensor(tensor, tol=1e-5):
    """
    Check if a multi-dimensional array is a valid probability tensor:
    - All elements are non-negative
    - The sum of all elements is (approximately) 1

    Parameters:
    - tensor: np.ndarray
    - tol: Tolerance for numerical error in the sum (default: 1e-8)

    Returns:
    - bool: True if tensor is valid, False otherwise
    """
    if not isinstance(tensor, np.ndarray):
        raise ValueError("Input must be a NumPy array")

    non_negative = np.all(tensor >= 0)
    total_sum = np.sum(tensor)
    sum_is_one = np.abs(total_sum - 1.0) < tol

    if not(sum_is_one):
        print(f"sum is... {total_sum}")

    return non_negative and sum_is_one