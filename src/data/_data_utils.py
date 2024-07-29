import numpy as np

def read_txt(path):
    with open(path, 'r', encoding="utf-8") as fp:
        data = fp.read().strip().splitlines()
    return data

def bounded_random(lower_bound, upper_bound, size=None):
    """
    Generate random numbers within a specified range [lower_bound, upper_bound).

    Parameters:
        lower_bound (float): The lower bound of the range.
        upper_bound (float): The upper bound of the range.
        size (int or tuple of ints, optional): Output shape. If the given shape is, 
                                               e.g., (m, n, k), then m * n * k samples 
                                               are drawn. Default is None, in which case
                                               a single value is returned.

    Returns:
        ndarray or float: Random values within the specified range.
    """
    if lower_bound == upper_bound:
        return lower_bound
    random_values = np.random.rand(*size) if size else np.random.rand()
    scaled_values = lower_bound + (upper_bound - lower_bound) * random_values
    return scaled_values