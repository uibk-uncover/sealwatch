import numpy as np
from scipy.stats import norm


def round(x):
    """ Helper method to mimic Matlab rounding

    |x| >= 0.5 => round away from zero

    :param x:
    :type x:
    :return: rounded x
    :rtype:

    :Example:

    >>> # TODO
    """
    # Preserve the input sign
    x_sign = np.sign(x)

    # Convert to absolute value
    x_abs = np.abs(x)

    # Round towards zero
    x_abs_floor = np.floor(x_abs)

    # Round away from zero
    x_abs_ceil = np.ceil(x_abs)

    # Compute difference between value and floored value
    abs_floor_diff = x_abs - x_abs_floor

    # Condition for rounding away from zero
    mask_ceil = (abs_floor_diff >= 0.5) | np.isclose(abs_floor_diff, 0.5)

    # Ceil or floor
    x_rounded = np.where(mask_ceil, x_abs_ceil, x_abs_floor)

    # Restore sign
    x_rounded *= x_sign

    return x_rounded


def randperm_naive(rng, n):
    """
    Generate random permutation using a simple implementation

    Note that np.random.RandomState(seed).rand() yields the same random values as Matlab's Mersenne twister PRNG.
    Hence, building a random permutation on top of the rand() method enables matching implementation in numpy and Matlab.

    :param rng: random number generator
    :param n: size of the permutation
    :return: random permutation of length n
    """
    random_numbers = rng.rand(n)
    return np.argsort(random_numbers)


def randi(low, high, size, rng):
    """
    Equivalent to Matlab's randi method
    :param low: minimum integer value to generate
    :param high: one above the maximum integer value to generate
    :param size: size of the array to generate
    :param rng: random number generator
    :return:
    """
    numbers = rng.rand(size) * (high - low) + low
    return np.floor(numbers).astype(int)


def randn(size, rng):
    """
    Equivalent to Matlab `norminv(rand(size), 0, 1)`
    :param size: shape of the random number array to be generated
    :param rng: random number generator
    :return: ndarray with random numbers from a normal distribution
    """

    if isinstance(size, int):
        # size is an integer
        return norm.ppf(rng.rand(size))

    return norm.ppf(rng.rand(*size))
