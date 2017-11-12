"""
Module implementing additional metrics for edge weights.

"""

__all__ = ['diff_medians', 'diff_medians_abs']

import numpy as np

def check_array(array):
    "Converts to flattened numpy arrays and ensures its not empty."

    if len(array) < 1:
        raise ValueError('Input array is empty! Must have atleast 1 element.')

    return np.array(array).flatten()


def diff_medians(array_one, array_two):
    """
    Computes the difference medians between two arrays of values.

    Given arrays will be flattened (to 1D array) regardless of dimension,
        and any bon-finite/NaN values will be ignored.

    Parameters
    ----------
    array_one, array_two : iterable
        Two arrays of values, possibly of different length.

    Returns
    -------
    diff_medians : float
        scalar measuring the difference in medians, ignoring NaNs/non-finite values.

    Raises
    ------
    ValueError
        If one or more of the arrays are empty.

    """

    array_one = check_array(array_one)
    array_two = check_array(array_two)
    diff_medians = np.median(array_one) - np.median(array_two)

    return diff_medians


def diff_medians_abs(array_one, array_two):
    """
    Computes the absolute difference (symmetric) medians between two arrays of values.

    Given arrays will be flattened (to 1D array) regardless of dimension,
        and any bon-finite/NaN values will be ignored.

    Parameters
    ----------
    array_one, array_two : iterable
        Two arrays of values, possibly of different length.

    Returns
    -------
    diff_medians : float
        scalar measuring the difference in medians, ignoring NaNs/non-finite values.

    Raises
    ------
    ValueError
        If one or more of the arrays are empty.

    """

    abs_diff_medians = np.abs(diff_medians(array_one, array_two))

    return abs_diff_medians