"""
Module implementing additional metrics for edge weights.

"""

from sys import version_info
if version_info.major==2 and version_info.minor==7:
    from .utils import check_array
elif version_info.major > 2:
    from hiwenet.utils import check_array
else:
    raise NotImplementedError('hiwenet supports only 2.7 or 3+. '
                              'Upgrade to Python 3+ is recommended.')

__all__ = ['diff_medians', 'diff_medians_abs',
           'diff_means', 'diff_means_abs']

import numpy as np


def diff_medians(array_one, array_two):
    """
    Computes the difference in medians between two arrays of values.

    Given arrays will be flattened (to 1D array) regardless of dimension,
        and any non-finite/NaN values will be ignored.

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
    diff_medians = np.ma.median(array_one) - np.ma.median(array_two)

    return diff_medians


def diff_medians_abs(array_one, array_two):
    """
    Computes the absolute (symmetric) difference in medians between two arrays of values.

    Given arrays will be flattened (to 1D array) regardless of dimension,
        and any non-finite/NaN values will be ignored.

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


def diff_means(array_one, array_two):
    """
    Computes the difference in means between two arrays of values.

    Given arrays will be flattened (to 1D array) regardless of dimension,
        and any non-finite/NaN values will be ignored.

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
    diff_means = np.ma.mean(array_one) - np.ma.mean(array_two)

    return diff_means


def diff_means_abs(array_one, array_two):
    """
    Computes the absolute (symmetric) difference in means between two arrays of values.

    Given arrays will be flattened (to 1D array) regardless of dimension,
        and any non-finite/NaN values will be ignored.

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

    abs_diff_means = np.abs(diff_means(array_one, array_two))

    return abs_diff_means