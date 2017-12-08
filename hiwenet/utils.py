"""
Common utilities and helpers.

"""

__all__ = ['compute_histogram', 'preprocess_histogram', 'compute_edge_weight', 'HiwenetWarning']

import numpy as np

def compute_histogram(values, edges, use_orig_distr=False):
    """Computes histogram (density) for a given vector of values."""

    if use_orig_distr:
        return values

    # ignoring invalid values: Inf and Nan
    values = check_array(values).compressed()

    hist, bin_edges = np.histogram(values, bins=edges, density=True)
    hist = preprocess_histogram(hist, values, edges)

    return hist


def preprocess_histogram(hist, values, edges):
    """Handles edge-cases and extremely-skewed histograms"""

    # working with extremely skewed histograms
    if np.count_nonzero(hist) == 0:
        # all of them above upper bound
        if np.all(values >= edges[-1]):
            hist[-1] = 1
        # all of them below lower bound
        elif np.all(values <= edges[0]):
            hist[0] = 1

    return hist


def compute_edge_weight(hist_one, hist_two, weight_func):
    """
    Computes the edge weight between the two histograms.

    Parameters
    ----------
    hist_one : sequence
        First histogram

    hist_two : sequence
        Second histogram

    weight_func : callable
        Identifying the type of distance (or metric) to compute between the pair of histograms.
        Must be one of the metrics implemented in medpy.metric.histogram, or another valid callable.

    Returns
    -------
    edge_value : float
        Distance or metric between the two histograms
    """

    edge_value = weight_func(hist_one, hist_two)

    return edge_value


class HiwenetWarning(Warning):
    """Exception to indicate the detection of non-fatal use of hiwenet."""
    pass


def check_array(array):
    "Converts to flattened numpy arrays and ensures its not empty."

    if len(array) < 1:
        raise ValueError('Input array is empty! Must have atleast 1 element.')

    return np.ma.masked_invalid(array).flatten()