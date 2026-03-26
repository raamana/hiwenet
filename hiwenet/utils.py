"""Common utilities and helpers."""

__all__ = [
    'compute_histogram',
    'precompute_group_histograms',
    'preprocess_histogram',
    'compute_edge_weight',
    'HiwenetWarning',
]

import numpy as np


def compute_histogram(values, edges, use_orig_distr=False):
    """Computes histogram (density) for a given vector of values."""

    if use_orig_distr:
        return values

    # ignoring invalid values: Inf and Nan
    values = check_array(values).compressed()
    counts = preprocess_histogram_counts(values, edges)
    return counts_to_density(counts, edges)


def precompute_group_histograms(features, groups, group_ids, edges, use_orig_distr=False):
    """Precomputes one histogram or original-value vector for each group."""

    return {
        group_id: compute_histogram(features[groups == group_id], edges, use_orig_distr)
        for group_id in group_ids
    }


def preprocess_histogram_counts(values, edges):
    """Builds histogram counts while handling degenerate and out-of-range inputs."""

    counts = np.histogram(values, bins=edges, density=False)[0].astype(float, copy=False)

    if np.count_nonzero(counts) == 0:
        if np.all(values >= edges[-1]):
            counts[-1] = float(len(values))
        elif np.all(values <= edges[0]):
            counts[0] = float(len(values))
        elif np.all(values == values[0]):
            # Constant-valued regions can produce all-zero counts if the value lies
            # exactly on the upper edge; keep the full mass in one bin deterministically.
            insert_at = np.searchsorted(edges, values[0], side="right") - 1
            insert_at = int(np.clip(insert_at, 0, len(counts) - 1))
            counts[insert_at] = float(len(values))

    return counts


def counts_to_density(counts, edges):
    """Converts histogram counts to density values matching numpy.histogram(density=True)."""

    total = float(np.sum(counts))
    if total <= 0:
        return counts

    bin_widths = np.diff(edges)
    density = counts / total
    density = np.divide(
        density,
        bin_widths,
        out=np.zeros_like(density, dtype=float),
        where=bin_widths > 0,
    )
    return density


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
