import sys
import os
import nibabel
import warnings
import networkx as nx
import numpy as np

list_medpy_histogram_metrics = np.array([
    'chebyshev', 'chebyshev_neg', 'chi_square',
    'correlate', 'correlate_1',
    'cosine', 'cosine_1', 'cosine_2', 'cosine_alt',
    'euclidean', 'fidelity_based',
    'histogram_intersection', 'histogram_intersection_1',
    'jensen_shannon', 'kullback_leibler', 'manhattan', 'minowski',
    'noelle_1', 'noelle_2', 'noelle_3', 'noelle_4', 'noelle_5',
    'relative_bin_deviation', 'relative_deviation'])

metric_list = [
    'kullback_leibler', 'manhattan', 'minowski', 'euclidean',
    'cosine_1',
    'noelle_2', 'noelle_4', 'noelle_5' ]

unknown_prop_list = ['histogram_intersection']
still_under_dev = ['quadratic_forms']
similarity_func = ['correlate', 'cosine', 'cosine_2', 'cosine_alt', 'fidelity_based']

semi_metric_list = [
    'jensen_shannon', 'chi_square',
    'chebyshev', 'chebyshev_neg',
    'histogram_intersection_1',
    'relative_deviation', 'relative_bin_deviation',
    'noelle_1', 'noelle_3',
    'correlate_1']

def extract(features, groups, weight_method='histogram_intersection',
            num_bins=100, trim_outliers=True, trim_percentile=5,
            return_networkx_graph=False):
    """
    Extracts the histogram weighted network.
    
    Parameters
    ----------
    features : numpy 1d array of length p
        with scalar values.
    groups : numpy 1d array
        Membership array of length p, each value specifying which group that particular node belongs to.
        For example, if you have a cortical thickness values for 1000 vertices belonging to 100 patches,
        this array could  have numbers 1 to 100 specifying which vertex belongs to which cortical patch.
        Grouping with numerical values (contiguous from 1 to num_patches) is strongly recommended for simplicity,
        but this could also be a list of strings of length p, in which case a tuple is
        returned identifying which weight belongs to which pair of patches.
    weight_method : string
        identifying the type of distance (or metric) to compute between the pair of histograms.
        It must be one of the methods implemented in medpy.metric.histogram: 
        [ 'chebyshev', 'chebyshev_neg', 'chi_square', 'correlate', 'correlate_1', 
        'cosine', 'cosine_1', 'cosine_2', 'cosine_alt', 'euclidean', 'fidelity_based', 
        'histogram_intersection', 'histogram_intersection_1', 'jensen_shannon', 'kullback_leibler', 
        'manhattan', 'minowski', 'noelle_1', 'noelle_2', 'noelle_3', 'noelle_4', 'noelle_5', 
        'relative_bin_deviation', 'relative_deviation'] except 'quadratic_forms'.
        Note only the following are metrics: [ 'kullback_leibler', 'manhattan', 'minowski', 'euclidean', 
        'cosine_1', 'noelle_2', 'noelle_4', 'noelle_5' ], 
        the following are semi-metrics: [ 'jensen_shannon', 'chi_square', 'chebyshev', 'chebyshev_neg', 'correlate_1' , 
        'histogram_intersection_1', 'relative_deviation', 'relative_bin_deviation', 'noelle_1', 'noelle_3']
        and the rest are similarity functions: 
            ['histogram_intersection', 'correlate', 'cosine', 'cosine_2', 'cosine_alt', 'fidelity_based']
    num_bins : scalar
        Number of bins to use when computing histogram within each patch/group.
        Note:
        1) Please ensure same number of bins are used across different subjects
        2) histogram shape can vary widely with number of bins (esp with fewer bins in the range of 3-20),
        and hence the features extracted based on them vary also.
        3) It is recommended to study the impact of this parameter on the final results of the experiment.
        This could also be optimized within an inner cross-validation loop if desired.
    trim_outliers : bool
        Whether to trim 5% outliers at the edges of feature range,
        when features are expected to contain extreme outliers (like 0 or eps or Inf).
        This is important to avoid numerical problems and also to stabilize the weight estimates.
    trim_percentile : float
        Small value specifying the percentile of outliers to trim.
        Default: 5 (5%). Must be in open interval (0, 100).
    return_networkx_graph : bool
        Specifies the need for a networkx graph populated with weights computed. Default: False.

    Returns
    -------
    edge_weights : numpy 2d array of pair-wise edge-weights. 
        Size: num_groups x num_groups, wherein num_groups is determined by the total number of unique values in groups.
        Only the upper triangular matrix is filled as the distance between node i and j would be the same as j and i.
        The edge weights from the upper triangular matrix can easily be obtained by
        weights_array = edge_weights[ np.triu_indices_from(edge_weights, 1) ]
    """

    # parameter check
    features, groups, num_bins, weight_method, group_ids, num_groups, num_links = __parameter_check(
        features, groups, num_bins, weight_method, trim_outliers, trim_percentile)

    # preprocess data
    if trim_outliers:
        # percentiles_to_keep = [ trim_percentile, 1.0-trim_percentile] # [0.05, 0.95]
        edges_of_edges = np.array([ np.percentile(features, trim_percentile),
                                    np.percentile(features, 100 - trim_percentile)])
    else:
        edges_of_edges = np.array([np.min(features), np.max(features)])

    # Edges computed using data from all nodes, in order to establish correspondence
    edges = np.linspace(edges_of_edges[0], edges_of_edges[1], num=num_bins, endpoint=True)

    if return_networkx_graph:
        nx_graph = nx.Graph()
        nx_graph.add_nodes_from(np.arange(num_groups))
    else:
        edge_weights = np.zeros([num_groups, num_groups], order='F')

    exceptions_list = list()
    for g1 in xrange(num_groups):
        index1 = groups == group_ids[g1]
        # compute histograms for each patch
        # using the same edges for all groups to ensure correspondence
        hist_one = __compute_histogram(features[index1], edges)

        for g2 in xrange(g1 + 1, num_groups, 1):
            index2 = groups == group_ids[g2]
            hist_two = __compute_histogram(features[index2], edges)

            try:
                edge_value = _compute_edge_weight(hist_one, hist_two, weight_method)
                if return_networkx_graph:
                    nx_graph.add_edge(group_ids[g1], group_ids[g2], weight=edge_value)
                else:
                    edge_weights[g1, g2] = edge_value
            except BaseException as exc:
                # numerical instabilities can cause trouble for histogram distance calculations
                exceptions_list.append(str(exc))
                warnings.warn('Unable to compute edge weight between {} and {}. Skipping it.'.format(group_ids[g1], group_ids[g2]))

    error_thresh = 0.5
    if len(exceptions_list) >= error_thresh*num_links:
        print('All exceptions encountered so far:\n {}'.format('\n'.join(exceptions_list)))
        raise ValueError('Weights for {:.2f}% of edges could not be computed.'.format(error_thresh*100))

    if return_networkx_graph:
        return nx_graph
    else:
        # triu_idx = np.triu_indices_from(edge_weights, 1)
        # return edge_weights[triu_idx]
        return edge_weights


def __compute_histogram(values, edges):
    """Computes histogram (density) for a given vector of values."""

    hist, bin_edges = np.histogram(values, bins=edges, density=True)
    hist = __preprocess_histogram(hist, values, edges)

    return hist


def __preprocess_histogram(hist, values, edges):
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


def _compute_edge_weight(hist_one, hist_two, weight_method_str):
    """
    Computes the edge weight between the two histograms.
    
    Parameters
    ----------
    hist_one : sequence
        First histogram
    hist_two : sequence
        Second histogram
    weight_method_str : string
        Identifying the type of distance (or metric) to compute between the pair of histograms.
        Must be one of the metrics implemented in medpy.metric.histogram
        
    Returns
    -------
    edge_value : float
        Distance or metric between the two histograms
    """

    from medpy.metric import histogram as medpy_hist_metrics

    weight_method = getattr(medpy_hist_metrics, weight_method_str)
    edge_value = weight_method(hist_one, hist_two)

    return edge_value


def _identify_groups(groups):
    """
    To compute number of unique elements in a given membership specification.
    
    Parameters
    ----------
    groups : numpy 1d array of length p, each value specifying which group that particular node belongs to.
        For examlpe, if you have a cortical thickness values for 1000 vertices belonging to 100 patches, 
        this array could  have numbers 1 to 100 specifying which vertex belongs to which cortical patch.
        Although grouping with numerical values (contiguous from 1 to num_patches) is strongly recommended for simplicity, 
        this could also be a list of strings of length p.

    Returns
    -------
    group_ids : numpy array of values identifying the unique groups specified
    num_groups : scalar value denoting the number of unique groups specified

    """

    group_ids = np.unique(groups)
    num_groups = len(group_ids)

    return group_ids, num_groups


def __parameter_check(features, groups, num_bins, weight_method, trim_outliers, trim_percentile):
    """Necessary check on values, ranges, and types."""

    num_bins = np.rint(num_bins)

    # TODO do some research on setting default values and ranges
    min_num_bins = 5
    if num_bins < min_num_bins:
        raise ValueError('Too few bins! The number of bins must be >= 5')

    if np.isnan(num_bins) or np.isinf(num_bins):
        raise ValueError('Invalid value for number of bins! Choose a natural number >= {}'.format(min_num_bins))

    if not isinstance(features, np.ndarray):
        features = np.array(features)

    if not isinstance(groups, np.ndarray):
        groups = np.array(groups)

    # memberships
    group_ids, num_groups = _identify_groups(groups)
    num_links = np.int64(num_groups * (num_groups - 1) / 2.0)

    num_values = len(features)
    if  num_values < num_groups:
        raise ValueError('Insufficient number of values in features (< number of nodes), or invalid membership!')

    if trim_outliers:
        if trim_percentile < 0 or trim_percentile >= 100:
            raise ValueError('percentile of tail values to trim must be in the semi-open interval [0,1).')

    if not trim_outliers:
        if num_values < 2:
            raise ValueError('too few features to compute minimum and maximum')

    if weight_method not in list_medpy_histogram_metrics:
        raise NotImplementedError('Chosen histogram distance/metric not implemented or invalid.')

    if num_groups < 2:
        raise ValueError('There must be atleast two nodes or groups in data, for pair-wise edge-weight calculations.')

    return features, groups, num_bins, weight_method, group_ids, num_groups, num_links