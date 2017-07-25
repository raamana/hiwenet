
import sys
import os
import nibabel
import warnings
import networkx

list_medpy_histogram_metrics = np.array([
    'chebyshev', 'chebyshev_neg', 'chi_square',
    'correlate', 'correlate_1',
    'cosine', 'cosine_1', 'cosine_2', 'cosine_alt',
    'euclidean', 'fidelity_based',
    'histogram_intersection', 'histogram_intersection_1',
    'jensen_shannon', 'kullback_leibler', 'manhattan', 'minowski',
    'noelle_1', 'noelle_2', 'noelle_3', 'noelle_4', 'noelle_5',
    'quadratic_forms', 'relative_bin_deviation', 'relative_deviation'])


def hiwenet(features, groups, weight_method = 'hist_int',
            num_bins = 100, trim_outliers = True, trim_percentile = 0.05,
            return_networkx_graph = False):
    """
    
    Parameters
    ----------
    features : numpy 1d array of length p 
        with scalar values. 
    groups : numpy 1d array 
        Membership array of length p, each value specifying which group that particular node belongs to.
        For examlpe, if you have a cortical thickness values for 1000 vertices belonging to 100 patches, 
        this array could  have numbers 1 to 100 specifying which vertex belongs to which cortical patch.
        Although grouping with numerical values (contiguous from 1 to num_patches) is strongly recommended for simplicity, 
        this could also be a list of strings of length p, 
            in which case a tuple is returned identifying which weight belongs to which pair of patches.
    weight_method : string 
        identifying the type of distance (or metric) to compute between the pair of histograms.
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
        Default: 0.05 (5%). Must be in open interval (0, 1).
    return_networkx_graph : bool 
        Specifies the need for a networkx graph populated with weights computed.

    Returns
    -------
    edge_weights : numpy 1d array of pair-wise edge-weights. 
        Size: num_groups x num_groups, wherein num_groups is determined by the total number of unique values in groups.
        This can be turned into a full matrix easily, if needed.

    """

    # parameter check
    num_bins = np.rint(num_bins)

    # TODO do some research on setting default values and ranges
    min_num_bins = 5
    if num_bins < min_num_bins:
        raise ValueError('Too few bins! The number of bins must be >= 5')

    if np.isnan(num_bins) or np.isinf(num_bins):
        raise ValueError('Invalid value for number of bins! Choose a natural number >= {}'.format(min_num_bins))

    if not isinstance(features, 'numpy.ndarray'):
        features = np.array(features)

    if not isinstance(groups, 'numpy.ndarray'):
        features = np.array(groups)

    if trim_percentile < 0.0 or trim_percentile >= 1.0:
        raise ValueError('percentile of tail values to trim must be in the semi-open interval [0,1).')

    if weight_method not in list_medpy_histogram_metrics:
        assert NotImplementedError('Chosen histogram distance/metric not implemented or invalid.')

    # preprocess data
    if trim_outliers:
        # percentiles_to_keep = [ trim_percentile, 1.0-trim_percentile] # [0.05, 0.95]
        edges_of_edges = [ np.percentile(features,     trim_percentile),
                           np.percentile(features, 1.0-trim_percentile)]
    else:
        edges_of_edges = np.array([ np.min(features), np.max(features)])

    # Edges computed using data from all groups, in order to establish correspondence
    edges = np.linspace(edges_of_edges[1], edges_of_edges[2], num = num_bins, endpoint = True)

    # memberships
    group_ids, num_groups = identify_groups(groups)

    edge_weights = np.empty([num_groups, num_groups])

    for g1 in num_groups:
        # TODO indexing needs to be implemented with care
        index1 = groups == group_ids[g1]
        # compute histograms for each patch
        # using the same edges for all groups to ensure correspondence
        hist_one = __compute_histogram(features[index1], edges)

        for g2 in xrange(g1+1, num_groups, 1):
            index2 = groups == group_ids[g2]
            hist_two = __compute_histogram(features[index2], edges)

            # TODO matrix or graph? nxG.add_edge(g1, g2, edge_value)
            edge_weights[g1, g2] = compute_edge_value(hist_one, hist_two, weight_method)

    return edge_weights


def __compute_histogram(values, edges):
    """Computes histogram (density) for a given vector of values."""

    hist = np.histogram(values, bins = edges, density=True)
    hist = __preprocess_histogram(hist, values, edges)

    return hist


def __preprocess_histogram(hist, values, edges):
    """Handles edge-cases and extremely-skewed histograms"""

    # working with extremely skewed histograms
    if np.count_nonzero(hist) == 0:
        # all of them above upper bound
        if np.all(values>=edges[-1]):
            hist[-1] = 1
        # all of them below lower bound
        elif np.all(values<=edges[0]):
            hist[0] = 1

    return hist


def compute_edge_value(hist_one, hist_two, weight_method_str):
    """
    
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


def identify_groups(groups):
    """
    To compute numner of unique elements in a given membership specification.
    
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

    group_ids = np.unique(groups_str)
    num_groups = len(group_ids)

    return group_ids, num_groups


