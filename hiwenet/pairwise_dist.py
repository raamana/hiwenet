__all__ = [
    'extract',
    'extract_with_config',
    'metric_list',
    'semi_metric_list',
    'metrics_on_original_features',
]

from collections.abc import Sequence
import traceback
import warnings
import logging
import networkx as nx
import numpy as np
from os.path import exists as pexists

from hiwenet import more_metrics, non_pairwise
from hiwenet.models import ExtractConfig, PreparedInputs, ResolvedMethod
from hiwenet.utils import precompute_group_histograms, HiwenetWarning

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
    'manhattan', 'minowski', 'euclidean',
    'noelle_2', 'noelle_4', 'noelle_5']

unknown_prop_list = ['histogram_intersection']
still_under_dev = ['quadratic_forms']
similarity_func = ['kullback_leibler', 'cosine_1', 'correlate',
                   'cosine', 'cosine_2', 'cosine_alt', 'fidelity_based']

semi_metric_list = [
    'jensen_shannon', 'chi_square',
    'chebyshev', 'chebyshev_neg',
    'histogram_intersection_1',
    'relative_deviation', 'relative_bin_deviation',
    'noelle_1', 'noelle_3',
    'correlate_1']

metrics_on_original_features = ['diff_medians', 'diff_medians_abs',
                                'diff_means',   'diff_means_abs',
                                'exp_diff_means_norm_std', 'ranksum_statistic']

list_hiwenet_histogram_metrics = np.array(['sorensen_dist', ])

symmetric_metrics_on_original_features = ['diff_medians_abs', 'diff_means_abs']

minimum_num_bins = 5

default_weight_method = 'manhattan'
default_num_bins = 25
default_edge_range = None
default_trim_percentile = 5
default_trim_behaviour = True
def compute_bin_edges(features, num_bins, edge_range, trim_outliers, trim_percentile,
                      use_orig_distr=False):
    "Compute the edges for the histogram bins to keep it the same for all nodes."

    if use_orig_distr:
        print('Using original distribution (without histogram) to compute edge weights!')
        return None

    if edge_range is None:
        if trim_outliers:
            # percentiles_to_keep = [ trim_percentile, 1.0-trim_percentile] # [0.05, 0.95]
            edges_of_edges = np.array([np.percentile(features, trim_percentile),
                                       np.percentile(features, 100 - trim_percentile)])
        else:
            edges_of_edges = np.array([np.min(features), np.max(features)])
    else:
        edges_of_edges = edge_range

    # Edges computed using data from all nodes, in order to establish correspondence
    edges = np.linspace(edges_of_edges[0], edges_of_edges[1], num=num_bins, endpoint=True)

    return edges


def extract(features, groups,
            weight_method=default_weight_method,
            num_bins=default_num_bins,
            edge_range=default_edge_range,
            trim_outliers=default_trim_behaviour,
            trim_percentile=default_trim_percentile,
            use_original_distribution=False,
            relative_to_all=False,
            asymmetric=False,
            return_networkx_graph=False):
    """
    Extracts the histogram-distance weighted adjacency matrix.

    Parameters
    ----------
    features : ndarray or str
        1d array of scalar values, either provided directly as a 1d numpy array,
        or as a path to a file containing these values

    groups : ndarray or str
        Membership array of same length as `features`, each value specifying which group that particular node belongs to.
        Input can be either provided directly as a 1d numpy array,or as a path to a file containing these values.

        For example, if you have cortical thickness values for 1000 vertices (`features` is ndarray of length 1000),
        belonging to 100 patches, the groups array (of length 1000) could  have numbers 1 to 100 (number of unique values)
        specifying which element belongs to which cortical patch.

        Grouping with numerical values (contiguous from 1 to num_patches) is strongly recommended for simplicity,
        but this could also be a list of strings of length p, in which case a tuple is returned,
        identifying which weight belongs to which pair of patches.

    weight_method : string or callable, optional
        Type of distance (or metric) to compute between the pair of histograms.
        It can either be a string identifying one of the weights implemented below, or a valid callable.

        If a string, it must be one of the following methods:

        - 'chebyshev'
        - 'chebyshev_neg'
        - 'chi_square'
        - 'correlate'
        - 'correlate_1'
        - 'cosine'
        - 'cosine_1'
        - 'cosine_2'
        - 'cosine_alt'
        - 'euclidean'
        - 'fidelity_based'
        - 'histogram_intersection'
        - 'histogram_intersection_1'
        - 'jensen_shannon'
        - 'kullback_leibler'
        - 'manhattan'
        - 'minowski'
        - 'noelle_1'
        - 'noelle_2'
        - 'noelle_3'
        - 'noelle_4'
        - 'noelle_5'
        - 'relative_bin_deviation'
        - 'relative_deviation'

        Note only the following are *metrics*:

        - 'manhattan'
        - 'minowski'
        - 'euclidean'
        - 'noelle_2'
        - 'noelle_4'
        - 'noelle_5'

        The following are *semi- or quasi-metrics*:

        - 'kullback_leibler'
        - 'jensen_shannon'
        - 'chi_square'
        - 'chebyshev'
        - 'cosine_1'
        - 'chebyshev_neg'
        - 'correlate_1'
        - 'histogram_intersection_1'
        - 'relative_deviation'
        - 'relative_bin_deviation'
        - 'sorensen_dist'
        - 'noelle_1'
        - 'noelle_3'

        The following are  classified to be similarity functions:

        - 'histogram_intersection'
        - 'correlate'
        - 'cosine'
        - 'cosine_2'
        - 'cosine_alt'
        - 'fidelity_based'

        *Default* choice: 'minowski'.

        The method can also be one of the following identifying metrics that operate on the original data directly -
         e.g. difference in the medians coming from the distributions of the pair of ROIs.

         - 'diff_medians'
         - 'diff_means'
         - 'diff_medians_abs'
         - 'diff_means_abs'
         - 'exp_diff_means_norm_std'
         - 'ranksum_statistic'

         Please note this can lead to adjacency matrices that may not be symmetric
            e.g. difference metric on two scalars is not symmetric).
            In this case, be sure to use the flag: allow_non_symmetric=True

        If weight_method is a callable, it must two accept two arrays as input and return one scalar as output.
            Example: ``diff_in_skew = lambda x, y: abs(scipy.stats.skew(x)-scipy.stats.skew(y))``
            NOTE: this method will be applied to histograms (not the original distribution of features from group/ROI).
            In order to apply this callable directly on the original distribution (without trimming and histogram binning),
            use ``use_original_distribution=True``.

    num_bins : scalar, optional
        Number of bins to use when computing histogram within each patch/group.

        Note:

        1) Please ensure same number of bins are used across different subjects
        2) histogram shape can vary widely with number of bins (esp with fewer bins in the range of 3-20), and hence the features extracted based on them vary also.
        3) It is recommended to study the impact of this parameter on the final results of the experiment.

        This could also be optimized within an inner cross-validation loop if desired.

    edge_range : tuple or None
        The range of edges within which to bin the given values.
        This can be helpful to ensure correspondence across multiple invocations of hiwenet (for different subjects),
        in terms of range across all bins as well as individual bin edges.
        Default is to automatically compute from the given values.

        Accepted format:

            - tuple of finite values: (range_min, range_max)
            - None, triggering automatic calculation (default)

        Notes : when controlling the ``edge_range``, it is not possible trim the tails (e.g. using the parameters
        ``trim_outliers`` and ``trim_percentile``) for the current set of features using its own range.

    trim_outliers : bool, optional
        Whether to trim a small percentile of outliers at the edges of feature range,
        when features are expected to contain extreme outliers (like 0 or eps or Inf).
        This is important to avoid numerical problems and also to stabilize the weight estimates.

    trim_percentile : float
        Small value specifying the percentile of outliers to trim.
        Default: 5 (5%). Must be in open interval (0, 100).

    use_original_distribution : bool, optional
        When using a user-defined callable, this flag
        1) allows skipping of pre-processing (trimming outliers) and histogram construction,
        2) enables the application of arbitrary callable (user-defined) on the original distributions coming from the two groups/ROIs/nodes directly.

        Example: ``diff_in_medians = lambda x, y: abs(np.median(x)-np.median(y))``

        This option is valid only when weight_method is a valid callable,
            which must take two inputs (possibly of different lengths) and return a single scalar.

    relative_to_all : bool
        Flag to instruct the computation of a grand histogram (distribution pooled from values in all ROIs),
        and compute distances (based on distance specified by ``weight_method``) by from each ROI to the grand mean.
        This would result in only N distances for N ROIs, instead of the usual N*(N-1) pair-wise distances.

    asymmetric : bool
        Flag to identify resulting adjacency matrix is expected to be non-symmetric.
        Note: this results in twice the computation time!
        Default: False , for histogram metrics implemented here are symmetric.

    return_networkx_graph : bool, optional
        Specifies the need for a networkx graph populated with weights computed. Default: False.

    Returns
    -------
    edge_weights : ndarray
        numpy 2d array of pair-wise edge-weights (of size: num_groups x num_groups),
        wherein num_groups is determined by the total number of unique values in `groups`.

        **Note**:

        - Only the upper triangular matrix is filled as the distance between node i and j would be the same as j and i.
        - The edge weights from the upper triangular matrix can easily be obtained by

        .. code-block:: python

            weights_array = edge_weights[ np.triu_indices_from(edge_weights, 1) ]

    """

    config = ExtractConfig(
        weight_method=weight_method,
        num_bins=num_bins,
        edge_range=edge_range,
        trim_outliers=trim_outliers,
        trim_percentile=trim_percentile,
        use_original_distribution=use_original_distribution,
        relative_to_all=relative_to_all,
        asymmetric=asymmetric,
        return_networkx_graph=return_networkx_graph,
    )
    return extract_with_config(features, groups, config)


def extract_with_config(features, groups, config: ExtractConfig):
    """Validated extraction entry point using an explicit config model."""

    prepared = check_params(
        features,
        groups,
        config.num_bins,
        config.edge_range,
        config.trim_outliers,
        config.trim_percentile,
    )
    method = check_weight_method(
        config.weight_method,
        config.use_original_distribution,
        config.asymmetric,
    )

    edges = compute_bin_edges(
        prepared.features,
        prepared.num_bins,
        prepared.edge_range,
        config.trim_outliers,
        config.trim_percentile,
        method.use_orig_distr,
    )

    if config.relative_to_all:
        return non_pairwise.relative_to_all(
            prepared.features,
            prepared.groups,
            edges,
            method.weight_func,
            method.use_orig_distr,
            prepared.group_ids,
            prepared.num_groups,
            config.return_networkx_graph,
        )

    return pairwise_extract(
        prepared.features,
        prepared.groups,
        edges,
        method.weight_func,
        method.use_orig_distr,
        prepared.group_ids,
        prepared.num_groups,
        prepared.num_links,
        method.non_symmetric,
        config.return_networkx_graph,
    )


def pairwise_extract(features, groups, edges, weight_func, use_orig_distr,
                     group_ids, num_groups, num_links,
                     non_symmetric, return_networkx_graph):
    """
    Core function to compute the pair-wise histogram distance between all ROIs.

    Parameters
    ----------
    features
    groups
    edges
    weight_func
    use_orig_distr
    group_ids
    num_groups
    num_links
    non_symmetric
    return_networkx_graph
    Returns
    -------
    result : object
        A networkx graph or numpy array depending on request

    """

    if return_networkx_graph:
        graph = nx.DiGraph() if non_symmetric else nx.Graph()
        graph.add_nodes_from(group_ids)
    else:
        edge_weights = np.full([num_groups, num_groups], np.nan)

    group_histograms = precompute_group_histograms(
        features,
        groups,
        group_ids,
        edges,
        use_orig_distr,
    )
    exceptions_list = list()
    for src in range(num_groups):
        hist_one = group_histograms[group_ids[src]]

        if non_symmetric:
            target_list = range(num_groups)
        else:
            # when symmetric, only upper tri matrix is computed/filled
            target_list = range(src + 1, num_groups, 1)

        for dest in target_list:
            if src == dest:
                continue

            hist_two = group_histograms[group_ids[dest]]

            try:
                edge_value = weight_func(hist_one, hist_two)
                if return_networkx_graph:
                    graph.add_edge(group_ids[src], group_ids[dest],
                                   weight=float(edge_value))
                else:
                    edge_weights[src, dest] = edge_value
            except (RuntimeError, RuntimeWarning) as runexc:
                # placeholder to ignore some runtime errors (such as medpy's logger issue)
                print(runexc)
            except Exception as exc:
                # numerical instabilities can cause trouble for histogram distance calculations
                traceback.print_exc()
                exceptions_list.append(str(exc))
                logging.warning('Unable to compute edge weight between '
                                ' {} and {}. Skipping it.'
                                ''.format(group_ids[src], group_ids[dest]))

    error_thresh = 0.05
    if len(exceptions_list) >= error_thresh * num_links:
        print('All exceptions encountered so far:\n'
              ' {}'.format('\n'.join(exceptions_list)))
        raise ValueError('Weights for atleast {:.2f}% of edges could not be computed.'
                         ''.format(error_thresh * 100))

    if return_networkx_graph:
        return graph
    else:
        return edge_weights


def identify_groups(groups):
    """
    To compute number of unique elements in a given membership specification.

    Parameters
    ----------
    groups : numpy 1d array of length p,
        each value specifying which group that particular node belongs to.
        For examlpe, if you have a cortical thickness values for 1000 vertices
        belonging to 100 patches, this array could  have numbers 1 to 100
        specifying which vertex belongs to which cortical patch. Although grouping
        with numerical values (contiguous from 1 to num_patches) is strongly
        recommended for simplicity, this could also be a list of strings of length p.

    Returns
    -------
    group_ids : numpy array of values identifying the unique groups specified
    num_groups : scalar value denoting the number of unique groups specified

    """

    group_ids = np.unique(groups)
    num_groups = len(group_ids)

    if num_groups < 2:
        raise ValueError('There must be atleast two nodes or groups in data, '
                         'for pair-wise edge-weight calculations.')

    return group_ids, num_groups


def check_param_ranges(num_bins, num_groups, num_values, trim_outliers, trim_percentile):
    """Ensuring the parameters are in valid ranges."""

    if num_bins < minimum_num_bins:
        raise ValueError('Too few bins! The number of bins must be >= 5')

    if num_values < num_groups:
        raise ValueError('Insufficient number of values '
                         'in features (< number of nodes), or invalid membership!')

    if trim_outliers:
        if trim_percentile < 0 or trim_percentile >= 100:
            raise ValueError('percentile of tail values to trim '
                             'must be in the semi-open interval [0,1).')
    elif num_values < 2:
        raise ValueError('too few features to compute minimum and maximum')

    return


def type_cast_params(num_bins, edge_range_spec, features, groups):
    """Casting inputs to required types."""

    if isinstance(num_bins, str):
        # possible when called from CLI
        num_bins = float(num_bins)

    if np.isnan(num_bins) or np.isinf(num_bins):
        raise ValueError('Invalid value for number of bins!'
                         ' Choose a natural/finite number >= {}'
                         ''.format(minimum_num_bins))

    # rounding it to ensure it is int
    num_bins = int(num_bins)

    if edge_range_spec is None:
        edge_range = edge_range_spec
    elif isinstance(edge_range_spec, Sequence):
        if len(edge_range_spec) != 2:
            raise ValueError('edge_range must be a tuple of two values: (min, max)')
        if edge_range_spec[0] >= edge_range_spec[1]:
            raise ValueError('edge_range : min {} is not less than the max {} !'
                             ''.format(edge_range_spec[0], edge_range_spec[1]))
        if not np.all(np.isfinite(edge_range_spec)):
            raise ValueError('Infinite or NaN values in edge range : {}'
                             ''.format(edge_range_spec))

        # converting it to tuple to make it immutable
        edge_range = tuple(edge_range_spec)
    else:
        raise ValueError('Invalid edge range!'
                         ' Must be a tuple of two values (min, max)')

    if not isinstance(features, np.ndarray):
        features = np.array(features)

    if not isinstance(groups, np.ndarray):
        groups = np.array(groups)

    return num_bins, edge_range, features, groups


def make_random_histogram(length=100, num_bins=10):
    "Returns a sequence of histogram density values that sum to 1.0"

    hist, bin_edges = np.histogram(np.random.random(length),
                                   bins=num_bins, density=True)

    # to ensure they sum to 1.0
    hist = hist / sum(hist)

    if len(hist) < 2:
        raise ValueError('Invalid histogram')

    return hist


def check_weight_method(weight_method_spec,
                        use_orig_distr=False,
                        allow_non_symmetric=False):
    "Check if weight_method is recognized and implemented, or ensure it is callable."

    if not isinstance(use_orig_distr, bool):
        raise TypeError('use_original_distribution flag must be boolean!')

    if not isinstance(allow_non_symmetric, bool):
        raise TypeError('allow_non_symmetric flag must be boolean')

    if isinstance(weight_method_spec, str):
        weight_method_spec = weight_method_spec.lower()

        if weight_method_spec in list_medpy_histogram_metrics:
            from medpy.metric import histogram as medpy_hist_metrics
            weight_func = getattr(medpy_hist_metrics, weight_method_spec)
            if use_orig_distr:
                warnings.warn('use_original_distribution must be False '
                              'when using builtin histogram metrics, '
                              'which expect histograms as input'
                              ' - setting it to False.', HiwenetWarning)
                use_orig_distr = False

        elif weight_method_spec in list_hiwenet_histogram_metrics:
            weight_func = getattr(more_metrics, weight_method_spec)

        elif weight_method_spec in metrics_on_original_features:
            weight_func = getattr(more_metrics, weight_method_spec)
            if not use_orig_distr:
                warnings.warn('use_original_distribution must be True'
                              ' when using builtin non-histogram metrics,'
                              ' which expect original feature values in ROI/node '
                              'as input - setting it to True.', HiwenetWarning)
                use_orig_distr = True

            if weight_method_spec in symmetric_metrics_on_original_features:
                print('Chosen metric is symmetric. Ignoring asymmetric=False flag.')
                allow_non_symmetric=False

        else:
            raise NotImplementedError('Chosen histogram distance/metric {}'
                                      ' not implemented or invalid.'
                                      ''.format(weight_method_spec))

    elif callable(weight_method_spec):
        # ensure 1) takes two ndarrays
        try:
            dummy_weight = weight_method_spec(make_random_histogram(), make_random_histogram())
        except Exception:
            raise TypeError('Error applying given callable on two input arrays.\n'
                            '{} must accept 2 arrays and return a 1 scalar value!')
        else:
            # and 2) returns only one number
            if not np.isscalar(dummy_weight):
                raise TypeError('Given callable does not return '
                                'a single scalar as output.')

        weight_func = weight_method_spec
    else:
        raise ValueError('Supplied method to compute edge weight is not recognized:\n'
                         'must be a string identifying one of the implemented methods\n{}'
                         '\n or a valid callable that accepts that two arrays '
                         'and returns 1 scalar.'.format(list_medpy_histogram_metrics))

    return ResolvedMethod(
        weight_func=weight_func,
        use_orig_distr=use_orig_distr,
        non_symmetric=allow_non_symmetric,
    )


def check_params(features_spec, groups_spec, num_bins, edge_range_spec,
                 trim_outliers, trim_percentile):
    """Necessary check on values, ranges, and types."""

    if isinstance(features_spec, str) and isinstance(groups_spec, str):
        features, groups = read_features_and_groups(features_spec, groups_spec)
    else:
        features, groups = features_spec, groups_spec

    num_bins, edge_range, features, groups = \
        type_cast_params(num_bins, edge_range_spec, features, groups)
    num_values = len(features)

    # memberships
    group_ids, num_groups = identify_groups(groups)
    num_links = np.int64(num_groups * (num_groups - 1) / 2.0)

    check_param_ranges(num_bins, num_groups, num_values, trim_outliers, trim_percentile)

    return PreparedInputs(
        features=features,
        groups=groups,
        group_ids=group_ids,
        num_groups=num_groups,
        num_links=int(num_links),
        num_bins=num_bins,
        edge_range=edge_range,
    )


def read_features_and_groups(features_path, groups_path):
    "Reader for data and groups"

    try:
        if not pexists(features_path):
            raise ValueError('non-existent features file')

        if not pexists(groups_path):
            raise ValueError('non-existent groups file')

        if isinstance(features_path, str):
            features = np.genfromtxt(features_path, dtype=float)
        else:
            raise ValueError('features input must be a file path ')

        if isinstance(groups_path, str):
            groups = np.genfromtxt(groups_path, dtype=str)
        else:
            raise ValueError('groups input must be a file path ')

    except Exception:
        raise IOError('error reading the specified features and/or groups.')

    if len(features) != len(groups):
        raise ValueError("lengths of features and groups do not match!")

    return features, groups
