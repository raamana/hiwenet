import os
from os.path import abspath

import networkx as nx
import numpy as np
import scipy.stats
import warnings

from hiwenet import extract as hiwenet
from hiwenet.pairwise_dist import metric_list, semi_metric_list
from hiwenet.utils import HiwenetWarning

list_weight_methods = metric_list + semi_metric_list

from pytest import raises, warns, mark

dimensionality = np.random.randint(500, 2500)
num_groups = np.random.randint(2, 100)


def make_features(dimensionality, num_groups):
    group_ids_init = np.arange(num_groups)
    random_indices_into_groups = np.random.randint(0, num_groups,
                                                   [1, dimensionality])
    groups = group_ids_init[random_indices_into_groups].flatten()

    group_ids = np.unique(groups)
    num_groups = len(group_ids)

    features = 10 * np.random.random(dimensionality)

    return features, groups, group_ids, num_groups


features, groups, group_ids, num_groups = make_features(dimensionality, num_groups)
num_links = np.int64(num_groups * (num_groups - 1) / 2.0)
# for ix1, gid1 in enumerate(group_ids):
#     for ix2, gid2 in enumerate(group_ids):
#         if ix1 != ix2:
#             dist = cdist(features[groups==gid1],features[groups==gid2],
#             'euclidean')

cur_dir = os.path.dirname(abspath(__file__))


# the following are mostly usage tests. Refer to test_medpy.py for scientific
# validity of histogram metrics

@mark.filterwarnings("ignore:.*usage will be deprecated.*:DeprecationWarning")
def test_directed_mat():
    ew = hiwenet(features, groups, asymmetric=True)
    assert len(ew) == num_groups
    assert ew.shape[0] == num_groups and ew.shape[1] == num_groups

    # ensure no self-loops
    diag_elements = ew.flatten()[::num_groups + 1]
    assert np.allclose(diag_elements, np.nan, equal_nan=True)


@mark.filterwarnings("ignore:.*usage will be deprecated.*:DeprecationWarning")
def test_directed_nx():
    graph = hiwenet(features, groups, asymmetric=True, return_networkx_graph=True)
    assert graph.is_directed()
    assert graph.number_of_nodes() == num_groups
    assert graph.number_of_edges() == 2 * num_links


@mark.filterwarnings("ignore:.*usage will be deprecated.*:DeprecationWarning")
def test_dimensions():
    ew = hiwenet(features, groups)
    assert len(ew) == num_groups
    assert ew.shape[0] == num_groups and ew.shape[1] == num_groups


@mark.filterwarnings("ignore:.*usage will be deprecated.*:DeprecationWarning")
def test_more_metrics():
    ew = hiwenet(features, groups, weight_method='diff_medians',
                 use_original_distribution=True)
    assert len(ew) == num_groups
    assert ew.shape[0] == num_groups and ew.shape[1] == num_groups

    ew_abs = hiwenet(features, groups, weight_method='diff_medians_abs',
                     use_original_distribution=True)
    assert np.allclose(np.abs(ew), ew_abs, equal_nan=True)

    with warns(HiwenetWarning):
        ew = hiwenet(features, groups, weight_method='diff_medians',
                     use_original_distribution=False)

    with warns(HiwenetWarning):
        ew = hiwenet(features, groups,
                     weight_method='manhattan',
                     use_original_distribution=True)


@mark.filterwarnings("ignore:.*usage will be deprecated.*:DeprecationWarning")
def test_too_few_groups():
    features, groups, group_ids, num_groups = make_features(100, 1)
    with raises(ValueError):
        hiwenet(features, groups)


@mark.filterwarnings("ignore:.*usage will be deprecated.*:DeprecationWarning")
def test_too_few_values():
    features, groups, group_ids, num_groups = make_features(10, 500)
    with raises(ValueError):
        hiwenet(features[:num_groups - 1], groups)


@mark.filterwarnings("ignore:.*usage will be deprecated.*:DeprecationWarning")
def test_invalid_trim_perc():
    with raises(ValueError):
        hiwenet(features, groups, trim_percentile=-1)

    with raises(ValueError):
        hiwenet(features, groups, trim_percentile=101)


@mark.filterwarnings("ignore:.*usage will be deprecated.*:DeprecationWarning")
def test_invalid_edge_range():
    with raises(ValueError):
        hiwenet(features, groups, edge_range=-1)

    with raises(ValueError):
        hiwenet(features, groups, edge_range=[])

    with raises(ValueError):
        hiwenet(features, groups, edge_range=[1, ])

    with raises(ValueError):
        hiwenet(features, groups, edge_range=[1, 2, 3])

    with raises(ValueError):
        hiwenet(features, groups, edge_range=(1, np.nan))

    with raises(ValueError):
        hiwenet(features, groups, edge_range=(2, 1))


@mark.filterwarnings("ignore:.*usage will be deprecated.*:DeprecationWarning")
def test_invalid_weight_method():
    with raises(NotImplementedError):
        hiwenet(features, groups, weight_method='dkjz.jscn')

    with raises(NotImplementedError):
        hiwenet(features, groups, weight_method='somerandomnamenoonewoulduse')


@mark.filterwarnings("ignore:.*usage will be deprecated.*:DeprecationWarning")
def test_trim_not_too_few_values():
    with raises(ValueError):
        hiwenet([0], [1], trim_outliers=False)


@mark.filterwarnings("ignore:.*usage will be deprecated.*:DeprecationWarning")
def test_trim_false_too_few_to_calc_range():
    with raises(ValueError):
        hiwenet([1], groups, trim_outliers=False)


@mark.filterwarnings("ignore:.*usage will be deprecated.*:DeprecationWarning")
def test_not_np_arrays():
    with raises(ValueError):
        hiwenet(list(), groups, trim_percentile=101)

    with raises(ValueError):
        hiwenet(features, list(), trim_percentile=101)


@mark.filterwarnings("ignore:.*usage will be deprecated.*:DeprecationWarning")
def test_invalid_nbins():
    with raises(ValueError):
        hiwenet(features, groups, num_bins=np.nan)

    with raises(ValueError):
        hiwenet(features, groups, num_bins=np.inf)

    with raises(ValueError):
        hiwenet(features, groups, num_bins=2)


@mark.filterwarnings("ignore:.*usage will be deprecated.*:DeprecationWarning")
def test_return_nx_graph():
    nxG = hiwenet(features, groups, return_networkx_graph=True)
    assert isinstance(nxG, nx.Graph)
    assert nxG.number_of_nodes() == num_groups
    assert nxG.number_of_edges() == num_links


@mark.filterwarnings("ignore:.*usage will be deprecated.*:DeprecationWarning")
def test_extreme_skewed():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = hiwenet(10 + np.zeros(dimensionality), groups)

    runtime_warnings = [
        warning for warning in caught
        if issubclass(warning.category, RuntimeWarning)
    ]
    assert not runtime_warnings
    assert result.shape == (num_groups, num_groups)


@mark.filterwarnings("ignore:.*usage will be deprecated.*:DeprecationWarning")
def test_input_callable():
    diff_skew = lambda x, y: abs(scipy.stats.skew(x) - scipy.stats.skew(y))
    ew = hiwenet(features, groups, weight_method=diff_skew)

    assert len(ew) == num_groups
    assert ew.shape[0] == num_groups and ew.shape[1] == num_groups


@mark.filterwarnings("ignore:.*usage will be deprecated.*:DeprecationWarning")
def test_input_callable_on_orig_data():
    diff_medians = lambda x, y: abs(np.median(x) - np.median(y))
    ew = hiwenet(features, groups, weight_method=diff_medians,
                 use_original_distribution=True)

    assert len(ew) == num_groups
    assert ew.shape[0] == num_groups and ew.shape[1] == num_groups

    # checking for invalid use : histogram metrics can not be applied on orig data
    # (without building histograms)
    with warns(HiwenetWarning):
        ew = hiwenet(features, groups,
                     weight_method='manhattan',
                     use_original_distribution=True)


@mark.filterwarnings("ignore:.*usage will be deprecated.*:DeprecationWarning")
def test_relative_to_all():
    ""

    for weight_method in list_weight_methods:
        ew = hiwenet(features, groups,
                     weight_method=weight_method,
                     relative_to_all=True)

        # it must be a vector
        assert ew.size == num_groups
        assert ew.shape[0] == num_groups
        assert ew.shape[1] == 1

        nxg = hiwenet(features, groups,
                      weight_method=weight_method,
                      relative_to_all=True,
                      return_networkx_graph=True)

        # additional node is the grand node (from all roi's)
        assert nxg.number_of_nodes() == num_groups + 1
        assert nxg.number_of_edges() == num_groups

        weights_from_nxg = np.array(
                [nxg[src]['whole']['weight'] for src in group_ids])
        assert np.allclose(ew.flatten(), weights_from_nxg)


# test_directed_nx()
# test_directed_mat()
# test_input_callable()

# test_more_metrics()

test_relative_to_all()
