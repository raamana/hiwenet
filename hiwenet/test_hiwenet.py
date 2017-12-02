import sys
import os
import shlex
import numpy as np
import scipy.stats
import networkx as nx
import matplotlib.pyplot as plt
from os.path import join as pjoin, exists as pexists, abspath
from sys import version_info

if version_info.major==2 and version_info.minor==7:
    from .pairwise_dist import extract as hiwenet
    from .pairwise_dist import run_cli as CLI
    from .pairwise_dist import metric_list, semi_metric_list
    from .utils import HiwenetWarning
elif version_info.major > 2:
    from hiwenet import extract as hiwenet
    from hiwenet import run_cli as CLI
    from hiwenet.pairwise_dist import metric_list, semi_metric_list
    from hiwenet.utils import HiwenetWarning
else:
    raise NotImplementedError('hiwenet supports only 2.7.13 or 3+. Upgrate to Python 3+ is recommended.')

list_weight_methods = metric_list + semi_metric_list
sys.dont_write_bytecode = True

from pytest import raises, warns, set_trace

dimensionality = np.random.randint(500, 2500)
num_groups = np.random.randint(2, 100)

def make_features(dimensionality, num_groups):
    num_links = int(num_groups*(num_groups-1)/2.0)

    group_ids_init = np.arange(num_groups)
    random_indices_into_groups = np.random.randint(0, num_groups, [1, dimensionality])
    groups = group_ids_init[random_indices_into_groups].flatten()

    group_ids = np.unique(groups)
    num_groups = len(group_ids)

    features = 10*np.random.random(dimensionality)

    return features, groups, group_ids, num_groups

features, groups, group_ids, num_groups = make_features(dimensionality, num_groups)
num_links = np.int64(num_groups * (num_groups - 1) / 2.0)
# for ix1, gid1 in enumerate(group_ids):
#     for ix2, gid2 in enumerate(group_ids):
#         if ix1 != ix2:
#             dist = cdist(features[groups==gid1],features[groups==gid2], 'euclidean')

cur_dir = os.path.dirname(abspath(__file__))

# the following are mostly usage tests. Refer to test_medpy.py for scientific validity of histogram metrics

def test_directed_mat():
    ew = hiwenet(features, groups, asymmetric=True)
    assert len(ew) == num_groups
    assert ew.shape[0] == num_groups and ew.shape[1] == num_groups

    # ensure no self-loops
    diag_elements = ew.flatten()[::num_groups+1]
    assert np.allclose(diag_elements, np.nan, equal_nan=True)

def test_directed_nx():
    graph = hiwenet(features, groups, asymmetric=True, return_networkx_graph=True)
    assert graph.is_directed()
    assert graph.number_of_nodes() == num_groups
    assert graph.number_of_edges() == 2*num_links

def test_dimensions():
    ew = hiwenet(features, groups)
    assert len(ew) == num_groups
    assert ew.shape[0] == num_groups and ew.shape[1] == num_groups

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

def test_too_few_groups():
    features, groups, group_ids, num_groups = make_features(100, 1)
    with raises(ValueError):
        ew = hiwenet(features, groups)

def test_too_few_values():
    features, groups, group_ids, num_groups = make_features(10, 500)
    with raises(ValueError):
        ew = hiwenet(features[:num_groups-1], groups)

def test_invalid_trim_perc():

    with raises(ValueError):
        ew = hiwenet(features, groups, trim_percentile= -1)

    with raises(ValueError):
        ew = hiwenet(features, groups, trim_percentile=101)

def test_invalid_edge_range():

    with raises(ValueError):
        ew = hiwenet(features, groups, edge_range= -1)

    with raises(ValueError):
        ew = hiwenet(features, groups, edge_range=[])

    with raises(ValueError):
        ew = hiwenet(features, groups, edge_range=[1, ])

    with raises(ValueError):
        ew = hiwenet(features, groups, edge_range=[1, 2, 3])

    with raises(ValueError):
        ew = hiwenet(features, groups, edge_range=(1, np.NaN))

    with raises(ValueError):
        ew = hiwenet(features, groups, edge_range=(2, 1))

def test_invalid_weight_method():

    with raises(NotImplementedError):
        ew = hiwenet(features, groups, weight_method= 'dkjz.jscn')

    with raises(NotImplementedError):
        ew = hiwenet(features, groups, weight_method= 'somerandomnamenoonewoulduse')

def test_trim_not_too_few_values():
    with raises(ValueError):
        ew = hiwenet( [0], [1], trim_outliers = False)

def test_trim_false_too_few_to_calc_range():
    with raises(ValueError):
        ew = hiwenet( [1], groups, trim_outliers = False)

def test_not_np_arrays():
    with raises(ValueError):
        ew = hiwenet(list(), groups, trim_percentile=101)

    with raises(ValueError):
        ew = hiwenet(features, list(), trim_percentile=101)

def test_invalid_nbins():
    with raises(ValueError):
        ew = hiwenet(features, groups, num_bins=np.NaN)

    with raises(ValueError):
        ew = hiwenet(features, groups, num_bins=np.Inf)

    with raises(ValueError):
        ew = hiwenet(features, groups, num_bins=2)

def test_return_nx_graph():
    nxG = hiwenet(features, groups, return_networkx_graph = True)
    assert isinstance(nxG, nx.Graph)
    assert nxG.number_of_nodes() == num_groups
    assert nxG.number_of_edges() == num_links

def test_extreme_skewed():
    # Not yet sure what to test for here!!
    ew = hiwenet(10+np.zeros(dimensionality), groups)


# CLI tests
def test_CLI_run():
    "function to hit the CLI lines."

    # first word is the script names (ignored)
    
    featrs_path = abspath(pjoin(cur_dir, '..', 'examples', 'features_1000.txt'))
    groups_path = abspath(pjoin(cur_dir, '..', 'examples', 'groups_1000.txt'))
    sys.argv = shlex.split('hiwenet -f {} -g {} -n 25'.format(featrs_path, groups_path))
    CLI()

def test_CLI_output_matches_API():
    " Ensuring results from the two interfaces matches within a tolerance"

    # turning groups into strings to correspond with CLI
    groups_str = np.array([str(grp) for grp in groups])

    featrs_path = abspath(pjoin(cur_dir, '..', 'examples', 'test_features.txt'))
    groups_path = abspath(pjoin(cur_dir, '..', 'examples', 'test_groups.txt'))
    result_path = abspath(pjoin(cur_dir, '..', 'examples', 'test_result.txt'))
    np.savetxt(featrs_path, features, fmt='%20.9f')
    np.savetxt(groups_path, groups,  fmt='%d')

    for weight_method in list_weight_methods:
        api_result = hiwenet(features, groups_str, weight_method=weight_method)

        sys.argv = shlex.split('hiwenet -f {} -g {} -o {} -w {}'.format(featrs_path, groups_path, result_path, weight_method))
        CLI()
        cli_result = np.genfromtxt(result_path, delimiter=',')

        if not bool(np.allclose(cli_result, api_result, rtol=1e-2, atol=1e-3, equal_nan=True)):
            raise ValueError('CLI results differ from API for {}'.format(weight_method))


def test_CLI_nonexisting_paths():
    "invalid paths"

    featrs_path = abspath(pjoin(cur_dir, '..', 'examples', 'features_1000.txt'))
    groups_path = 'NONEXISTING_groups_1000.txt'
    sys.argv = shlex.split('hiwenet -f {} -g {} -n 25'.format(featrs_path, groups_path))
    with raises(IOError):
        CLI()

    featrs_path = 'NONEXISTING_features_1000.txt'
    groups_path = abspath(pjoin(cur_dir, '..', 'examples', 'groups_1000.txt'))
    sys.argv = shlex.split('hiwenet -f {} -g {} -n 25'.format(featrs_path, groups_path))
    with raises(IOError):
        CLI()


def test_CLI_invalid_args():
    "invalid paths"

    featrs_path = 'NONEXISTING_features_1000.txt'
    # arg aaa or invalid_arg_name doesnt exist
    sys.argv = shlex.split('hiwenet --aaa {0} -f {0} -g {0}'.format(featrs_path))
    with raises(SystemExit):
        CLI()

    sys.argv = shlex.split('hiwenet --invalid_arg_name {0} -f {0} -g {0}'.format(featrs_path))
    with raises(SystemExit):
        CLI()


def test_CLI_too_few_args():
    "testing too few args"

    sys.argv = ['hiwenet ']
    with raises(SystemExit):
        CLI()

    sys.argv = ['hiwenet -f check']
    with raises(SystemExit):
        CLI()

    sys.argv = ['hiwenet -g check']
    with raises(SystemExit):
        CLI()


def test_input_callable():

    diff_skew = lambda x, y: abs(scipy.stats.skew(x)-scipy.stats.skew(y))
    ew = hiwenet(features, groups, weight_method=diff_skew)

    assert len(ew) == num_groups
    assert ew.shape[0] == num_groups and ew.shape[1] == num_groups


def test_input_callable_on_orig_data():

    diff_medians = lambda x, y: abs(np.median(x)-np.median(y))
    ew = hiwenet(features, groups, weight_method=diff_medians,
                 use_original_distribution=True)

    assert len(ew) == num_groups
    assert ew.shape[0] == num_groups and ew.shape[1] == num_groups

    # checking for invalid use : histogram metrics can not be applied on orig data (without building histograms)
    with warns(HiwenetWarning):
        ew = hiwenet(features, groups,
                     weight_method='manhattan',
                     use_original_distribution=True)

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
        assert nxg.number_of_nodes() == num_groups+1
        assert nxg.number_of_edges() == num_groups

        weights_from_nxg = np.array([ nxg[src]['whole']['weight'] for src in group_ids ])
        assert np.allclose(ew.flatten(), weights_from_nxg)



# test_directed_nx()
# test_directed_mat()
# test_CLI_output_matches_API()
# test_input_callable()

# test_more_metrics()

test_relative_to_all()