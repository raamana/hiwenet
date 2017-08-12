import sys
import os
import shlex
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from hiwenet import extract as hiwenet
from hiwenet import __run as CLI

sys.dont_write_bytecode = True

from pytest import raises, warns, set_trace

dimensionality = 1000
num_groups = 5

def make_features(dimensionality, num_groups):
    num_links = int(num_groups*(num_groups-1)/2.0)

    group_ids_init = np.arange(num_groups)
    random_indices_into_groups = np.random.randint(0, num_groups, [1, dimensionality])
    groups = group_ids_init[random_indices_into_groups].flatten()

    group_ids = np.unique(groups)
    num_groups = len(group_ids)

    features = 1000*np.random.random(dimensionality)

    return features, groups, group_ids, num_groups

features, groups, group_ids, num_groups = make_features(dimensionality, num_groups)
num_links = np.int64(num_groups * (num_groups - 1) / 2.0)

# the following tests are mostly usage tests
# scientific validity of histogram metrics need to be achieved via testing of medpy
# DISCLAIMER: so results are only as valid as the tests in medpy.metric.histogram

def test_dimensions():
    ew = hiwenet(features, groups)
    assert len(ew) == num_groups
    assert ew.shape[0] == num_groups and ew.shape[1] == num_groups

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
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    featrs_path = os.path.abspath(os.path.join(cur_dir, '..', 'examples', 'features_1000.txt'))
    groups_path = os.path.abspath(os.path.join(cur_dir, '..', 'examples', 'groups_1000.txt'))
    sys.argv = shlex.split('hiwenet -f {} -g {} -n 25'.format(featrs_path, groups_path))
    CLI()

def test_CLI_nonexisting_paths():
    "invalid paths"

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    featrs_path = os.path.abspath(os.path.join(cur_dir, '..', 'examples', 'features_1000.txt'))
    groups_path = 'NONEXISTING_groups_1000.txt'
    sys.argv = shlex.split('hiwenet -f {} -g {} -n 25'.format(featrs_path, groups_path))
    with raises(IOError):
        CLI()

    featrs_path = 'NONEXISTING_features_1000.txt'
    groups_path = os.path.abspath(os.path.join(cur_dir, '..', 'examples', 'groups_1000.txt'))
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
