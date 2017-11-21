"""

Script to analyze the relationships among different histogram distances,
    at different dimensionalities with different number of bins.

"""
import os
import pickle
from os.path import join as pjoin, exists as pexists

import numpy as np
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from medpy.metric import histogram

metric_list = ['manhattan', 'minowski', 'euclidean', 'noelle_2', 'noelle_4', 'noelle_5']
metric_list_to_doublecheck = ['cosine_1']
unknown_property = ['histogram_intersection']
still_under_dev  = ['quadratic_forms']
similarity_funcs = ['correlate', 'cosine', 'cosine_2', 'cosine_alt', 'fidelity_based']
semi_metric_list = ['jensen_shannon', 'chi_square', 'chebyshev', 'chebyshev_neg',
                    'histogram_intersection_1', 'relative_deviation', 'relative_bin_deviation',
                    'noelle_1', 'noelle_3', 'correlate_1']
excluded = ['kullback_leibler', ]

# all_methods = metric_list + similarity_funcs + semi_metric_list
# sticking to fewer methods with
all_methods = np.array(['chebyshev', 'chi_square', 'correlate', 'cosine', 'euclidean',
                           'histogram_intersection', 'jensen_shannon', 'manhattan', 'minowski',  'relative_deviation'])
all_methods.sort()

default_feature_dim = 1000
default_num_bins = 20
default_stdev = 0.1

range_dim = np.logspace(1, 3, num=20, dtype=int) # range(10, 100000, 1000)
range_num_bins = np.array(list(range(5, 200, 20)))
num_trials = 25

num_methods = len(all_methods)
num_dimensions = len(range_dim)
num_num_bins = len(range_num_bins)

num_rows = 3  # 7
num_cols = 7  # 3
fig_size = [14, 18]
num_ticks = 10

range_separability = [-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75]
center_one = 0.0

edge_range = (-1.0, 1.0)
common_bin_edges = lambda nb: np.linspace(edge_range[0], edge_range[1], nb)

num_sep = len(range_separability)

rng = np.random.RandomState(654321)

def get_distr(center=0.0, stdev=default_stdev, length=50):
    "Returns a PDF of a given length. "
    # distr = np.random.random(length)

    # sticking to normal distibution to easily control separability
    distr = rng.normal(center, stdev, size=[length, 1])

    return distr


def make_random_histogram(center=0.0, stdev=default_stdev, length=default_feature_dim, num_bins=default_num_bins):
    "Returns a sequence of histogram density values that sum to 1.0"

    hist, bin_edges = np.histogram(get_distr(center, stdev, length),
                                   range=edge_range, bins=num_bins, density=True)
    # to ensure they sum to 1.0
    hist = hist / sum(hist)

    if len(hist) < 2:
        raise ValueError('Invalid histogram')

    return hist, bin_edges


def dist_betn_rand_hist(method, center_one, center_two, feat_dim, num_bins):

    h1, _ = make_random_histogram(center_one, length=feat_dim, num_bins=num_bins)
    h2, _ = make_random_histogram(center_two, length=feat_dim, num_bins=num_bins)

    return method(h1, h2)


# -------------------

print('range of dimensions : {}'.format(range_dim))

out_dir = '/data1/strother_lab/praamana/dist_compare_hiwenet_lowerdim_finegrain'
if not pexists(out_dir):
    os.mkdir(out_dir)

out_dir_analysis = '/data1/strother_lab/praamana/dist_compare_hiwenet_lowerdim_finegrain/analysis'
if not pexists(out_dir_analysis):
    os.mkdir(out_dir_analysis)

print('Computation ...')
for separability in range_separability:
    print('separability : {}'.format(separability))

    center_two = center_one + separability

    expt_id = 'separability{}_stdev{}_distances_dim_num_bins'.format(separability, default_stdev)
    saved_path = pjoin(out_dir,'{}.pkl'.format(expt_id))
    try:
        if pexists(saved_path):
            print('reading from disk')
            with open(saved_path, 'rb') as df:
                distances = pickle.load(df)
        else:
            raise IOError(' no distaces saved ')
    except:
        print('recomputing.')
        distances = np.full([num_methods, num_dimensions, num_num_bins, num_trials ], np.nan)

        for mm, method_str in enumerate(all_methods):
            method = getattr(histogram, method_str)

            print('Analyzing {} ... dim : '.format(method_str))
            for dd, feat_dim in enumerate(range_dim):
                print(' {} '.format(feat_dim), end='')
                for nn, num_bins in enumerate(range_num_bins):
                    distances[mm, dd, nn, :] = Parallel(n_jobs=num_trials)(delayed(dist_betn_rand_hist)(method, center_one, center_two, feat_dim, num_bins) for tt in range(num_trials))

            print(' .. Done.')

        with open(saved_path, 'wb') as df:
            pickle.dump(distances, df)

    print('computation for separability {} done.'.format(separability))

distances = np.full([num_sep, num_methods, num_dimensions, num_num_bins, num_trials ], np.nan)
for ss, separability in enumerate(range_separability):
    expt_id = 'separability{}_stdev{}_distances_dim_num_bins'.format(separability, default_stdev)
    saved_path = pjoin(out_dir,'{}.pkl'.format(expt_id))
    with open(saved_path, 'rb') as df:
        distances[ss, :, :, :, :] = pickle.load(df)

# ---------
print('Generating figures now... ')
for ss, separability in enumerate(range_separability):

    fig, ax = plt.subplots(num_rows, num_cols, figsize=fig_size)

    expt_id = 'separability{}_stdev{}_distances_dim_num_bins'.format(separability, default_stdev)

    tmp_vec = distances[ss, :, :, :, :].flatten()
    clim_max = np.percentile(tmp_vec, 99.5)
    clim_min = np.percentile(tmp_vec, 0.05)
    for mm, method_str in enumerate(all_methods):
        img_data = np.squeeze(np.median(distances[ss, mm, :, :, :], axis=2))

        ax = plt.subplot(num_rows, num_cols, mm + 1)
        img = plt.imshow(img_data) # , vmin=clim_min, vmax=clim_max)

        plt.title('{}'.format(method_str))
        tick_loc_dim = range(1, num_dimensions, int(np.floor(num_dimensions/num_ticks)))
        tick_loc_bins = range(1, num_num_bins, int(np.floor(num_num_bins/num_ticks)))

        if mm in range(0, num_methods, num_cols):
            xtick_loc = tick_loc_dim
            xtick_str = range_dim[tick_loc_dim]
        else:
            xtick_loc = []
            xtick_str = []

        ax.set_yticks(xtick_loc)
        ax.set_yticklabels(xtick_str)

        if mm > num_methods-num_cols-1:
            ytick_loc = tick_loc_bins
            ytick_str = range_num_bins[tick_loc_bins]
        else:
            ytick_loc = []
            ytick_str = []

        ax.set_xticks(ytick_loc)
        ax.set_xticklabels(ytick_str, rotation='vertical')

        if mm == 0:
            plt.ylabel('dimensionality ')

    plt.xlabel('num bins')
    plt.suptitle('separability : {}'.format(separability))
    out_fig_path = pjoin(out_dir_analysis, 'fig_{}.pdf'.format(expt_id))
    fig.savefig(out_fig_path, dpi=300)
    plt.close()
    # plt.show(block=False)


# -----------------

num_rows2, num_cols2 = 2, 4
for mm, method_str in enumerate(all_methods):
    fig, ax = plt.subplots(num_rows2, num_cols2, figsize=fig_size)

    tmp_vec = distances[:, mm, :, :, :].flatten()
    clim_max = np.percentile(tmp_vec, 99.5)
    clim_min = np.percentile(tmp_vec, 0.05)

    for ss, separability in enumerate(range_separability):
        img_data = np.squeeze(np.median(distances[ss, mm, :, :, :], axis=2))
        ax = plt.subplot(num_rows2, num_cols2, ss + 1)
        img = plt.imshow(img_data, vmin=clim_min, vmax=clim_max)
        plt.title('sep = {}'.format(separability))

    plt.suptitle('method : {}'.format(method_str))
    out_fig_path = pjoin(out_dir_analysis, 'method_{}_diff_seprabilities.pdf'.format(method_str))
    fig.savefig(out_fig_path, dpi=300)
    plt.close()
