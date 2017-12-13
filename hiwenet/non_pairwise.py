"""
Module collecting all the special methods that fit in the category of inter-regional distance metrics, but not pair-wise.

"""

__all__ = ['relative_to_all', ]

import numpy as np
import networkx as nx
from sys import version_info
if version_info.major==2 and version_info.minor==7:
    from .utils import compute_histogram
elif version_info.major > 2:
    from hiwenet.utils import compute_histogram
else:
    raise NotImplementedError('hiwenet supports only 2.7 or 3+. Upgrade to Python 3+ is recommended.')


def relative_to_all(features, groups, bin_edges, weight_func,
                    use_orig_distr,
                    group_ids, num_groups,
                    return_networkx_graph, out_weights_path):
    """
    Computes the given function (aka weight or distance) between histogram from each of the groups to a "grand histogram" derived from all groups.

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

    bin_edges : list or ndarray
        Array of bin edges within which to compute the histogram in.

    weight_func : callable
        Function to compute the edge weight between groups/nodes.

    use_orig_distr : bool, optional
        When using a user-defined callable, this flag
        1) allows skipping of pre-processing (trimming outliers) and histogram construction,
        2) enables the application of arbitrary callable (user-defined) on the original distributions coming from the two groups/ROIs/nodes directly.

        Example: ``diff_in_medians = lambda x, y: abs(np.median(x)-np.median(y))``

        This option is valid only when weight_method is a valid callable,
            which must take two inputs (possibly of different lengths) and return a single scalar.

    group_ids : list
        List of unique group ids to construct the nodes from (must all be present in the `groups` argument)

    num_groups : int
        Number of unique groups in the `group_ids`

    return_networkx_graph : bool, optional
        Specifies the need for a networkx graph populated with weights computed. Default: False.

    out_weights_path : str, optional
        Where to save the extracted weight matrix. If networkx output is returned, it would be saved in GraphML format.
        Default: nothing saved unless instructed.

    Returns
    -------
    distance_vector : ndarray
        vector of distances between the grand histogram and the individual ROIs

    Raises
    ------
    ValueError
        If one or more of the arrays are empty.

    """

    # notice the use of all features without regard to group membership
    hist_whole = compute_histogram(features, bin_edges, use_orig_distr)

    # to identify the central node capturing distribution from all roi's
    whole_node = 'whole'

    if return_networkx_graph:
        graph = nx.Graph()
        graph.add_nodes_from(group_ids)
        graph.add_node(whole_node)
    else:
        edge_weights = np.full([num_groups, 1], np.nan)

    for src in range(num_groups):
        index_roi = groups == group_ids[src]
        hist_roi = compute_histogram(features[index_roi], bin_edges, use_orig_distr)
        edge_value = weight_func(hist_whole, hist_roi)
        if return_networkx_graph:
            graph.add_edge(group_ids[src], whole_node, weight=float(edge_value))
        else:
            edge_weights[src] = edge_value

    if return_networkx_graph:
        if out_weights_path is not None:
            graph.write_graphml(out_weights_path)
        return graph
    else:
        if out_weights_path is not None:
            np.savetxt(out_weights_path, edge_weights, delimiter=',', fmt='%.9f')
        return edge_weights



