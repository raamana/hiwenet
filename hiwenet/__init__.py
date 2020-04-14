"""

# Histogram-weighted Networks (hiwenet)

"""

__all__ = ['extract', 'pairwise_dist', 'run_cli', 'more_metrics', 'non_pairwise']

from sys import version_info

if version_info.major > 2:
    from hiwenet import more_metrics, non_pairwise
    from hiwenet.pairwise_dist import extract, run_cli
else:
    raise NotImplementedError('hiwenet supports only Python 3 or higher!')

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
