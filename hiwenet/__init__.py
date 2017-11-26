"""

# Histogram-weighted Networks (hiwenet)

"""

__all__ = ['extract', 'pairwise_dist', 'run_cli', 'more_metrics', 'non_pairwise']

from sys import version_info

if version_info.major==2 and version_info.minor==7:
    import more_metrics
    import non_pairwise
    from pairwise_dist import extract, run_cli
elif version_info.major > 2:
    from hiwenet import more_metrics, non_pairwise
    from hiwenet.pairwise_dist import extract, run_cli
else:
    raise NotImplementedError('hiwenet supports only 2.7 or 3+. Upgrade to Python 3+ is recommended.')

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
