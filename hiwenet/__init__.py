"""

# Histogram-weighted Networks (hiwenet)

"""

from sys import version_info

if version_info.major==2 and version_info.minor==7:
    from hiwenet import extract, run_cli
elif version_info.major > 2:
    from hiwenet.hiwenet import extract, run_cli
else:
    raise NotImplementedError('hiwenet supports only 2.7.13 or 3+. Upgrade to Python 3+ is recommended.')

__all__ = ['extract', 'hiwenet', 'run_cli']
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
