"""

# Histogram-weighted Networks (hiwenet)

"""

__all__ = [
    'ExtractConfig',
    'extract',
    'extract_with_config',
    'metric_list',
    'semi_metric_list',
    'metrics_on_original_features',
    'more_metrics',
    'non_pairwise',
    '__version__',
]

from importlib.metadata import PackageNotFoundError, version

from hiwenet import more_metrics, non_pairwise
from hiwenet.models import ExtractConfig
from hiwenet.pairwise_dist import (
    extract,
    extract_with_config,
    metric_list,
    semi_metric_list,
    metrics_on_original_features,
)

try:
    __version__ = version('hiwenet')
except PackageNotFoundError:
    __version__ = '0.6.0'
