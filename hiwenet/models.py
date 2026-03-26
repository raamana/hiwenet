from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


WeightFunction = Callable[[np.ndarray, np.ndarray], float]


@dataclass(frozen=True, slots=True)
class ExtractConfig:
    weight_method: str | WeightFunction = "manhattan"
    num_bins: int = 25
    edge_range: tuple[float, float] | None = None
    trim_outliers: bool = True
    trim_percentile: float = 5.0
    use_original_distribution: bool = False
    relative_to_all: bool = False
    asymmetric: bool = False
    return_networkx_graph: bool = False


@dataclass(frozen=True, slots=True)
class PreparedInputs:
    features: np.ndarray
    groups: np.ndarray
    group_ids: np.ndarray
    num_groups: int
    num_links: int
    num_bins: int
    edge_range: tuple[float, float] | None


@dataclass(frozen=True, slots=True)
class ResolvedMethod:
    weight_func: WeightFunction
    use_orig_distr: bool
    non_symmetric: bool
