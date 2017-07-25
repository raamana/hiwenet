# Histogram-weighted Networks (hiwenet)

Histogram-weighted Networks for Feature Extraction and Advance Analysis in Neuroscience

This package extracts single-subject (individualized, or intrinsic) networks from node-wise (ROI-wise, patch-wise or or another way to identify different graph nodes) by computing the edge weights based on histogram distance between the distributions of values within each node (or ROI or patch or cube). This is a great way to take advantage of the full distribution available within each node, compared to simply averaging it and then using it. 

A publication outlining one use case is here:
[Raamana, P.R. and Strother, S.C., 2016, June. Novel histogram-weighted cortical thickness networks and a multi-scale analysis of predictive power in Alzheimer's disease. In Pattern Recognition in Neuroimaging (PRNI), 2016 International Workshop on (pp. 1-4). IEEE.](http://ieeexplore.ieee.org/abstract/document/7552334/)

Another poster describing it can be found here: https://doi.org/10.6084/m9.figshare.5241616

Rough scheme of computation is shown below:
