import numpy as np
from hiwenet import hiwenet

dimensionality = 1000
num_groups = 10

random_indices_into_groups = np.random.randint(0, num_groups, [1, dimensionality])
group_ids = np.arange(num_groups)
groups = group_ids[random_indices_into_groups].flatten()

features = np.random.random(dimensionality)

vec = hiwenet(features, groups, weight_method = 'hist_int')