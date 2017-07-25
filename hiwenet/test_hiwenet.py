import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from hiwenet import hiwenet

dimensionality = 1000
num_groups = 5
num_links = int(num_groups*(num_groups-1)/2.0)

random_indices_into_groups = np.random.randint(0, num_groups, [1, dimensionality])
group_ids = np.arange(num_groups)
groups = group_ids[random_indices_into_groups].flatten()

features = 1000*np.random.random(dimensionality)

G = hiwenet(features, groups, weight_method = 'histogram_intersection')


