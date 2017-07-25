# Histogram-weighted Networks (hiwenet)

Histogram-weighted Networks for Feature Extraction and Advance Analysis in Neuroscience

This package extracts single-subject (individualized, or intrinsic) networks from node-wise (ROI-wise, patch-wise or or another way to identify different graph nodes) by computing the edge weights based on histogram distance between the distributions of values within each node (or ROI or patch or cube). This is a great way to take advantage of the full distribution available within each node, compared to simply averaging it and then using it. 

Rough scheme of computation is shown below:
![illustration](docs/illustration.png)

**Note on applicability** 
Although this technique was originally developed for cortical thickness, this is a generic and powerful technique that could be applied to any features such as gray matter density, PET uptake values, functional activation data or EEG features. All you need is a set of nodes/parcellation that has one-to-one correspondence across samples/subjects in your dataset.

## References
A publication outlining one use case is here:
[Raamana, P.R. and Strother, S.C., 2016, June. Novel histogram-weighted cortical thickness networks and a multi-scale analysis of predictive power in Alzheimer's disease. In Pattern Recognition in Neuroimaging (PRNI), 2016 International Workshop on (pp. 1-4). IEEE.](http://ieeexplore.ieee.org/abstract/document/7552334/)

Another poster describing it can be found here: https://doi.org/10.6084/m9.figshare.5241616

## Installation

`pip install -U hiwenet`

## Usage

This package computes single-subject networks, hence you may need loop over samples/subjects in your dataset to extract them for all the samples/subjects, and them proceed to your subsequent analysis (such as classification etc).

A rough example of usage can be:

```python
import hiwenet
from sklearn.ensemble import RandomForestClassifier
import numpy as np

num_subjects = len(subject_list)
num_ROIs = 50
edge_weights = np.empty(num_subjects, num_ROIs*(num_ROIs-1)/2.0)

out_folder = os.path.join(my_project, 'hiwenet')

for ss, subject in enumerate(subject_list):
  features = get_features(subject)
  edge_weights_subject = hiwenet(features)
  edge_weights[ii,:] = edge_weights_subject
  
  out_file = os.path.join(out_folder, 'hiwenet_{}.txt'.format(subject))
  np.save(out_file, edge_weights_subject)
  
  
# proceed to analysis

# very rough example for training a classifier (this is not cross validation)
rf = RandomForestClassifier(oob_score = True)
rf.fit(edge_weights, train_labels)
oob_error_train = rf.oob_score_


```
