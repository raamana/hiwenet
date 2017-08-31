# Histogram-weighted Networks (hiwenet)

[![status](http://joss.theoj.org/papers/df10a3a527fe169447a64c0cc810ff3c/status.svg)](http://joss.theoj.org/papers/df10a3a527fe169447a64c0cc810ff3c)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.839995.svg)](https://doi.org/10.5281/zenodo.839995)
[![travis](https://travis-ci.org/raamana/hiwenet.svg?branch=master)](https://travis-ci.org/raamana/hiwenet.svg?branch=master)
[![Code Health](https://landscape.io/github/raamana/hiwenet/master/landscape.svg?style=flat)](https://landscape.io/github/raamana/hiwenet/master)
[![codecov](https://codecov.io/gh/raamana/hiwenet/branch/master/graph/badge.svg)](https://codecov.io/gh/raamana/hiwenet)
[![PyPI version](https://badge.fury.io/py/hiwenet.svg)](https://badge.fury.io/py/hiwenet)


Histogram-weighted Networks for Feature Extraction and Advanced Analysis in Neuroscience

Network-level analysis of various features, esp. if it can be individualized for a single-subject, is proving to be quite a valuable tool in many applications. This package extracts single-subject (individualized, or intrinsic) networks from node-wise data by computing the edge weights based on histogram distance between the distributions of values within each node. Individual nodes could be an ROI or a patch or a cube, or any other unit of relevance in your application. This is a great way to take advantage of the full distribution of values available within each node, relative to the simpler use of averages (or another summary statistic). 

Rough scheme of computation is shown below:
![illustration](docs/illustration.png)

**Note on applicability** 

Although this technique was originally developed for cortical thickness, this is a generic and powerful technique that could be applied to any features such as gray matter density, PET uptake values, functional activation data or EEG features. All you need is a set of nodes/parcellation that has one-to-one correspondence across samples/subjects in your dataset.

## References

* A preprint outlining the use case: Raamana, P.R. and Strother, S.C., 2017, *Impact of spatial scale and edge weight on predictive power of cortical thickness networks* bioRxiv 170381 http://www.biorxiv.org/content/early/2017/07/31/170381. doi: https://doi.org/10.1101/170381 

## Installation

`pip install hiwenet`


As you continue to use or run into issues, try `pip install -U hiwenet` to get the latest bug-fixes, new features and optimized performance. 

## Usage

This package computes single-subject networks, hence you may need loop over samples/subjects in your dataset to extract them for all the samples/subjects, and them proceed to your subsequent analysis (such as classification etc).

A rough example of usage can be:

```python
from hiwenet import extract as hiwenet
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import nibabel
import os

# ----------------------------------------------------
# toy examples - modify for your application/need
my_project = '/data/myproject'
subject_list = ['a1', 'b2', 'c3', 'd4']
subject_labels = [1, 1, -1, -1]

num_subjects = len(subject_list)
# number of features (imaging vertex-wise cortical thickness values over the whole brain)
feature_dimensionality = 1000 
num_ROIs = 50
edge_weights = np.empty(num_subjects, num_ROIs*(num_ROIs-1)/2.0)

atlas = 'fsaverage'
# ----------------------------------------------------

def get_parcellation(atlas, parcel_param):
    "Placeholder to insert your own function to return parcellation in reference space."
    parc_path = os.path.join(atlas, 'parcellation_param{}.mgh'.format(parcel_param))
    parcel = nibabel.freesurfer.io.read_geometry(parc_path)
    
    return parcel


groups = get_parcellation(atlas, feature_dimensionality)

out_folder = os.path.join(my_project, 'hiwenet')

# choose a method from one among the three groups (metrics, semi-metrics and similarity functions)
metrics = [ 'manhattan', 'minowski', 'euclidean', 'noelle_2', 'noelle_4', 'noelle_5' ]

semi_metric_list = [
    'kullback_leibler', 'cosine_1', 
    'jensen_shannon', 'chi_square',
    'chebyshev', 'chebyshev_neg',
    'histogram_intersection_1',
    'relative_deviation', 'relative_bin_deviation',
    'noelle_1', 'noelle_3',
    'correlate_1']
similarity_func = ['correlate', 'cosine', 'cosine_2', 'cosine_alt', 'fidelity_based']



def get_features(subject_id):
    "Placeholder to insert your own function to read subject-wise features."
    features_path = os.path.join(my_project,'base_features', subject_id, 'features.txt')
    feature_vector = np.loadtxt(features_path)
    
    return feature_vector


def upper_tri_vec(matrix):
    "Returns the vectorized values of upper triangular part of a matrix"
    
    triu_idx = np.triu_indices_from(matrix, 1)
    return matrix[triu_idx]
    

for ss, subject in enumerate(subject_list):
  features = get_features(subject)
  edge_weights_subject = hiwenet(features, groups) # or add weight_method = metrics[ii] to use a various other metrics 
  edge_weights[ss,:] = upper_tri_vec(edge_weights_subject)
  
  out_file = os.path.join(out_folder, 'hiwenet_{}.txt'.format(subject))
  np.save(out_file, edge_weights_subject)
  
  
# proceed to analysis


# very rough example for training/evaluating a classifier
rf = RandomForestClassifier(oob_score = True)
scores = cross_val_score(rf, edge_weights, subject_labels)


```

## Citation

If you found it useful for your research, please cite it as:

 * Pradeep Reddy Raamana. (Version 2). Histogram-weighted Networks for Feature Extraction, Connectivity and Advanced Analysis in Neuroscience. Zenodo. http://doi.org/10.5281/zenodo.839995

## Support on Beerpay
Hey dude! Help me out for a couple of :beers:!

[![Beerpay](https://beerpay.io/raamana/hiwenet/badge.svg?style=beer-square)](https://beerpay.io/raamana/hiwenet)  [![Beerpay](https://beerpay.io/raamana/hiwenet/make-wish.svg?style=flat-square)](https://beerpay.io/raamana/hiwenet?focus=wish)
