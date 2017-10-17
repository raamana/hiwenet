---
title: 'Histogram-weighted Networks for Feature Extraction, Connectivity and Advanced Analysis in Neuroscience'
tags:
  - connectivity
  - neuroscience
  - graph
  - histogram
  - machine-learning
authors:
 - name: Pradeep Reddy Raamana
   orcid: 0000-0003-4662-0558
   affiliation: 1
 - name: Stephen C. Strother
   orcid: 0000-0002-3198-217X
   affiliation: 1, 2
affiliations:
 - name: Rotman Research Institute, Baycrest Health Sciences, Toronto, ON, Canada
   index: 1
 - name: Department of Medical Biophysics, University of Toronto, Toronto, ON, Canada
   index: 2
date: 21 August 2017
doi: 10.5281/zenodo.839995
bibliography: paper.bib
---

# Summary

Network-level analysis of various features, especially if it can be individualized for a single-subject, is proving to be a valuable tool in many applications[@Raamana170381; @evans2013networks; @palaniyappan2015abnormalities; @tijms2012similarity; @xu2017abnormalities; @raamana2015thickness; @lerch2006mapping; @he2007small]. This package extracts single-subject (individualized, or intrinsic) networks from node-wise data by computing the edge weights based on histogram distance between the distributions of values within each node. Individual nodes could be an ROI or a patch or a cube, or any other unit of relevance in your application. This is a great way to take advantage of the full distribution of values available within each node, relative to the simpler use of averages (or another summary statistic).

Rough scheme of computation is shown below:
![illustration](illustration.png)

## Note on applicability and target audience

Although this technique was originally developed for cortical thickness analysis in neuroimaging research, this is a generic and powerful technique that could be applied to any features such as gray matter density, PET uptake values, functional activation data, EEG features or any other domain. All that is needed is a set of nodes/parcellation that have one-to-one correspondence across samples/subjects in your dataset.

The target audience is users of almost all neuroimaging modalities who
 1. preprocessed dataset already,
 2. have some base features (node- or patch-wise, that are native to the given modality) extracted using other packages (metioned above), and
 3. who would like to analyze network-level or covariance-type or connectivity relations among the base features.

# What does the hiwenet package do?

 - This packages takes in vector of features and their membership labels (denoting which features belong to which groups - alternatively referred to as nodes in a graph), and computes their pair-wise histogram distances, using a chosen method.
 - This package is designed to be domain-agnostic, and hence a generic input format was chosen.
 - This `hiwenet.extract` could be used to extract advance covariance/connectome features in place of [MNE.extract_label_time_course](http://martinos.org/mne/stable/generated/mne.SourceEstimate.html#mne.SourceEstimate.extract_label_time_course) or nilearn.input_data.NiftiLabelsMasker.transform - see [here](http://nilearn.github.io/connectivity/functional_connectomes.html#extracting-signals-on-a-parcellation) and [here](http://nilearn.github.io/modules/generated/nilearn.input_data.NiftiLabelsMasker.html#nilearn.input_data.NiftiLabelsMasker.transform).
 - However, we plan to add interfaces to tools e.g. via a scikit-learn compatible API/interface is also in the works. Stay tuned.
 - Refer to [examples](../examples) directory and the [docs](hiwenet.readthedocs.io) for more detailed and usage examples.

## requirements

`hiwenet` package is built on the following packages: [@oliphant2007python; @walt2011numpy; @hagberg2008exploring; @oskar_maier_2017_886618]

# References
