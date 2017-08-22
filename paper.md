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

Network-level analysis of various features, esp. if it can be individualized for a single-subject, is proving to be quite a valuable tool in many applications. This package extracts single-subject (individualized, or intrinsic) networks from node-wise data by computing the edge weights based on histogram distance between the distributions of values within each node. Individual nodes could be an ROI or a patch or a cube, or any other unit of relevance in your application. This is a great way to take advantage of the full distribution of values available within each node, relative to the simpler use of averages (or another summary statistic). 

Rough scheme of computation is shown below:
![illustration](docs/illustration.png)

# References

 - Raamana, P.R. and Strother, S.C., 2017, *Impact of spatial scale and edge weight on predictive power of cortical thickness networks* bioRxiv 170381 http://www.biorxiv.org/content/early/2017/07/31/170381. doi: https://doi.org/10.1101/170381 
