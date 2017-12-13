# Histogram-weighted Networks (hiwenet)

[![status](http://joss.theoj.org/papers/df10a3a527fe169447a64c0cc810ff3c/status.svg)](http://joss.theoj.org/papers/df10a3a527fe169447a64c0cc810ff3c)
[![travis](https://travis-ci.org/raamana/hiwenet.svg?branch=master)](https://travis-ci.org/raamana/hiwenet.svg?branch=master)
[![Code Health](https://landscape.io/github/raamana/hiwenet/master/landscape.svg?style=flat)](https://landscape.io/github/raamana/hiwenet/master)
[![codecov](https://codecov.io/gh/raamana/hiwenet/branch/master/graph/badge.svg)](https://codecov.io/gh/raamana/hiwenet)
[![PyPI version](https://badge.fury.io/py/hiwenet.svg)](https://badge.fury.io/py/hiwenet)
[![Python versions](https://img.shields.io/badge/python-2.7%2C%203.5%2C%203.6-blue.svg)]

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1064012.svg)](https://doi.org/10.5281/zenodo.1064012)

Histogram-weighted Networks for Feature Extraction and Advanced Analysis in Neuroscience

Network-level analysis of various features, esp. if it can be individualized for a single-subject,
 is proving to be a valuable tool in many applications. Ability to extract the networks for a given subject individually on its own, would allow for feature extraction conducive to predictive modeling, unlike group-wise networks which can only be used for descriptive and explanatory purposes. This package extracts single-subject (individualized, or intrinsic) networks from node-wise data by computing the edge weights based on histogram distance between the distributions of values within each node. Individual nodes could be an ROI or a patch or a cube, or any other unit of relevance in your application. This is a great way to take advantage of the full distribution of values available within each node, relative to the simpler use of averages (or another summary statistic) to compare two nodes/ROIs within a given subject.

Rough scheme of computation is shown below:
![illustration](docs/illustration.png)

## Installation

`pip install -U hiwenet`

## Documentation

|||
|--:|---|
| Docs: |  http://hiwenet.readthedocs.io |
