--------------------------------------------------------------------------------------------------
About
--------------------------------------------------------------------------------------------------

Network-level analysis of various features, esp. if it can be individualized for a single-subject, is proving to be quite a valuable tool in many applications. This package extracts single-subject (individualized, or intrinsic) networks from node-wise data by computing the edge weights based on histogram distance between the distributions of values within each node. Individual nodes could be an ROI or a patch or a cube, or any other unit of relevance in your application. This is a great way to take advantage of the full distribution of values available within each node, relative to the simpler use of averages (or another summary statistic).

Rough scheme of computation is shown below:

.. image:: illustration.png

Applicability
-------------

Although this technique was originally developed for cortical thickness, this is a generic and powerful technique that could be applied to any features such as gray matter density, PET uptake values, functional activation data or EEG features. All that is needed is a set of nodes/parcellation that have one-to-one correspondence across samples/subjects in your dataset.

The target audience is users of almost all neuroimaging modalities who:

    1) preprocessed dataset already,
    2) have some base features extracted (node- or patch-wise, that are native to the given modality) using other packages (metioned above), and
    3) who would like to analyze network-level (i.e. covariance-type or connectivity) relations among the base features (either in space across the cortex or a relevant domain, or across time).
    4) This is similar to popular metrics of covariance like Correlation or LedoitWolf, and could be dropped in their place. Do you want to find out how histogram-based method compare to your own ideas?


Check the :doc:`usage_api` and :doc:`API` pages, and let me know your comments.

Thanks for checking out. Your feedback will be appreciated.

Acknowledgements
----------------

I would like to thank Oscar Esteban (@oesteban) for his volunteering of significant effort and time to review this package in great detail and offer constructive feedback, which has been very helpful in improving the software.