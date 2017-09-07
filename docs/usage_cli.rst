
Command line interface
-----------------------

The command line interface for hiwenet (although I recommend using it via API) is shown below. Check the bottom of this page for examples.

.. argparse::
   :ref: hiwenet.__get_parser
   :prog: hiwenet
   :nodefault:
   :nodefaultconst:

A rough example of usage can be:

.. code-block:: bash

    #!/bin/bash
    #$ -l mf=4G -q abaqus.q -wd /work/project/PBS -j yes -o /work/project/output/job.log
    cd /work/project/output
    hiwenet -f thickness/features_1000.txt -g thickness/groups_1000.txt -w manhattan -n 50 -o thickness/hiwenet_manhatten_n50.csv


The default behaviour of hiwenet is to trim the outliers, as I suspect their existence in the feature distributions of different ROIs. But if you choose not to do it, you can disable it like this with ``-t False`` flag:


.. code-block:: bash

    #!/bin/bash
    #$ -l mf=4G -q abaqus.q -wd /work/project/PBS -j yes -o /work/project/output/job.log
    cd /work/project/output
    hiwenet -f thickness/features_1000.txt -g thickness/groups_1000.txt -w manhattan -n 50 -t False -o thickness/hiwenet_manhatten_n50.csv

