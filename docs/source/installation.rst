Installation
============

.. code-block:: bash

    conda install -c axiom-data-science ioos_qc


Development and Testing
-----------------------

.. code-block:: bash

    conda create -c conda-forge -n ioosqc37 python=3.7
    conda activate ioosqc37
    conda install -c conda-forge --file requirements.txt --file tests/requirements.txt
