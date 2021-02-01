Installation
============

This package is a pure python implementation, but it is not (yet) on PyPI, so 
it must be installed from a local directory. To do this, first clone the 
github repository

.. code-block:: bash

    git clone https://github.com/rmnldwg/lymph.git
    cd lymph

From here you can either use `pip <http://www.pip-installer.org/>`_ 

.. code-block:: bash

    pip install .

or the file ``setup.py``

.. code-block:: bash

    python setup.py

.. note:: You will need to have `numpy <https://numpy.org/>`_ and 
    `pandas <https://pandas.pydata.org/>`_ installed and python 3.6 or higher.

In the future we plan to make the ``lymph`` package available on PyPI, so that 
it may be easier to install using `pip` or `conda <https://conda.io>`_.