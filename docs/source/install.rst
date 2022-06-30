Installation
============

The easiest way to install it is via pip. Note that due to a name clash, on the python packaging index the package isn't called ``lymph``, but ``lymph-model``:

.. code-block:: bash

    pip install lymph-model

From Source
-----------

To install the package from the source repository, start by cloning it.

.. code-block:: bash

    git clone https://github.com/rmnldwg/lymph.git
    cd lymph

From here you can either use `pip <http://www.pip-installer.org/>`_

.. code-block:: bash

    pip install .

or the file ``setup.py``

.. code-block:: bash

    python setup.py

.. note:: You will need to have  python 3.8 or higher installed.

In the future we plan to make the ``lymph`` package available on PyPI, so that it may be easier to install using `pip` or `conda <https://conda.io>`_.