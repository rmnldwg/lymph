Getting started
===============

This package is meant to be a relatively simple-to-use frontend. The math is done under the hood and one does not need to worry about it a lot. Below are the things that are actually necessary.

Graph
-----

The model is based on the assumption that one can represent the lymphatic system as a directed graph. Hence, the first thing to do is to define a graph that represents the drainage pathways of the lymphatic system aptly.

Here, this is done via a dictionary:

.. code-block:: python
    :linenos:

    graph = {'lvl 1': ['lvl 2'], 
             'lvl 2': []}

For every key in the dictionary, the :class:`system` will create a :class:`node` that represents a binary random variable. The values in the dictionary should then be the a list of names to which :class:`edges` from the current key should be created.

For each :class:`node` there is one parameter that indicates the *base probability* :math:`b`, i.e. the probability that the primary tumor will infect this lymph node level (LNL). For each :class:`edge` there is then another parameter - the *transition probability* :math:`t` - that indicates the probability that the parent node, once involved, will spread to its daughter node.

The current implementation also supports trinary random variables for the :class:`node`, but that has not been tested yet.

Observations
------------

After having defined the graph, one needs to define how many observational modalities will be attached to each :class:`node` and what sensitivity :math:`s_N` and specificity :math:`s_P` they will have. This is done via a :math:`3D` :class:`numpy` array:

.. code-block:: python
    :linenos:

    obs_table = np.array([[[   1,   0], 
                           [   0,   1]], 
                           
                          [[0.75, 0.2], 
                           [0.25, 0.8]]])

It is basically an array of :math:`2D` matrices that are of size :math:`2\times M` where :math:`M` is the number of states a LNL can take on (usually that's 2). The columns of those matrices must add up to one, since - in the binary case - they are composed like this:

.. math::
    
    \begin{pmatrix}
    s_P & 1-s_N \\
    1-s_P & s_N
    \end{pmatrix}

with those two parameters - the :code:`graph` and the :code:`obs_table` - defined, everything is automatically set up and one can in principle start the risk prediction.

Data
----

However, one usually does not have meaningful parameters to this specific model lying around. Therefore, we first need to feed some data to it and then use a sampler of our choice (e.g. :code:`emcee`) to infer parameters.