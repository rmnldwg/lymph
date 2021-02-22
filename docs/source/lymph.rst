.. module: lymph

.. _lymph:

Detailed API
============

The human lymph system (or rather parts of it) are modelled as directed graphs here. Hence, a :class:`System` consists of multiple :class:`Node` and :class:`Edge` instances, which are represented by a python class each.

Lymph system
------------

.. autoclass:: lymph.System
   :members:

Edge
----

Represents a lymphatic drainage pathway and therefore are spread probability.

.. autoclass:: lymph.Edge
   :members:

Node
----

Represents a lymph node level (LNL) or rather a random variable associated with it. It encodes the microscopic involvement of the LNL and - if involved - might spread along outgoing edges.

.. autoclass:: lymph.Node
   :members: