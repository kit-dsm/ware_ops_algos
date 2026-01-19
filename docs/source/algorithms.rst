Algorithms
==========

The algorithms package provides optimization algorithms for warehouse operations, including order batching, routing, and sequencing.


Overview
--------

The algorithms package is organized into specialized subpackages:

* **ItemAssignment**: Assign order items to storage locations.
* **OrderSelection**: Select a subset of orders from a larger order pool. 
* **Batching**: Group orders into batches to optimize picker workload.
* **Routing**: Find optimal picking routes through the warehouse.
* **Sequencing**: Determine the order in which tasks should be executed.

Base Classes
------------

Algorithm
~~~~~~~~~

All algorithms inherit from the base ``Algorithm`` class:

.. autoclass:: ware_ops_algos.algorithms.algorithm.Algorithm
   :members:
   :special-members: __init__
   :show-inheritance:

**Key Methods:**

* ``solve(input_data)`` - Main entry point to run the algorithm
* ``_run(input_data)`` - Abstract method implemented by subclasses

Batching Algorithms
-------------------

Order batching algorithms group orders into batches for efficient picking.

Priority-Based Batching
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ware_ops_algos.algorithms.batching.batching.FifoBatching
   :members:
   :show-inheritance:

.. autoclass:: ware_ops_algos.algorithms.batching.batching.DueDateBatching
   :members:
   :show-inheritance:

.. autoclass:: ware_ops_algos.algorithms.batching.batching.RandomBatching
   :members:
   :show-inheritance:


Savings-Based Batching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ware_ops_algos.algorithms.batching.batching.ClarkAndWrightBatching
   :members:
   :show-inheritance:

Savings-Based Batching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ware_ops_algos.algorithms.batching.batching.SeedBatching
   :members:
   :show-inheritance:

Local Search Batching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ware_ops_algos.algorithms.batching.batching.LocalSearchBatching
   :members:
   :show-inheritance:


Routing Algorithms
------------------

Routing algorithms determine the path through pick locations.

.. automodule:: ware_ops_algos.algorithms.routing.routing
   :members:
   :undoc-members:
   :show-inheritance:





