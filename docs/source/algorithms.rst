Algorithms
==========

Algorithms use a common ``solve`` interface and return solution objects with
runtime and result metadata. The repository contains implementations for item
assignment, batching, routing, integrated batching and routing, and scheduling.

Base interface
--------------

.. autoclass:: ware_ops_algos.algorithms.algorithm_interfaces.Algorithm
   :members:
   :show-inheritance:

Canonical configurable batching implementations
------------------------------------------------

.. autoclass:: ware_ops_algos.algorithms.batching.seed_batching.SeedBatching
   :members:
   :show-inheritance:

.. autoclass:: ware_ops_algos.algorithms.batching.local_search_batching.LocalSearchBatching
   :members:
   :show-inheritance:

Algorithm cards
---------------

Algorithm cards are stored under
``src/ware_ops_algos/algorithms/algorithm_cards``. They declare the subproblem,
objective, domain requirements, and parameters used to generate executable
algorithm configurations.
