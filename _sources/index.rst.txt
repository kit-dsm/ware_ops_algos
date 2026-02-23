.. 4D4L documentation master file

=====================================
ware_ops_algos
=====================================

.. image:: https://img.shields.io/badge/python-3.11+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :alt: License

**ware_ops_algos** is a library that aims to provide standardized implementations of algorithms for warehouse operations
such as order batching, picker routing, item assignment, and picker scheduling.


Key Features
============

🏗️ **Extensible Domain Model**
   Structured domain models for layouts, orders, storage, and resources

🔧 **Algorithm Repository**
   Ready-to-use implementations for batching, routing, assignment, and sequencing

🎯 **Algorithm Cards**
   Annotation of algorithm properties and requirements in the form of algorithm cards


Domain Model
------------

The library uses a flexible domain model to represent warehouse systems. Instead of enforcing a rigid data structure, domain objects capture varying contextual information organized by distinct aspects of the warehouse system.

.. list-table:: Domain Objects
   :header-rows: 1
   :widths: 15 40 45

   * - Domain
     - Description
     - Key Attributes
   * - **Layout**
     - Spatial structure of the warehouse
     - Type (conventional/unconventional), parametric form or explicit graph
   * - **Articles**
     - Master data of warehouse items
     - Weight, volume, handling classes
   * - **Orders**
     - Customer orders to be fulfilled
     - Order lines, arrival time, due date
   * - **Resources**
     - Picking agents (human or robotic)
     - Travel speed, handling time, pick cart configuration
   * - **Storage**
     - Warehousing configuration
     - Storage policy, inventory levels, location coordinates

Algorithm Repository
--------------------

All algorithms inherit from an abstract ``Algorithm`` base class providing a unified interface through ``solve()``, which handles timing and error management.

::

    Algorithm (abstract)
    ├── ItemAssignment      # Resolve order lines to storage locations
    ├── Batching            # Group orders into picker tours
    ├── Routing             # Determine paths through the warehouse
    └── Scheduling          # Sequence and assign tours to pickers

**Item Assignment** — For scattered storage where articles may be stored at multiple locations. Approaches include greedy allocation and nearest-neighbor selection.

**Batching** — Groups orders into batches subject to capacity constraints. Includes priority-based construction heuristics, seed-based methods, savings-based approaches (Clarke & Wright), and local search metaheuristics.

**Routing** — Determines picker traversal paths. Includes layout-based heuristics (S-Shape, Return, Midpoint, Largest Gap), nearest neighbor, and exact methods (TSP, Ratliff-Rosenthal).

**Scheduling** — Assigns tours to pickers using list scheduling with priority dispatching rules (SPT, LPT, EDD, ERD).

Algorithm Cards
---------------

Each algorithm is annotated with requirements through YAML-based algorithm cards, enabling automated selection based on warehouse context:

.. code-block:: yaml

    model_name: RatliffRosenthal
    problem_type: routing
    description: >
      Implementation of DP for SPRP

    requirements:
      layout:
        type:
          - conventional
        features:
          - start_node
          - end_node
          - closest_node_to_start
          - min_aisle_position
          - max_aisle_position
          - n_pick_locations
          - dist_pick_locations
          - dist_bottom_to_pick_location
          - n_blocks
        constraints:
          n_blocks:
            equals: 1
      resources:
        type:
          - human
        features:
      orders:
        type:
          - standard
        features:
      storage:
        type:
          - any
        features:
          - x
          - y

    objective: Distance

    implementation:
      class_name: RatliffRosenthal
      solver:
      type: heuristic




Quick Start
=======================

.. grid:: 2

   .. grid-item-card:: 📘 Getting Started
      :link: ./examples/getting_started
      :link-type: doc

      Start here with an basic example to learn how to model a warehouse and use algorithms.

   .. grid-item-card:: 🔬 Algorithm Reference
      :link: algorithms
      :link-type: doc

      Complete catalog of implemented algorithms.

.. grid:: 2

   .. grid-item-card:: 📚 API Documentation
      :link: autoapi/index
      :link-type: doc

      Detailed API reference for all modules and classes.

   .. grid-item-card:: 💡 Examples
      :link: ./examples
      :link-type: doc

      Examples and benchmark evaluations.


Citation
========

If you use ware_ops_algos in your research, please cite:

.. code-block:: bibtex

   @misc{bischoff2026ware_ops_algos,
    author = {Bischoff, Janik and Suba, Oezge Nur and Barlang, Maximilian and Kutabi, Hadi and Mohring, Uta and Dunke, Fabian and Meyer, Anne and Nickel, Stefan and Furmans, Kai},
    title = {ware_ops_algos},
    year = {2026},
    publisher = {GitHub},
    journal = {GitHub Repository},
    howpublished = {\url{https://github.com/kit-dsm/ware_ops_algos.git}},
}


Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Support
=======

* 📧 Email: janik.bischoff@kit.edu