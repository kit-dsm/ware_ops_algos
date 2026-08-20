Examples
========

This page is a map of the examples rather than a copy of their source code.
The main walkthrough is the :doc:`getting-started notebook <getting_started>`:
it builds a small warehouse, exposes the values that the algorithms consume,
matches algorithm cards, solves a routing problem, and draws the result.

The shorter Python scripts then isolate one idea at a time. The extension
example has its own page under :doc:`extending`, so it is linked here instead
of being repeated.

The main walkthrough: from warehouse facts to a route
------------------------------------------------------

Open ``examples/getting_started.ipynb`` after installing the notebook extra:

.. code-block:: bash

   uv run --locked --extra notebook jupyter lab examples/getting_started.ipynb

The notebook reproduces the published three-aisle single-picker routing
example. Its inputs are deliberately concrete:

* a conventional layout with **3 aisles**, **10 pick locations per aisle**, and
  a depot at **(3, 0)**;
* **9 articles** stored at **9 named pick nodes** and one unit-demand order;
* one human picker with a **9-item cart**;
* a domain object assembled from layout, articles, orders, resources, storage,
  and warehouse information.

The notebook makes the hand-off visible in four stages:

1. It plots the warehouse graph and highlights the depot and pick nodes.
2. It shows the domain-component types and feature counts, then turns the
   domain into a data card and lists the matching algorithm cards.
3. Greedy item assignment resolves the order into physical pick nodes, which
   are shown in a table.
4. Ratliff--Rosenthal computes the route and the notebook checks the published
   action sequence ``1pass, 11, 1pass, 22, top, 00`` and the total distance
   **36**. A second plot overlays the optimal tour, with line width showing
   whether a segment is traversed once or twice.

This is the useful starting point for changing the problem: alter the layout,
storage policy, order, or picker and observe which domain values, cards, pick
nodes, and route properties change. It demonstrates the algorithm library's
domain model without relying on a benchmark loader or duplicating the custom
algorithm extension.

Small runnable examples
-----------------------

The companion scripts use the same domain objects with smaller outputs:

.. list-table::
   :header-rows: 1
   :widths: 24 40 36

   * - Example
     - Question it answers
     - Observable result
   * - ``model_domain.py``
     - What does a minimal warehouse domain contain?
     - A 2-aisle layout, 4 articles, 4 orders, and cart capacities of 6 items,
       10 weight units, and 10 volume units.
   * - ``batch_orders.py``
     - How do resolved orders become feasible batches?
     - FIFO produces two batches: orders ``[1, 2]`` use 5 items, 8 weight, and
       5 volume; orders ``[3, 4]`` use 6 items, 10 weight, and 10 volume.

Run them from the repository root:

.. code-block:: bash

   uv run --locked python examples/model_domain.py
   uv run --locked python examples/batch_orders.py

The scripts assert the important invariant—the batches fit the pick cart—and
print the values so a failure is explainable. They are useful when debugging a
small domain in a terminal; the notebook is the better place to inspect the
relationships visually.

Where the extension belongs
---------------------------

Once the domain and batching flow are clear, see :doc:`extending` for the
separate ``custom_batching.py`` example. It demonstrates how to subclass an
algorithm interface and change order priority; it is intentionally kept apart
from this page so the getting-started example remains about modeling,
compatibility, values, and route behavior.
