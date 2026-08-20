Examples
========

The examples progress from constructing a warehouse domain directly in Python
to solving and extending a batching problem. The complete getting-started
notebook is the main walkthrough: it builds a published three-aisle instance,
shows the domain-component values and matched algorithm cards, resolves the
pick positions, verifies the optimal route distance of 36, and visualizes both
the layout and the optimal tour.

Getting started: values, algorithm cards, and a route visualization
-------------------------------------------------------------------

Open ``examples/getting_started.ipynb`` after installing the notebook extra:

.. code-block:: bash

   uv run --locked --extra notebook jupyter lab examples/getting_started.ipynb

The notebook is linked into the documentation as the Getting Started page.
It is intentionally a domain-model walkthrough rather than another algorithm
extension: the printed tables expose the input components, the card-matching
result, the physical pick nodes, and the expected action sequence before the
final plot explains how the dynamic program traverses the warehouse.

1. Model a warehouse domain
---------------------------

``model_domain.py`` defines a conventional layout, articles, storage locations,
orders, and a picker with item, weight, and volume capacities.

.. code-block:: bash

   uv run --locked python examples/model_domain.py

.. literalinclude:: ../../examples/model_domain.py
   :language: python

2. Batch orders subject to cart capacities
-------------------------------------------

``batch_orders.py`` resolves the order positions, applies FIFO batching, and
checks that every generated batch respects all three cart capacities.

.. code-block:: bash

   uv run --locked python examples/batch_orders.py

.. literalinclude:: ../../examples/batch_orders.py
   :language: python

3. Implement a batching rule
-----------------------------

The final example subclasses the common batching interface and changes the
order priority while reusing the same domain.

.. code-block:: bash

   uv run --locked python examples/custom_batching.py

.. literalinclude:: ../../examples/custom_batching.py
   :language: python
