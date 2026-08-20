Examples
========

The examples progress from constructing a warehouse domain directly in Python
to solving and extending a batching problem.

1. Model a warehouse domain
---------------------------

``model_domain.py`` defines a conventional layout, articles, storage locations,
orders, and a picker with item, weight, and volume capacities.

.. code-block:: bash

   uv run --frozen python examples/model_domain.py

.. literalinclude:: ../../examples/model_domain.py
   :language: python

2. Batch orders subject to cart capacities
-------------------------------------------

``batch_orders.py`` resolves the order positions, applies FIFO batching, and
checks that every generated batch respects all three cart capacities.

.. code-block:: bash

   uv run --frozen python examples/batch_orders.py

.. literalinclude:: ../../examples/batch_orders.py
   :language: python

3. Implement a batching rule
-----------------------------

The final example subclasses the common batching interface and changes the
order priority while reusing the same domain.

.. code-block:: bash

   uv run --frozen python examples/custom_batching.py

.. literalinclude:: ../../examples/custom_batching.py
   :language: python
