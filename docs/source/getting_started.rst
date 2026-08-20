Getting started
===============

Install the locked environment and run the bundled heuristic example:

.. code-block:: bash

   uv sync --frozen
   uv run --frozen python examples/getting_started.py

The example loads a small Foodmart instance, performs item assignment, and
batches the resolved orders.

.. literalinclude:: ../../examples/getting_started.py
   :language: python

Extending a batching algorithm
------------------------------

The extension example subclasses ``PriorityBatching`` and implements its order
priority without modifying the package:

.. code-block:: bash

   uv run --frozen python examples/custom_batching.py

.. literalinclude:: ../../examples/custom_batching.py
   :language: python
