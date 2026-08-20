Extending the algorithms
========================

The extension example subclasses ``PriorityBatching`` and implements its order
priority without modifying the package. It reuses the manually constructed
domain from ``model_domain.py``:

.. code-block:: bash

   uv run --frozen python examples/custom_batching.py

.. literalinclude:: ../../examples/custom_batching.py
   :language: python
