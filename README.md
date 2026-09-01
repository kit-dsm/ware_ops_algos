# Warehouse optimization algorithms

`ware_ops_algos` is a collection of reusable algorithms for warehouse
optimization built around a common domain model. It provides domain objects,
generic layout construction, algorithm interfaces and implementations,
algorithm cards, visualization, and a problem taxonomy.

The repository contains executable algorithms for item assignment, batching,
routing, integrated batching and routing, and scheduling. Algorithm cards state
the subproblem, objective, domain requirements, and configuration parameters of
these implementations and support card-based discovery and configuration.

## Installation

Python 3.13 and [uv](https://docs.astral.sh/uv/) are required. The interpreter
version is recorded in `.python-version` and enforced by `pyproject.toml`; the
full dependency resolution is pinned in `uv.lock`.

```bash
git clone https://github.com/kit-dsm/ware_ops_algos.git
cd ware_ops_algos
uv sync --locked
```

The [getting-started notebook](examples/getting_started.ipynb) constructs a
warehouse directly with the domain model, matches algorithm cards, reproduces a
published single-picker routing example, and visualizes the optimal tour. Start
it with:

```bash
uv sync --locked --extra notebook
uv run --locked --extra notebook jupyter lab examples/getting_started.ipynb
```

The command-line examples progress from domain modeling to algorithm use and
extension:

```bash
uv run --locked python examples/model_domain.py
uv run --locked python examples/batch_orders.py
uv run --locked python examples/custom_batching.py
```

These examples construct their inputs directly and use heuristic algorithms,
so they do not require a Gurobi license.

## Direct use

Algorithms expose a common `solve` method and return solution objects containing
the algorithm name, runtime, and problem-specific result. The complete workflow
is shown in the [getting-started notebook](examples/getting_started.ipynb).

```python
assignment = GreedyItemAssignment(domain.storage)
assignment_solution = assignment.solve(domain.orders.orders)

batching = OrderNrFifoBatching(
    pick_cart=domain.resources.resources[0].pick_cart,
    articles=domain.articles,
)
batching_solution = batching.solve(assignment_solution.resolved_orders)
```

## Extending the algorithm repository

[`examples/custom_batching.py`](examples/custom_batching.py) implements a small
priority-based batching algorithm by subclassing `PriorityBatching`. It can be
run without changing the package:

```bash
uv run --locked python examples/custom_batching.py
```

To make a new implementation available through card-based discovery, implement
the appropriate algorithm interface and add its algorithm card under
`src/ware_ops_algos/algorithms/algorithm_cards/`. The existing `FifoBatching`
implementation and `fifo_batching.yaml` card provide a compact example.

## Gurobi

`gurobipy` is a project dependency because the repository contains exact routing
and integrated batching-routing implementations. Running those implementations
requires a Gurobi license that supports the model size. Loading data, using the
domain model, and running the heuristic examples above do not require an active
license. See Gurobi's [Python installation instructions](https://support.gurobi.com/hc/en-us/articles/360044290292-How-do-I-install-Gurobi-for-Python)
and [academic licensing information](https://support.gurobi.com/hc/en-us/articles/12684663118993-How-do-I-obtain-a-Gurobi-license).

## License and citation

The source code is licensed under the BSD 3-Clause License. Citation metadata are provided in
[`CITATION.cff`](CITATION.cff), and software authorship is recorded in
[`AUTHORS`](AUTHORS).
