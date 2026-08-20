# Warehouse optimization algorithms for CASOP

`ware_ops_algos` provides the common domain model, benchmark data loaders,
algorithm implementations, algorithm cards, and problem taxonomy used by
[CASOP](https://github.com/kit-dsm/ware_ops_pipes), the framework presented in
*Context-Aware Synthesis of Optimization Pipelines for Warehouse Optimization*.

The repository contains executable algorithms for item assignment, batching,
routing, integrated batching and routing, and scheduling. Algorithm cards state
the subproblem, objective, domain requirements, and configuration parameters of
these implementations. `ware_ops_pipes` matches those cards to a data card and
composes compatible configurations into executable optimization pipelines.

## Repository structure

- `src/ware_ops_algos/domain_models/`: common warehouse domain model and data cards
- `src/ware_ops_algos/data_loaders/`: loaders for the benchmark formats
- `src/ware_ops_algos/algorithms/`: interfaces and implementations
- `src/ware_ops_algos/algorithms/algorithm_cards/`: algorithm cards and generated configurations
- `src/ware_ops_algos/domain_algo_mapper/`: matching of data cards and algorithm cards
- `src/ware_ops_algos/taxonomy/`: problem taxonomy
- `examples/`: runnable direct-use and extension examples
- `data/instances/`: small smoke-test instances only

## Installation

Python 3.11 or newer and [uv](https://docs.astral.sh/uv/) are recommended.

```bash
git clone https://github.com/kit-dsm/ware_ops_algos.git
cd ware_ops_algos
uv sync --frozen
```

Run the bundled example to verify the installation:

```bash
uv run --frozen python examples/getting_started.py
```

The example loads a small Foodmart instance, maps order positions to storage
locations, and batches the resulting warehouse orders. It uses heuristic
algorithms and does not solve a Gurobi model.

## Direct use

Algorithms expose a common `solve` method and return solution objects containing
the algorithm name, runtime, and problem-specific result. The complete runnable
example is [`examples/getting_started.py`](examples/getting_started.py).

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
uv run --frozen python examples/custom_batching.py
```

To expose a new implementation to CASOP, add its algorithm card under
`src/ware_ops_algos/algorithms/algorithm_cards/` and add the corresponding
CLS-Luigi component in `ware_ops_pipes`. The existing `FifoBatching`
implementation and `fifo_batching.yaml` card provide a compact example of this
pair.

## Gurobi

`gurobipy` is a project dependency because the repository contains exact routing
and integrated batching-routing implementations. Running those implementations
requires a Gurobi license that supports the model size. Loading data, using the
domain model, and running the heuristic examples above do not require an active
license. See Gurobi's [Python installation instructions](https://support.gurobi.com/hc/en-us/articles/360044290292-How-do-I-install-Gurobi-for-Python)
and [academic licensing information](https://support.gurobi.com/hc/en-us/articles/12684663118993-How-do-I-obtain-a-Gurobi-license).

## Benchmark files

Only small smoke-test instances are included. They originate from the benchmark
sets used in the CASOP paper:

- Foodmart: Valle et al. (2017), [doi:10.1016/j.ejor.2017.03.069](https://doi.org/10.1016/j.ejor.2017.03.069)
- HennWaescher: Henn et al. (2010), [doi:10.1007/BF03342717](https://doi.org/10.1007/BF03342717)
- MuterOencan: Muter and Öncan (2015), [doi:10.1080/0740817X.2014.991478](https://doi.org/10.1080/0740817X.2014.991478)
- SPRP: Heßler and Irnich (2024), [doi:10.1287/ijoc.2023.0075](https://doi.org/10.1287/ijoc.2023.0075)
- Kris: Briant et al. (2023), [arXiv:2303.17834](https://arxiv.org/abs/2303.17834)

The full collections are not included here. Their sources and terms are listed
in [`data/README.md`](data/README.md); the complete paper data preparation and
experiment workflow is maintained in `ware_ops_pipes`.

## License and citation

The source code is licensed under the BSD 3-Clause License. Third-party
benchmark files retain their original terms. Citation metadata are provided in
[`CITATION.cff`](CITATION.cff), and software authorship is recorded in
[`AUTHORS`](AUTHORS).
