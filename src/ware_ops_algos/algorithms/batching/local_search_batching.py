import time
from dataclasses import asdict

from ware_ops_algos.algorithms.algorithm_interfaces import (
    BatchObject,
    BatchingSolution,
    WarehouseOrder,
)
from ware_ops_algos.algorithms.batching.batching import Batching
from ware_ops_algos.algorithms.batching.moves import (
    LocalSearchStatistics,
    Neighborhood,
    ShiftNeighborhood,
    SwapNeighborhood,
    _SearchBatch,
    _SearchState,
)
from ware_ops_algos.algorithms.routing.routing import Routing
from ware_ops_algos.domain_models import PickCart, Articles


class LocalSearchBatching(Batching):
    """First-improvement local search over modular swap/shift neighborhoods."""

    def __init__(
        self,
        pick_cart: PickCart,
        articles: Articles,
        routing_class: type[Routing],
        routing_class_kwargs: dict,
        start_batching_class: type[Batching],
        neighborhood_classes: list[type[Neighborhood]] | None = None,
        start_batching_kwargs: dict | None = None,
        time_limit: float = 120.0,
        verbose: bool = False,
    ):
        super().__init__(pick_cart, articles)
        self.routing_class = routing_class
        self.routing_class_kwargs = routing_class_kwargs
        self.start_batching_class = start_batching_class
        self.start_batching_kwargs = start_batching_kwargs or {}
        self.neighborhood_classes = (
            [SwapNeighborhood, ShiftNeighborhood]
            if neighborhood_classes is None
            else neighborhood_classes
        )
        self.time_limit = time_limit
        self.verbose = verbose

        self._route_cache: dict[tuple[int, ...], float] = {}
        self._router = routing_class(**routing_class_kwargs)
        self._start_time: float | None = None
        self._order_consumptions: dict[int, tuple[float, ...]] = {}
        self._order_box_counts: dict[int, int] = {}
        self._statistics = LocalSearchStatistics()
        self.search_statistics: dict[str, int | float] = {}

        neighborhood_names = "_".join(
            neighborhood_class.__name__
            for neighborhood_class in self.neighborhood_classes
        )
        self.algo_name = (
            f"{routing_class.algo_name}_"
            f"{start_batching_class.algo_name}_"
            f"{neighborhood_names}_"
            "LocalSearchBatching"
        )

    def _run(self, input_data: list[WarehouseOrder]) -> BatchingSolution:
        self.order_list = input_data
        self._route_cache.clear()
        self._statistics = LocalSearchStatistics()
        self._precompute_order_capacity(input_data)
        start_batches = self._create_start_batches()
        state = self._create_search_state(start_batches)
        self._local_search(state)
        self.search_statistics = {
            **asdict(self._statistics),
            "final_objective": state.total_score,
            "final_batches": len(state.batches),
        }
        batches = [
            BatchObject(batch_id=batch.batch_id, orders=batch.orders)
            for batch in state.batches
        ]
        return BatchingSolution(batches=batches)

    def _precompute_order_capacity(
        self,
        orders: list[WarehouseOrder],
    ) -> None:
        self._order_consumptions = {
            id(order): self.capacity_checker.order_consumption(order)
            for order in orders
        }
        if self.pick_cart.box_can_mix_orders:
            self._order_box_counts = {id(order): 0 for order in orders}
        else:
            self._order_box_counts = {
                id(order): self.capacity_checker.order_box_count(order)
                for order in orders
            }

    def _order_consumption(
        self,
        order: WarehouseOrder,
    ) -> tuple[float, ...]:
        return self._order_consumptions[id(order)]

    def _order_box_count(self, order: WarehouseOrder) -> int:
        return self._order_box_counts[id(order)]

    def _create_start_batches(self) -> list[BatchObject]:
        batching_instance: Batching = self.start_batching_class(
            pick_cart=self.pick_cart,
            articles=self.articles,
            **self.start_batching_kwargs,
        )
        return batching_instance.solve(self.order_list).batches

    def _create_search_state(
        self,
        batches: list[BatchObject],
    ) -> _SearchState:
        search_batches: list[_SearchBatch] = []
        total_score = 0.0
        for batch in batches:
            consumption = tuple(
                sum(self._order_consumption(order)[dimension] for order in batch.orders)
                for dimension in range(self.pick_cart.n_dimension)
            )
            box_count = sum(
                self._order_box_count(order)
                for order in batch.orders
            )
            score = self._batch_cost_from_orders(batch.orders)
            total_score += score
            search_batches.append(_SearchBatch(
                batch_id=batch.batch_id,
                orders=list(batch.orders),
                consumption=consumption,
                box_count=box_count,
                score=score,
            ))
        return _SearchState(search_batches, total_score)

    def _local_search(self, state: _SearchState) -> None:
        self._start_time = time.time()
        initial_cost = state.total_score
        self._record_objective(initial_cost)
        neighborhoods = [
            neighborhood_class(
                capacity_checker=self.capacity_checker,
                cost_function=self._batch_cost_from_orders,
                order_consumption=self._order_consumption,
                order_box_count=self._order_box_count,
                statistics=self._statistics,
            )
            for neighborhood_class in self.neighborhood_classes
        ]

        if self.verbose:
            print(
                "Local Search Started: "
                f"{len(state.batches)} batches, cost {initial_cost:.2f}, "
                f"neighborhoods={[type(n).__name__ for n in neighborhoods]}"
            )

        iteration = 0
        while not self._time_limit_exceeded():
            iteration += 1
            overall_improved = False
            iteration_start_cost = state.total_score

            for neighborhood in neighborhoods:
                improved = True
                while improved and not self._time_limit_exceeded():
                    improved = neighborhood.improve(
                        state=state,
                        should_stop=self._time_limit_exceeded,
                    )
                    if improved:
                        overall_improved = True
                        self._record_objective(state.total_score)

                if self._time_limit_exceeded():
                    break

            if self.verbose and overall_improved:
                print(
                    f"Iteration {iteration}: cost {state.total_score:.2f} "
                    f"(delta {iteration_start_cost - state.total_score:+.2f}), "
                    f"{len(state.batches)} batches"
                )
            if not overall_improved:
                break

        if self.verbose:
            print(
                "Local Search Completed: "
                f"cost {state.total_score:.2f}, "
                f"{len(state.batches)} batches, "
                f"stats={asdict(self._statistics)}"
            )

    def _batch_cost_from_orders(self, orders: list[WarehouseOrder]) -> float:
        self._statistics.routing_score_requests += 1
        if not orders:
            return 0.0

        key = tuple(sorted(order.order_id for order in orders))
        try:
            score = self._route_cache[key]
        except KeyError:
            self._statistics.routing_cache_misses += 1
            picks = [
                position
                for order in orders
                for position in order.pick_positions
            ]
            score = self._router.score(picks)
            self._route_cache[key] = score
        else:
            self._statistics.routing_cache_hits += 1
        return score

    def _time_limit_exceeded(self) -> bool:
        return time.time() - self._start_time > self.time_limit
