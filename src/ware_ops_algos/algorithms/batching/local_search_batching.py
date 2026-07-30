import time

from ware_ops_algos.algorithms.algorithm_interfaces import (
    BatchObject,
    BatchingSolution,
    WarehouseOrder,
)
from ware_ops_algos.algorithms.batching.batching import Batching
from ware_ops_algos.algorithms.batching.moves import Neighborhood
from ware_ops_algos.algorithms.routing.routing import Routing
from ware_ops_algos.domain_models import PickCart, Articles


class LocalSearchBatchingModular(Batching):
    def __init__(
        self,
        pick_cart: PickCart,
        articles: Articles,
        routing_class: type[Routing],
        routing_class_kwargs: dict,
        start_batching_class: type[Batching],
        neighborhood_classes: list[type[Neighborhood]],
        start_batching_kwargs: dict | None = None,
        time_limit: float = 120.0,
        verbose: bool = False,
    ):
        super().__init__(pick_cart, articles)

        self.routing_class = routing_class
        self.routing_class_kwargs = routing_class_kwargs
        self.start_batching_class = start_batching_class
        self.start_batching_kwargs = start_batching_kwargs or {}
        self.time_limit = time_limit
        self.verbose = verbose

        self._route_cache = {}
        self._router = routing_class(**routing_class_kwargs)
        self._start_time = None

        self.neighborhoods = [
            neighborhood_class(
                capacity_checker=self.capacity_checker,
                cost_function=self._batch_cost_from_orders,
            )
            for neighborhood_class in neighborhood_classes
        ]

        neighborhood_names = "_".join(
            neighborhood.__class__.__name__
            for neighborhood in self.neighborhoods
        )

        self.algo_name = (
            f"{routing_class.algo_name}_"
            f"{start_batching_class.algo_name}_"
            f"{neighborhood_names}_"
            f"LocalSearchBatching"
        )

    def _run(self, input_data: list[WarehouseOrder]) -> BatchingSolution:
        self.order_list = input_data
        start_batches = self._create_start_batches()
        batches = self._local_search(start_batches)
        return BatchingSolution(batches=batches)

    def _create_start_batches(self) -> list[BatchObject]:
        batching_instance: Batching = self.start_batching_class(
            pick_cart=self.pick_cart,
            articles=self.articles,
            **self.start_batching_kwargs
        )
        batching_sol = batching_instance.solve(self.order_list)
        return batching_sol.batches

    def _local_search(
            self,
            batches: list[BatchObject],
    ) -> list[BatchObject]:
        self._start_time = time.time()

        initial_cost = sum(
            self._batch_cost_from_orders(batch.orders)
            for batch in batches
        )
        self._record_objective(initial_cost)

        if self.verbose:
            print(f"\n{'=' * 60}")
            print("Local Search Started")
            print(f"{'=' * 60}")
            print(
                f"Neighborhoods: "
                f"{[type(n).__name__ for n in self.neighborhoods]}"
            )
            print(
                f"Initial solution: {len(batches)} batches, "
                f"cost: {initial_cost:.2f}"
            )
            print(f"Time limit: {self.time_limit}s")
            print(f"{'=' * 60}\n")

        iteration = 0

        total_improvements = {
            type(neighborhood).__name__: 0
            for neighborhood in self.neighborhoods
        }

        while not self._time_limit_exceeded():
            iteration += 1
            overall_improved = False

            iteration_start_cost = sum(
                self._batch_cost_from_orders(batch.orders)
                for batch in batches
            )

            iteration_improvements = {
                type(neighborhood).__name__: 0
                for neighborhood in self.neighborhoods
            }

            # Exhaust each neighborhood in the configured order.
            for neighborhood in self.neighborhoods:
                neighborhood_name = type(neighborhood).__name__
                improved = True

                while improved and not self._time_limit_exceeded():
                    batches, improved = neighborhood.improve(
                        batches=batches,
                        should_stop=self._time_limit_exceeded,
                    )

                    if improved:
                        overall_improved = True
                        iteration_improvements[neighborhood_name] += 1
                        total_improvements[neighborhood_name] += 1

                        current_cost = sum(
                            self._batch_cost_from_orders(batch.orders)
                            for batch in batches
                        )
                        self._record_objective(current_cost)

                if self._time_limit_exceeded():
                    break

            if overall_improved:
                iteration_end_cost = sum(
                    self._batch_cost_from_orders(batch.orders)
                    for batch in batches
                )
                improvement = iteration_start_cost - iteration_end_cost

                if self.verbose:
                    elapsed = time.time() - self._start_time

                    counts = ", ".join(
                        f"{name}={count}"
                        for name, count in iteration_improvements.items()
                    )

                    print(
                        f"Iteration {iteration}: "
                        f"{counts} | "
                        f"cost: {iteration_end_cost:.2f} "
                        f"(Δ {improvement:+.2f}) | "
                        f"{len(batches)} batches | "
                        f"cache: {len(self._route_cache)} | "
                        f"{elapsed:.1f}s"
                    )

            # No neighborhood found an improvement.
            if not overall_improved:
                break

        final_cost = sum(
            self._batch_cost_from_orders(batch.orders)
            for batch in batches
        )
        total_cost_improvement = initial_cost - final_cost
        elapsed = time.time() - self._start_time

        if self.verbose:
            counts = ", ".join(
                f"{name}={count}"
                for name, count in total_improvements.items()
            )

            improvement_percentage = (
                100 * total_cost_improvement / initial_cost
                if initial_cost != 0
                else 0.0
            )

            print(f"\n{'=' * 60}")
            print("Local Search Completed")
            print(f"{'=' * 60}")
            print(f"Iterations: {iteration}")
            print(f"Improvements: {counts}")
            print(f"Initial cost: {initial_cost:.2f}")
            print(f"Final cost: {final_cost:.2f}")
            print(
                f"Total improvement: {total_cost_improvement:.2f} "
                f"({improvement_percentage:.1f}%)"
            )
            print(f"Final batches: {len(batches)}")
            print(f"Route cache size: {len(self._route_cache)}")
            print(f"Time elapsed: {elapsed:.2f}s")
            print(f"{'=' * 60}\n")

        return batches

    def _batch_cost_from_orders(self, orders: list[WarehouseOrder]) -> float:
        """Calculate routing cost for a list of orders with caching."""
        if not orders:
            return 0.0

        key = tuple(sorted(o.order_id for o in orders))
        if key not in self._route_cache:
            self._router.reset_parameters()
            pick_list = [pos for order in orders for pos in order.pick_positions]
            sol = self._router.solve(pick_list)
            self._route_cache[key] = sol.route.distance
        return self._route_cache[key]

    def _time_limit_exceeded(self) -> bool:
        """Check if time limit has been exceeded."""
        return time.time() - self._start_time > self.time_limit
