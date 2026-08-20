import heapq
import time
from itertools import combinations

from ware_ops_algos.algorithms.algorithm_interfaces import (
    BatchObject,
    BatchingSolution,
    WarehouseOrder,
)
from ware_ops_algos.algorithms.batching.batching import Batching
from ware_ops_algos.algorithms.routing.routing import Routing
from ware_ops_algos.domain_models import Articles, PickCart


class ClarkAndWrightBatching(Batching):
    """Clark-Wright savings batching with deterministic heap tie-breaking."""

    def __init__(
        self,
        pick_cart: PickCart,
        articles: Articles,
        routing_class: type[Routing],
        routing_class_kwargs: dict,
        time_limit: float | None = None,
    ):
        super().__init__(pick_cart, articles)
        self.routing_class = routing_class
        self.routing_class_kwargs = routing_class_kwargs
        self._router = routing_class(**routing_class_kwargs)
        self._route_cache: dict[tuple[int, ...], float] = {}
        self.time_limit = time_limit
        self.algo_name = f"{routing_class.algo_name}_SavingsBatching"

    def _batch_score(self, orders: list[WarehouseOrder]) -> float:
        key = tuple(sorted(order.order_id for order in orders))
        try:
            return self._route_cache[key]
        except KeyError:
            picks = [
                position
                for order in orders
                for position in order.pick_positions
            ]
            score = self._router.score(picks)
            self._route_cache[key] = score
            return score

    def _saving(self, batch_a: BatchObject, batch_b: BatchObject) -> float:
        merged_orders = batch_a.orders + batch_b.orders
        if not self.capacity_checker.orders_fit(merged_orders):
            return 0.0
        return (
            self._batch_score(batch_a.orders)
            + self._batch_score(batch_b.orders)
            - self._batch_score(merged_orders)
        )

    def _run(self, input_data: list[WarehouseOrder]) -> BatchingSolution:
        self._route_cache.clear()
        start_time = time.time()
        batches = {
            index: BatchObject(batch_id=index, orders=[order])
            for index, order in enumerate(input_data)
        }
        batch_counter = len(batches)
        savings_heap: list[tuple[float, int, int]] = []

        for id_a, id_b in combinations(batches, 2):
            saving = self._saving(batches[id_a], batches[id_b])
            if saving > 0:
                heapq.heappush(savings_heap, (-saving, id_a, id_b))

        while savings_heap:
            if self.time_limit and time.time() - start_time > self.time_limit:
                break

            neg_saving, id_a, id_b = heapq.heappop(savings_heap)
            if id_a not in batches or id_b not in batches:
                continue
            if -neg_saving <= 0:
                break

            batch_a = batches.pop(id_a)
            batch_b = batches.pop(id_b)
            merged_batch = BatchObject(
                batch_id=batch_counter,
                orders=batch_a.orders + batch_b.orders,
            )
            batches[batch_counter] = merged_batch

            for other_id, other_batch in batches.items():
                if other_id == batch_counter:
                    continue
                saving = self._saving(merged_batch, other_batch)
                if saving > 0:
                    first, second = sorted((batch_counter, other_id))
                    heapq.heappush(
                        savings_heap,
                        (-saving, first, second),
                    )
            batch_counter += 1

        return BatchingSolution(batches=list(batches.values()))

