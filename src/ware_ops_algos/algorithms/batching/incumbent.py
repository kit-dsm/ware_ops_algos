"""Batch new orders around selected draft tours without changing solver I/O."""

from __future__ import annotations

from ware_ops_algos.algorithms.algorithm_interfaces import (
    BatchObject, BatchingSolution, WarehouseOrder,
)
from ware_ops_algos.algorithms.batching.batching import Batching, FifoBatching
from ware_ops_algos.algorithms.routing.routing import Routing


class IncumbentStartBatching(Batching):
    """Keep draft membership, then admit new orders at minimum route cost."""

    algo_name = "IncumbentStartBatching"

    def __init__(self, *args, incumbent_batches: tuple[BatchObject, ...],
                 router: Routing, **kwargs):
        super().__init__(*args, **kwargs)
        self.incumbent_batches = incumbent_batches
        self.router = router

    def _route_cost(self, batch: BatchObject) -> float:
        self.router.reset_parameters()
        return self.router.solve(batch.pick_positions).route.distance

    def _run(self, input_data: list[WarehouseOrder]) -> BatchingSolution:
        if not self.incumbent_batches:
            return FifoBatching(self.pick_cart, self.articles)._run(input_data)
        by_id = {order.order_id: order for order in input_data}
        batches = [BatchObject(batch_id=batch.batch_id,
                               orders=[by_id[order_id]
                                       for order_id in sorted(batch.order_numbers)])
                   for batch in self.incumbent_batches]
        incumbent_ids = {order_id for batch in batches
                         for order_id in batch.order_numbers}
        unassigned = []
        for order in sorted((order for order in input_data
                             if order.order_id not in incumbent_ids),
                            key=lambda order: (order.order_date or 0.0,
                                               order.order_id)):
            choices = []
            for batch in batches:
                if not self.capacity_checker.can_add_order(batch.orders, order):
                    continue
                candidate = BatchObject(batch.batch_id, [*batch.orders, order])
                choices.append((self._route_cost(candidate) - self._route_cost(batch),
                                batch.batch_id, batch))
            if choices:
                _, _, chosen = min(choices, key=lambda item: (item[0], item[1]))
                chosen.orders.append(order)
            else:
                unassigned.append(order)
        if unassigned:
            new_batches = FifoBatching(self.pick_cart, self.articles)._run(unassigned).batches
            next_id = max(batch.batch_id for batch in batches) + 1
            batches.extend(BatchObject(next_id + index, batch.orders)
                           for index, batch in enumerate(new_batches))
        return BatchingSolution(batches=batches)
