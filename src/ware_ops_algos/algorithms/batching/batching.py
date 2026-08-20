import heapq
import random
import time
from abc import ABC, abstractmethod
from itertools import combinations
from typing import Type

from ware_ops_algos.algorithms import Algorithm
from ware_ops_algos.algorithms.batching.batching_utils import CapacityChecker
from ware_ops_algos.algorithms.routing.routing import Routing
from ware_ops_algos.algorithms.algorithm_interfaces import BatchingSolution, BatchObject, WarehouseOrder
from ware_ops_algos.domain_models import PickCart, Articles


class Batching(Algorithm[list[WarehouseOrder], BatchingSolution], ABC):
    """Batching class to batch orders"""

    def __init__(self, pick_cart: PickCart, articles: Articles, **kwargs):
        # instance attributes
        super().__init__(**kwargs)

        # self.picker = picker
        # self.picker_capa = pick_cart.
        self.execution_time = None
        self.pick_cart = pick_cart
        self.articles = articles
        self.capacity_checker = CapacityChecker(pick_cart=pick_cart,
                                                articles=articles)

    def _run(self, input_data: list[WarehouseOrder]) -> BatchingSolution:
        pass


class PriorityBatching(Batching):
    """Priority batching class to batch orders based on sorting criterion."""

    @abstractmethod
    def _sorted_orders(self) -> list[WarehouseOrder]:
        pass

    def _run(self, input_data: list[WarehouseOrder]) -> BatchingSolution:
        self.order_list = input_data
        sorted_orders = self._sorted_orders()
        batched_list: list[BatchObject] = []
        batch_number = 0
        current_batch: list[WarehouseOrder] = []

        for order in sorted_orders:
            # Check if we can add this order to current batch
            if self.capacity_checker.can_add_order(current_batch, order):
                current_batch.append(order)
            else:
                # Current batch is full - start new batch
                if current_batch:
                    batched_list.append(BatchObject(batch_id=batch_number, orders=current_batch))
                    batch_number += 1

                # Check if order fits alone
                if self.capacity_checker.can_add_order([], order):
                    current_batch = [order]
                else:
                    print(f"Order {order.order_id} exceeds capacity, excluded")
                    current_batch = []

        if current_batch:
            batched_list.append(BatchObject(batch_id=batch_number, orders=current_batch))

        return BatchingSolution(batches=batched_list)


class OrderNrFifoBatching(PriorityBatching):
    """First in First out batching based on order number."""
    algo_name = "OrderNrFiFoBatching"

    def _sorted_orders(self) -> list[WarehouseOrder]:
        return sorted(self.order_list, key=lambda o: o.order_id)


class FifoBatching(PriorityBatching):
    """
    First in First out batching class to batch orders
    """
    algo_name = "FiFoBatching"

    # def __init__(self, capacity):
    #     super().__init__(capacity)

    def _sorted_orders(self) -> list[WarehouseOrder]:
        return sorted(self.order_list, key=lambda o: o.order_date)


class RandomBatching(PriorityBatching):
    """
    First in First out batching class to batch orders
    """
    algo_name = "RandomBatching"

    def __init__(self, pick_cart: PickCart, articles: Articles, seed=44):
        super().__init__(pick_cart, articles)

        self.seed = seed
        self.batch_number = 0

    def _sorted_orders(self):
        """
        Sorting the batching list
        """
        shuffled = self.order_list.copy()
        random.Random(self.seed).shuffle(shuffled)
        return shuffled


class DueDateBatching(PriorityBatching):
    """
    First in First out batching class to batch orders
    """
    algo_name = "DueDateBatching"

    def __init__(self, pick_cart: PickCart, articles: Articles):
        super().__init__(pick_cart, articles)

    def _sorted_orders(self) -> list[WarehouseOrder]:
        return sorted(self.order_list, key=lambda o: o.due_date)


class SavingsBatching(Batching, ABC):
    """
    Base class for savings-based batching algorithms.
    """

    def __init__(self,
                 pick_cart: PickCart,
                 articles: Articles,
                 routing_class: Type[Routing],
                 routing_class_kwargs,
                 time_limit: float | None = None):
        super().__init__(pick_cart, articles)
        self.routing_class = routing_class
        self.routing_class_kwargs = routing_class_kwargs
        self._router = routing_class(**routing_class_kwargs)
        self._route_cache = {}
        self.time_limit = time_limit
        self.algo_name = f"{self.routing_class.algo_name}_SavingsBatching"

    def _calc_dist_with_routing_algo(self, orders: list[WarehouseOrder]) -> float:
        """Compute route distance for a list of orders, with caching."""
        key = tuple(sorted(o.order_id for o in orders))
        if key not in self._route_cache:
            # router = self.routing_class(
            #     **self.routing_class_kwargs
            # )
            self._router.reset_parameters()
            batches = [BatchObject(batch_id=0, orders=orders)]
            pick_lists = []
            for batch in batches:
                pick_list = []
                for order in batch.orders:
                    for pos in order.pick_positions:
                        pick_list.append(pos)
                pick_lists.append(pick_list)
            routing_sol = self._router.solve(pick_lists[0])
            self._route_cache[key] = routing_sol.route.distance
        return self._route_cache[key]

    def _calculate_saving(self, batch_a: BatchObject, batch_b: BatchObject) -> float:
        """Calculate saving from merging two batches."""
        orders_a = batch_a.orders
        orders_b = batch_b.orders
        merged_orders = orders_a + orders_b

        # Capacity check
        # total_amount = sum(pos.in_store for o in merged_orders for pos in o.pick_positions)
        # if total_amount > self.picker_capa:
        #     return 0
        if not self.capacity_checker.orders_fit(merged_orders):
            return 0

        dist_a = self._calc_dist_with_routing_algo(orders_a)
        dist_b = self._calc_dist_with_routing_algo(orders_b)
        dist_comb = self._calc_dist_with_routing_algo(merged_orders)

        return dist_a + dist_b - dist_comb


class ClarkAndWrightBatching(SavingsBatching):
    def _run(self, input_data: list[WarehouseOrder]) -> BatchingSolution:
        self.order_list = input_data
        start_time = time.time()


        batches = {i: BatchObject(batch_id=i, orders=[order])
                   for i, order in enumerate(self.order_list)}
        batch_counter = len(batches)

        savings_heap = []
        for id_a, id_b in combinations(batches.keys(), 2):
            saving = self._calculate_saving(batches[id_a], batches[id_b])
            if saving > 0:
                pair = (min(id_a, id_b), max(id_a, id_b))
                heapq.heappush(savings_heap, (-saving, pair[0], pair[1]))

        while savings_heap:
            if self.time_limit and (time.time() - start_time) > self.time_limit:
                break

            neg_saving, id_a, id_b = heapq.heappop(savings_heap)

            if id_a not in batches or id_b not in batches:
                continue

            if -neg_saving <= 0:
                break

            batch_a = batches.pop(id_a)
            batch_b = batches.pop(id_b)
            merged_batch = BatchObject(batch_id=batch_counter, orders=batch_a.orders + batch_b.orders)
            batches[batch_counter] = merged_batch

            for other_id in batches:
                if other_id == batch_counter:
                    continue
                saving = self._calculate_saving(merged_batch, batches[other_id])
                if saving > 0:
                    pair = (min(batch_counter, other_id), max(batch_counter, other_id))
                    heapq.heappush(savings_heap, (-saving, pair[0], pair[1]))

            batch_counter += 1

        return BatchingSolution(batches=list(batches.values()))
