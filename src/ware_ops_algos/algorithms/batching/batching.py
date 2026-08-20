import random
from abc import ABC, abstractmethod

from ..algorithm_interfaces import Algorithm
from ware_ops_algos.algorithms.batching.batching_utils import CapacityChecker
from ware_ops_algos.algorithms.algorithm_interfaces import (
    BatchingSolution,
    BatchObject,
    WarehouseOrder,
)
from ware_ops_algos.domain_models import DimensionType, PickCart, Articles


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
