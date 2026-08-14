import random
from abc import ABC, abstractmethod

from ..algorithm_interfaces import Algorithm
from ware_ops_algos.algorithms.batching.batching_utils import CapacityChecker
from ware_ops_algos.algorithms.algorithm_interfaces import (
    BatchingSolution,
    BatchObject,
    ResidualBatchingInput,
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


class ResidualFifoBatching(
    Algorithm[ResidualBatchingInput, BatchingSolution]
):
    """Fill genuinely unassigned order bins without changing existing owners."""

    algo_name = "ResidualFifoBatching"

    def _run(self, input_data: ResidualBatchingInput) -> BatchingSolution:
        cart = input_data.pick_cart
        if cart.box_can_mix_orders:
            raise ValueError(
                "Residual FIFO insertion does not support mixed-order bins"
            )
        if (
            cart.n_dimension != 1
            or cart.dimensions != [DimensionType.ORDERS]
        ):
            raise ValueError(
                "Residual FIFO insertion supports only one ORDERS dimension"
            )
        if not cart.capacities or cart.capacities[0] < 1:
            raise ValueError("Each order bin must accept at least one order")
        if cart.n_boxes != len(input_data.bin_order_ids):
            raise ValueError("Cart bin snapshot does not match PickCart.n_boxes")

        active_ids = set(input_data.active_batch.order_numbers)
        assignments = {
            bin_id: tuple(order_ids)
            for bin_id, order_ids in enumerate(input_data.bin_order_ids)
        }
        empty_bins = [
            bin_id
            for bin_id, order_ids in assignments.items()
            if not order_ids and bin_id not in input_data.locked_bin_ids
        ]
        candidates = sorted(
            (
                order
                for order in input_data.candidate_orders
                if order.order_id not in active_ids
            ),
            key=lambda order: (
                order.order_date if order.order_date is not None else 0,
                order.order_id,
            ),
        )
        inserted = candidates[:len(empty_bins)]
        for bin_id, order in zip(empty_bins, inserted):
            assignments[bin_id] = (order.order_id,)

        merged = BatchObject(
            batch_id=input_data.active_batch.batch_id,
            orders=list(input_data.active_batch.orders) + list(inserted),
            bin_assignments=assignments,
        )
        return BatchingSolution(
            batches=[merged],
        )


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
