from abc import ABC, abstractmethod
from typing import Callable

from .batching_utils import CapacityChecker
from ..algorithm_interfaces import BatchObject, WarehouseOrder


class Neighborhood(ABC):
    @abstractmethod
    def improve(
        self,
        batches: list[BatchObject],
        should_stop: Callable[[], bool]
    ) -> tuple[list[BatchObject], bool]:
        ...

class ShiftNeighborhood(Neighborhood):
    def __init__(self,
                 capacity_checker: CapacityChecker,
                 cost_function: Callable[[list[WarehouseOrder]], float]):
        super().__init__()
        self.capacity_checker = capacity_checker
        self.cost_function = cost_function

    def improve(
        self,
        batches: list[BatchObject],
        should_stop: Callable[[], bool],
    ) -> tuple[list[BatchObject], bool]:
        for i in range(len(batches)):
            batch_i = batches[i]

            for order_idx, order in enumerate(batch_i.orders):
                for j in range(len(batches)):
                    if should_stop():
                        return batches, False

                    if i == j:
                        continue

                    batch_j = batches[j]

                    # Check capacity BEFORE computing costs
                    temp_orders_i = batch_i.orders[:order_idx] + batch_i.orders[order_idx + 1:]
                    temp_orders_j = batch_j.orders + [order]

                    if not self.capacity_checker.orders_fit(temp_orders_i) or \
                            not self.capacity_checker.orders_fit(temp_orders_j):
                        continue

                    # Compute costs
                    old_cost_i = self.cost_function(batch_i.orders)
                    old_cost_j = self.cost_function(batch_j.orders)
                    old_total = old_cost_i + old_cost_j

                    new_cost_i = self.cost_function(temp_orders_i)
                    new_cost_j = self.cost_function(temp_orders_j)
                    new_total = new_cost_i + new_cost_j

                    if new_total < old_total - 1e-6:
                        # Apply improvement IN PLACE
                        batch_i.orders.pop(order_idx)
                        batch_j.orders.append(order)

                        # Clean up empty batches if needed
                        if not batch_i.orders:
                            batches.pop(i)

                        return batches, True

        return batches, False


class SwapNeighborhood(Neighborhood):
    def __init__(self,
                 capacity_checker: CapacityChecker,
                 cost_function: Callable[[list[WarehouseOrder]], float]):
        super().__init__()
        self.capacity_checker = capacity_checker
        self.cost_function = cost_function

    def improve(self,
                batches: list[BatchObject],
                should_stop: Callable[[], bool]) -> tuple[list[BatchObject], bool]:
        for i in range(len(batches)):
            for j in range(i + 1, len(batches)):
                if should_stop():
                    return batches, False

                batch_i = batches[i]
                batch_j = batches[j]

                old_cost_i = self.cost_function(batch_i.orders)
                old_cost_j = self.cost_function(batch_j.orders)
                old_total = old_cost_i + old_cost_j

                for i_idx, order_i in enumerate(batch_i.orders):
                    for j_idx, order_j in enumerate(batch_j.orders):
                        temp_orders_i = batch_i.orders[:i_idx] + [order_j] + batch_i.orders[i_idx + 1:]
                        temp_orders_j = batch_j.orders[:j_idx] + [order_i] + batch_j.orders[j_idx + 1:]

                        if not self.capacity_checker.orders_fit(temp_orders_i) or \
                                not self.capacity_checker.orders_fit(temp_orders_j):
                            continue

                        # Compute new costs
                        new_cost_i = self.cost_function(temp_orders_i)
                        new_cost_j = self.cost_function(temp_orders_j)
                        new_total = new_cost_i + new_cost_j

                        if new_total < old_total - 1e-6:
                            batch_i.orders[i_idx] = order_j
                            batch_j.orders[j_idx] = order_i
                            return batches, True

        return batches, False
