from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

from .batching_utils import CapacityChecker
from ..algorithm_interfaces import WarehouseOrder


@dataclass(slots=True)
class LocalSearchStatistics:
    candidate_moves_inspected: int = 0
    accepted_shifts: int = 0
    accepted_swaps: int = 0
    routing_score_requests: int = 0
    routing_cache_hits: int = 0
    routing_cache_misses: int = 0
    capacity_checks: int = 0
    full_capacity_rescans: int = 0
    full_objective_resums: int = 0
    candidate_batch_objects: int = 0


@dataclass(slots=True)
class _SearchBatch:
    batch_id: int
    orders: list[WarehouseOrder]
    consumption: tuple[float, ...]
    box_count: int
    score: float


@dataclass(slots=True)
class _SearchState:
    batches: list[_SearchBatch]
    total_score: float


OrderConsumption = Callable[[WarehouseOrder], tuple[float, ...]]
OrderBoxCount = Callable[[WarehouseOrder], int]
BatchScore = Callable[[list[WarehouseOrder]], float]


def _replace_consumption(
    current: tuple[float, ...],
    removed: tuple[float, ...],
    added: tuple[float, ...],
) -> tuple[float, ...]:
    return tuple(
        value - remove + add
        for value, remove, add in zip(current, removed, added)
    )


def _subtract_consumption(
    current: tuple[float, ...],
    removed: tuple[float, ...],
) -> tuple[float, ...]:
    return tuple(value - remove for value, remove in zip(current, removed))


def _add_consumption(
    current: tuple[float, ...],
    added: tuple[float, ...],
) -> tuple[float, ...]:
    return tuple(value + add for value, add in zip(current, added))


class Neighborhood(ABC):
    def __init__(
        self,
        capacity_checker: CapacityChecker,
        cost_function: BatchScore,
        order_consumption: OrderConsumption,
        order_box_count: OrderBoxCount,
        statistics: LocalSearchStatistics,
    ):
        self.capacity_checker = capacity_checker
        self.cost_function = cost_function
        self.order_consumption = order_consumption
        self.order_box_count = order_box_count
        self.statistics = statistics

    def _fits(
        self,
        consumption: tuple[float, ...],
        box_count: int,
    ) -> bool:
        self.statistics.capacity_checks += 1
        return self.capacity_checker.fits_consumption(
            consumption,
            box_count=box_count,
        )

    @abstractmethod
    def improve(
        self,
        state: _SearchState,
        should_stop: Callable[[], bool],
    ) -> bool:
        ...


class ShiftNeighborhood(Neighborhood):
    def improve(
        self,
        state: _SearchState,
        should_stop: Callable[[], bool],
    ) -> bool:
        for i, source in enumerate(state.batches):
            for order_idx, order in enumerate(source.orders):
                order_consumption = self.order_consumption(order)
                order_boxes = self.order_box_count(order)

                for j, destination in enumerate(state.batches):
                    if should_stop():
                        return False
                    if i == j:
                        continue

                    self.statistics.candidate_moves_inspected += 1
                    source_consumption = _subtract_consumption(
                        source.consumption,
                        order_consumption,
                    )
                    destination_consumption = _add_consumption(
                        destination.consumption,
                        order_consumption,
                    )
                    source_boxes = source.box_count - order_boxes
                    destination_boxes = destination.box_count + order_boxes

                    if not self._fits(source_consumption, source_boxes):
                        continue
                    if not self._fits(
                        destination_consumption,
                        destination_boxes,
                    ):
                        continue

                    source_orders = (
                        source.orders[:order_idx]
                        + source.orders[order_idx + 1:]
                    )
                    destination_orders = destination.orders + [order]
                    source_score = self.cost_function(source_orders)
                    destination_score = self.cost_function(destination_orders)
                    delta = (
                        source_score
                        + destination_score
                        - source.score
                        - destination.score
                    )

                    if delta < -1e-6:
                        source.orders.pop(order_idx)
                        destination.orders.append(order)
                        source.consumption = source_consumption
                        destination.consumption = destination_consumption
                        source.box_count = source_boxes
                        destination.box_count = destination_boxes
                        source.score = source_score
                        destination.score = destination_score
                        state.total_score += delta
                        self.statistics.accepted_shifts += 1

                        if not source.orders:
                            state.batches.pop(i)
                        return True

        return False


class SwapNeighborhood(Neighborhood):
    def improve(
        self,
        state: _SearchState,
        should_stop: Callable[[], bool],
    ) -> bool:
        for i, batch_i in enumerate(state.batches):
            for batch_j in state.batches[i + 1:]:
                if should_stop():
                    return False

                for i_idx, order_i in enumerate(batch_i.orders):
                    consumption_i = self.order_consumption(order_i)
                    boxes_i = self.order_box_count(order_i)

                    for j_idx, order_j in enumerate(batch_j.orders):
                        self.statistics.candidate_moves_inspected += 1
                        consumption_j = self.order_consumption(order_j)
                        boxes_j = self.order_box_count(order_j)
                        new_consumption_i = _replace_consumption(
                            batch_i.consumption,
                            consumption_i,
                            consumption_j,
                        )
                        new_consumption_j = _replace_consumption(
                            batch_j.consumption,
                            consumption_j,
                            consumption_i,
                        )
                        new_boxes_i = batch_i.box_count - boxes_i + boxes_j
                        new_boxes_j = batch_j.box_count - boxes_j + boxes_i

                        if not self._fits(new_consumption_i, new_boxes_i):
                            continue
                        if not self._fits(new_consumption_j, new_boxes_j):
                            continue

                        orders_i = (
                            batch_i.orders[:i_idx]
                            + [order_j]
                            + batch_i.orders[i_idx + 1:]
                        )
                        orders_j = (
                            batch_j.orders[:j_idx]
                            + [order_i]
                            + batch_j.orders[j_idx + 1:]
                        )
                        score_i = self.cost_function(orders_i)
                        score_j = self.cost_function(orders_j)
                        delta = (
                            score_i
                            + score_j
                            - batch_i.score
                            - batch_j.score
                        )

                        if delta < -1e-6:
                            batch_i.orders[i_idx] = order_j
                            batch_j.orders[j_idx] = order_i
                            batch_i.consumption = new_consumption_i
                            batch_j.consumption = new_consumption_j
                            batch_i.box_count = new_boxes_i
                            batch_j.box_count = new_boxes_j
                            batch_i.score = score_i
                            batch_j.score = score_j
                            state.total_score += delta
                            self.statistics.accepted_swaps += 1
                            return True

        return False
