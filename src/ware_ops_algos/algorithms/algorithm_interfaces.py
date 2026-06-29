from __future__ import annotations
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import count
from typing import Generic, TypeVar, Optional, Any, NamedTuple, Deque
import time
import logging

from ware_ops_algos.domain_models import Order, ResolvedOrderPosition, OrderPosition, Resource

I = TypeVar("I")  # input type
O = TypeVar("O")  # output type

@dataclass
class AlgorithmSolution:
    algo_name: str = ""
    execution_time: float = 0.0
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PickPosition:
    order_number: int
    article_id: int
    amount: int
    pick_node: tuple[int, int]
    in_store: int
    article_name: Optional[str] = None,


@dataclass
class WarehouseOrder:
    order_id: int
    parent_order_id: Optional[int] = None
    due_date: Optional[float] = None
    order_date: Optional[float] = None
    pick_positions: tuple[PickPosition, ...] = ()


@dataclass
class BatchObject:
    batch_id: int
    orders: list[WarehouseOrder]

    @property
    def pick_positions(self) -> list[PickPosition]:
        return [pp for o in self.orders for pp in o.pick_positions]

    @property
    def order_numbers(self) -> frozenset[int]:
        return frozenset(o.order_id for o in self.orders)

    @property
    def earliest_due_date(self) -> float:
        dues = [o.due_date for o in self.orders if o.due_date is not None]
        return min(dues) if dues else float("inf")

    @property
    def arrival_time(self) -> float:
        """The time at which all orders in this batch had arrived."""
        dates = [o.order_date for o in self.orders if o.order_date is not None]
        return max(dates) if dates else 0.0


class NodeType(Enum):
    PICK = auto()
    ROUTE = auto()


class RouteNode(NamedTuple):
    position: tuple[int, int]
    node_type: NodeType

@dataclass
class Route:
    distance: float
    route: Optional[list[tuple[float, float]]] = None
    item_sequence: Optional[list] = None
    batch: Optional[BatchObject] = None
    annotated_route: Optional[list[RouteNode]] = None
    service_time: Optional[float] = None

    @property
    def has_sequence(self) -> bool:
        return self.annotated_route is not None

    @property
    def node_sequence(self) -> list[tuple[int, int]]:
        if self.annotated_route is None:
            raise ValueError("Route was not constructed with sequence information.")
        return [n.position for n in self.annotated_route]

    # @property
    # def item_sequence(self) -> list[tuple[int, int]]:
    #     if self.annotated_route is None:
    #         raise ValueError("Route was not constructed with sequence information.")
    #     return [n.position for n in self.annotated_route if n.node_type == NodeType.PICK]

    # @property
    # def n_picks(self):
    #     return len(self.item_sequence)

    @property
    def earliest_due_date(self) -> float:
        return self.batch.earliest_due_date

    @property
    def arrival_time(self) -> float:
        return self.batch.arrival_time

    @property
    def order_numbers(self) -> frozenset[int]:
        return self.batch.order_numbers


@dataclass(frozen=True)
class Job:
    """
    Scheduler input. Stores only what's new at this stage:
    a stable id and the resource-dependent processing time.
    Everything else is forwarded from the route.
    """
    job_id: int
    processing_time: float
    release_time: float
    due_date: float
    n_picks: int
    route: Optional[Route] = None
    batch: Optional[BatchObject] = None

    @property
    def distance(self) -> float:
        return self.route.distance

    @property
    def order_numbers(self) -> frozenset[int]:
        return self.route.order_numbers


@dataclass(frozen=True)
class ScheduledJob:
    """
    A Job placed on a picker at a concrete start/end time.
    Performance metrics are properties.
    """
    job: Job
    picker_id: str
    start_time: float
    end_time: float

    @property
    def tardiness(self) -> float:
        return max(0.0, self.end_time - self.job.due_date)

    @property
    def lateness(self) -> float:
        return self.end_time - self.job.due_date

    @property
    def is_on_time(self) -> bool:
        return self.end_time <= self.job.due_date

    @property
    def order_numbers(self) -> frozenset[int]:
        return self.job.order_numbers


@dataclass
class ItemAssignmentSolution(AlgorithmSolution):
    resolved_orders: list[WarehouseOrder] = field(default_factory=list)


@dataclass
class BatchingSolution(AlgorithmSolution):
    batches: list[BatchObject] = field(default_factory=list)

@dataclass
class RoutingSolution(AlgorithmSolution):
    route: Route = field(default_factory=Route)


@dataclass
class CombinedRoutingSolution(AlgorithmSolution):
    routes: list[Route] = field(default_factory=list)


@dataclass
class SchedulingSolution(AlgorithmSolution):
    jobs: list[ScheduledJob] = field(default_factory=list)


class Algorithm(ABC, Generic[I, O]):
    """
    Abstract base class for all algorithms (routing, batching, etc.).
    Handles timing, algo naming, and ensures consistent result metadata.
    """

    algo_name: str = "Algorithm"

    def __init__(self, seed: Optional[int] = None):
        self._seed = seed
        self.logger = logging.getLogger(self.__class__.__name__)

    def solve(self, input_data: I) -> O:
        start_time = time.perf_counter()

        try:
            result: O = self._run(input_data)
        except Exception as e:
            raise RuntimeError(f"Algorithm '{self}' failed: {e}") from e

        elapsed = time.perf_counter() - start_time

        if not result.algo_name:
            result.algo_name = self.algo_name
        result.execution_time = elapsed

        return result

    @abstractmethod
    def _run(self, input_data: I) -> O:
        """
        Concrete algorithms implement this method.
        Must return a subclass of AlgorithmResult.
        """
        ...
