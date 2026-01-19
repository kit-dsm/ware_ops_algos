from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import count
from typing import Generic, TypeVar, Optional, Any, NamedTuple
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


@dataclass
class PickPosition:
    order_number: int
    article_id: int
    amount: int
    pick_node: tuple[int, int]
    in_store: int
    article_name: Optional[str] = None
    # dynamic info
    picked: Optional[bool] = None


@dataclass(frozen=True)
class WarehouseOrder:
    order_id: int
    due_date: Optional[float] = None
    order_date: Optional[float] = None
    pick_positions: Optional[list[PickPosition]] = None
    # dynamic info
    fulfilled: Optional[bool] = None


@dataclass
class ItemAssignmentSolution(AlgorithmSolution):
    resolved_orders: list[WarehouseOrder] = field(default_factory=list)


@dataclass
class BatchObject:
    batch_id: int
    orders: list[WarehouseOrder]


@dataclass
class BatchingSolution(AlgorithmSolution):
    batches: list[BatchObject] | None = None
    # pick_lists: list[list[PickPosition]] = None
    pick_lists: list[PickList] = None


@dataclass
class PickList:
    pick_positions: list[PickPosition]
    orders: list[WarehouseOrder]
    release: Optional[float] = None
    earliest_due_date: Optional[float] = None
    id: int = field(default_factory=count().__next__)

    @property
    def order_numbers(self) -> list[int]:
        if self.pick_positions is None:
            return []
        return list({pp.order_number for pp in self.pick_positions})


@dataclass
class PickerAssignment:
    picker: Resource
    pick_list: PickList


@dataclass
class AssignmentSolution(AlgorithmSolution):
    assignments: list[PickerAssignment] = field(default_factory=list)


class NodeType(Enum):
    PICK = auto()
    ROUTE = auto()


class RouteNode(NamedTuple):
    position: tuple[int, int]
    node_type: NodeType


@dataclass
class Route:
    route: list[tuple[int, int]] | None = None
    item_sequence: list[tuple[int, int]] | None = None
    distance: float = 0.0
    pick_list: Optional[PickList] = None
    annotated_route: Optional[list[RouteNode]] = None


@dataclass
class PickTour:
    pick_list: PickList
    route: Route
    assigned_picker: Resource
    starts_after: Optional[int] = None
    starts_before: Optional[int] = None

    planned_start: Optional[float] = None  # if scheduled
    planned_end: Optional[float] = None


@dataclass
class RoutingSolution(AlgorithmSolution):
    route: Optional[Route] = None


@dataclass
class CombinedRoutingSolution(AlgorithmSolution):
    routes: Optional[list[Route]] = None


@dataclass
class Sequencing:
    pick_list_sequence: list[int]


@dataclass
class Assignment:
    tour_id: int
    picker_id: int


@dataclass
class Job:
    batch_idx: int
    picker_id: int
    start_time: float
    end_time: float
    release_time: float
    distance: float
    n_picks: int
    travel_time: float
    handling_time: float
    route: Route


@dataclass
class SchedulingSolution(AlgorithmSolution):
    jobs: list[Job] | None = None

@dataclass
class OrderSelectionSolution(AlgorithmSolution):
    selected_orders: list[WarehouseOrder] = field(default_factory=list)

@dataclass
class PlanningState:
    item_assignment: Optional[ItemAssignmentSolution] = None
    batching_solutions: Optional[BatchingSolution] = None
    assignment_solutions: Optional[AssignmentSolution] = None
    routing_solutions: Optional[list[RoutingSolution]] = field(default_factory=list)
    sequencing_solutions: Optional[SchedulingSolution] = None
    order_selection_solutions: Optional[OrderSelectionSolution] = None
    provenance: dict[str, Any] = field(default_factory=dict)



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
