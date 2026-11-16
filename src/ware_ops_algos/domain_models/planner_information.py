"""Information available to algorithms, separate from realized simulation events.

Forecasts of exogenous processes, and the decisions a WMS recorded, which a
replay hands to the planner instead of computing them.
"""

from dataclasses import dataclass, fields
from enum import Enum
from math import isfinite
from typing import Any, ClassVar

from .base_domain_object import BaseDomainObject


class InformationType(str, Enum):
    PROCESS_INFORMATION = "process_information"


@dataclass(frozen=True)
class ExponentialSingleLineUniformLocationOrderStream:
    """Exponential arrivals, one order line, and uniformly distributed picks."""

    representation: ClassVar[str] = "exponential_single_line_uniform_location_order_stream"
    mean_interarrival_time_s: float

    def __post_init__(self) -> None:
        mean = self.mean_interarrival_time_s
        if isinstance(mean, bool) or not isinstance(mean, (int, float)) or not isfinite(mean) or mean <= 0:
            raise ValueError("mean_interarrival_time_s must be a finite positive number")


@dataclass(frozen=True)
class RecordedLine:
    """One recorded order line: its order, article and pick node."""

    order_id: int
    article_id: int
    pick_node: tuple[float, float]


@dataclass(frozen=True)
class RecordedTour:
    """The lines one picker picked in one tour, in the recorded sequence.

    ``start_s``, if given, is the tour's recorded start: a plan anchored to the
    record starts the tour no earlier. ``start_node`` is an explicit route
    origin when known; otherwise the layout's start node is used.
    """

    tour_id: int
    picker_id: int
    lines: tuple[RecordedLine, ...]
    start_s: float | None = None
    start_node: tuple[float, float] | None = None

    @property
    def order_ids(self) -> tuple[int, ...]:
        """The tour's orders in the order of their first recorded line."""
        return tuple(dict.fromkeys(line.order_id for line in self.lines))


@dataclass(frozen=True)
class RecordedPickingPlan:
    """The tours a WMS recorded, per picker in the order picked, with their lines.

    Feature ``tours`` is the number of tours, so that a card can require a
    recorded plan without its content.
    """

    representation: ClassVar[str] = "recorded_picking_plan"
    tours: tuple[RecordedTour, ...]

    def __post_init__(self) -> None:
        if len({tour.tour_id for tour in self.tours}) != len(self.tours):
            raise ValueError("recorded tour ids must be unique")
        if any(not tour.lines for tour in self.tours):
            raise ValueError("a recorded tour needs at least one line")
        orders = [order for tour in self.tours for order in tour.order_ids]
        if len(set(orders)) != len(orders):
            raise ValueError("an order may belong to one recorded tour only")

    def tour(self, tour_id: int) -> RecordedTour:
        return next(tour for tour in self.tours if tour.tour_id == tour_id)


@dataclass
class PlannerInformation(BaseDomainObject):
    """Named forecasts known to the decision maker at the current decision."""

    tpe: InformationType
    processes: dict[str, Any]

    def require(self, name: str, expected_type: type):
        description = self.processes[name]
        if not isinstance(description, expected_type):
            raise TypeError(f"{name} requires {expected_type.__name__}")
        return description

    def get_features(self) -> dict[str, Any]:
        features = {}
        for name, description in self.processes.items():
            features[f"{name}.type"] = description.representation
            for field in fields(description):
                value = getattr(description, field.name)
                if isinstance(value, tuple):
                    value = len(value)
                if value is not None:
                    features[f"{name}.{field.name}"] = value
        return features
