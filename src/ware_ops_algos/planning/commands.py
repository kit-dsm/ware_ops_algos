"""Typed, simulator-independent edits to a warehouse plan."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from ware_ops_algos.algorithms import BatchObject, Route, ScheduledJob


@dataclass(frozen=True)
class CreatePickList:
    batch: BatchObject


@dataclass(frozen=True)
class CreateTour:
    job: ScheduledJob


@dataclass(frozen=True)
class CancelTour:
    tour_id: int


@dataclass(frozen=True)
class ReplaceSuffix:
    tour_id: int
    route: Route
    added_order_ids: tuple[int, ...]


@dataclass(frozen=True)
class Reschedule:
    tour_id: int
    picker_id: int
    start_time: float
    end_time: float


@dataclass(frozen=True)
class CommitTour:
    tour_id: int | None = None
    creation_index: int | None = None

    def __post_init__(self) -> None:
        if (self.tour_id is None) == (self.creation_index is None):
            raise ValueError("CommitTour requires exactly one tour_id or creation_index")


Command: TypeAlias = (CreatePickList | CreateTour | CancelTour |
                      ReplaceSuffix | Reschedule | CommitTour)


class ViolationCode(str, Enum):
    STATE_INFORMATION = "state_information"
    ACCEPTS = "accepts"
    PHYSICAL = "physical"
    EXECUTED_PREFIX = "executed_prefix"
    MEMBERSHIP = "membership"


@dataclass(frozen=True)
class Violation:
    code: ViolationCode
    tour_id: int | None
    message: str
