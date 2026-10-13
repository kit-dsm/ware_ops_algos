"""Shared lifecycle vocabulary for warehouse planning and simulation.

The simulator derives statuses from its state managers. Algorithms may inspect
these values, but the physical limits below are enforced when CASIM commits a
decision.
"""

from enum import Enum


class TourStates(str, Enum):
    PLANNED = "planned"
    ASSIGNED = "assigned"
    SCHEDULED = "scheduled"
    RELEASED = "released"
    PENDING = "pending"
    STARTED = "started"
    DONE = "done"
    CANCELLED = "cancelled"


class OrderStates(str, Enum):
    BUFFERED = "buffered"
    BATCHED = "batched"
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    DONE = "done"


class PickListStates(str, Enum):
    BUFFERED = "buffered"
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    DONE = "done"


class ResourceStates(str, Enum):
    OFF_SHIFT = "off_shift"
    IDLE = "idle"
    RESERVED = "reserved"
    BUSY = "busy"


class Accepts(str, Enum):
    REBUILD = "rebuild"
    REVISE = "revise"
    INSERT = "insert"
    NONE = "none"


class TimeAttribute(str, Enum):
    DUE_DATE = "due_date"
    START_TIME = "start_time"
    ORDER_DATE = "order_date"


TOUR_ORDER = (
    TourStates.PLANNED,
    TourStates.ASSIGNED,
    TourStates.SCHEDULED,
    TourStates.RELEASED,
    TourStates.PENDING,
    TourStates.STARTED,
    TourStates.DONE,
)

_FREE = frozenset(Accepts)
_FROZEN = frozenset((Accepts.REVISE, Accepts.INSERT, Accepts.NONE))
PHYSICAL: dict[TourStates, frozenset[Accepts]] = {
    TourStates.PLANNED: _FREE,
    TourStates.ASSIGNED: _FREE,
    TourStates.SCHEDULED: _FREE,
    TourStates.RELEASED: _FREE,
    TourStates.PENDING: _FROZEN,
    TourStates.STARTED: _FROZEN,
    TourStates.DONE: frozenset((Accepts.NONE,)),
    TourStates.CANCELLED: frozenset((Accepts.NONE,)),
}


def accepts_for(
    status: TourStates,
    freeze_from: TourStates,
    frozen_accepts: Accepts,
) -> Accepts:
    """Return the revision right for a selected tour at this lifecycle stage."""
    if status is TourStates.CANCELLED:
        return Accepts.NONE
    frozen = TOUR_ORDER.index(status) >= TOUR_ORDER.index(freeze_from)
    return frozen_accepts if frozen else Accepts.REBUILD
