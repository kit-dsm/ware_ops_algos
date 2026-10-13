"""Lifecycle and physical revision limits for warehouse plans."""

from enum import Enum


class TourStates(str, Enum):
    PLANNED = "planned"
    SCHEDULED = "scheduled"
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


class ResourceStates(str, Enum):
    OFF_SHIFT = "off_shift"
    IDLE = "idle"
    RESERVED = "reserved"
    BUSY = "busy"


class Accepts(str, Enum):
    NONE = "none"
    INSERT = "insert"
    REBUILD = "rebuild"


class TimeAttribute(str, Enum):
    DUE_DATE = "due_date"
    START_TIME = "start_time"


TOUR_ORDER = (
    TourStates.PLANNED,
    TourStates.SCHEDULED,
    TourStates.PENDING,
    TourStates.STARTED,
    TourStates.DONE,
    TourStates.CANCELLED,
)

TOUR_TRANSITIONS: dict[TourStates, frozenset[TourStates]] = {
    TourStates.PLANNED: frozenset((TourStates.SCHEDULED, TourStates.CANCELLED)),
    TourStates.SCHEDULED: frozenset((TourStates.PENDING, TourStates.CANCELLED)),
    TourStates.PENDING: frozenset((TourStates.STARTED,)),
    TourStates.STARTED: frozenset((TourStates.DONE,)),
    TourStates.DONE: frozenset(),
    TourStates.CANCELLED: frozenset(),
}

PHYSICAL: dict[TourStates, Accepts] = {
    TourStates.PLANNED: Accepts.REBUILD,
    TourStates.SCHEDULED: Accepts.REBUILD,
    TourStates.PENDING: Accepts.INSERT,
    TourStates.STARTED: Accepts.INSERT,
    TourStates.DONE: Accepts.NONE,
    TourStates.CANCELLED: Accepts.NONE,
}

_ACCEPTS_ORDER = (Accepts.NONE, Accepts.INSERT, Accepts.REBUILD)


def accepts_at_least(actual: Accepts, required: Accepts) -> bool:
    """Whether a revision level permits the required change."""
    return _ACCEPTS_ORDER.index(actual) >= _ACCEPTS_ORDER.index(required)


def accepts_for(status: TourStates, freeze_from: TourStates,
                frozen_accepts: Accepts) -> Accepts:
    """Return the configured revision level at a lifecycle status."""
    if status in (TourStates.DONE, TourStates.CANCELLED):
        return Accepts.NONE
    if TOUR_ORDER.index(status) < TOUR_ORDER.index(freeze_from):
        return Accepts.REBUILD
    return frozen_accepts