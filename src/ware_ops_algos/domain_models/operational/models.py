from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Optional

from ware_ops_algos.domain_models.lifecycle import TourStates

if TYPE_CHECKING:
    from ware_ops_algos.algorithms.algorithm_interfaces import (
        BatchObject, PickPosition, Route, RouteNode, RoutingOrigin, ScheduledJob,
    )


@dataclass(frozen=True)
class RevisionRights:
    """Describes to what degree an operational plan can be revised"""
    membership: Literal["replace", "add_only", "fixed"] = "replace"
    route: Literal["replace", "remaining", "fixed"] = "replace"
    picker: bool = True
    timing: bool = True


FIXED = RevisionRights(
    membership="fixed",
    route="fixed",
    picker=False,
    timing=False,
)


@dataclass
class PlanEntry:
    id: int
    decision: BatchObject | Route | ScheduledJob
    revision: RevisionRights = field(default_factory=RevisionRights)
    released: bool = False
    tour_id: int | None = None

    @property
    def batch(self) -> BatchObject:
        from ware_ops_algos.algorithms.algorithm_interfaces import BatchObject, Route

        if isinstance(self.decision, BatchObject):
            return self.decision
        if isinstance(self.decision, Route):
            return self.decision.batch
        return self.decision.job.route.batch


@dataclass
class OperationalPlan:
    entries: dict[int, PlanEntry] = field(default_factory=dict)
    next_id: int = 1

    def add_batch(self, batch: BatchObject):
        entry = PlanEntry(
            id=self.next_id,
            decision=batch,
        )
        self.entries[entry.id] = entry
        self.next_id += 1
        return entry

    def replace_draft(self, jobs: list[ScheduledJob]) -> None:
        """Replace proposals while retaining released execution history."""
        self.entries = {key: entry for key, entry in self.entries.items()
                        if entry.released}
        order_ids: set[int] = set()
        for job in jobs:
            overlapping = order_ids.intersection(job.order_numbers)
            if overlapping:
                raise ValueError(f"Orders occur in more than one draft job: {overlapping}")
            order_ids.update(job.order_numbers)
            self.entries[self.next_id] = PlanEntry(id=self.next_id, decision=job)
            self.next_id += 1

    def invalidate_draft(self) -> None:
        self.entries = {key: entry for key, entry in self.entries.items()
                        if entry.released}

    def release_next_per_picker(self, picker_ids: set[int]) -> list[PlanEntry]:
        """Release at most one earliest draft job for each available picker."""
        from ware_ops_algos.algorithms.algorithm_interfaces import ScheduledJob

        chosen: list[PlanEntry] = []
        for picker_id in sorted(picker_ids):
            eligible = [entry for entry in self.entries.values()
                        if not entry.released
                        and isinstance(entry.decision, ScheduledJob)
                        and entry.decision.picker_id == picker_id]
            if eligible:
                entry = min(eligible, key=lambda e: (e.decision.start_time, e.id))
                entry.released = True
                chosen.append(entry)
        return chosen


@dataclass
class TourPlanningState:
    """
    Thin wrapper around a Route object.
    Keeps track of the planning state for a single tour.

    - route_nodes / pick_sequence are copies from the plan (immutable intent).
    - cursor / picks_left / version are the mutable execution state.
    - original_route is kept only for debugging/inspection (do not mutate).
    """
    tour_id: int

    # original plan (copied from Route)
    order_numbers: list[int]
    original_route: Route
    batch: BatchObject
    annotated_route: list[RouteNode]

    assigned_resource: Optional[int] = None
    start_time: Optional[float] = None
    processing_time: Optional[float] = None
    # planning_plan: Optional[TourPlanningState] = None
    end_time: Optional[float] = None
    end_time_planned: Optional[float] = None
    # execution state, mutable during picking
    cursor: int = 0                 # index into route_nodes
    status: str = TourStates.PLANNED
    route_version: int = 0
    remaining_picks: list[PickPosition] = field(default_factory=list)
    completed_picks: list[PickPosition] = field(default_factory=list)
    edge_origin: RouteNode | None = None
    edge_destination: RouteNode | None = None
    edge_start_time: float | None = None
    edge_end_time: float | None = None
    edge_distance: float | None = None
    picking_until: float | None = None
    cart_bins: dict[int, int] = field(default_factory=dict)
    routing_origin: RoutingOrigin | None = None
    positioning_mode: str | None = None
    positioning_hold: bool = False

    def position_at(self, time: float) -> tuple[float, float]:
        if self.edge_origin is None:
            return self.current_node().position
        duration = self.edge_end_time - self.edge_start_time
        fraction = 1.0 if duration == 0 else min(1.0, max(0.0, (time - self.edge_start_time) / duration))
        start, end = self.edge_origin.position, self.edge_destination.position
        return (start[0] + fraction * (end[0] - start[0]),
                start[1] + fraction * (end[1] - start[1]))

    def current_node(self) -> RouteNode:
        return self.annotated_route[self.cursor]

    def at_end(self) -> bool:
        """True if cursor is on the final node (typically the depot)."""
        return self.cursor >= len(self.annotated_route) - 1

    def next_node(self) -> RouteNode:
        return self.annotated_route[self.cursor + 1]


Node = tuple[float, float]
