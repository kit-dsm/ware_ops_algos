"""Information about uncertain processes available to a decision maker.

The objects here describe assumptions, never realized future events.  A process
identifier distinguishes, for example, an incoming-order stream from a future
picker-performance model; the value's concrete type states its semantics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import ClassVar


class ProcessInformation:
    """Marker for a typed, immutable description of one uncertain process."""

    representation: ClassVar[str]


@dataclass(frozen=True)
class ExponentialSingleLineUniformLocationOrderStream(ProcessInformation):
    """I.i.d. exponential interarrivals; one line and uniform location per order.

    The mean is a planner assumption.  No future order or random draw is held
    here.  These are exactly the assumptions of the analytical waiting method.
    """

    representation: ClassVar[str] = "exponential_single_line_uniform_location_order_stream"
    mean_interarrival_time_s: float

    def __post_init__(self) -> None:
        mean = self.mean_interarrival_time_s
        if isinstance(mean, bool) or not isinstance(mean, (int, float)) or not isfinite(mean) or mean <= 0:
            raise ValueError("mean_interarrival_time_s must be a finite positive number")


@dataclass(frozen=True)
class PlannerInformation:
    """Named process descriptions; a tuple prevents replacement or duplicates."""

    processes: tuple[tuple[str, ProcessInformation], ...]

    def __post_init__(self) -> None:
        entries = tuple(self.processes)
        names: set[str] = set()
        for entry in entries:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise ValueError("Each planner-information entry must be (process_id, description)")
            name, description = entry
            if not isinstance(name, str) or not name or not name.isidentifier():
                raise ValueError(f"Invalid process identifier: {name!r}")
            if name in names:
                raise ValueError(f"Duplicate process identifier: {name}")
            if not isinstance(description, ProcessInformation):
                raise TypeError(f"{name} must have a typed process description")
            names.add(name)
        object.__setattr__(self, "processes", entries)

    def require(self, name: str, expected_type: type[ProcessInformation]):
        """Return a process only when its representation matches exactly."""
        for process_name, description in self.processes:
            if process_name == name:
                if type(description) is not expected_type:
                    raise ValueError(f"{name} requires {expected_type.__name__}, got {type(description).__name__}")
                return description
        raise ValueError(f"Missing planner information for {name}")

    def get_type_value(self) -> str:
        return "process_information"

    def get_features(self) -> dict[str, str]:
        return {name: description.representation for name, description in self.processes}


def parse_planner_information(raw: Mapping | None) -> PlannerInformation | None:
    """Read the small data-card schema and reject unsupported representations."""
    if raw is None:
        return None
    if (not isinstance(raw, Mapping) or set(raw) != {"processes"}
            or not isinstance(raw["processes"], Sequence)
            or isinstance(raw["processes"], (str, bytes))):
        raise ValueError("information must contain a processes list")
    entries = []
    for process in raw["processes"]:
        if not isinstance(process, Mapping):
            raise ValueError("Each information process must be a mapping")
        name = process.get("id")
        representation = process.get("type")
        if representation != ExponentialSingleLineUniformLocationOrderStream.representation:
            raise ValueError(f"Unknown planner-information type: {representation!r}")
        expected = {"id", "type", "mean_interarrival_time_s"}
        if set(process) != expected:
            raise ValueError(f"{name}: expected fields {sorted(expected)}, got {sorted(process)}")
        entries.append((name, ExponentialSingleLineUniformLocationOrderStream(
            mean_interarrival_time_s=process["mean_interarrival_time_s"]
        )))
    return PlannerInformation(tuple(entries))


def information_card_section(raw: Mapping | None) -> dict:
    """Expose exact process representations to existing card matching."""
    information = parse_planner_information(raw)
    return {"type": information.get_type_value() if information else None,
            "features": information.get_features() if information else {}}
