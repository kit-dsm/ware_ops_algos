"""Forecasts available to algorithms, separate from realized simulation events."""

from dataclasses import dataclass
from math import isfinite
from typing import Any, ClassVar


@dataclass(frozen=True)
class ExponentialSingleLineUniformLocationOrderStream:
    """Exponential arrivals, one order line, and uniformly distributed picks."""

    representation: ClassVar[str] = "exponential_single_line_uniform_location_order_stream"
    mean_interarrival_time_s: float

    def __post_init__(self) -> None:
        mean = self.mean_interarrival_time_s
        if isinstance(mean, bool) or not isinstance(mean, (int, float)) or not isfinite(mean) or mean <= 0:
            raise ValueError("mean_interarrival_time_s must be a finite positive number")


@dataclass
class PlannerInformation:
    """Named forecasts known to the decision maker at the current decision."""

    processes: dict[str, Any]

    def require(self, name: str, expected_type: type):
        description = self.processes[name]
        if not isinstance(description, expected_type):
            raise TypeError(f"{name} requires {expected_type.__name__}")
        return description

    def get_type_value(self) -> str:
        return "process_information"

    def get_features(self) -> dict[str, str]:
        return {name: description.representation for name, description in self.processes.items()}
