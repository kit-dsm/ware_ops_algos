"""Validated single-line stochastic waiting cost and optimum."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log

from .interval_builder import IntervalData, build_interval_data
from . import geometry as g
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...algorithm_interfaces import WaitingInput


@dataclass(frozen=True)
class AnalyticWaitingResult:
    wait_duration: float
    expected_cost: float
    immediate_dispatch_cost: float
    route_duration: float
    critical_mean_interarrival_time: float | None


def _integral_quadratic_exponential(
    coefficients: tuple[float, float, float],
    rate: float,
    duration: float,
) -> float:
    c0, c1, c2 = coefficients
    exponential = exp(-rate * duration)
    scaled_duration = rate * duration
    i0 = (1.0 - exponential) / rate
    i1 = (
        1.0 - exponential * (1.0 + scaled_duration)
    ) / rate**2
    i2 = (
        2.0
        - exponential
        * (2.0 + 2.0 * scaled_duration + scaled_duration**2)
    ) / rate**3
    return c0 * i0 + c1 * i1 + c2 * i2


def _cost_constant(
    rate: float,
    intervals: list[IntervalData],
    route_duration: float,
    remaining_arrivals_after_miss: int,
) -> float:
    cost = 0.0
    for interval in intervals:
        duration = interval.t_end - interval.t_start
        integral = _integral_quadratic_exponential(
            interval.ed_coeffs, rate, duration
        )
        cost += rate * exp(-rate * interval.t_start) * integral

    cost_after_route = remaining_arrivals_after_miss / rate + route_duration
    return cost + cost_after_route * exp(-rate * route_duration)


def _initial_cost_derivative(
    mean_interarrival_time: float,
    intervals: list[IntervalData],
    route_duration: float,
    remaining_arrivals_after_miss: int,
) -> float:
    rate = 1.0 / mean_interarrival_time
    constant = _cost_constant(
        rate,
        intervals,
        route_duration,
        remaining_arrivals_after_miss,
    )
    detour_at_depot = intervals[0].ed_coeffs[0] if intervals else 0.0
    return 1.0 - rate * (constant - detour_at_depot)


def critical_mean_interarrival_time(
    data: WaitingInput,
    remaining_arrivals_after_miss: int = 3,
    search_bounds: tuple[float, float] = (1e-3, 1e5),
) -> float | None:
    """Find the interarrival mean where the optimal decision changes."""
    lower, upper = map(float, search_bounds)
    if lower <= 0.0 or upper <= lower:
        raise ValueError("search_bounds must be positive and increasing")
    intervals, route_duration = build_interval_data(data)

    def derivative(value: float) -> float:
        return _initial_cost_derivative(
            value,
            intervals,
            route_duration,
            remaining_arrivals_after_miss,
        )

    low_value = derivative(lower)
    high_value = derivative(upper)
    if low_value == 0.0:
        return lower
    if high_value == 0.0:
        return upper
    if low_value * high_value > 0.0:
        return None
    for _ in range(80):
        midpoint = (lower + upper) / 2.0
        mid_value = derivative(midpoint)
        if abs(mid_value) < 1e-12:
            return midpoint
        if low_value * mid_value <= 0.0:
            upper = midpoint
        else:
            lower = midpoint
            low_value = mid_value
    return (lower + upper) / 2.0


def expected_waiting_cost(
    wait_duration: float,
    mean_interarrival_time: float,
    intervals: list[IntervalData],
    route_duration: float,
    remaining_arrivals_after_miss: int = 3,
) -> float:
    """Evaluate the project_4D4L analytical objective ``J(wait)``."""
    if wait_duration < 0:
        raise ValueError("wait_duration must be non-negative")
    if mean_interarrival_time <= 0:
        raise ValueError("mean_interarrival_time must be positive")
    if remaining_arrivals_after_miss < 0:
        raise ValueError("remaining_arrivals_after_miss must be non-negative")

    rate = 1.0 / mean_interarrival_time
    constant = _cost_constant(
        rate,
        intervals,
        route_duration,
        remaining_arrivals_after_miss,
    )
    detour_at_depot = intervals[0].ed_coeffs[0] if intervals else 0.0
    return (
        wait_duration
        + detour_at_depot * (1.0 - exp(-rate * wait_duration))
        + constant * exp(-rate * wait_duration)
    )


def solve_optimal_wait(
    data: WaitingInput,
    mean_interarrival_time: float,
    remaining_arrivals_after_miss: int = 3,
) -> AnalyticWaitingResult:
    """Return the closed-form optimum of the validated single-line model."""
    if mean_interarrival_time <= 0:
        raise ValueError("mean_interarrival_time must be positive")
    if remaining_arrivals_after_miss < 0:
        raise ValueError("remaining_arrivals_after_miss must be non-negative")

    intervals, route_duration = build_interval_data(data)
    rate = 1.0 / mean_interarrival_time
    constant = _cost_constant(
        rate,
        intervals,
        route_duration,
        remaining_arrivals_after_miss,
    )
    detour_at_depot = intervals[0].ed_coeffs[0] if intervals else 0.0
    scale = rate * (constant - detour_at_depot)
    wait_duration = log(scale) / rate if scale > 1.0 else 0.0
    expected_cost = expected_waiting_cost(
        wait_duration,
        mean_interarrival_time,
        intervals,
        route_duration,
        remaining_arrivals_after_miss,
    )
    immediate_cost = expected_waiting_cost(
        0.0,
        mean_interarrival_time,
        intervals,
        route_duration,
        remaining_arrivals_after_miss,
    )
    return AnalyticWaitingResult(
        wait_duration=float(wait_duration),
        expected_cost=float(expected_cost),
        immediate_dispatch_cost=float(immediate_cost),
        route_duration=float(route_duration),
        critical_mean_interarrival_time=critical_mean_interarrival_time(
            data,
            remaining_arrivals_after_miss,
        ),
    )
