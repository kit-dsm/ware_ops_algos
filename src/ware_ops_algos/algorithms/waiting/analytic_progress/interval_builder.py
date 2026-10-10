"""Piecewise-quadratic representation of the single-line detour model."""

from __future__ import annotations

from dataclasses import dataclass

from . import continuous_calculation
from .core import compute_segment_times
from . import geometry as g
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...algorithm_interfaces import WaitingInput


@dataclass(frozen=True)
class IntervalData:
    """``E[D](s) = c0 + c1*s + c2*s**2`` on one route phase."""

    t_start: float
    t_end: float
    phase_label: str
    ed_coeffs: tuple[float, float, float]


_BOUNDARY_OFFSET = 1e-8


def _lagrange_quadratic(
    samples: list[tuple[float, float]],
) -> tuple[float, float, float]:
    if len(samples) == 1:
        return samples[0][1], 0.0, 0.0
    if len(samples) == 2:
        (s1, y1), (s2, y2) = samples
        if abs(s2 - s1) < 1e-15:
            return y1, 0.0, 0.0
        c1 = (y2 - y1) / (s2 - s1)
        return y1 - c1 * s1, c1, 0.0

    (s1, y1), (s2, y2), (s3, y3) = samples[:3]
    d12 = s1 - s2
    d13 = s1 - s3
    d23 = s2 - s3
    if abs(d12) < 1e-15 or abs(d13) < 1e-15 or abs(d23) < 1e-15:
        return _lagrange_quadratic(samples[:2])

    a1 = y1 / (d12 * d13)
    a2 = y2 / (-d12 * d23)
    a3 = y3 / (d13 * d23)
    return (
        a1 * s2 * s3 + a2 * s1 * s3 + a3 * s1 * s2,
        -(a1 * (s2 + s3) + a2 * (s1 + s3) + a3 * (s1 + s2)),
        a1 + a2 + a3,
    )


def _sample_phase(
    t_start: float,
    t_end: float,
    data: WaitingInput,
    times_in: dict[int, float],
    times_out: dict[int, float],
) -> tuple[float, float, float]:
    duration = t_end - t_start
    if duration <= 4.0 * _BOUNDARY_OFFSET:
        estimate = continuous_calculation.continuous_expected_detour(
            0.5 * (t_start + t_end), data, times_in, times_out
        )
        return float(estimate.expected_detour), 0.0, 0.0

    local_times = (
        0.25 * duration,
        0.5 * duration,
        duration - _BOUNDARY_OFFSET,
    )
    samples = []
    for local_time in local_times:
        estimate = continuous_calculation.continuous_expected_detour(
            t_start + local_time, data, times_in, times_out
        )
        samples.append((local_time, float(estimate.expected_detour)))
    return _lagrange_quadratic(samples)


def _interval(
    t_start: float,
    t_end: float,
    label: str,
    data: WaitingInput,
    times_in: dict[int, float],
    times_out: dict[int, float],
) -> IntervalData:
    return IntervalData(
        t_start=float(t_start),
        t_end=float(t_end),
        phase_label=label,
        ed_coeffs=_sample_phase(
            t_start, t_end, data, times_in, times_out
        ),
    )


def build_interval_data(
    data: WaitingInput,
) -> tuple[list[IntervalData], float]:
    """Build the route phases used by the validated analytical cost model."""
    times_in, times_out, route_duration = compute_segment_times(data)
    sequence = [
        index for index, _ in sorted(times_in.items(), key=lambda item: item[1])
    ]
    intervals: list[IntervalData] = []
    first = sequence[0]
    intervals.append(
        _interval(
            0.0,
            times_in[first],
            "phase_1_to_first_entry",
            data,
            times_in,
            times_out,
        )
    )

    has_return_aisle = g.k(data) % 2 == 1
    for rank, index in enumerate(sequence):
        t_in = times_in[index]
        t_out = times_out[index]
        if has_return_aisle and index == sequence[-1]:
            vertical_speed = (
                continuous_calculation.effective_vertical_speed_for_aisle(
                    data, index
                )
            )
            split = t_in + g.L(data) / vertical_speed
            intervals.append(
                _interval(
                    t_in,
                    split,
                    "phase_n_2_ret_up",
                    data,
                    times_in,
                    times_out,
                )
            )
            intervals.append(
                _interval(
                    split,
                    t_out,
                    "phase_n_1_ret_down",
                    data,
                    times_in,
                    times_out,
                )
            )
        else:
            intervals.append(
                _interval(
                    t_in,
                    t_out,
                    "phase_2_vertical",
                    data,
                    times_in,
                    times_out,
                )
            )

        if rank < g.k(data) - 1:
            next_index = sequence[rank + 1]
            intervals.append(
                _interval(
                    t_out,
                    times_in[next_index],
                    "phase_3_horizontal",
                    data,
                    times_in,
                    times_out,
                )
            )

    last = sequence[-1]
    intervals.append(
        _interval(
            times_out[last],
            route_duration,
            "phase_4_after_last_aisle",
            data,
            times_in,
            times_out,
        )
    )
    return intervals, float(route_duration)
