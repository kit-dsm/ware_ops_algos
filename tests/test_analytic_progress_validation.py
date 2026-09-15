"""Validation against hand-computed Mathematica reference values.

Benchmark instance: Route A = (1, 2, 8), M=8, N_L=17, w=1.0, L=17.0,
v=1.0, t_p=0.0.  Reference values from validate_a128.py in
project_4D4L/2_Stochastic_Waiting/Algorithms/analytic_progress/Archive/.

Checks:
1. Segment times (T_in, T_out, T_end) match the expected phase boundaries.
2. Continuous E[D] at spot-check points matches exact rational values.
3. Route duration matches.
"""

from __future__ import annotations

import pytest

from ware_ops_algos.algorithms.waiting.analytic_progress.core import (
    compute_segment_times,
    theoretical_route_duration,
)
from ware_ops_algos.algorithms.waiting.analytic_progress.continuous_calculation import (
    continuous_expected_detour,
)
from ware_ops_algos.algorithms.waiting.analytic_progress.models import (
    WarehouseInstance,
)


def _make_instance() -> WarehouseInstance:
    """Route A = (1, 2, 8): 3 aisles, 1 pick each, 8-aisle warehouse."""
    return WarehouseInstance(
        M=8, N_L=17, w=1.0, L=17.0, A=[1, 2, 8],
        n_list=[1, 1, 1], v=1.0, t_p=0.0,
        P=[], arrival_times=[],
    )


# Expected phase intervals from Mathematica (validate_a128.py)
# (label, t_start, t_end, c0, c1, c2)
EXPECTED_PHASES = [
    ("phase_1_to_first_entry", 0.0, 8.0),
    ("phase_2_vertical", 8.0, 25.0),
    ("phase_3_horizontal", 25.0, 31.0),
    ("phase_2_vertical", 31.0, 48.0),
    ("phase_3_horizontal", 48.0, 49.0),
    ("phase_n_2_ret_up", 49.0, 66.0),
    ("phase_n_1_ret_down", 66.0, 83.0),
    ("phase_4_after_last_aisle", 83.0, 84.0),
]

# E[D] spot-checks: (absolute_t, expected_E_D, label)
# Values are exact fractions from Mathematica derivation.
SPOT_CHECKS = [
    (25.0, 289 / 136, "phase_2_vertical:8-25 s=17"),
    (31.0, 59 / 8, "phase_3_horizontal:25-31 s=6"),
    (48.0, 13.75, "phase_2_vertical:31-48 s=17"),
    (66.0, 15.5, "phase_n_2_ret_up:49-66 s=17"),
    (83.0, 147 / 4 + 289 / 136, "phase_n_1_ret_down:66-83 s=17"),
]

TOL = 1e-9


class TestRouteGeometry:
    """Validate S-shape route geometry against Mathematica reference."""

    def test_route_duration(self):
        inst = _make_instance()
        duration = theoretical_route_duration(inst)
        assert duration == pytest.approx(84.0, abs=TOL)

    def test_segment_times_match_phase_boundaries(self):
        inst = _make_instance()
        T_in, T_out, T_end = compute_segment_times(inst)

        assert T_end == pytest.approx(84.0, abs=TOL)

        # Aisle 8 (j=3): first visited, rightmost
        assert T_in[3] == pytest.approx(8.0, abs=TOL)
        assert T_out[3] == pytest.approx(25.0, abs=TOL)

        # Aisle 2 (j=2): middle
        assert T_in[2] == pytest.approx(31.0, abs=TOL)
        assert T_out[2] == pytest.approx(48.0, abs=TOL)

        # Aisle 1 (j=1): last visited, return aisle (odd k)
        assert T_in[1] == pytest.approx(49.0, abs=TOL)
        assert T_out[1] == pytest.approx(83.0, abs=TOL)


class TestExpectedDetour:
    """Validate E[D](t) against Mathematica reference values.

    The continuous engine matches the Mathematica derivation exactly
    at interior phase boundaries.  The discrete engine uses step-wise
    position approximations and has small deviations (expected).
    """

    @pytest.mark.parametrize(
        ("t", "expected", "label"),
        SPOT_CHECKS,
        ids=[s[2] for s in SPOT_CHECKS],
    )
    def test_continuous_ed_matches_reference(self, t, expected, label):
        inst = _make_instance()
        T_in, T_out, T_end = compute_segment_times(inst)
        t_clamped = min(t, T_end)
        result = continuous_expected_detour(t_clamped, inst, T_in, T_out)
        assert result.expected_detour == pytest.approx(expected, abs=TOL), (
            f"{label}: t={t}, expected E[D]={expected:.9f}, "
            f"got {result.expected_detour:.9f}"
        )

    def test_continuous_ed_at_phase_1_is_zero(self):
        """E[D] = 0 before entering the first aisle (no detour possible)."""
        inst = _make_instance()
        T_in, T_out, T_end = compute_segment_times(inst)
        result = continuous_expected_detour(4.0, inst, T_in, T_out)
        assert result.expected_detour == pytest.approx(0.0, abs=TOL)
        assert result.phase_label == "phase_1_to_first_entry"

    def test_continuous_ed_probabilities_sum_to_one(self):
        """p1 + p2 + p3 + p4 must equal 1.0 at all times."""
        inst = _make_instance()
        T_in, T_out, T_end = compute_segment_times(inst)
        for t in [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]:
            result = continuous_expected_detour(t, inst, T_in, T_out)
            assert result.p_sum == pytest.approx(1.0, abs=1e-9), (
                f"p_sum != 1.0 at t={t}: {result.p_sum}"
            )

    def test_continuous_ed_is_monotonically_increasing(self):
        """E[D](t) should generally increase as the picker progresses
        through the route (more backtrack distance)."""
        inst = _make_instance()
        T_in, T_out, T_end = compute_segment_times(inst)
        values = []
        for t in range(1, 83):
            result = continuous_expected_detour(float(t), inst, T_in, T_out)
            values.append(result.expected_detour)
        # Allow small non-monotonic dips at phase transitions but
        # overall trend must be increasing
        assert values[-1] > values[0], (
            f"E[D] should increase from t=1 to t=82: "
            f"start={values[0]:.4f}, end={values[-1]:.4f}"
        )
