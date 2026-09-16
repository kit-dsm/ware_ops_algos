from dataclasses import dataclass
from math import comb, isclose
from typing import Dict, List, Tuple

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))

try:
    from Algorithms.analytic_progress import continuous_calculation as cc
    from Algorithms.analytic_progress import core as core_mod
    from Algorithms.analytic_progress.models import WarehouseInstance
except ImportError:
    import continuous_calculation as cc
    import core as core_mod
    from models import WarehouseInstance


@dataclass(frozen=True)
class MultilineDetourEstimate:
    expected_detour: float
    backtrack_part: float
    pass_horizontal_part: float
    vertical_part: float
    kappa: int
    replaceable: bool
    mhit_distribution: List[float] | None = None


def psi(kappa: int, p: float) -> float:
    """Erwarteter Backtrack-Faktor fuer kappa i.i.d. Linien."""
    if p <= 0.0:
        return 0.0
    return 2.0 - (2.0 / ((kappa + 1) * p)) * (1.0 - (1.0 - p) ** (kappa + 1))


def _aisle_weights(aisles: List[int], x: float, w: float) -> List[float]:
    """Horizontale Round-Trip-Laengen der Gassen, aufsteigend sortiert."""
    return sorted(2.0 * abs(float(a) - float(x)) * float(w) for a in aisles)


def _layer_cake_expected_max(weights: List[float], kappa: int, m: int) -> float:
    """Layer-Cake fuer den Maximumsoperator ueber kappa uniforme Ziehungen."""
    if not weights:
        return 0.0

    total = 0.0
    prev = 0.0
    n = len(weights)
    for idx, weight in enumerate(weights):
        n_ge = n - idx
        prob = 1.0 - (1.0 - (n_ge / m)) ** kappa
        total += (weight - prev) * prob
        prev = weight
    return total


def _mhit_prob(j: int, kappa: int, m: int, k: int) -> float:
    """Verteilung von M_hit per Inklusions-Exklusions-Formel."""
    g = m - k
    if j < 0 or j > g:
        return 0.0

    s = 0.0
    for i in range(0, j + 1):
        s += ((-1) ** i) * comb(j, i) * ((k + j - i) / m) ** kappa
    return comb(g, j) * s


def _mhit_distribution(m: int, k: int, kappa: int) -> List[float]:
    g = m - k
    return [_mhit_prob(j, kappa, m, k) for j in range(0, g + 1)]


def _vertical_operator(
    inst: WarehouseInstance,
    route_state,
    kappa: int,
) -> Tuple[float, bool, List[float]]:
    """Geteilter 2L-Operator fuer die Anzahl getroffener nicht besuchter Gassen."""
    m = inst.M
    k = inst.k
    g = m - k
    a_ret = inst.A[0] if (k % 2 == 1) else None
    eps = 1e-9
    if route_state.phase_label == "phase_1_to_first_entry":
        # Anfahrt: eine zusaetzliche Gasse wird ohne vertikalen Umweg integriert
        # (Outbound-Verlaengerung). Daher ersetzbar erzwingen und die
        # x>=a_ret-Bedingung in dieser Phase ueberschreiben.
        replaceable = (a_ret is not None)
    else:
        replaceable = (a_ret is not None) and (route_state.x > a_ret - eps) and (not route_state.is_return_phase)

    probs = _mhit_distribution(m, k, kappa)
    if g <= 0:
        return 0.0, replaceable, probs

    def num_2l(j: int) -> int:
        return (j // 2) if replaceable else ((j + 1) // 2)

    e_2l = sum(probs[j] * num_2l(j) for j in range(0, g + 1))
    return 2.0 * inst.L * e_2l, replaceable, probs


def _horizontal_operator(
    aisles: List[int],
    x: float,
    w: float,
    kappa: int,
    m: int,
) -> float:
    return _layer_cake_expected_max(_aisle_weights(aisles, x, w), kappa, m)


def _phase1_outbound_horizontal(
    g_front: List[int],
    inst: WarehouseInstance,
    kappa: int,
) -> float:
    """Outbound-Horizontale der Anfahrt (Spiegelung des Single-Line-Outbound-Zweigs).

    Nur Front-Gassen JENSEITS der letzten besuchten Gasse a_k = max(A) erzeugen einen
    Umweg, mit Round-Trip-Laenge 2*(a-a_k)*w. Reduziert bei kappa=1 auf
    p4*phase1_weighted und ist 0, sobald keine Gasse jenseits max(A) liegt, weil alle
    Front-Gassen ohnehin auf dem Hinweg erreicht werden.
    """
    if not inst.A:
        return 0.0
    a_k = max(inst.A)
    weights = sorted(2.0 * (float(a) - a_k) * float(inst.w) for a in g_front if a > a_k)
    return _layer_cake_expected_max(weights, kappa, inst.M)


def continuous_expected_detour_multiline(
    t: float,
    inst: WarehouseInstance,
    T_in: Dict[int, float],
    T_out: Dict[int, float],
    n_orderlines: int = 2,
) -> MultilineDetourEstimate:
    """Erwarteter Umweg fuer eine Order mit beliebig vielen Orderlines."""
    if n_orderlines < 1:
        raise ValueError("n_orderlines must be >= 1")

    p1, p2, p3, p4, p_sum, _, _ = cc.continuous_class_probabilities(t, T_in, T_out, inst)
    route_state = cc.estimate_continuous_route_state(t, T_in, T_out, inst)
    x = float(route_state.x)

    g_pass = core_mod.G_pass(x, t, inst)
    g_front = core_mod.G_front(x, t, inst)
    d_backtrack = cc.continuous_backtrack_detour(t, T_in, T_out, inst)

    backtrack_part = psi(n_orderlines, p2) * d_backtrack

    if route_state.phase_label == "phase_1_to_first_entry":
        pass_horizontal_part = _phase1_outbound_horizontal(g_front, inst, n_orderlines)
    else:
        pass_horizontal_part = _horizontal_operator(g_pass, x, inst.w, n_orderlines, inst.M)

    vertical_part, replaceable, mhit_distribution = _vertical_operator(inst, route_state, n_orderlines)
    expected = backtrack_part + pass_horizontal_part + vertical_part

    return MultilineDetourEstimate(
        expected_detour=float(expected),
        backtrack_part=float(backtrack_part),
        pass_horizontal_part=float(pass_horizontal_part),
        vertical_part=float(vertical_part),
        kappa=int(n_orderlines),
        replaceable=bool(replaceable),
        mhit_distribution=[float(p) for p in mhit_distribution],
    )


if __name__ == "__main__":
    inst = WarehouseInstance(
        M=8,
        N_L=16,
        w=1.0,
        L=17.0,
        A=[1, 2, 8],
        n_list=[1, 1, 1],
        v=1.0,
        t_p=0.0,
        P=[],
        arrival_times=[],
    )

    fake_state = cc.ContinuousRouteState(
        x=0.0,
        y=0.0,
        is_return_phase=True,
        phase_label="phase_2_vertical",
        active_j=None,
    )

    orig_probs = cc.continuous_class_probabilities
    orig_backtrack = cc.continuous_backtrack_detour
    orig_state = cc.estimate_continuous_route_state
    orig_g_pass = core_mod.G_pass
    orig_g_front = core_mod.G_front

    try:
        cc.continuous_class_probabilities = lambda t, T_in, T_out, inst: (0.0, 0.3309, 0.5, 0.125, 1.0, 0.0, 1.0)
        cc.continuous_backtrack_detour = lambda t, T_in, T_out, inst: 31.4
        cc.estimate_continuous_route_state = lambda t, T_in, T_out, inst: fake_state
        core_mod.G_pass = lambda x, t, inst: [2, 4, 5, 6]
        core_mod.G_front = lambda x, t, inst: [7]

        est2 = continuous_expected_detour_multiline(0.0, inst, {}, {}, n_orderlines=2)
        assert isclose(est2.backtrack_part, 18.488, abs_tol=1e-2)
        assert isclose(est2.pass_horizontal_part, 6.78125, abs_tol=1e-2)
        assert isclose(est2.vertical_part, 29.21875, abs_tol=1e-2)
        assert isclose(est2.expected_detour, 54.4879, abs_tol=1e-2)

        est1 = continuous_expected_detour_multiline(0.0, inst, {}, {}, n_orderlines=1)
        est3 = continuous_expected_detour_multiline(0.0, inst, {}, {}, n_orderlines=3)
        assert est1.expected_detour < est2.expected_detour < est3.expected_detour

        cc.estimate_continuous_route_state = lambda t, T_in, T_out, inst: cc.ContinuousRouteState(
            x=1.5,
            y=0.0,
            is_return_phase=False,
            phase_label="phase_2_vertical",
            active_j=None,
        )
        est_replaceable = continuous_expected_detour_multiline(0.0, inst, {}, {}, n_orderlines=1)
        assert isclose(est_replaceable.vertical_part, 0.0, abs_tol=1e-12)

        cc.continuous_class_probabilities = lambda t, T_in, T_out, inst: (0.0, 0.25, 0.5, 0.125, 1.0, 0.0, 1.0)
        cc.continuous_backtrack_detour = lambda t, T_in, T_out, inst: 10.0
        cc.estimate_continuous_route_state = lambda t, T_in, T_out, inst: fake_state
        core_mod.G_pass = lambda x, t, inst: [2]
        core_mod.G_front = lambda x, t, inst: [7]
        est_check = continuous_expected_detour_multiline(0.0, inst, {}, {}, n_orderlines=1)
        expected_check = psi(1, 0.25) * 10.0 + est_check.pass_horizontal_part + est_check.vertical_part
        assert isclose(est_check.expected_detour, expected_check, abs_tol=1e-12)

        print("Multiline self-tests passed.")
    finally:
        cc.continuous_class_probabilities = orig_probs
        cc.continuous_backtrack_detour = orig_backtrack
        cc.estimate_continuous_route_state = orig_state
        core_mod.G_pass = orig_g_pass
        core_mod.G_front = orig_g_front
