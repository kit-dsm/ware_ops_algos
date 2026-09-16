from dataclasses import dataclass
from math import comb, isclose
from typing import Dict, List, Tuple

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../..")))

try:
    from Algorithms.analytic_progress import detour as detour_mod
    from Algorithms.analytic_progress import continuous_calculation as cc
    from Algorithms.analytic_progress import core as core_mod
    from Algorithms.analytic_progress.models import WarehouseInstance
except ImportError:
    import detour as detour_mod
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
    """Erwarteter Backtrack-Faktor fuer kappa i.i.d. Orderlines."""
    if p <= 0.0:
        return 0.0
    return 2.0 - (2.0 / ((kappa + 1) * p)) * (1.0 - (1.0 - p) ** (kappa + 1))


def _horizontal_weights(g_list: List[int], x_start: float, w: float) -> List[float]:
    """Exakte horizontale Round-Trip-Laengen pro Gasse."""
    return sorted(2.0 * abs(float(a) - float(x_start)) * float(w) for a in g_list)


def _layer_cake(weights: List[float], kappa: int, m: int) -> float:
    """Layer-Cake-Maximumsoperator ueber kappa uniforme Ziehungen."""
    weights = sorted(weights)
    if not weights:
        return 0.0

    total = 0.0
    prev = 0.0
    n = len(weights)
    for idx, weight in enumerate(weights):
        n_ge = n - idx
        total += (weight - prev) * (1.0 - (1.0 - (n_ge / m)) ** kappa)
        prev = weight
    return total


def _lambda_horizontal(g_list: List[int], x_start: float, w: float, kappa: int, m: int) -> float:
    """Layer-Cake-Operator ueber die exakten Gassenabstaende (Round-Trip ab x_start)."""
    return _layer_cake(_horizontal_weights(g_list, x_start, w), kappa, m)


def _phase1_outbound_horizontal(g_front: List[int], inst: WarehouseInstance, kappa: int) -> float:
    """Outbound-Horizontale der Anfahrt: nur Front-Gassen JENSEITS a_k=max(A),
    Round-Trip-Laenge 2*(a-a_k)*w. 0, sobald keine Gasse jenseits max(A) liegt."""
    if not inst.A:
        return 0.0
    a_k = max(inst.A)
    weights = [2.0 * (float(a) - a_k) * float(inst.w) for a in g_front if a > a_k]
    return _layer_cake(weights, kappa, inst.M)


def _mhit_prob(j: int, kappa: int, m: int, k: int) -> float:
    """Wahrscheinlichkeit fuer genau j getroffene nicht besuchte Gassen."""
    g = m - k
    if j < 0 or j > g:
        return 0.0

    s = 0.0
    for i in range(0, j + 1):
        s += ((-1) ** i) * comb(j, i) * ((k + j - i) / m) ** kappa
    return comb(g, j) * s


def _vertical_operator(
    inst: WarehouseInstance,
    x_start: float,
    is_return_phase: bool,
    kappa: int,
    outbound: bool = False,
) -> Tuple[float, bool, List[float]]:
    """Geteilte 2L-Regel ueber die Verteilung der distinct getroffenen Gassen."""
    m = inst.M
    k = inst.k
    g = m - k
    a_ret = sorted(inst.A)[0] if (k % 2 == 1) else None
    eps = 1e-9
    if outbound:
        # Anfahrt: Outbound-Integration einer zusaetzlichen Gasse ist "frei"
        # -> ersetzbar erzwingen, x>=a_ret-Bedingung in dieser Phase ueberschreiben.
        replaceable = (a_ret is not None)
    else:
        replaceable = (a_ret is not None) and (x_start > a_ret - eps) and (not is_return_phase)

    probs = [_mhit_prob(j, kappa, m, k) for j in range(0, g + 1)]
    if g <= 0:
        return 0.0, replaceable, probs

    def num_2l(j: int) -> int:
        return (j // 2) if replaceable else ((j + 1) // 2)

    e_2l = sum(probs[j] * num_2l(j) for j in range(0, g + 1))
    return 2.0 * inst.L * e_2l, replaceable, probs


def calculate_detour_multiline(
    inst: WarehouseInstance,
    t: float,
    xStart: float,
    yStart: float,
    gFront: List[int],
    gPass: List[int],
    v_nodes_count: int,
    r_nodes_count: int,
    n_orderlines: int = 2,
    isReturnPhase: bool = False,
    outbound: bool | None = None,
) -> MultilineDetourEstimate:
    """Erwarteten Umweg fuer mehrere Orderlines auf Basis der exakten diskreten Eingaben."""
    if n_orderlines < 1:
        raise ValueError("n_orderlines must be >= 1")

    total_nodes = inst.N_L * inst.M
    if total_nodes <= 0:
        raise ValueError("N = N_L * M must be > 0")

    p2 = v_nodes_count / total_nodes
    d_backtrack = 2.0 * detour_mod.compute_detour_route_back(inst, xStart, yStart, t, isReturnPhase)
    backtrack_part = psi(n_orderlines, p2) * d_backtrack

    if outbound is None:
        T_in, T_out, _ = core_mod.compute_segment_times(inst)
        route_state = cc.estimate_continuous_route_state(t, T_in, T_out, inst)
        outbound = route_state.phase_label == "phase_1_to_first_entry"
    if outbound:
        lambda_horizontal = _phase1_outbound_horizontal(gFront, inst, n_orderlines)
    else:
        lambda_horizontal = _lambda_horizontal(gPass, xStart, inst.w, n_orderlines, inst.M)

    vertical_part, replaceable, mhit_distribution = _vertical_operator(
        inst, xStart, isReturnPhase, n_orderlines, outbound=outbound
    )
    expected = backtrack_part + lambda_horizontal + vertical_part

    return MultilineDetourEstimate(
        expected_detour=float(expected),
        backtrack_part=float(backtrack_part),
        pass_horizontal_part=float(lambda_horizontal),
        vertical_part=float(vertical_part),
        kappa=int(n_orderlines),
        replaceable=bool(replaceable),
        mhit_distribution=[float(p) for p in mhit_distribution],
    )


if __name__ == "__main__":
    inst = WarehouseInstance(
        M=8,
        N_L=12500,
        w=1.0,
        L=17.0,
        A=[1, 2, 8],
        n_list=[0, 0, 0],
        v=1.0,
        t_p=0.0,
        P=[],
        arrival_times=[],
    )

    orig_route_back = detour_mod.compute_detour_route_back

    try:
        detour_mod.compute_detour_route_back = lambda inst, xStart, yStart, t, isReturnPhase: 15.7

        est2 = calculate_detour_multiline(
            inst=inst,
            t=0.0,
            xStart=0.0,
            yStart=0.0,
            gFront=[7],
            gPass=[2, 4, 5, 6],
            v_nodes_count=33090,
            r_nodes_count=0,
            n_orderlines=2,
            isReturnPhase=False,
            outbound=False,
        )
        assert isclose(est2.backtrack_part, 18.488, abs_tol=1e-2)
        assert isclose(est2.pass_horizontal_part, 6.78125, abs_tol=1e-2)
        assert isclose(est2.vertical_part, 29.21875, abs_tol=1e-2)
        assert isclose(est2.expected_detour, 54.4879, abs_tol=1e-2)

        est1 = calculate_detour_multiline(
            inst=inst,
            t=0.0,
            xStart=0.0,
            yStart=0.0,
            gFront=[7],
            gPass=[2, 4, 5, 6],
            v_nodes_count=33090,
            r_nodes_count=0,
            n_orderlines=1,
            isReturnPhase=False,
            outbound=False,
        )
        est3 = calculate_detour_multiline(
            inst=inst,
            t=0.0,
            xStart=0.0,
            yStart=0.0,
            gFront=[7],
            gPass=[2, 4, 5, 6],
            v_nodes_count=33090,
            r_nodes_count=0,
            n_orderlines=3,
            isReturnPhase=False,
            outbound=False,
        )
        assert est1.expected_detour < est2.expected_detour < est3.expected_detour

        est_replaceable = calculate_detour_multiline(
            inst=inst,
            t=0.0,
            xStart=2.0,
            yStart=0.0,
            gFront=[7],
            gPass=[2, 4, 5, 6],
            v_nodes_count=33090,
            r_nodes_count=0,
            n_orderlines=1,
            isReturnPhase=False,
            outbound=False,
        )
        assert isclose(est_replaceable.vertical_part, 0.0, abs_tol=1e-12)

        print("Multiline detour self-tests passed.")
    finally:
        detour_mod.compute_detour_route_back = orig_route_back