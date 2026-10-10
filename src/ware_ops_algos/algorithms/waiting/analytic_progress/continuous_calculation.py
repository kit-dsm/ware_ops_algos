from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Dict, Tuple

from . import core as core_mod
from . import detour as detour_mod
from . import geometry as g
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...algorithm_interfaces import WaitingInput


@dataclass(frozen=True)
class ContinuousDetourEstimate:
    expected_detour: float
    detour_route_back: float
    detour_apass: float
    detour_aup: float
    p1_route: float
    p2_backtrack: float
    p3_pass: float
    p4_front: float
    p_sum: float
    phase_label: str
    progress: float
    x: float
    y: float
    is_return_phase: bool
    g_pass_count: int
    g_front_count: int
    v_cont: float
    r_cont: float


@dataclass(frozen=True)
class ContinuousRouteState:
    x: float
    y: float
    is_return_phase: bool
    phase_label: str
    active_j: int | None


def _route_progress(t: float, t_end: float) -> float:
    """Kontinuierlicher Routenfortschritt in [0, 1]."""
    if t_end <= 0.0:
        return 1.0
    return max(0.0, min(1.0, float(t) / float(t_end)))


def effective_vertical_speed_for_aisle(data: WaitingInput, j: int) -> float:
    """Effektive Vertikalgeschwindigkeit in Gasse j inkl. Pickzeit der Gasse."""
    if j < 1 or j > g.k(data):
        raise ValueError(f"aisle index j={j} out of range")
    n_j = g.n_list(data)[j - 1]
    return g.L(data) / (g.L(data) / g.v(data) + n_j * g.t_p(data))


def _ordered_aisles_by_entry(T_in: Dict[int, float]) -> list[int]:
    return [j for j, _ in sorted(T_in.items(), key=lambda item: item[1])]


def _get_return_aisle_index(data: WaitingInput) -> int | None:
    """1-basierter Index der Return-Gasse im Besuchsset A (nur bei ungeradem k)."""
    if g.k(data) % 2 == 0 or g.k(data) == 0:
        return None
    # In diesem Modell ist die Return-Gasse die kleinste besuchte Gasse (a_1).
    return 1


def estimate_continuous_route_state(
    t: float,
    T_in: Dict[int, float],
    T_out: Dict[int, float],
    data: WaitingInput,
) -> ContinuousRouteState:
    """Schaetzt die kontinuierliche Position (x,y) entlang der S-Shape-Route ueber T_in/T_out."""
    if g.k(data) <= 0:
        return ContinuousRouteState(x=0.0, y=0.0, is_return_phase=False, phase_label="empty", active_j=None)

    _, _, t_end = core_mod.compute_segment_times(data)
    clamped_t = max(0.0, min(float(t), t_end))
    seq = _ordered_aisles_by_entry(T_in)
    first_j = seq[0]
    first_aisle = float(g.A(data)[first_j - 1])
    eps = 1e-12

    def horizontal_x(x_from: float, x_to: float, t_start: float, t_current: float) -> float:
        if g.w(data) <= 0.0:
            return x_to
        direction = 1.0 if x_to >= x_from else -1.0
        x = x_from + direction * ((t_current - t_start) * g.v(data) / g.w(data))
        if direction > 0.0:
            return min(x, x_to)
        return max(x, x_to)

    if clamped_t <= T_in[first_j] + eps:
        x = horizontal_x(0.0, first_aisle, 0.0, clamped_t)
        return ContinuousRouteState(x=x, y=0.0, is_return_phase=False, phase_label="phase_1_to_first_entry", active_j=None)

    a_ret = g.A(data)[0] if (g.k(data) % 2 == 1) else None

    for pos, j in enumerate(seq):
        aisle_x = float(g.A(data)[j - 1])
        t_in_j = T_in[j]
        t_out_j = T_out[j]
        direction = core_mod.delta_j(data, j, curr_ret=False)
        entry_y = 0.0 if direction == +1 else g.L(data)
        exit_y = g.L(data) if direction == +1 else 0.0

        if t_in_j - eps <= clamped_t <= t_out_j + eps:
            local_t = max(0.0, clamped_t - t_in_j)

            if (a_ret is not None) and (g.A(data)[j - 1] == a_ret):
                v_eff_up = effective_vertical_speed_for_aisle(data, j)
                up_duration = g.L(data) / v_eff_up
                down_duration = g.L(data) / g.v(data)
                if local_t <= up_duration + eps:
                    travelled = min(g.L(data), max(0.0, local_t * v_eff_up))
                    y = entry_y + direction * travelled
                    y = max(0.0, min(g.L(data), y))
                    return ContinuousRouteState(
                        x=aisle_x,
                        y=y,
                        is_return_phase=False,
                        phase_label="phase_n_2_ret_up",
                        active_j=j,
                    )

                down_t = min(max(0.0, local_t - up_duration), down_duration)
                travelled = min(g.L(data), max(0.0, down_t * g.v(data)))
                y = exit_y - direction * travelled
                y = max(0.0, min(g.L(data), y))
                return ContinuousRouteState(
                    x=aisle_x,
                    y=y,
                    is_return_phase=True,
                    phase_label="phase_n_1_ret_down",
                    active_j=j,
                )

            v_eff = effective_vertical_speed_for_aisle(data, j)
            travelled = min(g.L(data), max(0.0, local_t * v_eff))
            y = entry_y + direction * travelled
            y = max(0.0, min(g.L(data), y))
            return ContinuousRouteState(
                x=aisle_x,
                y=y,
                is_return_phase=False,
                phase_label="phase_2_vertical",
                active_j=j,
            )

        if pos + 1 < len(seq):
            j_next = seq[pos + 1]
            t_next = T_in[j_next]
            if t_out_j + eps < clamped_t < t_next - eps:
                x_to = float(g.A(data)[j_next - 1])
                x = horizontal_x(aisle_x, x_to, t_out_j, clamped_t)
                return ContinuousRouteState(
                    x=x,
                    y=exit_y,
                    is_return_phase=False,
                    phase_label="phase_3_horizontal",
                    active_j=None,
                )

    x = horizontal_x(float(g.A(data)[0]), 0.0, T_out[1], clamped_t)
    return ContinuousRouteState(x=x, y=0.0, is_return_phase=False, phase_label="phase_4_after_last_aisle", active_j=None)


def _continuous_vr_state(
    t: float,
    T_in: Dict[int, float],
    T_out: Dict[int, float],
    data: WaitingInput,
) -> Tuple[float, float, str, int | None, float]:
    """Kontinuierliche V/R-Anteile auf den besuchten Gassen, normalisiert auf [0,1]."""
    k = g.k(data)
    if k <= 0:
        return 0.0, 1.0, "empty", None, 0.0

    seq = _ordered_aisles_by_entry(T_in)
    first_j = seq[0]
    eps = 1e-12

    if t <= T_in[first_j] + eps:
        return 0.0, 1.0, "phase_1_to_first_entry", None, 0.0

    route_state = estimate_continuous_route_state(t, T_in, T_out, data)
    a_ret = g.A(data)[0] if (k % 2 == 1) else None

    for i, j in enumerate(seq):
        t_in_j = T_in[j]
        t_out_j = T_out[j]

        if t_in_j - eps <= t <= t_out_j + eps:
            local = 0.0
            aisle_idx = g.A(data)[j - 1]
            is_same_aisle = int(round(route_state.x)) == aisle_idx

            if t >= t_out_j - eps:
                local = 1.0
            elif is_same_aisle and route_state.y > 0.0 and route_state.y < g.L(data):
                if (a_ret is not None) and (aisle_idx == a_ret):
                    # Phase n-2 (Aufstieg a_ret): konstant.
                    # Phase n-1 (Rueckweg a_ret): Fortschritt bis 1.0.
                    local = (g.L(data) - route_state.y) / g.L(data) if route_state.is_return_phase else 0.0
                else:
                    local = route_state.y / g.L(data) if core_mod.delta_j(data, j, curr_ret=False) == +1 else (g.L(data) - route_state.y) / g.L(data)
                local = max(0.0, min(1.0, local))

            v_cont = (i + local) / k
            r_cont = 1.0 - v_cont
            if (a_ret is not None) and (aisle_idx == a_ret):
                label = "phase_n_1_ret_down" if route_state.is_return_phase else "phase_n_2_ret_up"
            else:
                label = "phase_2_vertical"
            return v_cont, r_cont, label, j, local

        if i + 1 < len(seq):
            j_next = seq[i + 1]
            if t_out_j + eps < t < T_in[j_next] - eps:
                v_cont = (i + 1) / k
                r_cont = 1.0 - v_cont
                return v_cont, r_cont, "phase_3_horizontal", None, 0.0

    return 1.0, 0.0, "phase_4_after_last_aisle", None, 1.0


def _horizontal_phase_bounds(
    phase_label: str,
    t: float,
    T_in: Dict[int, float],
    T_out: Dict[int, float],
    data: WaitingInput,
) -> tuple[float, float] | None:
    """
    Liefert (x_min, x_max) der aktuellen horizontalen Phase für Alternative A.
    Falls die Phase keine horizontale ist, wird None zurückgegeben.

    Annahme:
    - phase_3_horizontal: Bewegung zwischen zwei benachbarten besuchten Gassen.
    - phase_4_after_last_aisle: Bewegung von letzter Gasse zurück zum Depot (x=0).
    """

    # Gassen nach Entry-Zeit sortieren (wie im continuous_code)
    seq = _ordered_aisles_by_entry(T_in)
    k = g.k(data)

    if phase_label.startswith("phase_3"):
        eps = 1e-9
        for idx, j in enumerate(seq):
            if idx + 1 >= len(seq):
                break
            j_next = seq[idx + 1]
            t_out_j = T_out[j]
            t_in_next = T_in[j_next]
            if t_out_j + eps < t <= t_in_next + eps:
                # Horizontal von A[j] nach A[j_next]
                x1 = float(g.A(data)[j - 1])
                x2 = float(g.A(data)[j_next - 1])
                return (min(x1, x2), max(x1, x2))
        return None

    if phase_label.startswith("phase_4_after_last_aisle"):
        j_last = seq[-1]
        x1 = float(g.A(data)[j_last - 1])
        x2 = 0.0
        return (min(x1, x2), max(x1, x2))

    return None


def _continuous_p3_p4_length_based(
    x: float,
    phase_label: str,
    t: float,
    T_in: Dict[int, float],
    T_out: Dict[int, float],
    data: WaitingInput,
) -> tuple[float, float]:


    # Alle Gassen und nicht-besuchte Kandidaten
    I_all = set(range(1, g.M(data) + 1))
    A_set = set(g.A(data))
    candidates = I_all - A_set
    num_cand = len(candidates)
    if num_cand <= 0:
        return 0.0, 0.0

    # Bounds der horizontalen Phase holen
    bounds = _horizontal_phase_bounds(phase_label, t, T_in, T_out, data)
    if bounds is None:
        # Keine horizontale Phase: hier keine Alternative-A-Anpassung
        return 0.0, 0.0

    x_min, x_max = bounds
    if x_max <= x_min:
        return 0.0, 0.0


    cand_share = num_cand / g.M(data)
    candidates_left_max = [c for c in candidates if c < x_max]
    candidates_left_min = [c for c in candidates if c < x_min]

    candidates_in_segment = [c for c in candidates if x_min <= c <= x_max]
    boundary_extra = 1 if candidates_in_segment else 0

    p4_max = (len(candidates_left_max) + boundary_extra) / g.M(data)
    p4_min = len(candidates_left_min) / g.M(data)

    u = (x - x_min) / (x_max - x_min)

    # p3 wächst mit u, p4 ist der Rest der Kandidaten
    p4 = p4_min + (p4_max - p4_min) * u
    p3 = cand_share - p4
    return float(p3), float(p4)

def continuous_class_probabilities(
    t: float,
    T_in: Dict[int, float],
    T_out: Dict[int, float],
    data: WaitingInput,
) -> Tuple[float, float, float, float, float, float, float]:
    """
    Kontinuierliche Approximation von p1..p4.
    Rückgabe: (p1_route, p2_backtrack, p3_pass, p4_front, p_sum).
    """
    if g.k(data) <= 0:
        raise ValueError("Instance has no visited aisles (k=0).")

    v_cont, r_cont, phase_label, active_j, local_progress = _continuous_vr_state(t, T_in, T_out, data)

    route_share = g.k(data) / g.M(data)

    p1 = route_share * r_cont
    p2 = route_share * v_cont

    route_state = estimate_continuous_route_state(t, T_in, T_out, data)
    x = route_state.x
    g_pass = core_mod.G_pass(x, t, data)
    g_front = core_mod.G_front(x, t, data)

    num_non_base = g.M(data) - g.k(data)
    if num_non_base > 0:
        if phase_label.startswith("phase_3") or phase_label.startswith("phase_4"):
            p3, p4 = _continuous_p3_p4_length_based(x, phase_label, t, T_in, T_out, data)

        else:
            p3 = (len(g_pass) / num_non_base) * ((g.M(data) - g.k(data)) / g.M(data))  # = len(g_pass)/M
            p4 = (len(g_front) / num_non_base) * ((g.M(data) - g.k(data)) / g.M(data))  # = len(g_front)/M
    else:
        p3 = 0.0
        p4 = 0.0
    p_sum = p1 + p2 + p3 + p4

    if not isclose(p_sum, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"Probability sum is not 1.0 (got {p_sum:.12f}). "
            "Check continuous progress and G_pass/G_front partitioning."
        )
    return p1, p2, p3, p4, p_sum, v_cont, r_cont


def continuous_backtrack_detour(
    t: float,
    T_in: Dict[int, float],
    T_out: Dict[int, float],
    data: WaitingInput,
) -> float:
    """
    Kontinuierliche Backtrack-Approximation (Round-Trip, Faktor 2 bereits enthalten)
    nach der gewichteten
    Formel ueber bereits vertikal durchlaufene Strecke:

      L_ges = L_y + A_comp * L
      D_rb  = (L_y / L_ges) * L_y
              + Sum_i (L / L_ges) * (L + 2 * dx_i)

    mit i ueber alle bereits vollstaendig durchschrittenen Gassen.

    Hinweise zur Modellierung:
      - A_comp folgt der x-Positionslogik: Gassen in A mit a_i > x_akt.
      - In der Return-Gasse zaehlt der Aufstieg nicht als besucht; beim
        Rueckweg wird L_y = L - y verwendet.
    """
    route_state = estimate_continuous_route_state(t, T_in, T_out, data)
    phase_label = route_state.phase_label
    if phase_label == "phase_1_to_first_entry":
        return 0.0

    A = g.A(data)
    w = g.w(data)
    L = g.L(data)
    x = route_state.x
    y = route_state.y
    active_j = route_state.active_j
    i_ret = _get_return_aisle_index(data)
    eps = 1e-9

    # Spezialfall: ungerades k, letzte Phase nach der Return-Gasse auf y=0 und links von a_ret.
    # Dann muss fuer alle Nicht-Return-Gassen ein zusaetzlicher 2L-Anteil beruecksichtigt werden,
    # weil die Return-Gasse fuer den Backtrack durchquert werden muss.
    if (
        (g.k(data) % 2 == 1)
        and (phase_label == "phase_4_after_last_aisle")
        and (abs(y) <= eps)
        and (i_ret is not None)
    ):
        a_ret = g.A(data)[i_ret - 1]
        if x < a_ret - eps:
            k = len(A)
            if k <= 0:
                return 0.0
            L_ges_special = k * L
            total = 0.0
            for a_i in A:
                dx = abs(x - float(a_i)) * w
                # Mittlerer vertikaler Anteil in einer Zielgasse + horizontaler Hin/Rueckweg.
                one_way_mean = L + 2.0 * dx
                # Fuer alle Gassen ausser a_ret faellt zusaetzlich das Queren von a_ret an.
                extra_return_cross = 0.0 if a_i == a_ret else 2.0 * L
                total += (L / L_ges_special) * (one_way_mean + extra_return_cross)
            # Nach Verlassen der Return-Gasse (x < a_ret, y=0) ist ein zusaetzlicher
            # 2L-Anteil immer unvermeidlich, unabhaengig von der Zielgasse.
            if k == 1:
                return total
            return total + 2.0 * L

    def _delta_dir(j: int) -> int:
        return core_mod.delta_j(data, j, curr_ret=False)

    # Vertikal bereits durchlaufene Strecke in der aktuellen Gasse.
    L_y = 0.0
    if active_j is not None and phase_label in ("phase_2_vertical", "phase_n_2_ret_up", "phase_n_1_ret_down"):
        if (i_ret is not None) and (active_j == i_ret):
            if route_state.is_return_phase:
                L_y = max(0.0, min(L, L - y))
            else:
                L_y = 0.0
        else:
            d = _delta_dir(active_j)
            L_y = max(0.0, min(L, y if d == +1 else (L - y)))

    completed_aisles = [a_i for a_i in A if a_i > (x + eps)]
    A_comp = len(completed_aisles)

    L_ges = L_y + (A_comp * L)
    if L_ges <= eps:
        return 0.0

    current_weighted = (L_y / L_ges) * L_y if L_y > eps else 0.0

    completed_weighted = 0.0
    for idx, a_i in enumerate(completed_aisles):
        dx = abs(x - float(a_i)) * w
        if route_state.is_return_phase:
            completed_weighted += (L / L_ges) * (L + 2.0 * (dx + idx * L))
        else:
            completed_weighted += (L / L_ges) * (L + 2.0 * (dx + L_y + idx * L))

    return current_weighted + completed_weighted

def _continuous_detour_apass(
        data: WaitingInput,
        x: float,
        y: float,
        t: float,
        T_in: Dict[int, float],
        T_out: Dict[int, float],
        g_pass,
        is_return_phase: bool,
) -> float:
    """
    Kontinuierliche A_pass-Berechnung mit phasenweiser Logik.
    """
    if not g_pass:
        return 0.0

    w = g.w(data)
    L = g.L(data)
    A_sorted = sorted(g.A(data))
    k = len(A_sorted)
    eps = 1e-9

    _, _, phase_label, _, _ = _continuous_vr_state(t, T_in, T_out, data)

    if phase_label == "phase_1_to_first_entry":
        return 0.0

    is_horizontal = (
            phase_label.startswith("phase_3")
            or phase_label.startswith("phase_4")
    )

    if is_horizontal and detour_mod._to_picking_on_lower_cross_aisle(
            data, t, y, is_return_phase):
        return 0.0

    non_visited = sorted(set(range(1, g.M(data) + 1)) - set(g.A(data)))

    def dx_at(pos_x: float) -> float:
        candidates = [aisle for aisle in non_visited if float(aisle) > pos_x]
        if not candidates:
            return 0.0
        return sum(abs(float(aisle) - pos_x) * w for aisle in candidates) / len(candidates)


    if phase_label.startswith("phase_3") or phase_label.startswith("phase_4"):
        bounds = _horizontal_phase_bounds(phase_label, t, T_in, T_out, data)
        if bounds is not None:
            x_min, x_max = bounds
            if x_max > x_min + eps:
                u = max(0.0, min(1.0, (x - x_min) / (x_max - x_min)))
                dx_min = dx_at(x_min)
                dx_max = dx_at(x_max)
                dx = dx_min + (dx_max - dx_min) * u
            else:
                dx = dx_at(x)
        else:
            dx = dx_at(x)
    else:
        dx = dx_at(x)

    if dx <= eps:
        return 0.0

    if k % 2 == 0:
        return 2.0 * dx + 2.0 * L

    a_ret = A_sorted[0]
    in_return_aisle = abs(x - a_ret) < eps
    if in_return_aisle:
        # Return-Gasse: hoch -> 2dx, runter -> 2(dx+L)
        return (2.0 * dx + 2.0 * L) if is_return_phase else (2.0 * dx)

    # Explizite diskrete Spiegelung fuer Phase 4 bei ungeradem k.
    if phase_label.startswith("phase_4_after_last_aisle") and (x < a_ret - eps):
        return 2.0 * dx + 2.0 * L

    if x > a_ret or (abs(x - a_ret) < eps and abs(y) < eps and (not is_return_phase)):
        return 2.0 * dx
    return 2.0 * dx + 2.0 * L


def _continuous_detour_aup(
    data: WaitingInput,
    x: float,
    y: float,
    g_front: list[int],
    outbound: bool,
    is_return_phase: bool,
) -> float:
    """
    Kontinuierliche Variante von compute_detour_aup.
    """
    L = g.L(data)
    w = g.w(data)
    A = g.A(data)
    A_sorted = sorted(A)
    k = len(A_sorted)
    max_a = A_sorted[-1]
    eps = 1e-9

    if not g_front:
        return 0.0

    phase1_from_start = outbound and (x <= A_sorted[-1] + eps)

    def phase1_weighted_horizontal_detour(a_k: int) -> float:
        # In Phase 1 tragen nur Gassen rechts von a_k horizontal bei;
        # Erwartungswert ist ueber alle D_front-Gassen gewichtet.
        total = len(g_front)
        if total == 0:
            return 0.0
        right_sum = sum(abs(aisle - a_k) * w for aisle in g_front if aisle > a_k)
        return 2.0 * (right_sum / total)

    def dist_x(a1: float, a2: int) -> float:
        return abs(a2 - a1) * w

    def return_aisle() -> int | None:
        if k % 2 == 1:
            return A_sorted[0]
        return None

    # k=1-Sonderfall mit exakter Eintrittsgrenze (a_ret,0):
    # - am Eingangspunkt noch front-Logik
    # - in der Return-Gasse (y>0) hoch: 0, runter: 2L
    if k == 1:
        a_ret = A_sorted[0]
        at_entry = (abs(x - a_ret) <= eps) and (abs(y) <= eps)
        in_ret_aisle = (abs(x - a_ret) <= eps) and (y > eps)

        if at_entry:
            # Beim Verlassen der Return-Gasse (ret=True, y=0) bleibt der Umweg 2L.
            if is_return_phase:
                return 2.0 * L
            return phase1_weighted_horizontal_detour(a_ret)

        if in_ret_aisle:
            return 2.0 * L if is_return_phase else 0.0

        if x < a_ret - eps:
            if outbound:
                # Phase 1: gewichteter Horizontalanteil
                return phase1_weighted_horizontal_detour(a_ret)

            # Phase 4 (Rueckweg zum Depot): diskret-konsistent 2L
            return 2.0 * L

        return 0.0

    relevant_aisles = [aisle for aisle in g_front if aisle > max_a]
    dx_front = 0.0
    if relevant_aisles:
        a_k = A_sorted[-1]
        dx_list = [dist_x(float(a_k), aisle) for aisle in relevant_aisles]
        dx_front = sum(dx_list) / len(dx_list)

    if phase1_from_start:
        horizontal_detour = phase1_weighted_horizontal_detour(A_sorted[-1])
    else:
        horizontal_detour = 2.0 * dx_front if outbound else 0.0

    if k % 2 == 0:
        return (2.0 * L) + horizontal_detour

    a_ret = A_sorted[0]
    # Exakter Eintrittspunkt der Return-Gasse auf der unteren Quergasse:
    # Die Front-Gassen bleiben noch in der Horizontal-Logik, erst innerhalb
    # der Return-Gasse (y>0) wird der Front-Umweg 0.
    if abs(x - a_ret) < eps and abs(y) < eps and (not is_return_phase):
        return horizontal_detour

    in_return_aisle = abs(x - a_ret) < eps
    if in_return_aisle:
        # Return-Gasse: hoch -> 0, runter -> 2L
        return 2.0 * L if is_return_phase else 0.0

    a_ret = return_aisle()
    if a_ret is None:
        return (2.0 * L) + horizontal_detour


    if outbound:
        return horizontal_detour
    if a_ret is not None and x > a_ret:
        return 0.0

    return 2.0 * L


def continuous_expected_detour(
    t: float,
    data: WaitingInput,
    T_in: Dict[int, float],
    T_out: Dict[int, float],
) -> ContinuousDetourEstimate:
    """
    Kontinuierliche Approximation der erwarteten Detour E[D(t)]:
      E[D] = p2 * D_backtrack_cont + p3 * D_pass_disc + p4 * D_front_disc

    Konvention:
      - D_backtrack_cont ist bereits als Hin- und Rueckweg modelliert
        (kein zusaetzlicher Faktor 2 in E[D]).

    - Wahrscheinlichkeiten p1..p4: über Längenanteile + G_pass/G_front.
    - D_backtrack_cont: Integral-basierte Approximation aus p_eff(t).
    - D_pass_disc, D_front_disc: weiterhin über compute_detour_apass/compute_detour_aup,
      gemittelt über alle beteiligten Gassen.
    """

    p1, p2, p3, p4, p_sum, v_cont, r_cont = continuous_class_probabilities(t, T_in, T_out, data)

    _, _, t_end = core_mod.compute_segment_times(data)
    progress = _route_progress(t, t_end)

    route_state = estimate_continuous_route_state(t, T_in, T_out, data)
    point = core_mod.RoutePoint(route_state.x, route_state.y, route_state.is_return_phase)
    g_pass = core_mod.G_pass(point.x, t, data)
    g_front = core_mod.G_front(point.x, t, data)

    # Phasenbasierte Outbound-Definition (vermeidet Fehlklassifikation ueber all(T_out>t)).
    outbound = route_state.phase_label == "phase_1_to_first_entry"

    d_rb = continuous_backtrack_detour(t, T_in, T_out, data)

    d_pass = 0.0
    if g_pass:
        d_pass = _continuous_detour_apass(
                data,
                point.x,
                point.y,
                t,
                T_in,
                T_out,
                g_pass,
                route_state.is_return_phase,
            )

    d_front = 0.0
    if g_front:
        d_front = _continuous_detour_aup(
            data,
            point.x,
            point.y,
            g_front,
            outbound,
            point.is_return_phase,
        )

    expected = (p2 * d_rb) + (p3 * d_pass) + (p4 * d_front)


    return ContinuousDetourEstimate(
        expected_detour=float(expected),
        detour_route_back=float(d_rb),
        detour_apass=float(d_pass),
        detour_aup=float(d_front),
        p1_route=float(p1),
        p2_backtrack=float(p2),
        p3_pass=float(p3),
        p4_front=float(p4),
        p_sum=float(p_sum),
        phase_label=route_state.phase_label,
        progress=float(progress),
        x=float(point.x),
        y=float(point.y),
        is_return_phase=bool(point.is_return_phase),
        g_pass_count=len(g_pass),
        g_front_count=len(g_front),
        v_cont=v_cont,
        r_cont=r_cont
    )
