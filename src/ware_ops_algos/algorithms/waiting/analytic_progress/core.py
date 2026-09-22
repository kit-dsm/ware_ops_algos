from __future__ import annotations
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
from math import floor, isclose
from typing import Dict, List, Optional, Tuple

from .models import WarehouseInstance


def compute_segment_times(inst: WarehouseInstance) -> Tuple[Dict[int, float], Dict[int, float], float]:
    """
    Compute T_in^(j), T_out^(j), and T_end for S-shape routing with optional return aisle.
    """
    k = inst.k
    A = inst.A
    n = inst.n_list
    v = inst.v
    L = inst.L
    w = inst.w

    T_in = {j: 0.0 for j in range(1, k + 1)}
    T_out = {j: 0.0 for j in range(1, k + 1)}

    has_ret = (k % 2 == 1)
    a_ret = A[0] if has_ret else None

    def vertical_time(j: int) -> float:
        if has_ret and A[j - 1] == a_ret:
            return 2.0 * L / v
        return L / v

    # Time to reach the first visited aisle from depot row.
    T_in[k] = (A[-1] * w) / v
    T_out[k] = T_in[k] + vertical_time(k) + n[k - 1] * inst.t_p

    for j in range(k - 1, 0, -1):
        a_j = A[j - 1]
        a_j1 = A[j]
        T_in[j] = T_out[j + 1] + ((a_j1 - a_j) * w) / v
        T_out[j] = T_in[j] + vertical_time(j) + n[j - 1] * inst.t_p

    T_end = T_out[1] + (A[0] * w) / v
    return T_in, T_out, T_end


def theoretical_route_duration(inst: WarehouseInstance) -> float:
    _, _, T_end = compute_segment_times(inst)
    return T_end


@dataclass(frozen=True)
class RoutePoint:
    x: float
    y: float
    is_return_phase: bool


@dataclass(frozen=True)
class _RouteCache:
    route: List[RoutePoint]
    node_first_visit_index: Dict[Tuple[int, int], int]
    node_first_visit_time: Dict[Tuple[int, int], float]
    time_by_index: List[float]
    segment_move_time: List[float]
    segment_pick_delay: List[float]
    time_to_position: Dict[float, Tuple[float, float]]
    t_end: float


def _level_step(inst: WarehouseInstance) -> float:
    return inst.L / (inst.N_L + 1)


def delta_j(inst: WarehouseInstance, j: int, curr_ret: bool = False) -> int:
    k = inst.k
    if curr_ret:
        return -1
    if k % 2 == 0:
        return +1 if (j % 2 == 0) else -1
    return +1 if (j % 2 == 1) else -1


def _append_route_point(route: List[RoutePoint], x: float, y: float, is_return_phase: bool) -> None:
    if not route:
        route.append(RoutePoint(x=x, y=y, is_return_phase=is_return_phase))
        return
    last = route[-1]
    if isclose(last.x, x, abs_tol=1e-9) and isclose(last.y, y, abs_tol=1e-9) and last.is_return_phase == is_return_phase:
        return
    route.append(RoutePoint(x=x, y=y, is_return_phase=is_return_phase))


def _move_horizontal(route: List[RoutePoint], x_from: float, x_to: float, y: float, is_return_phase: bool) -> float:
    if isclose(x_from, x_to, abs_tol=1e-9):
        return x_from
    step = 1.0 if x_to > x_from else -1.0
    x = x_from
    while not isclose(x, x_to, abs_tol=1e-9):
        x += step
        if (step > 0 and x > x_to) or (step < 0 and x < x_to):
            x = x_to
        _append_route_point(route, x, y, is_return_phase)
    return x


def _move_vertical(route: List[RoutePoint], x: float, y_from: float, y_to: float, step_y: float, is_return_phase: bool) -> float:
    if isclose(y_from, y_to, abs_tol=1e-9):
        return y_from
    direction = 1.0 if y_to > y_from else -1.0
    y = y_from
    while not isclose(y, y_to, abs_tol=1e-9):
        y += direction * step_y
        if (direction > 0 and y > y_to) or (direction < 0 and y < y_to):
            y = y_to
        _append_route_point(route, x, y, is_return_phase)
    return y


def build_discrete_route(inst: WarehouseInstance) -> List[RoutePoint]:
    """Baut die positionsdiskrete S-Shape-Route (Start -> letzte Gasse -> ... -> Depot)."""
    route: List[RoutePoint] = [RoutePoint(x=0.0, y=0.0, is_return_phase=False)]
    if inst.k == 0:
        return route

    step_y = _level_step(inst)
    x = 0.0
    y = 0.0

    # Start (0,0) -> erster Entry an der rechtesten Besuchsgasse.
    x = _move_horizontal(route, x, float(inst.A[-1]), y, False)

    for j in range(inst.k, 0, -1):
        aisle = float(inst.A[j - 1])
        if not isclose(x, aisle, abs_tol=1e-9):
            x = _move_horizontal(route, x, aisle, y, False)

        upward = delta_j(inst, j, curr_ret=False) == +1
        y_target = inst.L if upward else 0.0
        y = _move_vertical(route, x, y, y_target, step_y, False)

        if j > 1:
            x = _move_horizontal(route, x, float(inst.A[j - 2]), y, False)

    # Ungerade k: Return-Gasse wird in der Rueckphase von oben nach unten gelaufen.
    if inst.k % 2 == 1 and not isclose(y, 0.0, abs_tol=1e-9):
        y = _move_vertical(route, x, y, 0.0, step_y, True)

    # Von der letzten Position horizontal zum Depot zurueck.
    x = _move_horizontal(route, x, 0.0, y, False)
    if not isclose(y, 0.0, abs_tol=1e-9):
        y = _move_vertical(route, x, y, 0.0, step_y, False)
    _append_route_point(route, x, y, False)

    return route


def _node_from_xy(inst: WarehouseInstance, x: float, y: float) -> Optional[Tuple[int, int]]:
    aisle = int(round(x))
    if aisle not in inst.aisle_to_j:
        return None
    lv = _level_step(inst)
    raw = y / lv if lv > 0 else -1.0
    y_node = int(round(raw))
    if 1 <= y_node <= inst.N_L and isclose(raw, y_node, abs_tol=1e-6):
        return aisle, y_node
    return None


def _build_route_cache(inst: WarehouseInstance) -> _RouteCache:
    route = build_discrete_route(inst)

    # Deterministische Pickzeiten je Knoten (Duplikate in P erhoehen die Anzahl).
    pick_count_by_node: Counter[Tuple[int, int]] = Counter(inst.P)
    picked_nodes: set[Tuple[int, int]] = set()

    time_by_index: List[float] = [0.0]
    time_to_position: Dict[float, Tuple[float, float]] = {0.0: (route[0].x, route[0].y)}

    # Erste Phase: Bewegungszeiten je Segment ohne Pickzeiten.
    dt_list: List[float] = [0.0]
    segment_move_time: List[float] = [0.0]
    segment_pick_delay: List[float] = [0.0]
    for idx in range(1, len(route)):
        prev = route[idx - 1]
        curr = route[idx]
        dx = abs(curr.x - prev.x) * inst.w
        dy = abs(curr.y - prev.y)
        dt = (dx + dy) / inst.v
        dt_list.append(dt)
        segment_move_time.append(dt)
        segment_pick_delay.append(0.0)

    # Zweite Phase: Identifiziere, an welcher Gasse Picks stattfinden und addiere t_p zur NÄCHSTEN Position.
    for idx in range(1, len(route)):
        prev = route[idx - 1]
        # Prüfe, ob an der vorherigen Position (prev) ein Pick stattfand.
        prev_node = _node_from_xy(inst, prev.x, prev.y)
        if prev_node is not None and prev_node not in picked_nodes:
            picks_here = int(pick_count_by_node.get(prev_node, 0))
            if picks_here > 0:
                # Addiere die Pickzeit zur aktuellen (nächsten) Position, nicht zur vorherigen.
                pick_delay = picks_here * inst.t_p
                dt_list[idx] += pick_delay
                segment_pick_delay[idx] += pick_delay
                picked_nodes.add(prev_node)

    # Dritte Phase: Berechne kumulierte Zeitstempel.
    for idx in range(1, len(route)):
        curr_time = time_by_index[-1] + dt_list[idx]
        time_by_index.append(curr_time)
        # Cache fuer schnellen Globalzugriff: Zeitstempel -> bereits erreichte Position.
        time_to_position[curr_time] = (route[idx].x, route[idx].y)
    t_end = time_by_index[-1]

    node_first_visit_index: Dict[Tuple[int, int], int] = {}
    node_first_visit_time: Dict[Tuple[int, int], float] = {}

    # In der Return-Gasse a_ret (nur bei ungeradem k) zaehlt der erste Aufstieg
    # noch nicht als "visited". Erst beim Herunterlaufen in der Return-Phase
    # werden die Knoten schrittweise von R nach V ueberfuehrt.
    has_ret = (inst.k % 2 == 1)
    a_ret = inst.A[0] if has_ret else None
    ret_fallback_first_seen: Dict[Tuple[int, int], int] = {}

    for idx, point in enumerate(route):
        node = _node_from_xy(inst, point.x, point.y)
        if node is None or node in node_first_visit_index:
            continue

        if has_ret and node[0] == a_ret:
            ret_fallback_first_seen.setdefault(node, idx)
            if point.is_return_phase:
                node_first_visit_index[node] = idx
                node_first_visit_time[node] = time_by_index[idx]
            continue

        node_first_visit_index[node] = idx
        node_first_visit_time[node] = time_by_index[idx]

    # Safety-Fallback: falls fuer einzelne a_ret-Knoten keine Return-Phase erkannt wurde,
    # verwende den ersten Sichtkontakt, damit kein Knoten dauerhaft in R verbleibt.
    for node, idx in ret_fallback_first_seen.items():
        if node not in node_first_visit_index:
            node_first_visit_index[node] = idx
            node_first_visit_time[node] = time_by_index[idx]

    return _RouteCache(
        route=route,
        node_first_visit_index=node_first_visit_index,
        node_first_visit_time=node_first_visit_time,
        time_by_index=time_by_index,
        segment_move_time=segment_move_time,
        segment_pick_delay=segment_pick_delay,
        time_to_position=time_to_position,
        t_end=t_end,
    )


def _get_route_cache(inst: WarehouseInstance) -> _RouteCache:
    cache = getattr(inst, "_route_cache", None)
    if cache is None:
        cache = _build_route_cache(inst)
        setattr(inst, "_route_cache", cache)
    return cache


def map_time_to_route_index(t: float, inst: WarehouseInstance) -> int:
    """Mappt kontinuierliche Zeit auf den zuletzt bereits erreichten diskreten Routenindex."""
    cache = _get_route_cache(inst)
    if not cache.route:
        return 0
    if cache.t_end <= 0:
        return 0

    clamped_t = max(0.0, min(float(t), cache.t_end))
    idx = bisect_right(cache.time_by_index, clamped_t) - 1
    return max(0, min(idx, len(cache.route) - 1))


def map_route_index_to_time(index: int, inst: WarehouseInstance) -> float:
    cache = _get_route_cache(inst)
    if not cache.route:
        return 0.0
    idx = max(0, min(index, len(cache.route) - 1))
    return cache.time_by_index[idx]


def route_point_at_index(index: int, inst: WarehouseInstance) -> RoutePoint:
    cache = _get_route_cache(inst)
    idx = max(0, min(index, len(cache.route) - 1))
    return cache.route[idx]


def route_point_at_time(t: float, inst: WarehouseInstance) -> RoutePoint:
    idx = map_time_to_route_index(t, inst)
    return route_point_at_index(idx, inst)


def active_aisle_index(
    t: float,
    T_in: Dict[int, float],
    T_out: Dict[int, float],
    inst: WarehouseInstance,
) -> Optional[int]:
    x, y = picker_position_2d(t, inst)
    if x is None or y is None:
        return None
    if y <= 0.0 or y >= inst.L:
        return None
    aisle = int(round(x))
    return inst.aisle_to_j.get(aisle)


def p_j(t: float, j: int, inst: WarehouseInstance, T_in: Dict[int, float]) -> float:
    x, y = picker_position_2d(t, inst, T_in, {})
    if x is None or y is None:
        return 0.0
    aisle = int(round(x))
    if inst.A[j - 1] != aisle:
        return 0.0
    return max(0.0, min(inst.L, y))

def G_pass(x: float, t: float, inst: WarehouseInstance) -> List[int]:
    """Nicht-besuchte, bereits passierte Gassen relativ zur Bewegungsphase."""
    I_all = set(range(1, inst.M + 1))
    A_set = set(inst.A)
    candidates = I_all - A_set

    # k=1-Spezialfall:
    # Am exakten Eingangspunkt (a_ret, 0) bleiben rechte Gassen noch in G_front.
    # Erst sobald y>0 in der Return-Gasse ist, wechseln sie nach G_pass.
    if inst.k == 1:
        a_ret = inst.A[0]
        _x, y_pos = picker_position_2d(t, inst)
        y_val = 0.0 if y_pos is None else float(y_pos)
        in_ret_after_entry = (x > float(a_ret) + 1e-9) or (
            abs(x - float(a_ret)) <= 1e-9 and y_val > 1e-9
        )
        if in_ret_after_entry:
            return sorted(i for i in candidates if i > a_ret)

    V_nodes, _, _, _ = V_R_global(t, {}, {}, inst)
    outbound = len(V_nodes) == 0
    if outbound:
        return []
    x_idx = int(floor(x + 1e-9))
    return sorted(i for i in candidates if i > x_idx)


def G_front(x: float, t: float, inst: WarehouseInstance) -> List[int]:
    """Nicht-besuchte, vor dem Picker liegende Gassen relativ zur Bewegungsphase."""

    I_all = set(range(1, inst.M + 1))
    A_set = set(inst.A)
    candidates = I_all - A_set

    # k=1-Spezialfall:
    # Am exakten Eingangspunkt (a_ret, 0) gehoeren rechte Gassen noch zu G_front.
    # Erst ab y>0 wechseln sie aus G_front nach G_pass.
    if inst.k == 1:
        a_ret = inst.A[0]
        _x, y_pos = picker_position_2d(t, inst)
        y_val = 0.0 if y_pos is None else float(y_pos)
        in_ret_after_entry = (x > float(a_ret) + 1e-9) or (
            abs(x - float(a_ret)) <= 1e-9 and y_val > 1e-9
        )
        if in_ret_after_entry:
            return sorted(i for i in candidates if i < a_ret)

    V_nodes, _, _, _ = V_R_global(t, {}, {}, inst)
    outbound = len(V_nodes) == 0
    if outbound:
        return sorted(candidates)
    x_idx = int(floor(x + 1e-9))
    return sorted(i for i in candidates if i <= x_idx)


def T_out_norm(j: int, T_in: Dict[int, float], inst: WarehouseInstance) -> float:
    return T_in.get(j, 0.0)


def p_eff_j(
    t: float,
    j: int,
    inst: WarehouseInstance,
) -> Tuple[float, bool]:
    point = route_point_at_time(t, inst)
    aisle = int(round(point.x))
    if inst.A[j - 1] != aisle:
        return 0.0, point.is_return_phase
    return max(0.0, min(inst.L, point.y)), point.is_return_phase


def s_y(y: int, inst: WarehouseInstance) -> float:
    return (y / inst.N_L) * inst.L


def horizontal_segment_info(
    t: float,
    T_in: Dict[int, float],
    T_out: Dict[int, float],
    inst: WarehouseInstance,
) -> Optional[Tuple[int, int]]:
    j_act = active_aisle_index(t, T_in, T_out, inst)
    if j_act is None:
        return None
    if j_act > 1:
        return j_act, j_act - 1
    return None


def picker_position_2d(
    t: float,
    inst: WarehouseInstance,
) -> Tuple[Optional[float], Optional[float]]:
    point = route_point_at_time(t, inst)
    return point.x, point.y


def nodes_in_aisle_order(inst: WarehouseInstance, j: int, curr_ret: bool = False) -> List[Tuple[int, int]]:
    a_j = inst.A[j - 1]
    d = delta_j(inst, j, curr_ret)
    y_indices = range(1, inst.N_L + 1) if d == +1 else range(inst.N_L, 0, -1)
    return [(a_j, y) for y in y_indices]


def visited_and_remaining_nodes_in_current_aisle(
    t: float,
    j: int,
    T_in: Dict[int, float],
    T_out: Dict[int, float],
    inst: WarehouseInstance,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], float, bool]:
    cache = _get_route_cache(inst)
    clamped_t = max(0.0, min(float(t), cache.t_end))
    p_eff, curr_ret = p_eff_j(t, j, T_in, T_out, inst)
    V_j_nodes: List[Tuple[int, int]] = []
    R_j_nodes: List[Tuple[int, int]] = []
    for y_node in range(1, inst.N_L + 1):
        node = (inst.A[j - 1], y_node)
        first_t = cache.node_first_visit_time.get(node, float("inf"))
        if clamped_t + 1e-9 >= first_t:
            V_j_nodes.append(node)
        else:
            R_j_nodes.append(node)
    return V_j_nodes, R_j_nodes, p_eff, curr_ret


def V_R_global(
    t: float,
    T_in: Dict[int, float],
    T_out: Dict[int, float],
    inst: WarehouseInstance,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], float, bool]:
    cache = _get_route_cache(inst)
    idx = map_time_to_route_index(t, inst)
    clamped_t = max(0.0, min(float(t), cache.t_end))
    curr_ret = cache.route[idx].is_return_phase
    j_act = active_aisle_index(t, T_in, T_out, inst)
    V_nodes: List[Tuple[int, int]] = []
    R_nodes: List[Tuple[int, int]] = []

    for aisle in inst.A:
        for y_node in range(1, inst.N_L + 1):
            node = (aisle, y_node)
            first_t = cache.node_first_visit_time.get(node, float("inf"))
            if clamped_t + 1e-9 >= first_t:
                V_nodes.append(node)
            else:
                R_nodes.append(node)

    p_eff = 0.0
    if j_act is not None:
        p_eff = max(0.0, min(inst.L, cache.route[idx].y))
    return V_nodes, R_nodes, p_eff, curr_ret
