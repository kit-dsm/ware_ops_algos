from math import floor
from typing import Dict, List, Sequence, Tuple

try:
    from . import mdp_based_formulation
    from .core import (
        G_front,
        G_pass,
        V_R_global,
        _get_route_cache,
        compute_segment_times,
        map_route_index_to_time,
        map_time_to_route_index,
        picker_position_2d,
        theoretical_route_duration,
    )
    from .detour import (
        calculate_detour_estimate,
        compute_detour_apass,
        compute_detour_aup,
        compute_detour_route_back_exact,
    )
    from .models import BatchComparisonResult, BatchWindow, OrderInput, TimeSampleResult, WarehouseInstance
    from .optimizer_common import (
        EXPECTED_INTERARRIVAL,
        build_batch_windows,
        create_warehouse_instance_from_orders,
        expanded_pick_nodes_for_order,
        load_article_mapping,
        load_orders,
        order_pick_service_time,
    )
except ImportError:  # Direktes Ausfuehren als Skript
    import mdp_based_formulation
    from core import (
        G_front,
        G_pass,
        V_R_global,
        _get_route_cache,
        compute_segment_times,
        map_route_index_to_time,
        map_time_to_route_index,
        picker_position_2d,
        theoretical_route_duration,
    )
    from detour import (
        calculate_detour_estimate,
        compute_detour_apass,
        compute_detour_aup,
        compute_detour_route_back_exact,
    )
    from models import BatchComparisonResult, BatchWindow, OrderInput, TimeSampleResult, WarehouseInstance
    from optimizer_common import (
        EXPECTED_INTERARRIVAL,
        build_batch_windows,
        create_warehouse_instance_from_orders,
        expanded_pick_nodes_for_order,
        load_article_mapping,
        load_orders,
        order_pick_service_time,
    )


_expanded_pick_nodes_for_order = expanded_pick_nodes_for_order
_order_pick_service_time = order_pick_service_time


def _actual_insert_interval(window: BatchWindow, base_route_duration: float) -> float:
    start_time = max(order.arrival_time for order in window.base_orders)
    raw_interval = window.insert_order.arrival_time - start_time
    return max(0.0, raw_interval)


def _actual_insert_step(window: BatchWindow, base_route_duration: float) -> int:
    """Diskretes Intervall: kontinuierliche Zeit wird via floor auf Positionsindex gemappt."""
    return int(floor(_actual_insert_interval(window, base_route_duration)))


def _detour_state_for_t(
    base_inst: WarehouseInstance,
    window: BatchWindow,
    article_mapping: Dict[str, Tuple[int, int]],
    T_in: Dict[int, float],
    T_out: Dict[int, float],
    t_idx: int,
) -> Tuple[float, float, str, float]:
    t_float = map_route_index_to_time(t_idx, base_inst)

    V_nodes, R_nodes, p_eff, curr_ret = V_R_global(t_float, T_in, T_out, base_inst)
    x, y = picker_position_2d(t_float, base_inst)
    if x is None or y is None:
        raise ValueError(f"Keine Picker-Position fuer t={t_float:.3f} bestimmbar")

    pick_nodes = _expanded_pick_nodes_for_order(window.insert_order, article_mapping)
    if len(pick_nodes) != 1:
        return float(x), float(y), "full_recompute", float("nan")

    actual_pick = pick_nodes[0]
    g_front = G_front(x, t_float, base_inst)
    g_pass = G_pass(x, t_float, base_inst)
    outbound = len(V_nodes) == 0

    if actual_pick in R_nodes:
        return float(x), float(y), "route", 0.0
    if actual_pick in V_nodes:
        detour = compute_detour_route_back_exact(
            base_inst,
            x,
            y,
            target_aisle=actual_pick[0],
            target_y_node=actual_pick[1],
            t=t_float,
            isReturnPhase=curr_ret,
        )
        return float(x), float(y), "backtrack", float(detour)
    if actual_pick[0] in g_pass:
        detour = compute_detour_apass(
            base_inst,
            x,
            y,
            t_float,
            g_pass,
            curr_ret,
            target_aisle=actual_pick[0],
        )
        return float(x), float(y), "pass", float(detour)
    if actual_pick[0] in g_front:
        detour = compute_detour_aup(
            base_inst,
            x,
            y,
            g_front,
            outbound,
            curr_ret,
            target_aisle=actual_pick[0],
        )
        return float(x), float(y), "front", float(detour)

    return float(x), float(y), "unclassified", 0.0


def _find_best_actual_integration_from_t(
    base_inst: WarehouseInstance,
    window: BatchWindow,
    article_mapping: Dict[str, Tuple[int, int]],
    T_in: Dict[int, float],
    T_out: Dict[int, float],
    start_t_idx: int,
    end_t_idx: int,
) -> Tuple[int, float, float, str, float]:
    pick_nodes = _expanded_pick_nodes_for_order(window.insert_order, article_mapping)
    if len(pick_nodes) != 1:
        x0, y0, case0, detour0 = _detour_state_for_t(base_inst, window, article_mapping, T_in, T_out, start_t_idx)
        return start_t_idx, x0, y0, case0, detour0

    best_t = start_t_idx
    best_x, best_y, best_case, best_detour = _detour_state_for_t(
        base_inst, window, article_mapping, T_in, T_out, start_t_idx
    )
    for t in range(start_t_idx + 1, end_t_idx + 1):
        x, y, case, detour = _detour_state_for_t(base_inst, window, article_mapping, T_in, T_out, t)
        if detour < best_detour - 1e-9:
            best_t = t
            best_x, best_y, best_case, best_detour = x, y, case, detour
    return best_t, best_x, best_y, best_case, best_detour

def analyze_batch_window(
    window: BatchWindow,
    article_mapping: Dict[str, Tuple[int, int]],
    *,
    M: int,
    N_L: int,
    w: float,
    L: float,
    v: float,
    t_p: float,

) -> BatchComparisonResult:
    base_order_arrival_times = [float(order.arrival_time) for order in window.base_orders]
    insert_order_arrival_time = float(window.insert_order.arrival_time)
    base_order_positions = [
        _expanded_pick_nodes_for_order(order, article_mapping)
        for order in window.base_orders
    ]
    insert_order_positions = _expanded_pick_nodes_for_order(window.insert_order, article_mapping)

    base_inst = create_warehouse_instance_from_orders(
        window.base_orders,
        article_mapping,
        M=M,
        N_L=N_L,
        w=w,
        L=L,
        v=v,
        t_p=t_p,
    )
    candidate_orders = window.all_orders
    candidate_inst = create_warehouse_instance_from_orders(
        candidate_orders,
        article_mapping,
        M=M,
        N_L=N_L,
        w=w,
        L=L,
        v=v,
        t_p=t_p,
    )

    T_in, T_out, base_route_duration = compute_segment_times(base_inst)
    insert_pick_service_time = _order_pick_service_time(window.insert_order, t_p)
    avg_actual_arrival_times = sum(candidate_inst.arrival_times) / len(candidate_inst.arrival_times)
    if len(window.base_orders) >= 3:
        arrival_time_order_3 = float(window.base_orders[2].arrival_time)
    elif window.base_orders:
        arrival_time_order_3 = float(window.base_orders[-1].arrival_time)
    else:
        arrival_time_order_3 = 0.0
    initial_completion_time = base_route_duration + arrival_time_order_3


    # Phase 2: tatsächlicher Einfügezeitpunkt (Sekunden) -> Positionsindex.
    actual_t_seconds = _actual_insert_interval(window, base_route_duration)

    # Positionsdiskret über die gesamte Basisroute (aus Cache, nicht neu erzeugt).
    cache = _get_route_cache(base_inst)
    effective_min_idx = 0
    effective_max_idx = len(cache.route) - 1

    # -------------------------------------------------------------------------
    # Phase 1 – SAMPLING: t_actual ist zu diesem Zeitpunkt NICHT bekannt.
    # Die Ankunftszeit von Order 4 wird geschätzt als:
    #   geschätzte_Ankunft = Ankunftszeit_letzte_bekannte_Order + EXPECTED_INTERARRIVAL (28.8 s)
    # Der geschaetzte Einfuegezeitpunkt fuer diskrete Entscheidungen bleibt gerundet,
    # aber die Wartezeit wird auf Basis von 28.8s berechnet.
    # -------------------------------------------------------------------------
    estimated_t_insert_idx = min(map_time_to_route_index(EXPECTED_INTERARRIVAL, base_inst), effective_max_idx)
    estimated_t_waiting: float = min(EXPECTED_INTERARRIVAL, float(base_route_duration))

    def _build_sample(idx: int) -> TimeSampleResult | None:
        t_float = map_route_index_to_time(idx, base_inst)
        V_nodes, R_nodes, p_eff, curr_ret = V_R_global(t_float, T_in, T_out, base_inst)
        x, y = picker_position_2d(t_float, base_inst)
        if x is None or y is None:
            return None

        g_front = G_front(x, t_float, base_inst)
        g_pass = G_pass(x, t_float, base_inst)
        estimate = calculate_detour_estimate(
            inst=base_inst,
            t=t_float,
            xStart=x,
            yStart=y,
            gFront=g_front,
            gPass=g_pass,
            v_nodes_count=len(V_nodes),
            r_nodes_count=len(R_nodes),
            isReturnPhase=curr_ret,
        )

        predicted_route_duration = base_route_duration + estimate.expected_detour + insert_pick_service_time
        # Wartezeit bis zum geschätzten Eintreffen der Order (Phase 1, t_actual unbekannt)
        waiting_time_to_estimated = float(max(0.0, estimated_t_waiting - t_float))
        predicted_add_time = waiting_time_to_estimated + estimate.expected_detour + insert_pick_service_time
        predicted_completion_time = arrival_time_order_3 + base_route_duration + predicted_add_time
        # Ausgabefelder (neue Formel: Routendauer-Anteil pro Order)
        predicted_completion_time_per_order = (predicted_completion_time - arrival_time_order_3) / 4.0
        initial_completion_time_per_order = (initial_completion_time - arrival_time_order_3) / 3.0

        base_avg_arrival_time = estimate.avg_arrival_time

        predicted_avg_arrival_time = (sum(order.arrival_time for order in window.base_orders) +
                                      (EXPECTED_INTERARRIVAL + arrival_time_order_3)) / (len(window.base_orders) + 1)

        mean_initial_order_completion_time = initial_completion_time - base_avg_arrival_time
        mean_predicted_order_completion_time = predicted_completion_time - predicted_avg_arrival_time

        should_integrate = mean_predicted_order_completion_time < mean_initial_order_completion_time

        if should_integrate:
            effective_policy = "integrate"
            effective_detour = estimate.expected_detour + insert_pick_service_time
            effective_route_duration = predicted_route_duration
            effective_completion_time = predicted_completion_time
        else:
            effective_policy = "not_integrated"
            effective_detour = 0.0
            effective_route_duration = base_route_duration
            effective_completion_time = initial_completion_time

        phase = "hin" if len(V_nodes) == 0 else "rueck"
        return_flag = "ret" if curr_ret else "-"

        return TimeSampleResult(
            batch_index=window.batch_index,
            idx = idx,
            t=t_float,
            x=float(x),
            y=float(y),
            phase=phase,
            return_flag=return_flag,
            base_route_duration=float(base_route_duration),
            expected_detour=float(estimate.expected_detour),
            waiting_time_to_estimated=waiting_time_to_estimated,
            predicted_add_time=float(predicted_add_time),
            predicted_route_duration=float(predicted_route_duration),
            arrival_time_order_3=float(arrival_time_order_3),
            initial_completion_time=float(initial_completion_time),
            predicted_completion_time=float(predicted_completion_time),
            mean_initial_order_completion_time=float(mean_initial_order_completion_time),
            mean_predicted_order_completion_time=float(mean_predicted_order_completion_time),
            effective_detour=float(effective_detour),
            effective_route_duration=float(effective_route_duration),
            effective_completion_time=float(effective_completion_time),
            effective_policy=effective_policy,
            detour_route_back=float(estimate.detour_route_back),
            detour_apass=float(estimate.detour_apass),
            detour_aup=float(estimate.detour_aup),
            p1_route=float(estimate.p1_route),
            p2_backtrack=float(estimate.p2_backtrack),
            p3_pass=float(estimate.p3_pass),
            p4_front=float(estimate.p4_front),
            p_sum=float(estimate.p_sum),
            v_nodes=len(V_nodes),
            r_nodes=len(R_nodes),
            g_pass_count=len(g_pass),
            g_front_count=len(g_front),
            predicted_completion_time_per_order=predicted_completion_time_per_order,
            initial_completion_time_per_order=initial_completion_time_per_order,
            should_integrate=should_integrate
        )

    samples: List[TimeSampleResult] = []
    for idx in range(effective_min_idx, effective_max_idx):
        sample = _build_sample(idx)
        if sample is not None:
            samples.append(sample)

    if not samples:
        raise ValueError(f"Keine Zeitstichproben fuer Batch {window.batch_index} erzeugt")

    # Phase 1: bestes erwartetes Mittel und Integrations-Gate gegen den Initialzustand.
    phase1_best_sample = min(samples, key=lambda sample: (sample.mean_predicted_order_completion_time, sample.t))

    mdp_based_formulation_sample = mdp_based_formulation.phase_1_discrete(samples)

    best_integration_t_idx = phase1_best_sample.idx
    phase1_should_integrate = (phase1_best_sample.mean_predicted_order_completion_time
                               < phase1_best_sample.mean_initial_order_completion_time)

    best_integration_time = map_route_index_to_time(best_integration_t_idx, base_inst)

    mean_initial_order_completion_time = phase1_best_sample.mean_initial_order_completion_time
    initial_completion_time = phase1_best_sample.initial_completion_time


    if best_integration_time >= EXPECTED_INTERARRIVAL:
        actual_waiting_time = 0.0
    else:
        actual_waiting_time = EXPECTED_INTERARRIVAL - best_integration_time

    # Phase 2: tatsächliche Ankunft + optionales Warten aus Phase 1 -> aktuelle Position.
    phase2_start_seconds = actual_t_seconds + actual_waiting_time
    phase2_start_idx = map_time_to_route_index(phase2_start_seconds, base_inst)
    phase2_start_x, phase2_start_y, _, _ = _detour_state_for_t(
        base_inst,
        window,
        article_mapping,
        T_in,
        T_out,
        phase2_start_idx,
    )

    actual_t = map_route_index_to_time(phase2_start_idx, base_inst)
    actual_x = float(phase2_start_x)
    actual_y = float(phase2_start_y)
    actual_case = "not_integrated"
    actual_detour = 0.0
    actual_route_duration = base_route_duration
    actual_completion_time = initial_completion_time
    actual_mean_order_completion_time = mean_initial_order_completion_time

    if phase2_start_seconds < base_route_duration:
        best_actual_t_idx, best_actual_x, best_actual_y, best_actual_case, best_actual_detour = _find_best_actual_integration_from_t(
            base_inst,
            window,
            article_mapping,
            T_in,
            T_out,
            start_t_idx=phase2_start_idx,
            end_t_idx=effective_max_idx,
        )

        if best_actual_case == "full_recompute":
            trial_route_duration = theoretical_route_duration(candidate_inst)
            trial_detour = trial_route_duration - base_route_duration
            trial_completion_time = trial_route_duration + avg_actual_arrival_times
            trial_mean_order_completion_time = (trial_completion_time - arrival_time_order_3) / 4.0
            trial_case = "full_recompute"
        else:
            trial_route_duration = (base_route_duration + best_actual_detour +
                                    insert_pick_service_time + actual_waiting_time)
            trial_detour = best_actual_detour + insert_pick_service_time
            trial_completion_time = trial_route_duration + arrival_time_order_3
            trial_mean_order_completion_time = trial_completion_time - avg_actual_arrival_times
            trial_case = best_actual_case

        if trial_mean_order_completion_time < mean_initial_order_completion_time:
            actual_t = map_route_index_to_time(best_actual_t_idx, base_inst)
            actual_x = float(best_actual_x)
            actual_y = float(best_actual_y)
            actual_case = trial_case
            actual_detour = float(trial_detour)
            actual_route_duration = float(trial_route_duration)
            actual_completion_time = float(trial_completion_time)
            actual_mean_order_completion_time = float(trial_mean_order_completion_time)

        sample_match = next((sample for sample in samples if sample.idx == best_actual_t_idx), None)
        actual_t_mean_predicted_order_completion_time = (
            float(sample_match.mean_predicted_order_completion_time) if sample_match is not None else float("nan")
        )
        actual_t_estimated_detour = float(sample_match.expected_detour) if sample_match is not None else float("nan")

    else:
        actual_t_mean_predicted_order_completion_time = -1.0
        actual_t_estimated_detour = -1.0

    return BatchComparisonResult(
        batch_index=window.batch_index,
        base_order_ids=[order.order_id for order in window.base_orders],
        insert_order_id=window.insert_order.order_id,
        base_order_arrival_times=base_order_arrival_times,
        insert_order_arrival_time=insert_order_arrival_time,
        base_order_positions=base_order_positions,
        insert_order_positions=insert_order_positions,
        base_route_duration=float(base_route_duration),
        estimated_t_insert_idx=estimated_t_insert_idx,
        phase1_should_integrate=phase1_should_integrate,
        phase1_best_integration_t_idx=best_integration_t_idx,
        phase1_best_integration_t=best_integration_time,
        actual_integration_t=actual_t,
        actual_waiting_time=float(actual_waiting_time),
        mean_initial_completion_time=float(mean_initial_order_completion_time),
        actual_x=float(actual_x),
        actual_y=float(actual_y),
        actual_case=actual_case,
        actual_route_duration=float(actual_route_duration),
        actual_detour=float(actual_detour),
        mean_actual_order_completion_time=float(actual_mean_order_completion_time),
        avg_arrival_time=float(avg_actual_arrival_times),
        phase1_best_t=phase1_best_sample.idx,
        phase1_best_x=phase1_best_sample.x,
        phase1_best_y=phase1_best_sample.y,
        phase1_predicted_best_detour=float(phase1_best_sample.expected_detour),
        phase1_predicted_route_duration=float(phase1_best_sample.predicted_route_duration),
        phase1_predicted_completion_time=float(phase1_best_sample.predicted_completion_time),
        actual_completion_time=float(actual_completion_time),
        samples=samples,
        actual_t_estimated_detour=actual_t_estimated_detour,
        actual_t_mean_predicted_OCT = actual_t_mean_predicted_order_completion_time
    )


def analyze_batches(
    order_json_path: str,
    mapping_json_path: str,
    *,
    M: int,
    N_L: int,
    w: float,
    L: float,
    v: float,
    t_p: float,
    known_size: int = 3,
    total_size: int = 4,
    max_batches: int | None = None,
) -> List[BatchComparisonResult]:
    orders = load_orders(order_json_path)
    article_mapping = load_article_mapping(mapping_json_path)
    results: List[BatchComparisonResult] = []
    start_index = 0

    while start_index + known_size < len(orders):
        if max_batches is not None and len(results) >= max_batches:
            break

        current = list(orders[start_index:start_index + total_size])
        if len(current) < total_size:
            break

        window = BatchWindow(
            batch_index=len(results) + 1,
            base_orders=current[:known_size],
            insert_order=current[known_size],
        )

        result = analyze_batch_window(
            window,
            article_mapping,
            M=M,
            N_L=N_L,
            w=w,
            L=L,
            v=v,
            t_p=t_p,
        )
        results.append(result)

        if result.actual_case == "not_integrated":
            start_index += known_size
        else:
            start_index += total_size

    return results

def batch_summary_rows(results: Sequence[BatchComparisonResult]) -> List[Dict[str, float | int | str]]:
    rows: List[Dict[str, float | int | str]] = []
    for result in results:
        # Bei not_integrated gibt es keine tatsaechliche Detour-Einfuegung;
        # daher werden die zugehoerigen Fehlergroessen nicht berechnet.
        if result.actual_case == "not_integrated":
            detour_error = float("nan")
            route_duration_error = float("nan")
            completion_time_error = float("nan")
        else:
            detour_error = result.actual_detour - result.actual_t_estimated_detour
            route_duration_error = float("nan")
            completion_time_error = result.mean_actual_order_completion_time - result.actual_t_mean_predicted_OCT

        rows.append(
            {
                "batch": result.batch_index,
                "base_orders": ",".join(str(order_id) for order_id in result.base_order_ids),
                "insert_order": result.insert_order_id,
                "base_order_arrival_times": ",".join(str(arrival_time) for arrival_time in result.base_order_arrival_times),
                "insert_order_arrival_time": result.insert_order_arrival_time,
                "base_order_positions": ",".join(str(pos) for pos in result.base_order_positions),
                "insert_order_position": ",".join(str(pos) for pos in result.insert_order_positions),
                "base_route_duration": result.base_route_duration,
                "estimated_t_insert_idx": result.estimated_t_insert_idx,
                "phase1_should_integrate": bool(result.phase1_should_integrate),
                "phase1_best_integration_t_idx": result.phase1_best_integration_t_idx,
                "phase1_best_integration_t": result.phase1_best_integration_t,
                "actual_integration_t": result.actual_integration_t,
                "actual_waiting_time": result.actual_waiting_time,
                "mean_initial_completion_time": result.mean_initial_completion_time,
                "actual_x": result.actual_x,
                "actual_y": result.actual_y,
                "actual_case": result.actual_case,
                "actual_route_duration": result.actual_route_duration,
                "actual_detour": result.actual_detour,
                "mean_actual_order_completion_time": result.mean_actual_order_completion_time,
                "avg_actual_arrival_time": result.avg_arrival_time,
                "phase1_best_t": result.phase1_best_t,
                "phase1_best_x": result.phase1_best_x,
                "phase1_best_y": result.phase1_best_y,
                "phase1_predicted_best_detour": result.phase1_predicted_best_detour,
                "phase1_predicted_route_duration": result.phase1_predicted_route_duration,
                "phase1_predicted_completion_time": result.phase1_predicted_completion_time,
                "actual_completion_time": result.actual_completion_time,
                "actual_t_estimated_detour": result.actual_t_estimated_detour,
                "actual_t_mean_predicted_OCT": result.actual_t_mean_predicted_OCT,
                "detour_error": detour_error,
                "route_duration_error": route_duration_error,
                "completion_time_error": completion_time_error,
            }
        )
    return rows


def batch_sample_rows(results: Sequence[BatchComparisonResult]) -> List[Dict[str, float | int | str]]:
    rows: List[Dict[str, float | int | str]] = []
    for result in results:
        for sample in result.samples:
            rows.append(
                {
                    "batch": result.batch_index,
                    "base_orders": ",".join(str(order_id) for order_id in result.base_order_ids),
                    "insert_order": result.insert_order_id,
                    "actual_integration_t": result.actual_integration_t,
                    "actual_waiting_time": result.actual_waiting_time,
                    "phase1_should_integrate": int(result.phase1_should_integrate),
                    "phase1_best_integration_t": result.phase1_best_integration_t,
                    "actual_case": result.actual_case,
                    "t": sample.t,
                    "x": sample.x,
                    "y": sample.y,
                    "phase": sample.phase,
                    "return_flag": sample.return_flag,
                    "E[D]": sample.expected_detour,
                    "base_route_duration": sample.base_route_duration,
                    "predicted_route_duration": sample.predicted_route_duration,
                    "waiting_time_to_estimated": sample.waiting_time_to_estimated,
                    "predicted_add_time": sample.predicted_add_time,
                    "arrival_time_order_3": sample.arrival_time_order_3,
                    "predicted_completion_time": sample.predicted_completion_time,
                    "predicted_mean_order_completion_time": sample.mean_predicted_order_completion_time,
                    "initial_completion_time": sample.initial_completion_time,
                    "mean_initial_completion_time": sample.mean_initial_order_completion_time,
                    "completion_time_no_insert": sample.initial_completion_time,
                    "effective_detour": sample.effective_detour,
                    "effective_route_duration": sample.effective_route_duration,
                    "effective_completion_time": sample.effective_completion_time,
                    "effective_policy": sample.effective_policy,
                    "d_rb": sample.detour_route_back,
                    "d_pass": sample.detour_apass,
                    "d_front": sample.detour_aup,
                    "p1": sample.p1_route,
                    "p2": sample.p2_backtrack,
                    "p3": sample.p3_pass,
                    "p4": sample.p4_front,
                    "p_sum": sample.p_sum,
                    "|V|": sample.v_nodes,
                    "|R|": sample.r_nodes,
                    "|Gp|": sample.g_pass_count,
                    "|Gf|": sample.g_front_count,
                }
            )
    return rows




