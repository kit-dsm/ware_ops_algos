"""
Kontinuierliche Variante des Batch-Optimizers.

Phase 1 (Planung): E[D(t)] wird kontinuierlich aus continuous_calculation berechnet.
Phase 2 (Ausführung): Nutzt exakte diskrete Detour-Funktionen für Vergleichbarkeit.
"""

from __future__ import annotations

import logging
from math import floor, isnan
from typing import Dict, List, Tuple

from . import mdp as mdp_based_formulation, continuous_calculation
from .core import (
    G_front,
    G_pass,
    V_R_global,
    _get_route_cache,
    compute_segment_times,
    map_route_index_to_time,
    map_time_to_route_index,
    theoretical_route_duration,
)
from .detour import (
    compute_detour_apass,
    compute_detour_aup,
    compute_detour_route_back_exact,
)
from .models import (
    BatchComparisonResult, BatchWindow, OrderInput, TimeSampleResult, WarehouseInstance,
    TIME_TOLERANCE, INTEGRATION_THRESHOLD,
)
from .optimizer_common import (
    EXPECTED_INTERARRIVAL,
    build_batch_windows,
    create_warehouse_instance_from_orders,
    expanded_pick_nodes_for_order,
    load_article_mapping,
    load_orders,
    order_pick_service_time,
)

logger = logging.getLogger(__name__)

# Lokale Aliasnamen für bessere Lesbarkeit
_expanded_pick_nodes_for_order = expanded_pick_nodes_for_order
_order_pick_service_time = order_pick_service_time


def _compute_actual_insert_arrival_delay(window: BatchWindow) -> float:
    """
    Berechnet die zeitliche Verzögerung zwischen der letzten bekannten Ankunftszeit
    und der tatsächlichen Ankunftszeit der Insert-Order (Phase 2).

    Returns:
        float: Nicht-negative Verzögerung in Sekunden.
    """
    last_known_arrival = max(order.arrival_time for order in window.base_orders)
    delay = window.insert_order.arrival_time - last_known_arrival
    return max(0.0, delay)



def _classify_detour_for_position(
    base_inst: WarehouseInstance,
    window: BatchWindow,
    article_mapping: Dict[str, Tuple[int, int]],
    T_in: Dict[int, float],
    T_out: Dict[int, float],
    t_idx: int,
) -> Tuple[float, float, str, float]:
    """
    **Phase 2 Helper:** Klassifiziert den Umweg für eine gegebene Routenposition.

    Nutzt exakte (diskrete) Detour-Berechnung aus core.detour für Vergleichbarkeit
    mit der diskreten Variante. Dies ist notwendig, da die Position im Lager (x,y)
    genau bestimmt werden muss.

    Args:
        base_inst: Warehouse-Instanz mit 3 bekannten Orders
        window: Batch mit 3 Base-Orders + 1 Insert-Order
        article_mapping: Pick-Positionen für alle Artikel
        T_in, T_out: Segment-Timing für S-Shape Routing
        t_idx: Route-Position als diskreter Index

    Returns:
        Tuple: (x, y, detour_case, detour_cost)
               - detour_case: "route" | "backtrack" | "pass" | "front" | "full_recompute" | "unclassified"
               - detour_cost: >= 0.0 oder NaN bei full_recompute
    """
    t_float = map_route_index_to_time(t_idx, base_inst)
    route_state = continuous_calculation.estimate_continuous_route_state(t_float, T_in, T_out, base_inst)
    x, y = float(route_state.x), float(route_state.y)
    pick_nodes = _expanded_pick_nodes_for_order(window.insert_order, article_mapping)

    # V_R_global erforderlich für Phase 2 Detour-Klassifikation
    V_nodes, R_nodes, p_eff, curr_ret = V_R_global(t_float, T_in, T_out, base_inst)

    # Multi-Pick Orders können nicht mit exakter Formel berechnet werden
    if len(pick_nodes) != 1:
        logger.debug(f"Multi-pick order detected ({len(pick_nodes)} nodes), full recompute")
        return float(x), float(y), "full_recompute", float("nan")

    actual_pick = pick_nodes[0]
    g_front = G_front(x, t_float, base_inst)
    g_pass = G_pass(x, t_float, base_inst)
    outbound = len(V_nodes) == 0

    # Exakte Klassifikation nach Knoten-Zustand
    if actual_pick in R_nodes:
        return float(x), float(y), "route", 0.0
    if actual_pick in V_nodes:
        detour = compute_detour_route_back_exact(
            base_inst, x, y,
            target_aisle=actual_pick[0],
            target_y_node=actual_pick[1],
            t=t_float,
            isReturnPhase=curr_ret,
        )
        return float(x), float(y), "backtrack", float(detour)
    if actual_pick[0] in g_pass:
        detour = compute_detour_apass(
            base_inst, x, y, t_float, g_pass, curr_ret,
            target_aisle=actual_pick[0],
        )
        return float(x), float(y), "pass", float(detour)
    if actual_pick[0] in g_front:
        detour = compute_detour_aup(
            base_inst, x, y, g_front, outbound, curr_ret,
            target_aisle=actual_pick[0],
        )
        return float(x), float(y), "front", float(detour)

    logger.warning(f"Unclassified pick node {actual_pick} at t_idx={t_idx}")
    return float(x), float(y), "unclassified", 0.0


def _search_optimal_integration_position(
    base_inst: WarehouseInstance,
    window: BatchWindow,
    article_mapping: Dict[str, Tuple[int, int]],
    T_in: Dict[int, float],
    T_out: Dict[int, float],
    start_t_idx: int,
    end_t_idx: int,
) -> Tuple[int, float, float, str, float]:
    """
    **Phase 2 Helper:** Findet die beste Integrationsstelle (minimale Umwegkosten).

    Iteriert über alle Routenpositionen von start bis end und wählt die Position
    mit dem kleinsten Umweg. Falls Multi-Pick-Order, Fallback auf Start-Position.

    Args:
        base_inst: Warehouse-Instanz
        window: Batch-Fenster
        article_mapping: Pick-Mapping
        T_in, T_out: Timing-Dictionaries
        start_t_idx: Suchstart (Route-Index)
        end_t_idx: Suchende (Route-Index)

    Returns:
        Tuple: (best_t_idx, x, y, detour_case, min_detour)
    """
    pick_nodes = _expanded_pick_nodes_for_order(window.insert_order, article_mapping)
    if len(pick_nodes) != 1:
        # Fallback: keine exakte Suche möglich
        x0, y0, case0, detour0 = _classify_detour_for_position(
            base_inst, window, article_mapping, T_in, T_out, start_t_idx
        )
        logger.debug(f"Multi-pick: returning start position {start_t_idx}")
        return start_t_idx, x0, y0, case0, detour0

    best_t = start_t_idx
    best_x, best_y, best_case, best_detour = _classify_detour_for_position(
        base_inst, window, article_mapping, T_in, T_out, start_t_idx
    )

    # Suche beste Position (minimaler Umweg)
    for t in range(start_t_idx + 1, end_t_idx + 1):
        x, y, case, detour = _classify_detour_for_position(
            base_inst, window, article_mapping, T_in, T_out, t
        )
        if not isnan(detour) and detour < best_detour - TIME_TOLERANCE:
            best_t = t
            best_x, best_y, best_case, best_detour = x, y, case, detour

    return best_t, best_x, best_y, best_case, best_detour


def _initialize_batch_data(
    window: BatchWindow,
    article_mapping: Dict[str, Tuple[int, int]],
    M: int, N_L: int, w: float, L: float, v: float, t_p: float,
) -> Tuple[Dict, float, float]:
    """
    Initialisiert die Batch-Daten: Route-Instanzen und Segment-Timing.

    Returns:
        Tuple: (base_inst, candidate_inst, T_in, T_out, base_route_duration)
    """
    base_inst = create_warehouse_instance_from_orders(
        window.base_orders, article_mapping,
        M=M, N_L=N_L, w=w, L=L, v=v, t_p=t_p,
    )
    candidate_orders = window.all_orders
    candidate_inst = create_warehouse_instance_from_orders(
        candidate_orders, article_mapping,
        M=M, N_L=N_L, w=w, L=L, v=v, t_p=t_p,
    )

    T_in, T_out, base_route_duration = compute_segment_times(base_inst)

    return {
        'base_inst': base_inst,
        'candidate_inst': candidate_inst,
        'T_in': T_in,
        'T_out': T_out,
    }, base_route_duration, candidate_inst.arrival_times


def _calculate_reference_times(window: BatchWindow, base_route_duration: float) -> Tuple[float, float]:
    """
    Berechnet Referenzzeiten für die Baseline ohne Integration.

    Returns:
        (arrival_time_order_3, initial_completion_time)
    """
    if len(window.base_orders) >= 3:
        arrival_time_order_3 = float(window.base_orders[2].arrival_time)
    elif window.base_orders:
        arrival_time_order_3 = float(window.base_orders[-1].arrival_time)
    else:
        arrival_time_order_3 = 0.0

    return arrival_time_order_3, base_route_duration + arrival_time_order_3


def _run_phase1_sampling(
    base_inst: WarehouseInstance,
    window: BatchWindow,
    article_mapping: Dict[str, Tuple[int, int]],
    T_in: Dict[int, float],
    T_out: Dict[int, float],
    base_route_duration: float,
    insert_pick_service_time: float,
    arrival_time_order_3: float,
    initial_completion_time: float,
    cache,
    effective_min_idx: int,
    effective_max_idx: int,
) -> Tuple[List[TimeSampleResult], float, float]:
    """
    **Phase 1 – SAMPLING:** Generiert Zeitstichproben unter der Annahme einer
    geschätzten Ankunftszeit (EXPECTED_INTERARRIVAL).

    Dies ist der Planungshorizont: die tatsächliche Ankunftszeit der Order 4 ist NICHT bekannt.
    Wir nehmen an: t_estimate = t_last_known + EXPECTED_INTERARRIVAL (28.8s)

    Returns:
        Tuple: (samples, estimated_t_insert_idx, estimated_t_waiting)
    """
    estimated_t_insert_idx = min(
        map_time_to_route_index(EXPECTED_INTERARRIVAL, base_inst),
        effective_max_idx
    )
    estimated_t_waiting: float = min(EXPECTED_INTERARRIVAL, float(base_route_duration))

    def _build_single_sample(idx: int, t_float: float) -> TimeSampleResult | None:
        """Baut einen einzelnen Zeitstichpunkt für Phase 1."""
        estimate = continuous_calculation.continuous_expected_detour(
            t_float, base_inst, T_in, T_out)

        E_D_cont = estimate.expected_detour
        x = estimate.x
        y = estimate.y

        predicted_route_duration = base_route_duration + E_D_cont + insert_pick_service_time

        # Wartezeit bis zur prognostizierten Ankunft (EXPECTED_INTERARRIVAL = 28.8s)
        waiting_time_to_estimated = float(max(0.0, estimated_t_waiting - t_float))
        predicted_add_time = waiting_time_to_estimated + E_D_cont + insert_pick_service_time
        predicted_completion_time = arrival_time_order_3 + base_route_duration + predicted_add_time

        predicted_completion_time_per_order = (predicted_completion_time - arrival_time_order_3) / 4.0
        initial_completion_time_per_order = (initial_completion_time - arrival_time_order_3) / 3.0

        # Für den Mittelwert der Ankunftszeiten benutzen wir das gleiche Schema wie diskret
        base_avg_arrival_time = sum(order.arrival_time for order in window.base_orders) / len(window.base_orders)
        predicted_avg_arrival_time = (
            sum(order.arrival_time for order in window.base_orders)
            + (EXPECTED_INTERARRIVAL + arrival_time_order_3)
        ) / (len(window.base_orders) + 1)

        mean_initial_order_completion_time = initial_completion_time - base_avg_arrival_time
        mean_predicted_order_completion_time = predicted_completion_time - predicted_avg_arrival_time

        should_integrate = mean_predicted_order_completion_time < mean_initial_order_completion_time

        if should_integrate:
            effective_policy = "integrate"
            effective_detour = E_D_cont + insert_pick_service_time
            effective_route_duration = predicted_route_duration
            effective_completion_time = predicted_completion_time
        else:
            effective_policy = "not_integrated"
            effective_detour = 0.0
            effective_route_duration = base_route_duration
            effective_completion_time = initial_completion_time

        phase = "hin" if estimate.phase_label == "phase_1_to_first_entry" else "rueck"
        return_flag = "ret" if estimate.is_return_phase else "-"

        return TimeSampleResult(
            batch_index=window.batch_index,
            idx=idx,
            t=t_float,
            x=float(x),
            y=float(y),
            phase=phase,
            return_flag=return_flag,
            base_route_duration=float(base_route_duration),
            expected_detour=float(E_D_cont),
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
            v_nodes=float(estimate.v_cont),
            r_nodes=float(estimate.r_cont),
            g_pass_count=int(estimate.g_pass_count),
            g_front_count=int(estimate.g_front_count),
            predicted_completion_time_per_order=predicted_completion_time_per_order,
            initial_completion_time_per_order=initial_completion_time_per_order,
            should_integrate=should_integrate,
        )

    samples: List[TimeSampleResult] = []
    for idx in range(effective_min_idx, effective_max_idx):
        t_float = map_route_index_to_time(idx, base_inst)
        sample = _build_single_sample(idx, t_float)
        if sample is not None:
            samples.append(sample)

    if not samples:
        raise ValueError(
            f"Batch {window.batch_index}: Keine Zeitstichproben erzeugt. "
            f"Prüfen Sie effective_min_idx={effective_min_idx}, effective_max_idx={effective_max_idx}"
        )

    return samples, estimated_t_insert_idx, estimated_t_waiting


def _run_phase2_integration(
    base_inst: WarehouseInstance,
    candidate_inst: WarehouseInstance,
    window: BatchWindow,
    article_mapping: Dict[str, Tuple[int, int]],
    T_in: Dict[int, float],
    T_out: Dict[int, float],
    base_route_duration: float,
    insert_pick_service_time: float,
    arrival_time_order_3: float,
    initial_completion_time: float,
    samples: List[TimeSampleResult],
    mean_initial_order_completion_time: float,
    phase1_best_sample: TimeSampleResult,
    actual_t_seconds: float,
    effective_max_idx: int,
) -> Tuple[float, float, float, str, float, float, float, float, float, float, float]:
    """
    **Phase 2 – INTEGRATION:** Nutzt tatsächliche Ankunftszeit zur optimalen Integrationsstelle.

    Verarbeitet das Outcome der Phase 1 gegen die reale Ankunftszeit.

    Returns:
        Tuple: (actual_t, actual_x, actual_y, actual_case, actual_detour,
                actual_route_duration, actual_completion_time, actual_mean_order_completion_time,
                actual_t_mean_predicted_order_completion_time, actual_t_estimated_detour, actual_waiting_time)
    """
    # Best Integration Time aus Phase 1
    best_integration_t_idx = phase1_best_sample.idx
    best_integration_time = map_route_index_to_time(best_integration_t_idx, base_inst)

    # Warte-Logik: Falls Phase 1 später als EXPECTED_INTERARRIVAL, kein Warten nötig
    if best_integration_time >= EXPECTED_INTERARRIVAL:
        actual_waiting_time = 0.0
    else:
        actual_waiting_time = EXPECTED_INTERARRIVAL - best_integration_time

    # Phase 2 Startposition: tatsächliche Ankunftszeit + optionales Warten
    phase2_start_seconds = actual_t_seconds + actual_waiting_time
    phase2_start_idx = map_time_to_route_index(phase2_start_seconds, base_inst)

    # Initiale Defaults (nicht integriert)
    actual_t = map_route_index_to_time(phase2_start_idx, base_inst)
    actual_x, actual_y, _, _ = _classify_detour_for_position(
        base_inst, window, article_mapping, T_in, T_out, phase2_start_idx,
    )
    actual_case = "not_integrated"
    actual_detour = 0.0
    actual_route_duration = base_route_duration
    actual_completion_time = initial_completion_time
    actual_mean_order_completion_time = mean_initial_order_completion_time
    actual_t_mean_predicted_order_completion_time = float("nan")
    actual_t_estimated_detour = float("nan")

    # Suche beste Integrationsstelle, wenn noch Zeit in der Route vorhanden
    if phase2_start_seconds < base_route_duration:
        best_actual_t_idx, best_actual_x, best_actual_y, best_actual_case, best_actual_detour = (
            _search_optimal_integration_position(
                base_inst, window, article_mapping, T_in, T_out,
                start_t_idx=phase2_start_idx,
                end_t_idx=effective_max_idx,
            )
        )

        # Berechne Trial-Kosten für gefundene beste Position
        if best_actual_case == "full_recompute":
            trial_route_duration = theoretical_route_duration(candidate_inst)
            trial_detour = trial_route_duration - base_route_duration
            avg_actual_arrival_times = sum(candidate_inst.arrival_times) / len(candidate_inst.arrival_times)
            trial_completion_time = trial_route_duration + avg_actual_arrival_times
            trial_mean_order_completion_time = (trial_completion_time - arrival_time_order_3) / 4.0
            trial_case = "full_recompute"
            logger.debug(f"Batch {window.batch_index}: full_recompute trial")
        else:
            trial_route_duration = (
                base_route_duration + best_actual_detour + insert_pick_service_time + actual_waiting_time
            )
            trial_detour = best_actual_detour + insert_pick_service_time
            trial_completion_time = trial_route_duration + arrival_time_order_3
            avg_actual_arrival_times = sum(candidate_inst.arrival_times) / len(candidate_inst.arrival_times)
            trial_mean_order_completion_time = trial_completion_time - avg_actual_arrival_times
            trial_case = best_actual_case

        # Nur integrieren, wenn Trial besser als Baseline
        if trial_mean_order_completion_time < mean_initial_order_completion_time:
            actual_t = map_route_index_to_time(best_actual_t_idx, base_inst)
            actual_x = float(best_actual_x)
            actual_y = float(best_actual_y)
            actual_case = trial_case
            actual_detour = float(trial_detour)
            actual_route_duration = float(trial_route_duration)
            actual_completion_time = float(trial_completion_time)
            actual_mean_order_completion_time = float(trial_mean_order_completion_time)
            logger.debug(f"Batch {window.batch_index}: Integrating at idx={best_actual_t_idx}, case={trial_case}")

        # Lookup sampel-Match für diagnostische Werte
        sample_match = next((s for s in samples if s.idx == best_actual_t_idx), None)
        if sample_match is not None:
            actual_t_mean_predicted_order_completion_time = float(sample_match.mean_predicted_order_completion_time)
            actual_t_estimated_detour = float(sample_match.expected_detour)
        else:
            actual_t_mean_predicted_order_completion_time = INTEGRATION_THRESHOLD
            actual_t_estimated_detour = INTEGRATION_THRESHOLD
    else:
        actual_t_mean_predicted_order_completion_time = INTEGRATION_THRESHOLD
        actual_t_estimated_detour = INTEGRATION_THRESHOLD

    return (actual_t, actual_x, actual_y, actual_case, actual_detour, actual_route_duration,
            actual_completion_time, actual_mean_order_completion_time,
            actual_t_mean_predicted_order_completion_time, actual_t_estimated_detour, actual_waiting_time)


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
    """
    Hauptfunktion: Kontinuierliche Optimierung für ein Batch-Fenster.

    **Ablauf:**
    1. Initialisiere Batch-Daten (Warehouse-Instanzen, Timings)
    2. Phase 1: Sampling mit geschätzter Ankunftszeit (EXPECTED_INTERARRIVAL)
    3. Phase 1: Finde beste Integrationsstelle gemäß E[D(t)]
    4. Phase 2: Nutze tatsächliche Ankunftszeit und exakte Detour-Klassifikation
    5. Baue Vergleichs-Result
    """
    # Extrahiere Positionen der Orders
    base_order_arrival_times = [float(order.arrival_time) for order in window.base_orders]
    insert_order_arrival_time = float(window.insert_order.arrival_time)

    base_order_positions = [
        _expanded_pick_nodes_for_order(order, article_mapping)
        for order in window.base_orders
    ]
    insert_order_positions = _expanded_pick_nodes_for_order(window.insert_order, article_mapping)

    # =========================================================================
    # Initialisierung
    # =========================================================================
    batch_data, base_route_duration, candidate_arrival_times = _initialize_batch_data(
        window, article_mapping, M, N_L, w, L, v, t_p,
    )
    base_inst = batch_data['base_inst']
    candidate_inst = batch_data['candidate_inst']
    T_in = batch_data['T_in']
    T_out = batch_data['T_out']

    insert_pick_service_time = _order_pick_service_time(window.insert_order, t_p)
    avg_actual_arrival_times = sum(candidate_arrival_times) / len(candidate_arrival_times)

    arrival_time_order_3, initial_completion_time = _calculate_reference_times(
        window, base_route_duration
    )

    # Tatsächliche (reale) Ankunftsverzögerung der Insert-Order
    actual_t_seconds = _compute_actual_insert_arrival_delay(window)

    cache = _get_route_cache(base_inst)
    effective_min_idx = 0
    effective_max_idx = len(cache.route) - 1

    # =========================================================================
    # Phase 1: Sampling mit geschätzter Ankunftszeit
    # =========================================================================
    try:
        samples, estimated_t_insert_idx, estimated_t_waiting = _run_phase1_sampling(
            base_inst,
            window,
            article_mapping,
            T_in,
            T_out,
            base_route_duration,
            insert_pick_service_time,
            arrival_time_order_3,
            initial_completion_time,
            cache,
            effective_min_idx,
            effective_max_idx,
        )
    except ValueError as e:
        logger.error(f"Phase 1 sampling failed for batch {window.batch_index}: {e}")
        raise

    # =========================================================================
    # Phase 1: Beste erwartete Integration vs. Baseline
    # =========================================================================
    phase1_best_sample = min(samples, key=lambda sample: (sample.mean_predicted_order_completion_time, sample.t))
    _ = mdp_based_formulation.phase_1_discrete(samples)  # Optionaler Vergleich diskret vs. kontinuierlich

    best_integration_t_idx = phase1_best_sample.idx
    phase1_should_integrate = (
        phase1_best_sample.mean_predicted_order_completion_time
        < phase1_best_sample.mean_initial_order_completion_time
    )
    mean_initial_order_completion_time = phase1_best_sample.mean_initial_order_completion_time
    initial_completion_time = phase1_best_sample.initial_completion_time

    logger.debug(
        f"Batch {window.batch_index}: Phase1 best_sample at idx={best_integration_t_idx}, "
        f"should_integrate={phase1_should_integrate}"
    )

    # =========================================================================
    # Phase 2: Tatsächliche Integration mit realer Ankunftszeit
    # =========================================================================
    (actual_t, actual_x, actual_y, actual_case, actual_detour, actual_route_duration,
     actual_completion_time, actual_mean_order_completion_time,
     actual_t_mean_predicted_order_completion_time, actual_t_estimated_detour,
     actual_waiting_time) = _run_phase2_integration(
        base_inst,
        candidate_inst,
        window,
        article_mapping,
        T_in,
        T_out,
        base_route_duration,
        insert_pick_service_time,
        arrival_time_order_3,
        initial_completion_time,
        samples,
        mean_initial_order_completion_time,
        phase1_best_sample,
        actual_t_seconds,
        effective_max_idx,
    )

    # =========================================================================
    # Result zusammenstellen
    # =========================================================================
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
        phase1_best_integration_t=map_route_index_to_time(best_integration_t_idx, base_inst),
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
        actual_t_mean_predicted_OCT=actual_t_mean_predicted_order_completion_time,
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
