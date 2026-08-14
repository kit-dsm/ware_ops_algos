"""Small, repeatable cProfile study for representative ware_ops_algos paths.

This is a diagnostic harness, not a benchmark framework.  It deliberately
uses the same cached fixture and algorithm wiring as tests/benchmarks.
"""

from __future__ import annotations

import cProfile
import argparse
import json
import pstats
import statistics
import time
from pathlib import Path

from ware_ops_algos.algorithms import (
    ClarkAndWrightBatching,
    GreedyItemAssignment,
    LargestGapRouting,
    LocalSearchBatching,
    MidpointRouting,
    NearestNeighbourhoodRouting,
    OrderNrFifoBatching,
    PickListRouting,
    RatliffRosenthalRouting,
    ReturnRouting,
    SShapeRouting,
    UShapeRouting,
)
from ware_ops_algos.algorithms.scheduling.scheduling import LPTScheduling
from ware_ops_algos.data_loaders import HesslerIrnichLoader


ROOT = Path(__file__).resolve().parents[1]
INSTANCES = ROOT / "data" / "instances"
CACHES = INSTANCES / "caches"


def load_fixture(name: str):
    fixtures = {
        "henn": ("HennWaescherUniform", "1l-20-30-0.txt"),
        "muter": ("MuterOencan", "100_48_5.txt"),
    }
    instance_set, instance_file = fixtures[name]
    loader = HesslerIrnichLoader(
        instances_dir=INSTANCES / instance_set,
        cache_dir=CACHES / instance_set,
    )
    domain = loader.load(instance_file)
    orders = GreedyItemAssignment(domain.storage).solve(domain.orders.orders).resolved_orders
    return domain, orders


def routing_kwargs(domain):
    ln = domain.layout.layout_network
    return {
        "start_node": ln.start_node,
        "end_node": ln.end_node,
        "closest_node_to_start": ln.closest_node_to_start,
        "min_aisle_position": ln.min_aisle_position,
        "max_aisle_position": ln.max_aisle_position,
        "distance_matrix": ln.distance_matrix,
        "predecessor_matrix": ln.predecessor_matrix,
        "picker": domain.resources.resources,
        "gen_tour": False,
        "gen_item_sequence": False,
        "node_list": ln.node_list,
        "node_to_idx": {node: idx for idx, node in enumerate(ln.graph.nodes)},
        "idx_to_node": {idx: node for idx, node in enumerate(ln.graph.nodes)},
    }


def rr_routing_kwargs(domain):
    ln = domain.layout.layout_network
    gd = domain.layout.graph_data
    return {
        "start_node": ln.start_node,
        "end_node": ln.end_node,
        "closest_node_to_start": ln.closest_node_to_start,
        "min_aisle_position": ln.min_aisle_position,
        "max_aisle_position": ln.max_aisle_position,
        "distance_matrix": ln.distance_matrix,
        "predecessor_matrix": ln.predecessor_matrix,
        "picker": domain.resources.resources,
        "n_aisles": gd.n_aisles,
        "n_pick_locations": gd.n_pick_locations,
        "dist_aisle": gd.dist_aisle,
        "dist_pick_locations": gd.dist_pick_locations,
        "dist_aisle_location": gd.dist_bottom_to_pick_location,
        "dist_start": gd.dist_start,
        "dist_end": gd.dist_end,
        "gen_tour": False,
        "gen_item_sequence": False,
    }


def profile(name, fn, repeats=3):
    """Measure cold calls; profile one matching cold call."""
    elapsed = []
    cache_sizes = []
    for _ in range(repeats):
        started = time.perf_counter()
        obj = fn()
        elapsed.append(time.perf_counter() - started)
        cache_sizes.append(getattr(obj, "_route_cache", None) and len(obj._route_cache))

    profiler = cProfile.Profile()
    profiler.enable()
    profiled_obj = fn()
    profiler.disable()
    stats = pstats.Stats(profiler)
    rows = []
    for (file, line, func), (cc, nc, tt, ct, callers) in stats.stats.items():
        if "ware_ops_algos" not in file:
            continue
        rows.append({
            "function": f"{Path(file).name}:{line}({func})",
            "calls": nc,
            "self_s": tt,
            "cum_s": ct,
        })
    by_cum = sorted(rows, key=lambda x: x["cum_s"], reverse=True)[:15]
    by_self = sorted(rows, key=lambda x: x["self_s"], reverse=True)[:15]
    interest_terms = (
        "_batch_cost_from_orders", "_calc_dist_with_routing_algo",
        "_calculate_saving", "orders_fit", "_compute_consumption",
        "_compute_order_consumption", "_get_article_dim_index",
        "_get_distance", "_get_next_nearest_node_by_dijkstra",
        "_nearest_kernel", "_distance_of_visit_order", "_compute_visit_order",
        "_materialize_output", "score",
        "_solve_direct_dp", "_prepare_aisles", "_propagate_aisle_layer",
        "_propagate_cross_aisle_layer", "_backtrack_decisions",
        "build_state_space", "_add_aisle_transitions", "_add_cross_aisle_transitions",
        "_construct_picker_tour", "_build_annotated_route", "<listcomp>",
    )
    interesting = [row for row in rows if any(term in row["function"] for term in interest_terms)]
    interesting.sort(key=lambda x: x["cum_s"], reverse=True)
    route_request_functions = {
        "_batch_cost_from_orders", "_calc_dist_with_routing_algo",
    }
    route_requests = sum(
        nc for (_file, _line, func), (_cc, nc, _tt, _ct, _callers)
        in stats.stats.items()
        if func in route_request_functions
    )
    score_rows = [
        (nc, ct)
        for (file, _line, func), (_cc, nc, _tt, ct, _callers)
        in stats.stats.items()
        if func == "score" and Path(file).name == "routing.py"
    ]
    score_calls = sum(nc for nc, _ in score_rows)
    score_cumulative_s = sum(ct for _, ct in score_rows)
    return {
        "name": name,
        "repeats": repeats,
        "cold_s": {"min": min(elapsed), "median": statistics.median(elapsed), "max": max(elapsed)},
        "route_cache_entries": cache_sizes,
        "profile_total_s": stats.total_tt,
        "top_cumulative": by_cum,
        "top_self": by_self,
        "selected_hotspots": interesting[:30],
        "profiled_cache_entries": getattr(profiled_obj, "_route_cache", None) and len(profiled_obj._route_cache),
        "route_score_requests": route_requests,
        "route_score_cache_misses": score_calls if route_requests else None,
        "route_score_cache_hits": route_requests - score_calls if route_requests else None,
        "routing_score_cumulative_s": score_cumulative_s,
    }


def run_final_routing(orders, cart, articles, router):
    batches = OrderNrFifoBatching(cart, articles).solve(orders).batches
    for batch in batches:
        router.solve(batch.pick_positions)
    return router


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", choices=("henn", "muter"), default="henn")
    parser.add_argument("--expensive-repeats", type=int, default=3)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--filter", action="append", default=[])
    args = parser.parse_args()
    domain, orders = load_fixture(args.fixture)
    cart = domain.resources.resources[0].pick_cart
    articles = domain.articles
    nn_kwargs = routing_kwargs(domain)
    rr_kwargs = rr_routing_kwargs(domain)
    all_picks = [pos for order in orders for pos in order.pick_positions]

    results = []
    results.append(profile(
        "constructive_fifo",
        lambda: (lambda algo: (algo.solve(orders), algo)[1])(OrderNrFifoBatching(cart, articles)),
        repeats=12,
    ))
    for routing_class in (
        SShapeRouting,
        ReturnRouting,
        MidpointRouting,
        LargestGapRouting,
        NearestNeighbourhoodRouting,
        UShapeRouting,
        PickListRouting,
    ):
        label = routing_class.__name__.removesuffix("Routing").lower()
        score_router = routing_class(**nn_kwargs)
        semantic_router = routing_class(**nn_kwargs)
        tour_router = routing_class(**(
            nn_kwargs | {"gen_tour": True, "gen_item_sequence": True}
        ))
        results.append(profile(
            f"{label}_score_all_fixture_picks",
            lambda router=score_router: (router.score(all_picks), router)[1],
            repeats=12,
        ))
        results.append(profile(
            f"{label}_solve_semantic_all_fixture_picks",
            lambda router=semantic_router: (router.solve(all_picks), router)[1],
            repeats=8,
        ))
        results.append(profile(
            f"{label}_solve_tour_all_fixture_picks",
            lambda router=tour_router: (router.solve(all_picks), router)[1],
            repeats=8,
        ))

    rr_score_router = RatliffRosenthalRouting(**rr_kwargs)
    rr_semantic_router = RatliffRosenthalRouting(**rr_kwargs)
    rr_tour_router = RatliffRosenthalRouting(**(
        rr_kwargs | {"gen_tour": True, "gen_item_sequence": True}
    ))
    results.append(profile(
        "ratliff_rosenthal_score_all_fixture_picks",
        lambda: (rr_score_router.score(all_picks), rr_score_router)[1],
        repeats=20,
    ))
    results.append(profile(
        "ratliff_rosenthal_solve_semantic_all_fixture_picks",
        lambda: (rr_semantic_router.solve(all_picks), rr_semantic_router)[1],
        repeats=12,
    ))
    results.append(profile(
        "ratliff_rosenthal_solve_tour_all_fixture_picks",
        lambda: (rr_tour_router.solve(all_picks), rr_tour_router)[1],
        repeats=12,
    ))
    results.append(profile(
        "local_search_batching_rr_5s_limit",
        lambda: (lambda algo: (algo.solve(orders), algo)[1])(LocalSearchBatching(
            cart, articles, RatliffRosenthalRouting, rr_kwargs, OrderNrFifoBatching, time_limit=5.0)),
        repeats=args.expensive_repeats,
    ))
    results.append(profile(
        "clark_wright_batching_nn",
        lambda: (lambda algo: (algo.solve(orders), algo)[1])(ClarkAndWrightBatching(
            cart, articles, NearestNeighbourhoodRouting, nn_kwargs)),
        repeats=args.expensive_repeats,
    ))
    final_router = NearestNeighbourhoodRouting(**(
        nn_kwargs | {"gen_item_sequence": True}
    ))
    results.append(profile(
        "picker_routing_final_fifo_batches_nn",
        lambda: run_final_routing(
            orders, cart, articles, final_router,
        ),
        repeats=12,
    ))

    # Build ordinary scheduling input from the representative orders; scheduling
    # is intentionally measured separately from routing/batching.
    # This fixture leaves setup time unspecified; scheduling requires it, so the
    # benchmark supplies the neutral value for this standalone stage.
    for resource in domain.resources.resources:
        if resource.tour_setup_time is None:
            resource.tour_setup_time = 0.0
    # Explicit Job construction keeps the scheduling stage independent from the
    # preceding routing/batching output details.
    from ware_ops_algos.algorithms.algorithm_interfaces import Job
    jobs = [
        Job(i, 10.0 + i, float(i % 4), 80.0 + i, len(order.pick_positions))
        for i, order in enumerate(orders)
    ]
    results.append(profile(
        "lpt_scheduling_fixture_jobs",
        lambda: (lambda algo: (algo.solve(jobs), algo)[1])(LPTScheduling(domain.resources)),
        repeats=30,
    ))
    payload = {
        "fixture": args.fixture,
        "orders": len(orders),
        "pick_positions": len(all_picks),
        "results": results,
    }
    if args.filter:
        payload["results"] = [
            result for result in payload["results"]
            if any(term in result["name"] for term in args.filter)
        ]
    if args.summary_only:
        payload["results"] = [
            {
                "name": result["name"],
                "median_ms": result["cold_s"]["median"] * 1000,
                "route_score_requests": result["route_score_requests"],
                "route_score_cache_hits": result["route_score_cache_hits"],
                "route_score_cache_misses": result["route_score_cache_misses"],
                "routing_score_cumulative_ms": result["routing_score_cumulative_s"] * 1000,
                "top_cumulative": result["top_cumulative"][:5],
                "selected_hotspots": result["selected_hotspots"][:10],
            }
            for result in payload["results"]
        ]
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
