import importlib
from pathlib import Path

import pytest

from ware_ops_algos.algorithms import (
    ExactTSPRoutingDistance,
    ProfitableSprpRouting,
    ProfitableSprpRoutingRG,
    RatliffRosenthalScatteredRouting,
)
from ware_ops_algos.data_loaders import HesslerIrnichLoader

routing_module = importlib.import_module("ware_ops_algos.algorithms.routing.routing")


TRANSIENT = {
    "state_graph", "_aisle_ys", "_position_edges", "pick_list",
    "current_order", "mdl", "x", "x_start", "x_end", "T", "distance",
}


def _routing_kwargs(domain):
    network = domain.layout.layout_network
    return {
        "start_node": network.start_node,
        "end_node": network.end_node,
        "closest_node_to_start": network.closest_node_to_start,
        "min_aisle_position": network.min_aisle_position,
        "max_aisle_position": network.max_aisle_position,
        "picker": domain.resources.resources,
    }


def _rr_kwargs(domain):
    data = domain.layout.graph_data
    return {
        **_routing_kwargs(domain),
        "n_aisles": data.n_aisles,
        "n_pick_locations": data.n_pick_locations,
        "dist_aisle": data.dist_aisle,
        "dist_pick_locations": data.dist_pick_locations,
        "dist_aisle_location": data.dist_bottom_to_pick_location,
        "dist_start": data.dist_start,
        "dist_end": data.dist_end,
    }


def test_exact_router_reuse_does_not_retain_model_state(
    muter_domain, muter_resolved_orders, monkeypatch,
):
    network = muter_domain.layout.layout_network
    router = ExactTSPRoutingDistance(
        **_routing_kwargs(muter_domain),
        distance_matrix=network.distance_matrix,
        predecessor_matrix=network.predecessor_matrix,
        gen_tour=False,
        gen_item_sequence=False,
        big_m=1000,
        set_time_limit=30,
    )
    first = muter_resolved_orders[0].pick_positions[:4]
    second = muter_resolved_orders[1].pick_positions[:3]

    first_score = router.score(first)
    assert router.solve(second).route.distance == pytest.approx(router.score(second))
    assert router.solve(first).route.distance == pytest.approx(first_score)
    assert TRANSIENT.isdisjoint(router.__dict__)

    def fail(*args, **kwargs):
        raise AssertionError("exact score allocated a Route")

    monkeypatch.setattr(routing_module, "Route", fail)
    assert router.score(second) > 0


def test_scattered_score_solve_and_reuse_are_stateless(monkeypatch):
    instance_dir = Path("data/instances/SPRP-SS").resolve()
    domain = HesslerIrnichLoader(instance_dir).load(
        str(instance_dir / "unit_F2_m5_C30_a3_3.txt"), use_cache=False,
    )
    inputs = domain.orders.orders[0].order_positions
    router = RatliffRosenthalScatteredRouting(
        storage_locations=domain.storage,
        **_rr_kwargs(domain),
    )

    score = router.score(inputs)

    def fail(*args, **kwargs):
        raise AssertionError("physical scattered tour constructed")

    monkeypatch.setattr(router, "_construct_scattered_tour", fail)
    assert router.score(inputs) == pytest.approx(score)
    assert router.solve(inputs).route.distance == pytest.approx(score)
    assert TRANSIENT.isdisjoint(router.__dict__)

    tour_router = RatliffRosenthalScatteredRouting(
        storage_locations=domain.storage,
        gen_tour=True,
        **_rr_kwargs(domain),
    )
    route = tour_router.solve(inputs).route
    assert route.route
    assert route.route[0] == tour_router.start_node
    assert route.route[-1] == tour_router.end_node
    assert route.distance == pytest.approx(score)
    assert TRANSIENT.isdisjoint(tour_router.__dict__)


@pytest.mark.parametrize("routing_class", (ProfitableSprpRouting, ProfitableSprpRoutingRG))
def test_profitable_prepared_state_belongs_to_context(routing_class):
    router = routing_class(
        start_node=(0, 0), end_node=(0, 0), closest_node_to_start=(0, 0),
        min_aisle_position=1, max_aisle_position=10, picker=[],
        n_aisles=4, n_pick_locations=10, dist_aisle=2,
        dist_pick_locations=1, dist_aisle_location=1,
        dist_start=1, dist_end=1,
    )
    before = set(router.__dict__)
    context = router.prepare(
        [[(1, 2), (1, 8)], [(4, 4)], [(2, 3)]],
        [2, 2, 3],
        4,
    )

    first, first_distance = router.solve_with_scores(context, [5, 4, 1])
    second, second_distance = router.solve_with_scores(context, [1, 1, 10])
    assert first.tolist() == [0, 1]
    assert second.tolist() == [2]
    assert first_distance == pytest.approx(34.0)
    assert second_distance == pytest.approx(6.0)
    assert set(router.__dict__) == before
    assert TRANSIENT.isdisjoint(router.__dict__)
    assert context["graph"] is not None
    with pytest.raises(TypeError, match="prepare"):
        router.score([])
    with pytest.raises(TypeError, match="prepare"):
        router.solve([])
