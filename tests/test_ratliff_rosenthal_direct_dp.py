import networkx as nx
import pandas as pd
import pytest

from ware_ops_algos.algorithms import ExactTSPRoutingDistance, RatliffRosenthalRouting
from ware_ops_algos.algorithms.algorithm_interfaces import PickPosition
from ware_ops_algos.algorithms.routing.routing import _RRDecision
from ware_ops_algos.algorithms.routing.dynamic_programming_helpers import aisle_mapping
from ware_ops_algos.domain_models import Resource


def _picks(nodes):
    return [
        PickPosition(1, index + 1, 1, node, 1)
        for index, node in enumerate(nodes)
    ]


def _kwargs(*, depot=0, gen_tour=False, gen_item_sequence=False):
    return {
        "start_node": (depot, 0),
        "end_node": (depot, 0),
        "closest_node_to_start": (depot, 0),
        "min_aisle_position": 1,
        "max_aisle_position": 10,
        "picker": [Resource(id=1)],
        "n_aisles": 4,
        "n_pick_locations": 10,
        "dist_aisle": 2,
        "dist_pick_locations": 1,
        "dist_aisle_location": 1,
        "dist_start": 1,
        "dist_end": 1,
        "gen_tour": gen_tour,
        "gen_item_sequence": gen_item_sequence,
    }


def _decision_signature(decisions):
    return [
        (
            decision.action,
            decision.action_node,
            decision.from_state,
            decision.to_state,
        )
        for decision in decisions
    ]


def _direct_result(router, pick_list):
    _, aisle_info = router._prepare_aisles(pick_list)
    distance, decisions = router._solve_direct_dp(aisle_info, backtrack=True)
    path = [(1, ("0", "0", "0C"), "-")]
    for decision in decisions:
        if decision.kind == "aisle":
            path.append((decision.aisle, decision.to_state, "+"))
        else:
            path.append((decision.aisle + 1, decision.to_state, "-"))
    return distance, decisions, path


@pytest.mark.parametrize(
    ("nodes", "depot", "expected_distance", "expected_actions"),
    [
        (
            [(1, 2), (1, 8), (4, 4)],
            0,
            34,
            ["one_pass", (1, 1), "void", (1, 1), "void", (1, 1), "one_pass", (0, 0)],
        ),
        (
            [(3, 2), (3, 6), (3, 9)],
            3,
            20,
            ["void", (0, 0), "void", (0, 0), "bottom", (0, 0), "void", (0, 0)],
        ),
        (
            [(1, 2), (1, 9), (2, 5), (3, 2), (3, 8), (4, 6)],
            2,
            56,
            ["one_pass", (1, 1), "bottom", (1, 1), "gap", (1, 1), "one_pass", (0, 0)],
        ),
        (
            [(1, 7), (1, 8), (2, 5), (3, 5), (3, 8), (4, 5)],
            4,
            54,
            ["top", (2, 0), "one_pass", (1, 1), "one_pass", (0, 2), "bottom", (0, 0)],
        ),
    ],
)
def test_direct_dp_preserves_captured_distance_actions_and_ties(
    nodes,
    depot,
    expected_distance,
    expected_actions,
):
    pick_list = _picks(nodes)
    kwargs = _kwargs(depot=depot)
    router = RatliffRosenthalRouting(**kwargs)
    solution = router.solve(pick_list)
    direct_distance, decisions, path = _direct_result(router, pick_list)
    actions = [
        decision.action
        if isinstance(decision.action, tuple)
        else aisle_mapping[decision.action]
        for decision in decisions
    ]

    assert solution.route.distance == direct_distance == expected_distance
    assert actions == expected_actions
    repeated_distance, repeated_decisions, repeated_path = _direct_result(
        router, pick_list
    )
    assert repeated_distance == direct_distance
    assert repeated_path == path
    assert _decision_signature(repeated_decisions) == _decision_signature(decisions)
    assert not hasattr(router, "decisions")
    assert not hasattr(router, "path")


@pytest.mark.parametrize(
    ("gen_tour", "gen_item_sequence"),
    [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ],
)
def test_output_reconstruction_is_skipped_only_for_distance_scoring(
    gen_tour,
    gen_item_sequence,
):
    pick_list = _picks([(1, 2), (1, 8), (4, 4)])
    router = RatliffRosenthalRouting(**_kwargs(
        gen_tour=gen_tour,
        gen_item_sequence=gen_item_sequence,
    ))
    route = router.solve(pick_list).route

    if gen_item_sequence:
        expected = (
            [(4, 4), (1, 8), (1, 2)]
            if gen_tour
            else [(1, 2), (1, 8), (4, 4)]
        )
        assert route.item_sequence == expected
    else:
        assert route.item_sequence == []

    if gen_tour:
        assert route.route == [
            (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (4, 4),
            (4, 11), (3, 11), (2, 11), (1, 11), (1, 8), (1, 2),
            (1, 0), (0, 0),
        ]
        assert router._rr_route_distance(route.route) == pytest.approx(
            route.distance
        )
    else:
        assert route.route == []
        if gen_item_sequence:
            assert [node.position for node in route.annotated_route] == route.item_sequence
            assert all(node.node_type.name == "PICK" for node in route.annotated_route)
        else:
            assert route.annotated_route == []
    assert not hasattr(router, "T")
    assert not hasattr(router, "state_graph")


def test_two_pass_transition_is_propagated_directly():
    router = RatliffRosenthalRouting(**_kwargs())
    _, aisle_info = router._prepare_aisles(_picks([(2, 5)]))

    costs, predecessors = router._propagate_aisle_layer(
        2,
        {("E", "0", "1C"): 0.0},
        aisle_info,
    )

    assert ("E", "E", "1C") in costs
    assert predecessors[("E", "E", "1C")][1] == 5


@pytest.mark.parametrize(
    ("state", "expected_actions"),
    [
        (("U", "U", "1C"), {(1, 1)}),
        (("E", "0", "1C"), {(2, 0), (2, 2), (0, 0)}),
        (("0", "E", "1C"), {(0, 2), (2, 2), (0, 0)}),
        (("E", "E", "1C"), {(2, 0), (0, 2), (2, 2), (0, 0)}),
    ],
)
def test_cross_aisle_alternatives_are_propagated(state, expected_actions):
    router = RatliffRosenthalRouting(**_kwargs())
    _, predecessors = router._propagate_cross_aisle_layer(1, {state: 0.0})
    assert {predecessor[1] for predecessor in predecessors.values()} == expected_actions


def test_depot_aisle_cross_connectivity_is_preserved():
    router = RatliffRosenthalRouting(**_kwargs(depot=2))
    _, predecessors = router._propagate_cross_aisle_layer(
        2,
        {("E", "0", "1C"): 0.0},
    )
    assert {predecessor[1] for predecessor in predecessors.values()} == {(2, 2)}


def test_cross_aisle_edge_multiplicity_is_preserved():
    router = RatliffRosenthalRouting(**_kwargs())
    state = ("E", "E", "1C")
    graph = router._construct_picker_tour((
        _RRDecision(1, "cross", state, state, (2, 2), None, 8.0),
    ))

    assert graph.number_of_edges((1, 0), (2, 0)) == 2
    assert graph.number_of_edges((1, 11), (2, 11)) == 2


def test_non_unit_pick_spacing_matches_exact_tsp():
    n_aisles = 4
    n_pick_locations = 10
    dist_aisle = 2.0
    dist_pick_locations = 2.0
    dist_aisle_location = 1.0
    start = (0, 0)
    end = (5, 0)
    graph = nx.Graph()
    for aisle in range(1, n_aisles + 1):
        for y in range(n_pick_locations + 1):
            source, target = (aisle, y), (aisle, y + 1)
            weight = (
                dist_aisle_location
                if y in (0, n_pick_locations)
                else dist_pick_locations
            )
            graph.add_edge(source, target, weight=weight)
    for aisle in range(1, n_aisles):
        graph.add_edge((aisle, 0), (aisle + 1, 0), weight=dist_aisle)
        graph.add_edge(
            (aisle, n_pick_locations + 1),
            (aisle + 1, n_pick_locations + 1),
            weight=dist_aisle,
        )
    graph.add_edge(start, (1, 0), weight=0.0)
    graph.add_edge(end, (1, 0), weight=0.0)
    nodes = list(graph.nodes)
    distance_matrix = pd.DataFrame(
        [[nx.shortest_path_length(graph, a, b, weight="weight") for b in nodes]
         for a in nodes],
        index=nodes,
        columns=nodes,
    )
    picks = _picks([(1, 2), (1, 8), (3, 3), (3, 9), (4, 6)])
    common = {
        "start_node": start,
        "end_node": end,
        "closest_node_to_start": (1, 0),
        "min_aisle_position": 1,
        "max_aisle_position": 10,
        "picker": [Resource(id=1)],
    }
    rr = RatliffRosenthalRouting(
        **common,
        n_aisles=n_aisles,
        n_pick_locations=n_pick_locations,
        dist_aisle=dist_aisle,
        dist_pick_locations=dist_pick_locations,
        dist_aisle_location=dist_aisle_location,
        dist_start=0.0,
        dist_end=0.0,
    )
    tsp = ExactTSPRoutingDistance(
        **common,
        distance_matrix=distance_matrix,
        predecessor_matrix=None,
        gen_tour=False,
        gen_item_sequence=False,
        big_m=1000,
        set_time_limit=30,
    )

    assert rr.score(picks) == pytest.approx(tsp.score(picks))
