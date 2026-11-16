"""GivenSequenceRouting visits positions in the given order over shortest paths."""

import numpy as np
import pandas as pd
import pytest
from scipy.sparse.csgraph import floyd_warshall

from ware_ops_algos.algorithms import GivenSequenceRouting, NodeType, PickPosition
from ware_ops_algos.algorithms.algorithm_cards import load_packaged_algo_cards
from ware_ops_algos.domain_models import (
    InformationType, PlannerInformation, RecordedLine, RecordedPickingPlan, RecordedTour,
)

# A line of three nodes, 1 m apart; the depot is at (0, 0).
NODES = [(0, 0), (1, 0), (2, 0)]


def _router(sequence):
    weights = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    distances, predecessors = floyd_warshall(weights, return_predecessors=True)
    index = {node: i for i, node in enumerate(NODES)}
    return GivenSequenceRouting(
        start_node=(0, 0), end_node=(0, 0), closest_node_to_start=(0, 0),
        min_aisle_position=0, max_aisle_position=0, picker=[],
        gen_tour=True, gen_item_sequence=True, node_list=NODES,
        node_to_idx=index, idx_to_node=dict(enumerate(NODES)),
        distance_matrix=pd.DataFrame(distances, index=NODES, columns=NODES),
        predecessor_matrix=predecessors, sequence=sequence)


def _position(order, article, node):
    return PickPosition(order_number=order, article_id=article, amount=1, pick_node=node, in_store=1)


def test_positions_are_visited_in_the_given_sequence_not_by_order():
    a1, b1, a2 = _position(1, 10, (2, 0)), _position(2, 20, (1, 0)), _position(1, 11, (2, 0))
    sequence = [(1, 10, (2, 0)), (2, 20, (1, 0)), (1, 11, (2, 0))]
    route = _router(sequence).solve([a1, a2, b1]).route
    # depot -> (2,0) pick -> (1,0) pick -> (2,0) pick -> depot: 2 + 1 + 1 + 2
    assert route.distance == 6
    assert route.item_sequence == [(2, 0), (1, 0), (2, 0)]
    picks = [node.position for node in route.annotated_route if node.node_type == NodeType.PICK]
    assert picks == [(2, 0), (1, 0), (2, 0)]
    assert route.annotated_route[0].position == (0, 0)
    assert route.annotated_route[-1].position == (0, 0)


def test_two_lines_at_one_node_in_a_row_are_two_picks_without_walking():
    first, second = _position(1, 10, (1, 0)), _position(1, 11, (1, 0))
    route = _router([(1, 11, (1, 0)), (1, 10, (1, 0))]).solve([first, second]).route
    assert route.distance == 2
    assert [n.node_type for n in route.annotated_route].count(NodeType.PICK) == 2


def test_a_sequence_that_does_not_match_the_positions_is_rejected():
    # Algorithm.solve wraps the router's ValueError
    with pytest.raises(RuntimeError, match="missing from the sequence"):
        _router([(1, 10, (1, 0))]).solve([_position(1, 10, (1, 0)), _position(1, 11, (2, 0))])
    with pytest.raises(RuntimeError, match="not among the pick positions"):
        _router([(1, 12, (1, 0))]).solve([_position(1, 10, (1, 0))])


def test_the_card_requires_a_recorded_plan():
    card = next(c for c in load_packaged_algo_cards() if c.algo_name == "GivenSequence")
    assert card.requirements["information"]["constraints"] == {
        "recorded_plan.type": {"equals": "recorded_picking_plan"}}


def test_a_recorded_plan_exposes_its_type_and_size_as_features():
    plan = RecordedPickingPlan(tours=(
        RecordedTour(tour_id=0, picker_id=3, lines=(RecordedLine(1, 10, (2, 0)),
                                                     RecordedLine(2, 20, (1, 0)),
                                                     RecordedLine(1, 11, (2, 0)))),))
    assert plan.tours[0].order_ids == (1, 2)
    features = PlannerInformation(InformationType.PROCESS_INFORMATION,
                                  {"recorded_plan": plan}).get_features()
    assert features == {"recorded_plan.type": "recorded_picking_plan", "recorded_plan.tours": 1}
    with pytest.raises(ValueError, match="one recorded tour only"):
        RecordedPickingPlan(tours=(RecordedTour(0, 1, (RecordedLine(1, 10, (1, 0)),)),
                                   RecordedTour(1, 1, (RecordedLine(1, 11, (1, 0)),))))
