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


def _plan():
    line = lambda order, article: RecordedLine(order, article, (1, 0))  # noqa: E731
    return RecordedPickingPlan(tours=(
        RecordedTour(tour_id=7, picker_id=1, lines=(line(2, 20), line(1, 10), line(2, 21))),
        RecordedTour(tour_id=3, picker_id=0, lines=(line(3, 30),)),
        RecordedTour(tour_id=5, picker_id=1, lines=(line(4, 40),)),
    ))


def test_recorded_batching_makes_one_batch_per_recorded_tour():
    from ware_ops_algos.algorithms import RecordedBatching, WarehouseOrder
    from ware_ops_algos.domain_models import ArticleType, Articles, DimensionType, PickCart
    cart = PickCart(n_dimension=1, capacities=[1], dimensions=[DimensionType.ORDERS], n_boxes=1)
    orders = [WarehouseOrder(order_id=i) for i in (1, 2, 3, 4)]
    batches = RecordedBatching(pick_cart=cart, articles=Articles(ArticleType.STANDARD, []), plan=_plan()).solve(orders).batches
    assert [(b.batch_id, [o.order_id for o in b.orders]) for b in batches] == [
        (7, [2, 1]), (3, [3]), (5, [4])]
    with pytest.raises(RuntimeError, match="not in the recorded plan"):
        RecordedBatching(pick_cart=cart, articles=Articles(ArticleType.STANDARD, []), plan=_plan()).solve(
            [*orders, WarehouseOrder(order_id=9)])


def test_recorded_scheduling_chains_each_pickers_tours_in_recorded_order():
    from ware_ops_algos.algorithms import BatchObject, Job, RecordedScheduling, Route
    from ware_ops_algos.domain_models import Resource, Resources, ResourceType
    pickers = Resources(ResourceType.HUMAN, [
        Resource(id=0, speed=1.0, time_per_pick=2.0, tour_setup_time=1.0),
        Resource(id=1, speed=1.0, time_per_pick=2.0, tour_setup_time=1.0)])

    def job(batch_id, distance, picks):
        route = Route(route=[], item_sequence=[(1, 0)] * picks, distance=distance)
        route.batch = BatchObject(batch_id=batch_id, orders=[])
        return Job(job_id=batch_id, processing_time=0.0, release_time=10.0,
                   due_date=0.0, n_picks=picks, route=route)

    jobs = [job(5, 4.0, 1), job(3, 6.0, 2), job(7, 10.0, 3)]
    scheduled = RecordedScheduling(pickers, plan=_plan(), available_from={1: 12.0}).solve(jobs).jobs
    # picker 1: tour 7 from max(10, 12) = 12, 1 setup + 10 walk + 3 x 2 picks -> 29; tour 5 at 29 -> 29 + 1 + 4 + 2 = 36
    # picker 0: tour 3 from 10 -> 10 + 1 + 6 + 4 = 21
    assert [(j.job.route.batch.batch_id, j.picker_id, j.start_time, j.end_time) for j in scheduled] == [
        (7, 1, 12.0, 29.0), (3, 0, 10.0, 21.0), (5, 1, 29.0, 36.0)]


def test_recorded_scheduling_starts_an_anchored_tour_at_its_recorded_start():
    from dataclasses import replace
    from ware_ops_algos.algorithms import BatchObject, Job, RecordedScheduling, Route
    from ware_ops_algos.domain_models import Resource, Resources, ResourceType
    pickers = Resources(ResourceType.HUMAN, [
        Resource(id=0, speed=1.0, time_per_pick=2.0, tour_setup_time=1.0),
        Resource(id=1, speed=1.0, time_per_pick=2.0, tour_setup_time=1.0)])
    plan = _plan()
    plan = RecordedPickingPlan(tours=(plan.tours[0], plan.tours[1],
                                      replace(plan.tours[2], start_s=50.0)))
    route = Route(route=[], item_sequence=[(1, 0)], distance=4.0)
    route.batch = BatchObject(batch_id=5, orders=[])
    job = Job(job_id=5, processing_time=0.0, release_time=10.0, due_date=0.0, n_picks=1, route=route)
    scheduled = RecordedScheduling(pickers, plan=plan).solve([job]).jobs
    # without tour 7, picker 1 is free at 10, but tour 5 is anchored at 50: 50 + 1 + 4 + 2
    assert [(j.start_time, j.end_time) for j in scheduled] == [(50.0, 57.0)]


def test_a_line_pick_time_model_enters_the_planned_processing_time():
    from ware_ops_algos.algorithms import BatchObject, PickPosition, Route, WarehouseOrder
    from ware_ops_algos.algorithms.scheduling.scheduling import _processing_time
    from ware_ops_algos.domain_models import LinePickTime, Resource, pick_time
    constant = Resource(id=0, speed=2.0, time_per_pick=19.0)
    model = Resource(id=0, speed=2.0, time_per_pick=19.0,
                     pick_time_model=LinePickTime(base_s=10.0, per_unit_s=0.5))
    positions = (PickPosition(1, 10, 6, (1, 0), 1), PickPosition(1, 11, 24, (2, 0), 1))
    route = Route(route=[], item_sequence=[(1, 0), (2, 0)], distance=8.0)
    route.batch = BatchObject(batch_id=0, orders=[WarehouseOrder(order_id=1, pick_positions=positions)])
    assert pick_time(constant, 6) == 19.0 and pick_time(model, 6) == 13.0
    # walk 8 / 2 = 4; constant 2 x 19; model (10 + 3) + (10 + 12)
    assert _processing_time(route, constant) == 42.0
    assert _processing_time(route, model) == 39.0
