"""Exact small-instance makespan optimization for the Lorenz OBSRP-R model."""

from __future__ import annotations

import math
import time
from collections.abc import Iterable

from ware_ops_algos.algorithms.algorithm_interfaces import (
    BatchObject,
    Job,
    NodeType,
    PickPosition,
    Route,
    RouteNode,
    ScheduledJob,
    SchedulingSolution,
    WarehouseOrder,
)
from ware_ops_algos.domain_models import LayoutData, Resource


def _items(
    orders: list[WarehouseOrder],
) -> tuple[list[tuple[int, PickPosition]], list[int]]:
    items: list[tuple[int, PickPosition]] = []
    masks: list[int] = []
    for order_index, order in enumerate(orders):
        mask = 0
        for position in order.pick_positions:
            for _ in range(int(position.picked_quantity)):
                mask |= 1 << len(items)
                items.append((order_index, position))
        if not mask:
            raise ValueError(f"Order {order.order_id} contains no pickable items")
        masks.append(mask)
    return items, masks


def _distance(layout: LayoutData, source, target) -> float:
    matrix = layout.layout_network.distance_matrix
    try:
        return float(matrix.at[source, target])
    except AttributeError:
        nodes = layout.layout_network.node_list
        return float(matrix[nodes.index(source), nodes.index(target)])


def _shortest_path(
    layout: LayoutData,
    source,
    target,
) -> list[tuple[int, int]]:
    if source == target:
        return [source]
    network = layout.layout_network
    predecessors = network.predecessor_matrix
    if isinstance(predecessors, dict):
        path = [target]
        current = target
        while current != source:
            current = predecessors[source][current]
            path.append(current)
        return list(reversed(path))

    source_index = network.node_list.index(source)
    current_index = network.node_list.index(target)
    path_indices = [current_index]
    while current_index != source_index:
        current_index = int(predecessors[source_index, current_index])
        if current_index < 0:
            raise ValueError(f"No path from {source} to {target}")
        path_indices.append(current_index)
    return [network.node_list[index] for index in reversed(path_indices)]


def _make_route(
    layout: LayoutData,
    batch: BatchObject,
    positions: list[PickPosition],
) -> Route:
    network = layout.layout_network
    annotated = [RouteNode(network.start_node, NodeType.ROUTE)]
    previous = network.start_node
    distance = 0.0
    for position in positions:
        target = position.pick_node
        path = _shortest_path(layout, previous, target)
        distance += _distance(layout, previous, target)
        if previous == target:
            annotated.append(RouteNode(target, NodeType.PICK))
        else:
            annotated.extend(RouteNode(node, NodeType.ROUTE) for node in path[1:])
            annotated[-1] = RouteNode(target, NodeType.PICK)
        previous = target

    path = _shortest_path(layout, previous, network.end_node)
    distance += _distance(layout, previous, network.end_node)
    annotated.extend(RouteNode(node, NodeType.ROUTE) for node in path[1:])
    return Route(
        distance=distance,
        route=[node.position for node in annotated],
        item_sequence=[position.pick_node for position in positions],
        batch=batch,
        annotated_route=annotated,
    )


def exact_lorenz_makespan(
    orders: Iterable[WarehouseOrder],
    layout: LayoutData,
    picker: Resource,
    batch_capacity_orders: int,
    *,
    start_time: float = 0.0,
    max_states: int = 2_000_000,
    max_runtime_s: float = 60.0,
) -> SchedulingSolution:
    """Solve the supplied OBSRP-R instance by forward Bellman recursion.

    The state is ``(last item, open-batch order count, unpicked open-batch
    items, pending orders)``. A label stores the earliest attainable time.
    Closing a batch includes its return to the depot.
    """
    started = time.perf_counter()
    ordered = sorted(list(orders), key=lambda order: int(order.order_id))
    if not ordered:
        return SchedulingSolution(
            algo_name="LorenzMakespanDP",
            jobs=[],
            solver_status="optimal",
            objective_value=float(start_time),
            is_optimal=True,
            explored_states=1,
        )
    if batch_capacity_orders < 1:
        raise ValueError("batch_capacity_orders must be positive")
    if picker.speed is None or picker.speed <= 0:
        raise ValueError("A positive picker speed is required")
    if picker.time_per_pick is None or picker.time_per_pick < 0:
        raise ValueError("A non-negative item-picking time is required")
    if picker.tour_setup_time not in (None, 0, 0.0):
        raise ValueError("The Lorenz exact DP requires zero tour setup time")
    if max_states < 1:
        raise ValueError("max_states must be positive")
    if max_runtime_s <= 0:
        raise ValueError("max_runtime_s must be positive")

    items, order_item_masks = _items(ordered)
    order_ids = [int(order.order_id) for order in ordered]
    releases = [float(order.order_date or 0.0) for order in ordered]
    all_pending = (1 << len(ordered)) - 1
    initial = (-1, 0, 0, all_pending)
    labels = {initial: float(start_time)}
    keys = {initial: ()}
    predecessors: dict[
        tuple[int, int, int, int],
        tuple[tuple[int, int, int, int], tuple[int, bool]],
    ] = {}
    explored = 1
    eps = 1e-9

    for _stage in range(len(items)):
        next_labels: dict[tuple[int, int, int, int], float] = {}
        next_keys: dict[tuple[int, int, int, int], tuple] = {}
        for state in sorted(labels):
            if time.perf_counter() - started > max_runtime_s:
                raise RuntimeError(
                    f"EXACT_LIMIT: max_runtime_s={max_runtime_s} exceeded; "
                    "no unproven incumbent was returned"
                )
            last_item, open_count, remaining, pending = state
            clock = labels[state]
            current = (
                layout.layout_network.start_node
                if open_count == 0
                else items[last_item][1].pick_node
            )
            choices = remaining
            for order_index in range(len(ordered)):
                if (
                    pending & (1 << order_index)
                    and open_count < batch_capacity_orders
                ):
                    choices |= order_item_masks[order_index]

            for item_index in range(len(items)):
                bit = 1 << item_index
                if not choices & bit:
                    continue
                order_index, position = items[item_index]
                is_new_order = bool(pending & (1 << order_index))
                if not is_new_order and not (remaining & bit):
                    continue

                new_count = open_count + int(is_new_order)
                new_pending = pending & ~(1 << order_index)
                new_remaining = remaining
                if is_new_order:
                    new_remaining |= order_item_masks[order_index]
                new_remaining &= ~bit
                arrival = (
                    clock
                    + _distance(layout, current, position.pick_node)
                    / float(picker.speed)
                )
                pick_end = (
                    max(arrival, releases[order_index])
                    + float(picker.time_per_pick)
                )

                can_keep_open = (
                    new_count < batch_capacity_orders
                    and (new_pending != 0 or new_remaining != 0)
                )
                closures = (
                    (False, True)
                    if not new_remaining and can_keep_open
                    else (not new_remaining,)
                )
                for close in closures:
                    if close:
                        value = (
                            pick_end
                            + _distance(
                                layout,
                                position.pick_node,
                                layout.layout_network.end_node,
                            )
                            / float(picker.speed)
                        )
                        new_state = (-1, 0, 0, new_pending)
                    else:
                        value = pick_end
                        new_state = (
                            item_index,
                            new_count,
                            new_remaining,
                            new_pending,
                        )
                    action_key = (
                        order_ids[order_index],
                        int(position.article_id),
                        item_index,
                        int(close),
                    )
                    path_key = keys[state] + (action_key,)
                    old = next_labels.get(new_state)
                    if old is None or value < old - eps or (
                        abs(value - old) <= eps
                        and path_key < next_keys[new_state]
                    ):
                        next_labels[new_state] = value
                        next_keys[new_state] = path_key
                        predecessors[new_state] = (
                            state,
                            (item_index, close),
                        )
        labels = next_labels
        keys = next_keys
        explored += len(labels)
        if explored > max_states:
            raise RuntimeError(
                f"EXACT_LIMIT: max_states={max_states} exceeded; "
                "no unproven incumbent was returned"
            )

    final = (-1, 0, 0, 0)
    if final not in labels:
        raise RuntimeError("The exact DP found no complete depot-to-depot plan")

    actions: list[tuple[int, bool]] = []
    state = final
    while state != initial:
        previous, action = predecessors[state]
        actions.append(action)
        state = previous
    actions.reverse()

    batches: list[list[int]] = []
    current_batch: list[int] = []
    for item_index, close in actions:
        current_batch.append(item_index)
        if close:
            batches.append(current_batch)
            current_batch = []
    if current_batch:
        raise RuntimeError("Reconstructed DP plan does not close its final batch")

    jobs: list[ScheduledJob] = []
    clock = float(start_time)
    for batch_index, item_indices in enumerate(batches):
        batch_order_indices = sorted(
            {items[index][0] for index in item_indices}
        )
        batch = BatchObject(
            batch_id=batch_index,
            orders=[ordered[index] for index in batch_order_indices],
        )
        positions = [items[index][1] for index in item_indices]
        route = _make_route(layout, batch, positions)
        batch_start = clock
        previous = layout.layout_network.start_node
        picking_times = []
        for item_index in item_indices:
            order_index, position = items[item_index]
            clock += (
                _distance(layout, previous, position.pick_node)
                / float(picker.speed)
            )
            clock = (
                max(clock, releases[order_index])
                + float(picker.time_per_pick)
            )
            picking_times.append(clock)
            previous = position.pick_node
        clock += (
            _distance(
                layout,
                previous,
                layout.layout_network.end_node,
            )
            / float(picker.speed)
        )
        route.service_time = clock - batch_start
        route.picking_times = picking_times
        job = Job(
            job_id=batch_index,
            processing_time=clock - batch_start,
            release_time=batch_start,
            due_date=math.inf,
            n_picks=len(item_indices),
            route=route,
            batch=batch,
        )
        jobs.append(
            ScheduledJob(
                job=job,
                picker_id=picker.id,
                start_time=batch_start,
                end_time=clock,
            )
        )

    runtime = time.perf_counter() - started
    return SchedulingSolution(
        algo_name="LorenzMakespanDP",
        execution_time=runtime,
        jobs=jobs,
        solver_status="optimal",
        objective_value=float(labels[final]),
        is_optimal=True,
        explored_states=explored,
    )
