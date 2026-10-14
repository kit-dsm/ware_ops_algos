"""Shared empty-cart positioning mechanics."""

from __future__ import annotations

from abc import ABC, abstractmethod

import networkx as nx

from ware_ops_algos.domain_models import LayoutData

from ..algorithm_interfaces import (
    Algorithm, BatchObject, Job, NodeType, PositioningInput, PositioningSolution, Route, RouteNode, ScheduledJob,
)


from ..routing.patrol import all_aisles_patrol_route


class Positioning(Algorithm[PositioningInput, PositioningSolution], ABC):
    """Add ordinary empty-cart jobs to the proposed schedule."""

    @abstractmethod
    def _route(self, layout: LayoutData) -> Route:
        """Choose the route of an empty-cart tour."""

    def _run(self, input_data: PositioningInput) -> PositioningSolution:
        jobs = list(input_data.schedule.jobs)
        if input_data.input_closed:
            return PositioningSolution(jobs=jobs)
        assigned = {job.picker_id for job in jobs} | set(input_data.occupied_picker_ids)
        for picker in input_data.resources:
            if picker.id in assigned or picker.speed <= 0:
                continue
            route = self._route(input_data.layout)
            duration = route.distance / picker.speed
            job = Job(job_id=-picker.id - 1, processing_time=duration,
                      release_time=input_data.current_time,
                      due_date=float("inf"), n_picks=0,
                      route=route, batch=route.batch)
            jobs.append(ScheduledJob(job=job, picker_id=picker.id,
                                     start_time=input_data.current_time,
                                     end_time=input_data.current_time + duration))
        return PositioningSolution(jobs=jobs)


def center_relocation_route(layout: LayoutData) -> Route:
    """Move to an accessible central graph node and hold there."""
    network = layout.layout_network
    graph = network.graph
    pick_nodes = [node for node in graph if 1 <= node[1] <= layout.graph_data.n_pick_locations]
    if not pick_nodes:
        raise ValueError("Center relocation needs accessible pick-location nodes")
    midpoint = (sum(node[0] for node in pick_nodes) / len(pick_nodes),
                sum(node[1] for node in pick_nodes) / len(pick_nodes))
    center = min(pick_nodes, key=lambda node:
                 ((node[0] - midpoint[0]) ** 2 + (node[1] - midpoint[1]) ** 2,
                  node))
    nodes = nx.shortest_path(graph, network.start_node, center, weight="weight")
    distance = nx.path_weight(graph, nodes, weight="weight")
    return Route(distance=float(distance), route=nodes, item_sequence=[],
                 annotated_route=[RouteNode(node, NodeType.ROUTE) for node in nodes],
                 batch=BatchObject(batch_id=0, orders=[]))


class AislePatrolPositioning(Positioning):
    algo_name = "AislePatrolPositioning"

    def _route(self, layout: LayoutData) -> Route:
        return all_aisles_patrol_route(layout)


class CenterRelocationPositioning(Positioning):
    algo_name = "CenterRelocationPositioning"

    def _route(self, layout: LayoutData) -> Route:
        return center_relocation_route(layout)
