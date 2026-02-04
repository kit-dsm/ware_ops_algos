from collections import defaultdict
from typing import Type

import pandas as pd

from ware_ops_algos.algorithms import Algorithm, WarehouseOrder, OrderSelectionSolution, RouteNode, TourPlanningState, \
    TourStates, Routing
from ware_ops_algos.domain_models import Resource, ResourceType


class OrderSelection(Algorithm[list[WarehouseOrder], OrderSelectionSolution]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _run(self, input_data: list[WarehouseOrder]) -> OrderSelectionSolution:
        pass


class DummyOrderSelection(OrderSelection):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _run(self, input_data: list[WarehouseOrder]) -> OrderSelectionSolution:
        selected_orders = input_data
        solution = OrderSelectionSolution(selected_orders=selected_orders)
        return solution


class GreedyOrderSelection(OrderSelection):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _run(self, input_data: list[WarehouseOrder]) -> OrderSelectionSolution:
        selected_orders = [input_data[0]]
        solution = OrderSelectionSolution(selected_orders=selected_orders)
        return solution


class MinDistOrderSelection(OrderSelection):
    def __init__(self,
                 picker_position: tuple[float, float],
                 dima: pd.DataFrame,
                 **kwargs):
        super().__init__(**kwargs)
        self.picker_position = picker_position.position if isinstance(picker_position, RouteNode) else picker_position
        self.dima = dima

    def _select_order(self, pending_orders: list[WarehouseOrder]) -> WarehouseOrder:
        return min(
            pending_orders,
            key=lambda o: min(
                self.dima.at[self.picker_position, pos.pick_node]
                for pos in o.pick_positions
            ),
        )

    def _run(self, input_data: list[WarehouseOrder]) -> OrderSelectionSolution:
        selected_order = self._select_order(input_data)
        solution = OrderSelectionSolution(selected_orders=[selected_order])
        return solution


class MinAisleOrderSelection(OrderSelection):
    def __init__(self, congestion_info: dict, **kwargs):
        super().__init__(**kwargs)
        self.congestion_info = congestion_info

    @staticmethod
    def get_order_aisles(order: WarehouseOrder) -> set:
        return {pos.pick_node[0] for pos in order.pick_positions}

    def _select_order(self, pending_orders: list[WarehouseOrder], congestion: dict) -> WarehouseOrder:
        return min(
            pending_orders,
            key=lambda o: sum(congestion.get(a, 0) for a in self.get_order_aisles(o))
        )

    def _run(self, input_data: list[WarehouseOrder]) -> OrderSelectionSolution:
        selected_order = self._select_order(input_data, self.congestion_info)
        solution = OrderSelectionSolution(selected_orders=[selected_order])
        return solution


class MinMaxArticlesCobotSelection(OrderSelection):
    def __init__(self, resource: Resource, **kwargs):
        super().__init__(**kwargs)
        self.resource = resource

    @staticmethod
    def get_order_aisles(order: WarehouseOrder) -> set:
        return {pos.pick_node[0] for pos in order.pick_positions}

    @staticmethod
    def _select_order_max(pending_orders: list[WarehouseOrder]) -> WarehouseOrder:
        return max(pending_orders, key=lambda o: len(o.pick_positions))

    @staticmethod
    def _select_order_min(pending_orders: list[WarehouseOrder]) -> WarehouseOrder:
        return min(pending_orders, key=lambda o: len(o.pick_positions))

    def _run(self, input_data: list[WarehouseOrder]) -> OrderSelectionSolution:
        if self.resource.tpe == ResourceType.COBOT:
            selected_order = self._select_order_min(input_data)
        elif self.resource.tpe == ResourceType.HUMAN:
            selected_order = self._select_order_max(input_data)
        else:
            raise ValueError(f"Unknown resource type: {self.resource.tpe}")
        solution = OrderSelectionSolution(selected_orders=[selected_order])
        return solution


class MinMaxAisleOrderSelection(OrderSelection):
    def __init__(self, congestion_info: dict, resource: Resource, **kwargs):
        super().__init__(**kwargs)
        self.congestion_info = congestion_info
        self.resource = resource

    @staticmethod
    def get_order_aisles(order: WarehouseOrder) -> set:
        return {pos.pick_node[0] for pos in order.pick_positions}

    def _select_order_min(self, pending_orders: list[WarehouseOrder], congestion: dict) -> WarehouseOrder:
        return min(
            pending_orders,
            key=lambda o: sum(congestion.get(a, 0) for a in self.get_order_aisles(o))
        )

    def _select_order_max(self, pending_orders: list[WarehouseOrder], congestion: dict) -> WarehouseOrder:
        return max(
            pending_orders,
            key=lambda o: sum(congestion.get(a, 0) for a in self.get_order_aisles(o))
        )

    def _run(self, input_data: list[WarehouseOrder]) -> OrderSelectionSolution:
        if self.resource.tpe == ResourceType.HUMAN:
            selected_order = self._select_order_min(input_data, self.congestion_info)
        elif self.resource.tpe == ResourceType.COBOT:
            selected_order = self._select_order_max(input_data, self.congestion_info)
        else:
            raise ValueError(f"Unknown resource type: {self.resource.tpe}")
        solution = OrderSelectionSolution(selected_orders=[selected_order])
        return solution


class TimeIndexedMinConflictSelection(OrderSelection):
    """Select order that minimizes predicted time-indexed aisle conflicts"""

    def __init__(self,
                 active_tours: list[TourPlanningState],  # Tours from state transformer
                 resource: Resource,
                 resources: list[Resource],
                 picker_position: RouteNode,
                 distance_matrix: pd.DataFrame,
                 routing_class: Type[Routing],
                 routing_class_kwargs: dict,
                 current_time: float = 0.0,
                 slot_duration: float = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.active_tours = active_tours
        self.resource = resource
        self.resources = resources
        self.picker_position = picker_position.position if isinstance(picker_position, RouteNode) else picker_position
        self.distance_matrix = distance_matrix
        self.routing_class = routing_class
        self.routing_class_kwargs = routing_class_kwargs
        self.current_time = current_time
        self.slot_duration = slot_duration

    def _build_occupancy_tracker(self) -> dict[int, dict[float, int]]:
        """Build time-slotted occupancy from active tours: {slot_index: {aisle: count}}"""
        occupancy = defaultdict(lambda: defaultdict(int))

        print(f"\n=== Building Occupancy Tracker ===")
        print(f"Current time: {self.current_time}")
        print(f"Selecting for resource: {self.resource.id} (type: {self.resource.tpe})")
        print(f"Total active tours: {len(self.active_tours)}")

        tours_processed = 0
        tours_skipped = 0

        for tour in self.active_tours:
            print(f"\nProcessing tour {tour.tour_id}:")
            print(f"  Status: {tour.status}")
            # Skip tours that are done
            if tour.status == TourStates.DONE:
                print(f"  -> Skipped (DONE)")
                tours_skipped += 1
                continue

            # Get tour resource
            tour_resource_id = tour.assigned_resource
            tour_resource = self.resources[tour_resource_id]
            print(f"  Assigned resource: {tour_resource}")

            # Exclude self for cobots, include all for humans
            if self.resource.tpe == ResourceType.COBOT and tour_resource.id == self.resource.id:
                print(f"  -> Skipped (self-exclusion for cobot)")
                tours_skipped += 1
                continue

            if tour_resource.tpe != ResourceType.COBOT:
                print(f"  -> Skipped (not a cobot: {tour_resource.tpe})")
                tours_skipped += 1
                continue

            print(f"  -> Processing tour segments...")
            # Extract segments from tour's remaining route
            segments = self._extract_tour_segments(tour)
            print(f"  Extracted {len(segments)} segments")

            for aisle, start_time, end_time in segments:
                start_slot = int(start_time / self.slot_duration)
                end_slot = int(end_time / self.slot_duration)

                for slot in range(start_slot, end_slot):
                    occupancy[slot][aisle] += 1

            tours_processed += 1

        print(f"\n=== Occupancy Summary ===")
        print(f"Tours processed: {tours_processed}, Tours skipped: {tours_skipped}")
        print(f"Time slots with occupancy: {len(occupancy)}")

        # Print sample of occupancy
        if occupancy:
            sample_slots = sorted(occupancy.keys())[:5]
            print(f"Sample occupancy (first 5 slots):")
            for slot in sample_slots:
                print(f"  Slot {slot} (t={slot * self.slot_duration}): {dict(occupancy[slot])}")
        else:
            print("WARNING: No occupancy tracked!")

        return occupancy

    def _extract_tour_segments(self, tour: TourPlanningState) -> list[tuple]:
        """Extract (aisle, start_time, end_time) segments from a tour"""
        segments = []

        # Get route and cursor position
        route = tour.annotated_route
        cursor = tour.cursor

        print(f"    Route length: {len(route)}, Cursor at: {cursor}")

        # Get resource info (speed, pick_time)
        tour_picker = self.resources[tour.assigned_resource]
        resource_speed = tour_picker.speed
        resource_pick_time = tour_picker.time_per_pick
        print(f"    Resource speed: {resource_speed}, pick_time: {resource_pick_time}")
        current_time = self.current_time

        # Process remaining route from cursor onwards
        for i in range(cursor, len(route) - 1):
            current_node = route[i]
            next_node = route[i + 1]
            aisle = current_node.position[0]

            dist = self.distance_matrix.at[current_node.position, next_node.position]
            travel_time = dist / resource_speed
            pick_time = resource_pick_time if hasattr(next_node, 'node_type') and next_node.node_type == 'PICK' else 0

            segment_end = current_time + travel_time + pick_time
            segments.append((aisle, current_time, segment_end))

            current_time = segment_end

        print(f"    Created {len(segments)} segments from cursor {cursor} to end")
        if segments:
            print(f"    First segment: aisle {segments[0][0]}, t={segments[0][1]:.1f}-{segments[0][2]:.1f}")
            print(f"    Last segment: aisle {segments[-1][0]}, t={segments[-1][1]:.1f}-{segments[-1][2]:.1f}")

        return segments

    def _calculate_conflicts(self, order: WarehouseOrder, occupancy: dict) -> tuple[float, float]:
        """Calculate conflict score using actual routing"""
        print(f"\n--- Calculating conflicts for order {order.order_id} ---")

        router = self.routing_class(**self.routing_class_kwargs)
        route_solution = router.solve(order.pick_positions)
        annotated_route = route_solution.route.annotated_route

        print(f"  Route length: {len(annotated_route)}")

        current_time = self.current_time
        conflict_score = 0
        segments_with_conflicts = 0

        for i in range(len(annotated_route) - 1):
            current_node = annotated_route[i]
            next_node = annotated_route[i + 1]
            aisle = current_node.position[0]

            dist = self.distance_matrix.at[current_node.position, next_node.position]
            travel_time = dist / self.resource.speed
            pick_time = self.resource.time_per_pick if hasattr(next_node,
                                                               'node_type') and next_node.node_type == 'PICK' else 0

            segment_end = current_time + travel_time + pick_time

            # Sum conflicts across time slots
            start_slot = int(current_time / self.slot_duration)
            end_slot = int(segment_end / self.slot_duration)

            segment_conflicts = 0
            for slot in range(start_slot, end_slot):
                slot_conflict = occupancy[slot].get(aisle, 0)
                segment_conflicts += slot_conflict

            if segment_conflicts > 0:
                segments_with_conflicts += 1

            conflict_score += segment_conflicts
            current_time = segment_end

        print(f"  Total conflict score: {conflict_score}")
        print(f"  Segments with conflicts: {segments_with_conflicts}/{len(annotated_route) - 1}")

        return (conflict_score, route_solution.route.distance)

    def _run(self, input_data: list[WarehouseOrder]) -> OrderSelectionSolution:
        print(f"\n{'=' * 60}")
        print(f"TimeIndexedMinConflictSelection - Starting order selection")
        print(f"{'=' * 60}")
        print(f"Candidate orders: {len(input_data)}")

        occupancy = self._build_occupancy_tracker()

        selected_order = min(
            input_data,
            key=lambda o: self._calculate_conflicts(o, occupancy)
        )

        print(f"\n{'=' * 60}")
        print(f"SELECTED ORDER: {selected_order.order_id}")
        print(f"{'=' * 60}\n")

        return OrderSelectionSolution(selected_orders=[selected_order])
