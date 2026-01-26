from abc import abstractmethod
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Type, Callable

import numpy as np
import pandas as pd

from ware_ops_algos.algorithms import Algorithm, ItemAssignmentSolution, PickPosition, WarehouseOrder, Routing, \
    RatliffRosenthalRouting
from ware_ops_algos.data_loaders import HesslerIrnichLoader
from ware_ops_algos.domain_models import Order, ResolvedOrderPosition, StorageLocations, Location


class ItemAssignment(Algorithm[list[Order], ItemAssignmentSolution]):
    def __init__(self, storage_locations: StorageLocations, **kwargs):
        super().__init__(**kwargs)
        self.storage_locations = storage_locations

    def _run(self, input_data: list[Order]) -> ItemAssignmentSolution:
        pass


class GreedyItemAssignment(ItemAssignment):
    def _run(self, input_data: list[Order]) -> ItemAssignmentSolution:
        orders = deepcopy(input_data)
        warehouse_orders = []
        for order in orders:
            resolved = []
            for pos in order.order_positions:
                locs = self.storage_locations.get_locations_by_article_id(pos.article_id)
                sorted_locs = sorted(locs, key=lambda l: -l.amount)

                remaining = pos.amount
                fulfilled = 0
                used_locations = []

                for loc in sorted_locs:
                    if remaining <= 0:
                        break

                    pick_qty = min(remaining, loc.amount)

                    # resolved.append(ResolvedOrderPosition(
                    #     pos,
                    #     (loc.x, loc.y),
                    #     fulfilled=pick_qty,
                    #     picked=False
                    # ))

                    resolved.append(PickPosition(
                        order_number=pos.order_number,
                        article_id=pos.article_id,
                        amount=pos.amount,
                        pick_node=(loc.x, loc.y),
                        in_store=pick_qty,
                        article_name=pos.article_name,
                        picked=False
                    ))

                    fulfilled += pick_qty
                    remaining -= pick_qty
                    used_locations.append((loc.x, loc.y))
            # order.order_positions = resolved
            warehouse_orders.append(WarehouseOrder(order_id=order.order_id,
                                                   due_date=order.due_date,
                                                   order_date=order.order_date,
                                                   pick_positions=resolved,
                                                   fulfilled=False,
                                                   ))
        return ItemAssignmentSolution(resolved_orders=warehouse_orders)


class NearestNeighborItemAssignment(ItemAssignment):
    def __init__(
            self,
            storage_locations: StorageLocations,
            distance_matrix: pd.DataFrame,
            start_node: tuple[int, int] = (0, 0),
            **kwargs
    ):
        super().__init__(storage_locations=storage_locations, **kwargs)
        self.distance_matrix = distance_matrix
        self.start_node = start_node

        # Build article-location mappings
        self.article_location_mapping, self.location_article_mapping = self._build_mappings()

    def _build_mappings(self):
        """Build article-location and location-article mappings."""
        article_location_mapping = {}
        location_article_mapping = {}

        for loc in self.storage_locations.locations:
            if loc.article_id not in article_location_mapping:
                article_location_mapping[loc.article_id] = []
            article_location_mapping[loc.article_id].append(loc)

        for loc in self.storage_locations.locations:
            if (loc.x, loc.y) not in location_article_mapping:
                location_article_mapping[(loc.x, loc.y)] = loc.article_id

        return article_location_mapping, location_article_mapping

    def _run(self, input_data: list[Order]) -> ItemAssignmentSolution:
        orders = deepcopy(input_data)
        warehouse_orders = []
        for order in orders:
            resolved = self._select_for_order(order)
            warehouse_orders.append(WarehouseOrder(order_id=order.order_id,
                                                   due_date=order.due_date,
                                                   order_date=order.order_date,
                                                   pick_positions=resolved,
                                                   fulfilled=False,
                                                   ))

        return ItemAssignmentSolution(resolved_orders=warehouse_orders)

    def _select_for_order(self, order: Order) -> list[PickPosition]:
        """Select pick locations for a single order using nearest neighbor."""
        resolved = []
        current_loc = self.start_node

        for pos in order.order_positions:
            sku = pos.article_id
            demand = pos.amount
            fulfilled = 0

            candidates = self.storage_locations.get_locations_by_article_id(sku).copy()

            while fulfilled < demand and candidates:
                # Select location with min distance to current_loc
                nearest = min(
                    candidates,
                    key=lambda loc: self.distance_matrix.at[current_loc, (loc.x, loc.y)]
                )

                pick_qty = min(demand - fulfilled, nearest.amount)

                resolved.append(PickPosition(
                        order_number=pos.order_number,
                        article_id=pos.article_id,
                        amount=pos.amount,
                        pick_node=(nearest.x, nearest.y),
                        in_store=pick_qty,
                        article_name=pos.article_name,
                        picked=False
                    ))

                fulfilled += pick_qty
                current_loc = (nearest.x, nearest.y)

                # Remove this location from candidates
                candidates.remove(nearest)

            if fulfilled < demand:
                print(f"Could not fully fulfill article {sku}. Needed {demand}, got {fulfilled}")

        return resolved


class PriorityItemAssignment(ItemAssignment):
    """Base class for Weidinger selection algorithms."""

    def __init__(self, storage_locations: StorageLocations, distance_matrix: pd.DataFrame, **kwargs):
        super().__init__(storage_locations, **kwargs)
        self.storage_locations = storage_locations
        self.distance_matrix = distance_matrix
        self.q_max = max((loc.amount for loc in storage_locations.locations), default=1)

    def _run(self, input_data: list[Order]) -> ItemAssignmentSolution:
        orders = deepcopy(input_data)
        warehouse_orders = []

        for order in orders:
            pick_positions = self._select_for_order(order)
            warehouse_orders.append(WarehouseOrder(
                order_id=order.order_id,
                due_date=order.due_date,
                order_date=order.order_date,
                pick_positions=pick_positions,
                fulfilled=False
            ))

        return ItemAssignmentSolution(resolved_orders=warehouse_orders)

    @abstractmethod
    def _select_for_order(self, order: Order) -> list[PickPosition]:
        pass

    def _select_for_sku(
        self,
        sku_locations: list[Location],
        demand: int,
        priority_fn,
        preselected: list[Location] = None,
        initial_qty: int = 0
    ) -> list[Location]:
        preselected = preselected or []

        if initial_qty >= demand:
            return preselected

        candidates = [loc for loc in sku_locations if loc not in preselected]
        prioritized = sorted(candidates, key=priority_fn)

        selected = list(preselected)
        total_qty = initial_qty

        for loc in prioritized:
            if total_qty >= demand:
                break
            selected.append(loc)
            total_qty += loc.amount

        # deselection phase: remove redundant (we may have added to many storage locations -> "overshooting"
        sorted_desc = sorted(selected, key=priority_fn, reverse=True)
        total_qty = sum(loc.amount for loc in sorted_desc)

        final = []
        for loc in sorted_desc:
            if total_qty - loc.amount >= demand:
                total_qty -= loc.amount
            else:
                final.append(loc)

        return final

    def _assign_quantities(self, selected: list[Location], demand: int) -> list[tuple[Location, int]]:
        result = []
        remaining = demand
        for loc in selected:
            pick_qty = min(remaining, loc.amount)
            result.append((loc, pick_qty))
            remaining -= pick_qty
            if remaining <= 0:
                break
        return result



    def _to_pick_positions(self, selection: dict, order: Order) -> list[PickPosition]:
        pos_by_sku = {pos.article_id: pos for pos in order.order_positions}
        pick_positions = []

        for sku, locs_and_qtys in selection.items():
            order_pos = pos_by_sku.get(sku)
            if not order_pos:
                continue
            for loc, pick_qty in locs_and_qtys:
                pick_positions.append(PickPosition(
                    order_number=order_pos.order_number,
                    article_id=sku,
                    amount=order_pos.amount,
                    pick_node=(loc.x, loc.y),
                    in_store=pick_qty,
                    article_name=order_pos.article_name,
                    picked=False
                ))

        return pick_positions


class SinglePositionItemAssignment(PriorityItemAssignment):
    """Algorithm 2"""
    def __init__(self,
                 storage_locations: StorageLocations,
                 distance_matrix: pd.DataFrame,
                 routing_class: Type[Routing],
                 routing_class_kwargs: dict, **kwargs):

        super().__init__(storage_locations, distance_matrix, **kwargs)
        self.routing_class = routing_class
        self.routing_class_kwargs = routing_class_kwargs
        self.start_node = routing_class_kwargs["start_node"]

    def _select_for_order(self, order: Order) -> list[PickPosition]:
        demand_per_sku = {pos.article_id: pos.amount for pos in order.order_positions}
        if not demand_per_sku:
            return []

        # Find sku with least nr of storage locations (most constrained -> we have to visit anyway)
        rarest_sku = min(
            demand_per_sku.keys(),
            key=lambda sku: len(self.storage_locations.get_locations_by_article_id(sku) or [])
        )
        rarest_locations = self.storage_locations.get_locations_by_article_id(rarest_sku)

        best_selection = None
        best_tour_length = float('inf')

        for anchor in rarest_locations:
            selection = self._select_with_anchor(anchor, demand_per_sku)
            pick_positions = self._to_pick_positions(selection, order)
            tour_length = self._calc_tour_length(pick_positions)

            if tour_length < best_tour_length:
                best_tour_length = tour_length
                best_selection = selection

        return self._to_pick_positions(best_selection or {}, order)

    def _select_with_anchor(self, anchor: Location, demand_per_sku: dict) -> dict:
        selection = {}

        for sku, demand in demand_per_sku.items():
            sku_locations = self.storage_locations.get_locations_by_article_id(sku)

            preselected = [anchor] if anchor.article_id == sku else []
            initial_qty = anchor.amount if anchor.article_id == sku else 0

            priority_fn = lambda loc, a=anchor: self._priority_single(loc, a)

            selected = self._select_for_sku(
                sku_locations, demand, priority_fn, preselected, initial_qty
            )
            selection[sku] = self._assign_quantities(selected, demand)

        return selection

    def _priority_single(self, loc: Location, anchor: Location) -> float:
        dist = self.distance_matrix.at[(loc.x, loc.y), (anchor.x, anchor.y)]
        tiebreaker = (self.q_max - loc.amount) / self.q_max
        return dist + tiebreaker

    def _calc_tour_length(self, pick_positions: list[PickPosition]) -> float:
        if not pick_positions:
            return float('inf')
        router = self.routing_class(**self.routing_class_kwargs)
        return router.solve(pick_positions).route.distance


class MinMaxItemAssignment(PriorityItemAssignment):
    """
    Algorithm 1: MinMax

    Process SKUs by fewest locations first.
    Priority = MAX distance to any already-selected position.
    """

    def __init__(self,
                 storage_locations: StorageLocations,
                 distance_matrix: pd.DataFrame,
                 start_node: tuple[float, float],
                 **kwargs):

        super().__init__(storage_locations, distance_matrix, **kwargs)
        self.start_node = start_node

    def _select_for_order(self, order: Order) -> list[PickPosition]:
        demand_per_sku = {pos.article_id: pos.amount for pos in order.order_positions}
        if not demand_per_sku:
            return []

        selected_positions: set[tuple[int, int]] = {self.start_node}

        skus_sorted = sorted(
            demand_per_sku.keys(),
            key=lambda sku: len(self.storage_locations.get_locations_by_article_id(sku) or [])
        )

        selection = {}

        for sku in skus_sorted:
            demand = demand_per_sku[sku]
            sku_locations = self.storage_locations.get_locations_by_article_id(sku)

            # Priority: MAX distance to Ñ (capture current state)
            current_selected = set(selected_positions)
            priority_fn = lambda loc, sel=current_selected: self._priority_minmax(loc, sel)

            selected = self._select_for_sku(sku_locations, demand, priority_fn)
            selection[sku] = self._assign_quantities(selected, demand)

            # Update Ñ
            for loc in selected:
                selected_positions.add((loc.x, loc.y))

        return self._to_pick_positions(selection, order)

    def _priority_minmax(self, loc: Location, selected_positions: set) -> float:
        loc_pos = (loc.x, loc.y)
        max_dist = max(self.distance_matrix.at[loc_pos, pos] for pos in selected_positions)
        tiebreaker = (self.q_max - loc.amount) / self.q_max
        return max_dist + tiebreaker


class MinMinItemAssignment(PriorityItemAssignment):
    """
    Algorithm 1: MinMin

    Process SKUs by fewest locations first.
    Priority = MIN distance to any already-selected position.
    Creates tight clusters.
    """
    def __init__(self,
                 storage_locations: StorageLocations,
                 distance_matrix: pd.DataFrame,
                 start_node: tuple[float, float],
                 **kwargs):

        super().__init__(storage_locations, distance_matrix, **kwargs)
        self.start_node = start_node

    def _select_for_order(self, order: Order) -> list[PickPosition]:
        demand_per_sku = {pos.article_id: pos.amount for pos in order.order_positions}
        if not demand_per_sku:
            return []

        selected_positions: set[tuple[int, int]] = {self.start_node}

        skus_sorted = sorted(
            demand_per_sku.keys(),
            key=lambda sku: len(self.storage_locations.get_locations_by_article_id(sku) or [])
        )

        selection = {}

        for sku in skus_sorted:
            demand = demand_per_sku[sku]
            sku_locations = self.storage_locations.get_locations_by_article_id(sku)

            current_selected = set(selected_positions)
            priority_fn = lambda loc, sel=current_selected: self._priority_minmin(loc, sel)

            selected = self._select_for_sku(sku_locations, demand, priority_fn)
            selection[sku] = self._assign_quantities(selected, demand)

            for loc in selected:
                selected_positions.add((loc.x, loc.y))

        return self._to_pick_positions(selection, order)

    def _priority_minmin(self, loc: Location, selected_positions: set) -> float:
        """P(i) = min{dist(i, j) for j in Ñ} + tiebreaker"""
        loc_pos = (loc.x, loc.y)
        min_dist = min(self.distance_matrix.at[loc_pos, pos] for pos in selected_positions)
        tiebreaker = (self.q_max - loc.amount) / self.q_max
        return min_dist + tiebreaker


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    instances_base = DATA_DIR / "instances"
    cache_base = DATA_DIR / "instances" / "caches"

    instance_set = "SPRP-SS"
    il = HesslerIrnichLoader(instances_dir=instances_base / instance_set)
    domain = il.load(filepath="varying_F10_m50_C180_a30_49.txt",
                     use_cache=False)

    orders = domain.orders
    layout = domain.layout
    resources = domain.resources
    articles = domain.articles
    storage_locations = domain.storage

    layout_network = layout.layout_network
    graph = layout_network.graph
    graph_params = layout.graph_data
    dima = layout_network.distance_matrix

    rr_kwargs = {"start_node": layout_network.start_node,
                 "end_node": layout_network.end_node,
                 "closest_node_to_start": layout_network.closest_node_to_start,
                 "min_aisle_position": layout_network.min_aisle_position,
                 "max_aisle_position": layout_network.max_aisle_position,
                 "distance_matrix": layout_network.distance_matrix,
                 "predecessor_matrix": layout_network.predecessor_matrix,
                 "picker": resources.resources,
                 "n_aisles": graph_params.n_aisles,
                 "n_pick_locations": graph_params.n_pick_locations,
                 "dist_aisle": graph_params.dist_aisle,
                 "dist_pick_locations": graph_params.dist_pick_locations,
                 "dist_aisle_location": graph_params.dist_bottom_to_pick_location,
                 "dist_start": graph_params.dist_start,
                 "dist_end": graph_params.dist_end,
                 "gen_tour": False,
                 "gen_item_sequence": False
                 }

    gia = GreedyItemAssignment(
        storage_locations=storage_locations
    )

    gia_sol = gia.solve(domain.orders.orders)

    nnia = NearestNeighborItemAssignment(
        storage_locations=storage_locations,
        distance_matrix=dima,
        start_node=layout_network.start_node
    )

    nnia_sol = nnia.solve(domain.orders.orders)

    single_pos = SinglePositionItemAssignment(
        storage_locations=storage_locations,
        distance_matrix=dima,
        routing_class=RatliffRosenthalRouting,
        routing_class_kwargs=rr_kwargs
    )

    single_pos_sol = single_pos.solve(domain.orders.orders)

    minmin = MinMinItemAssignment(
        storage_locations=storage_locations,
        distance_matrix=dima,
        start_node=layout_network.start_node
    )

    minmin_sol = minmin.solve(domain.orders.orders)

    minmax = MinMaxItemAssignment(
        storage_locations=storage_locations,
        distance_matrix=dima,
        start_node=layout_network.start_node
    )

    minmax_sol = minmax.solve(domain.orders.orders)

    rr_routing = RatliffRosenthalRouting(
        **rr_kwargs
    )

    for sol in [gia_sol, nnia_sol, single_pos_sol, minmax_sol, minmin_sol]:
        pick_list = []
        for o in sol.resolved_orders:
            for pp in o.pick_positions:
                pick_list.append(pp)

        routing_sol = rr_routing.solve(pick_list)
        print(routing_sol.route.distance)
