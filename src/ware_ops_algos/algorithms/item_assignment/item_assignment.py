from collections import defaultdict
from copy import deepcopy

import pandas as pd

from ware_ops_algos.algorithms import Algorithm, ItemAssignmentSolution, PickPosition, WarehouseOrder
from ware_ops_algos.domain_models import Order, ResolvedOrderPosition, StorageLocations


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
        self.orders = input_data

        for order in self.orders:
            resolved = self._select_for_order(order)
            order.order_positions = resolved

        return ItemAssignmentSolution(resolved_orders=self.orders)

    def _select_for_order(self, order: Order) -> list[ResolvedOrderPosition]:
        """Select pick locations for a single order using nearest neighbor."""
        resolved = []
        current_loc = self.start_node

        for pos in order.order_positions:
            sku = pos.article_id
            demand = pos.amount
            fulfilled = 0

            # Create working list of candidate locations for this SKU
            candidates = self.storage_locations.get_locations_by_article_id(sku).copy()

            while fulfilled < demand and candidates:
                # Select location with min distance to current_loc
                nearest = min(
                    candidates,
                    key=lambda loc: self.distance_matrix.at[current_loc, (loc.x, loc.y)]
                )

                pick_qty = min(demand - fulfilled, nearest.amount)

                resolved.append(ResolvedOrderPosition(
                    pos,
                    pick_node=(nearest.x, nearest.y),
                    fulfilled=pick_qty,
                    picked=False
                ))

                fulfilled += pick_qty
                current_loc = (nearest.x, nearest.y)

                # Remove this location from candidates
                candidates.remove(nearest)

            if fulfilled < demand:
                print(f"Could not fully fulfill article {sku}. Needed {demand}, got {fulfilled}")

        return resolved
