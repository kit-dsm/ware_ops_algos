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


class MinAislesPickLocationSelector(ItemAssignment):
    def _run(self, input_data: list[Order]) -> ItemAssignmentSolution:
        orders = deepcopy(input_data)
        warehouse_orders = []
        # Idea: Start picking in the aisle that contains the most required items and is closest
        # then add the aisle wich contains the rest and so on
        # aisle 1: (A1, A2, A4)
        # aisle 2: (A2, A3)

        self.demand = defaultdict(int)
        for order in orders:
            for position in order.order_positions:
                self.demand[position.article_id] += 1

        aisle_content = defaultdict(list)
        total_warehouse_supply = defaultdict(int)
        aisle_total_supply = defaultdict(lambda: defaultdict(int))

        relevant_articles = set(self.demand.keys())

        for article_id in relevant_articles:
            locations = self.storage_locations.get_locations_by_article_id(article_id)
            for loc in locations:
                # loc.x = aisle, loc.y = position
                aisle_idx = int(loc.x)
                aisle_content[aisle_idx].append({
                    'y': int(loc.y),
                    'article_id': loc.article_id
                })
                self.total_warehouse_supply[article_id] += 1
                self.aisle_total_supply[aisle_idx][article_id] += 1
        resolved = []
        sorted_aisles = sorted(self.aisle_total_supply, key=lambda l: len(n_items))

        best_aisle = max(
            aisle_item_counts.keys(),
            key=lambda aisle: (
                aisle_item_counts[aisle],  # Primary: item count
                -_aisle_distance(aisle, aisle_locations[aisle], current_position)  # Tie-break: nearest
            )
        )

        for aisle_idx in sorted_aisles:
            pass
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

# class NearestNeighborPickLocationSelector:
#     def __init__(
#         self,
#         order: Order,
#         storage_locations: StorageLocations,
#         distance_matrix: pd.DataFrame,
#         start_node: tuple[int, int] = (0, 0)
#     ):
#         self.order = order
#         self.storage_locations = storage_locations
#         self.distance_matrix = distance_matrix
#         self.start_node = start_node
#
#     def select(self) -> list[ResolvedOrderPosition]:
#         resolved = []
#         current_loc = self.start_node
#
#         for pos in self.order.order_positions:
#             sku = pos.article_id
#             demand = pos.amount
#             fulfilled = 0
#
#             # Create working list of candidate locations for this SKU
#             candidates = self.storage_locations.get_locations_by_article_id(sku).copy()
#
#             while fulfilled < demand and candidates:
#                 # Select location with min distance to current_loc
#                 nearest = min(
#                     candidates,
#                     key=lambda loc: self.distance_matrix.at[current_loc, (loc.x, loc.y)]
#                 )
#
#                 pick_qty = min(demand - fulfilled, nearest.amount)
#
#                 resolved.append(ResolvedOrderPosition(
#                     pos,
#                     pick_node=(nearest.x, nearest.y),
#                     fulfilled=pick_qty
#                 ))
#
#                 fulfilled += pick_qty
#                 current_loc = (nearest.x, nearest.y)
#
#                 # Remove this location from candidates
#                 candidates.remove(nearest)
#
#             if fulfilled < demand:
#                 print(f"Could not fully fulfill article {sku}. Needed {demand}, got {fulfilled}")
#
#         return resolved


class NNPickLocationSelector(ItemAssignment):
    def __init__(self, order, article_location_mapping, location_inventory_mapping, distance_matrix, start_node):
        super().__init__(order, article_location_mapping, location_inventory_mapping, distance_matrix, start_node)

    def select(self):
        selected = {}
        current_loc = self.start_node
        for sku in self.order["article_id"].unique():
            # TODO: Hier müsste man theoretisch demand handlen -> Inventory maintainen und Bestellung abziehen
            # bzw checken welche locations genug inventory haben
            candidates = self.article_location_mapping[sku]
            nearest = min(candidates, key=lambda item: self.distance_matrix.at[current_loc, item])
            selected[sku] = nearest
            current_loc = nearest

        return selected


class GeneralDemandPickLocationSelector(ItemAssignment):
    def __init__(self, order, article_location_mapping, location_inventory_mapping, distance_matrix, start_node):
        super().__init__(order, article_location_mapping, location_inventory_mapping, distance_matrix, start_node)
        self.sorted_sku_locations = {
            sku: sorted(locations, key=lambda loc: location_inventory_mapping[loc], reverse=True)
            for sku, locations in article_location_mapping.items()
        }

    def select(self):
        selected = {}
        current_loc = self.start_node
        for sku in self.order["article_id"].unique():
            demand = self.order[self.order["article_id"] == sku]["amount"].item()
            fulfilled = 0
            selected_locs = []
            # Get the sorted locations for the SKU based on inventory quantity
            candidates = self.sorted_sku_locations[sku]  # you created this earlier
            while fulfilled < demand and candidates:
                # Find the nearest candidate (based on current location)
                nearest = min(candidates, key=lambda item: self.distance_matrix.at[current_loc, item])
                available = self.location_inventory_mapping[nearest]
                qty_needed = demand - fulfilled
                take = min(qty_needed, available)
                if take > 0:
                    selected_locs.append((nearest, take))
                    fulfilled += take
                    current_loc = nearest
                # Remove this location from candidates to avoid re-selection
                candidates.remove(nearest)
            if fulfilled < demand:
                print(f"Warning: Could not fully fulfill SKU {sku}. Needed {demand}, got {fulfilled}")
            selected[sku] = selected_locs
        return selected
