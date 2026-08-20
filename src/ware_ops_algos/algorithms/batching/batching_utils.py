import math

from ..algorithm_interfaces import WarehouseOrder
from ware_ops_algos.domain_models import Order, DimensionType, Articles, PickCart


class CapacityChecker:
    def __init__(self, pick_cart: PickCart, articles: Articles):
        self.pick_cart = pick_cart

        # Only load article dimensions if needed (non-ITEMS/ORDERLINES dimensions)
        if any(dim not in [DimensionType.ITEMS, DimensionType.ORDERLINES, DimensionType.ORDERS]
               for dim in pick_cart.dimensions):
            self.article_dimensions = self._load_article_dimensions(articles)
        else:
            self.article_dimensions = None

    def _compute_boxes_needed(self, order: WarehouseOrder) -> int:
        consumption = self._compute_order_consumption(order)
        return max(
            math.ceil(consumption[d] / self.pick_cart.capacities[d])
            for d in range(self.pick_cart.n_dimension)
        )

    def _load_article_dimensions(self, article_domain: Articles) -> dict[int, list[float]]:
        dimensions_map = {}

        for article in article_domain.articles:
            dims = []

            for dim_type in self.pick_cart.dimensions:
                if dim_type in [DimensionType.ITEMS, DimensionType.ORDERLINES, DimensionType.ORDERS]:
                    # Skip - we don't need article data for ITEMS or ORDERLINES
                    continue
                elif dim_type == DimensionType.WEIGHT:
                    if article.weight is None:
                        raise ValueError(f"Article {article.article_id} missing weight")
                    dims.append(article.weight)
                elif dim_type == DimensionType.VOLUME:
                    if article.volume is None:
                        raise ValueError(f"Article {article.article_id} missing volume")
                    dims.append(article.volume)
                else:
                    raise ValueError(f"Unknown dimension type: {dim_type}")

            dimensions_map[article.article_id] = dims

        return dimensions_map

    # def can_add_order(self, current_batch: list[WarehouseOrder], new_order: WarehouseOrder) -> bool:
    #     # Compute consumption of current batch
    #     current_consumption = self._compute_consumption(current_batch)
    #
    #     # Compute consumption of new order
    #     new_consumption = self._compute_order_consumption(new_order)
    #
    #     # Total capacity
    #     total_capacity = [
    #         cap * self.pick_cart.n_boxes
    #         for cap in self.pick_cart.capacities
    #     ]
    #
    #     # Check: current + new ≤ capacity
    #     for d in range(self.pick_cart.n_dimension):
    #         if current_consumption[d] + new_consumption[d] > total_capacity[d]:
    #             return False
    #
    #     return True
    def can_add_order(self, current_batch: list[WarehouseOrder], new_order: WarehouseOrder) -> bool:
        return self.orders_fit(current_batch + [new_order])

    def orders_fit(self, orders: list[WarehouseOrder]) -> bool:
        if self.pick_cart.box_can_mix_orders:
            consumption = self._compute_consumption(orders)
            return self.fits_consumption(consumption)
        else:
            total_boxes = sum(self.order_box_count(o) for o in orders)
            return self.fits_consumption(
                (),
                box_count=total_boxes,
            )

    def order_consumption(self, order: WarehouseOrder) -> tuple[float, ...]:
        """Return the capacity vector contributed by one semantic order."""
        return tuple(self._compute_order_consumption(order))

    def order_box_count(self, order: WarehouseOrder) -> int:
        """Return the boxes occupied by an order when boxes cannot be mixed."""
        return self._compute_boxes_needed(order)

    def fits_consumption(
        self,
        consumption: tuple[float, ...] | list[float],
        *,
        box_count: int | None = None,
    ) -> bool:
        """Check an already-computed batch capacity state.

        ``box_count`` is required for non-mixing carts because aggregate
        dimension consumption cannot represent per-order box rounding.
        """
        if not self.pick_cart.box_can_mix_orders:
            if box_count is None:
                raise ValueError(
                    "box_count is required when cart boxes cannot mix orders"
                )
            return box_count <= self.pick_cart.n_boxes

        return all(
            value <= capacity * self.pick_cart.n_boxes
            for value, capacity in zip(
                consumption,
                self.pick_cart.capacities,
            )
        )

    def _compute_consumption(self, orders: list[WarehouseOrder]) -> list[float]:
        total = [0.0] * self.pick_cart.n_dimension

        for order in orders:
            consumption = self._compute_order_consumption(order)
            for d in range(self.pick_cart.n_dimension):
                total[d] += consumption[d]

        return total

    def _compute_order_consumption(self, order: WarehouseOrder) -> list[float]:
        consumption = [0.0] * self.pick_cart.n_dimension

        for d, dim_type in enumerate(self.pick_cart.dimensions):
            if dim_type == DimensionType.ITEMS:
                # Count items
                consumption[d] = sum(
                    pos.picked_quantity for pos in order.pick_positions
                )
            elif dim_type == DimensionType.ORDERLINES:
                # A line may be split over several storage locations. Within
                # the current domain model an order line is identified by its
                # order number and article.
                consumption[d] = len({
                    (pos.order_number, pos.article_id)
                    for pos in order.pick_positions
                })
            elif dim_type == DimensionType.ORDERS:
                # Count orders
                consumption[d] = 1
            else:
                # Sum (quantity × article_dimension)
                for pos in order.pick_positions:
                    article_id = pos.article_id
                    quantity = pos.picked_quantity
                    article_dims = self.article_dimensions[article_id]
                    dim_idx = self._get_article_dim_index(d)
                    consumption[d] += quantity * article_dims[dim_idx]

        return consumption

    def _get_article_dim_index(self, pick_cart_dim_index: int) -> int:
        """Map pick cart dimension index to article dimension index."""
        count = 0
        for i in range(pick_cart_dim_index):
            if self.pick_cart.dimensions[i] not in [DimensionType.ITEMS, DimensionType.ORDERS,
                                                    DimensionType.ORDERLINES]:
                count += 1
        return count

    def get_item_consumption(self, article_id: int, quantity: int = 1) -> list[float]:
        """
        Get consumption for an article.

        Args:
            article_id: Article ID
            quantity: Number of units (default: 1)

        Returns:
            Consumption vector [dim0, dim1, ...]
        """
        consumption = [0.0] * self.pick_cart.n_dimension

        for d, dim_type in enumerate(self.pick_cart.dimensions):
            if dim_type == DimensionType.ITEMS:
                consumption[d] = quantity
            elif dim_type == DimensionType.ORDERLINES:
                consumption[d] = 1  # Each position counts as 1 orderline
            elif dim_type == DimensionType.ORDERS:
                consumption[d] = 0  # Not counted per item
            else:
                article_dims = self.article_dimensions[article_id]
                dim_idx = self._get_article_dim_index(d)
                consumption[d] = article_dims[dim_idx] * quantity

        return consumption


def latest_order_arrival(orders: list[WarehouseOrder]) -> float:
    if any(o.order_date is not None for o in orders):
        arrivals = [o.order_date for o in orders]
        return max(arrivals) if arrivals else 0.0
    else:
        return 0.0


def first_due_date(orders: list[WarehouseOrder]) -> float:
    if any(o.order_date is not None for o in orders):
        due_dates = [o.order_date for o in orders]
        return min(due_dates) if due_dates else float("inf")
    else:
        return 0.0


# def build_pick_lists(orders: list[WarehouseOrder]):
#     # build pick lists
#     pick_positions = []
#     for order in orders:
#         for pos in order.pick_positions:
#             pick_positions.append(pos)
#
#     pick_list = PickList(
#         pick_positions=pick_positions,
#         release=latest_order_arrival(orders),
#         earliest_due_date=first_due_date(orders),
#         orders=orders
#     )
#     return pick_list
