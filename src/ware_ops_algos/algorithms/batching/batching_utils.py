import math

from ware_ops_algos.algorithms import WarehouseOrder
from ware_ops_algos.domain_models import Order, DimensionType, Articles, PickCart


class CapacityChecker:
    def __init__(self, pick_cart: PickCart, articles: Articles):
        self.pick_cart = pick_cart

        # Only load article dimensions if needed (non-ITEMS/ORDERLINES dimensions)
        if any(dim not in [DimensionType.ITEMS, DimensionType.ORDERLINES]
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
            total_capacity = [
                cap * self.pick_cart.n_boxes
                for cap in self.pick_cart.capacities
            ]
            return all(
                consumption[d] <= total_capacity[d]
                for d in range(self.pick_cart.n_dimension)
            )
        else:
            total_boxes = sum(self._compute_boxes_needed(o) for o in orders)
            return total_boxes <= self.pick_cart.n_boxes

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
                consumption[d] = sum(pos.in_store for pos in order.pick_positions)
            elif dim_type == DimensionType.ORDERLINES:
                # Count order lines (positions)
                consumption[d] = len(order.pick_positions)
            elif dim_type == DimensionType.ORDERS:
                # Count orders
                consumption[d] = 1
            elif dim_type == DimensionType.WEIGHT:
                for pos in order.pick_positions:
                    article_id = pos.article_id
                    quantity = pos.amount
                    article_dims = self.article_dimensions[article_id]
                    dim_idx = self._get_article_dim_index(d)
                    consumption[d] += quantity * article_dims[dim_idx]
            else:
                # Sum (quantity × article_dimension)
                for pos in order.pick_positions:
                    article_id = pos.order_position.article_id
                    quantity = pos.in_store
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