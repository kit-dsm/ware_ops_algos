from abc import ABC, abstractmethod
from typing import List, Tuple
from datetime import datetime
from Domain.entities import Order
import logging

logger = logging.getLogger(__name__)

class InitialRouteStrategy(ABC):
    """
    Abstract base class for strategies that generate an initial route for a vehicle when it's idle.
    Subclasses only implement generate_nodes() to define the specific route generation logic.
    """
    def __init__(self, warehouse, shift_end: datetime = None):
        """
        Initialize the initial route strategy.

        Args:
            warehouse: The warehouse object containing the graph and other information.
            shift_end: The datetime indicating the end of the shift. This is used to set the due time for dummy orders.
        """
        self.warehouse = warehouse
        self.shift_end = shift_end

    @abstractmethod
    def generate_nodes(self) -> List[Tuple[int, int]] :
        """
        Generates the node sequence for the initial route. Has to be implemented in every subclass.

        Returns:
            A list of nodes to be visited.
        """
        pass

    def generate_dummy_order(self, current_time, order_id: int = -1) -> Order:
        """
        Open method: Creates a dummy order with all nodes of the initial route as SKU with Quantity=0.
        Internally, generate_nodes() is called first to generate the node sequence.
        """

        nodes = self.generate_nodes()
        now   = current_time
        return self._make_dummy_order_from_nodes(
            nodes=nodes,
            sku_allocation=self.warehouse.allocation,
            arrival_time=now,
            due_time=self.shift_end,
            order_id=order_id
        )

    @staticmethod
    def _make_dummy_order_from_nodes(
            nodes: List[Tuple[int, int]],
            sku_allocation: dict,
            arrival_time: datetime,
            due_time: datetime,
            order_id: int = -1,
    ) -> Order:
        """
        Private helper: Creates the dummy order from a list of nodes.
        Works same for all strategies.

        Args:
            nodes: List of (x,y) nodes to be visited
            sku_allocation: Dict mapping SKU -> (x,y)-Location.
            order_id: ID or Dummy-Order (default -1).
            arrival_time: now
            due_time: shift_end

        Returns:
            A Order with items={ sku1:0, sku2:0, … }.
        """

        loc2sku = {loc: sku for sku, loc in sku_allocation.items()}
        items = {}
        for idx, node in enumerate(nodes):
            sku = loc2sku.get(node)
            if sku is None:
                logger.sim(
                    f'{arrival_time} | {__class__.__module__} | {__class__.__name__}: '
                    f'Node {node} not in sku_allocation. Skipping.')
                continue
            items[sku] = 0

        return Order(
            id=order_id,
            items=items,
            arrival_time=arrival_time,
            due_time=due_time,
            dummy_order=True
        )


class AllNodes(InitialRouteStrategy):
    """
    A strategy for generating an initial route for a vehicle when it's idle.
    This strategy creates a route that visits all nodes in the warehouse.
    """
    def generate_nodes(self) -> List[Tuple[int, int]]:
        """
        Generate a route that visits all nodes in the warehouse.

        Returns:
            A list of nodes to be visited in sequence.
        """
        # Get all nodes in the warehouse graph except the start and end nodes
        all_nodes = list(self.warehouse.graph.nodes())
        if self.warehouse.start_node in all_nodes:
            all_nodes.remove(self.warehouse.start_node)
        if self.warehouse.end_node in all_nodes:
            all_nodes.remove(self.warehouse.end_node)

        # Reverse the tour to match the expected format
        return all_nodes


def debug_sku_allocation_consistency(warehouse) -> dict:
    """
    Debug helper to check consistency between graph nodes and SKU allocation.

    Args:
        warehouse: The warehouse object to analyze

    Returns:
        Dictionary with diagnostic information
    """
    graph_nodes = set(warehouse.graph.nodes())
    sku_locations = set(warehouse.allocation.values())

    # Check for duplicate locations (multiple SKUs per location)
    from collections import Counter
    location_counts = Counter(warehouse.allocation.values())
    locations_with_multiple_skus = {
        loc: count for loc, count in location_counts.items() if count > 1
    }

    diagnostics = {
        'total_graph_nodes': len(graph_nodes),
        'total_skus': len(warehouse.allocation),
        'unique_pick_locations': len(sku_locations),
        'nodes_without_sku': len(graph_nodes - sku_locations),
        'sku_locations_not_in_graph': len(sku_locations - graph_nodes),
        'locations_with_multiple_skus': len(locations_with_multiple_skus),
        'sample_multi_sku_locations': dict(list(locations_with_multiple_skus.items())[:5])
    }

    # Log Diagnostics Header
    logger.info("=" * 70)
    logger.info("SKU ALLOCATION DIAGNOSTICS")
    logger.info("=" * 70)

    # Log each diagnostic item
    for key, value in diagnostics.items():
        logger.info(f"{key:.<50} {value}")

    logger.info("=" * 70)

    # Warning für mehrere SKUs pro Location
    if diagnostics['locations_with_multiple_skus'] > 0:
        logger.warning(
            f"{diagnostics['locations_with_multiple_skus']} locations have multiple SKUs assigned. "
            f"This is handled correctly by the fixed _make_dummy_order_from_nodes()")

        # Log sample locations
        if diagnostics['sample_multi_sku_locations']:
            logger.info(f"Sample multi-SKU locations: {diagnostics['sample_multi_sku_locations']}")

    # Warnung für Nodes ohne SKU
    if diagnostics['nodes_without_sku'] > 0:
        logger.info(
            f"{diagnostics['nodes_without_sku']} graph nodes have no SKU assignment "
            f"(likely corridor/transit nodes)")

    # Warnung für SKU-Locations nicht im Graph
    if diagnostics['sku_locations_not_in_graph'] > 0:
        logger.warning(
            f"{diagnostics['sku_locations_not_in_graph']} SKU locations are NOT in the warehouse graph! "
            f"This may cause routing issues.")

    return diagnostics
