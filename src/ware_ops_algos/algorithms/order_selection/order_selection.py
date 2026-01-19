from ware_ops_algos.algorithms import Algorithm, WarehouseOrder, OrderSelectionSolution
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