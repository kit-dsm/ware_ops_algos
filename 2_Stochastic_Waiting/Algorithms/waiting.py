import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, Any
from Domain.events import InitialRouteEvent


logger = logging.getLogger(__name__)

class WaitingPolicy(ABC):
    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name

    @abstractmethod
    def make_decision(self, batch, state: 'SimulationState') -> bool:
        pass


class WaitingOrderCountPolicy(WaitingPolicy):
    def __init__(self, min_orders: int, policy_name: str = "WaitingOrderCount"):
        super().__init__(policy_name)
        self.min_orders = min_orders

    def make_decision(self, state: 'SimulationState', batch: Optional['Batch'] = None) -> bool:
        batch_ready = bool(batch and len(batch.orders) >= self.min_orders and any(v.available for v in state.vehicles)
                           and not state.ready_batches)

        if batch:
            logger.sim(
                f'{state.current_time} | {self.__class__.__module__} | '
                f'{self.__class__.__bases__[0].__name__} | {self.__class__.__name__}: '
                f'Batch {batch.id} meets conditions to start: {batch_ready}')

        else:
            logger.sim(
                f'{state.current_time} | {self.__class__.__module__} | '
                f'{self.__class__.__bases__[0].__name__} | {self.__class__.__name__}: '
                f'No batch passed')

        return batch_ready


class StartImmediatelyPolicy(WaitingPolicy):
    """
    A waiting policy that starts a vehicle immediately when it becomes idle.
    Since there are no orders for the vehicle at this time, an initial route is needed.
    This policy uses an initial route strategy to create a route for the vehicle.

    The policy will stop creating immediate start events after the shift end time.
    """
    def __init__(self, initial_route_strategy, shift_end: datetime = None, policy_name: str = "StartImmediately"):
        super().__init__(policy_name)
        self.initial_route_strategy = initial_route_strategy
        self.shift_end = shift_end
        self._initial_order_cache = None

    def get_initial_route(self, current_time: datetime):
        if not self._initial_order_cache:
            self._initial_order_cache = self.initial_route_strategy.generate_dummy_order(current_time=current_time)

            logger.sim(
                f'{current_time} | {self.__class__.__module__} | '
                f'{self.__class__.__bases__[0].__name__} | {self.__class__.__name__}: '
                f'Initial nodes created in instance of Order class.')

        return self._initial_order_cache

    def make_decision(self, state: 'SimulationState', batch: Optional['Batch'] = None) -> Any:
        # Check if we've reached the shift end and if there is no open batch left
        #04.08.2025
        if self.shift_end and state.current_time >= self.shift_end and not batch:

            logger.sim(
            f'{state.current_time} | {self.__class__.__module__} | '
            f'{self.__class__.__bases__[0].__name__} | {self.__class__.__name__}: '
            f'Shift end reached, no more immediate starts.')

            return []

        # If there's an available vehicle and no ready batches, start the vehicle with an initial route
        available_vehicles = [v for v in state.vehicles if v.available]
        if available_vehicles and not state.ready_batches and batch:
            batch_ready = bool(batch and any(v.available for v in state.vehicles) and not state.ready_batches)

            logger.sim(
                f'{state.current_time} | {self.__class__.__module__} | '
                f'{self.__class__.__bases__[0].__name__} | {self.__class__.__name__}: '
                f'Batch {batch.id} meets conditions to start: {batch_ready}')

            return batch_ready

        elif available_vehicles and not state.ready_batches and not batch:

            logger.sim(
                f'{state.current_time} | {self.__class__.__module__} | '
                f'{self.__class__.__bases__[0].__name__} | {self.__class__.__name__}: '
                f'No batch with orders, but vehicle(s) available. Creating InitialRoute Event(s).')

            if isinstance(self.initial_route_strategy, type):
                self.initial_route_strategy = self.initial_route_strategy(warehouse=state.warehouse,
                                                        shift_end=self.shift_end)

            events = []
            for vehicle in available_vehicles:
                initial_order = self.get_initial_route(current_time=state.current_time)
                events.append(InitialRouteEvent(state.current_time, vehicle, initial_order))
            return events
        else:

            logger.sim(
                f'{state.current_time} | {self.__class__.__module__} | '
                f'{self.__class__.__bases__[0].__name__} | {self.__class__.__name__}: '
                f'No immediate start necessary.')

            return []




