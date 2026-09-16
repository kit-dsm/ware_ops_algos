from datetime import datetime
from typing import List
from Domain.entities import Order, Vehicle, Batch
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Event:
    event_counter = 0

    def __init__(self, time: datetime):
        self.time = time
        self.id = Event.event_counter
        Event.event_counter += 1

    def __eq__(self, other: 'Event'):
        """used for sorting events in heaps"""
        return self.time == other.time and self.id == other.id

    def __le__(self, other: 'Event'):
        return self.time <= other.time

    def __lt__(self, other: 'Event'):
        if self.time == other.time:
            return self.id < other.id
        return self.time < other.time

    def __ge__(self, other: 'Event'):
        return self.time >= other.time

    def __gt__(self, other: 'Event'):
        return self.time > other.time

    def handle(self, state: 'SimulationState') -> List['Event']:
        if self.time > state.current_time:
            state.current_time = self.time
        return []


class ActionEvent(Event):
    """
    Represents a decision-making point in the simulation.

    This event is responsible for making decisions about order processing,
    such as dispatching vehicles or rescheduling orders.

    Attributes:
        time (datetime): The time at which the action should be taken.
        order (Order): The order associated with this action.
        decision_mode (str): The type of decision to be made (e.g., "dispatch").
    """
    def __init__(self, time: datetime, entity, decision_mode: str, abort_event: bool = False):
        super().__init__(time)
        self.decision_mode = decision_mode
        self.entity = entity
        self.abort_event = abort_event

    def handle(self, state: 'SimulationState') -> List['Event']:
        """
        Handle the action event based on the decision mode.

        Args:
            state (SimulationState): The current state of the simulation.

        Returns:
            List[Event]: A list containing the event based on the decision or no event if no action is necessary.

        Side effects:
            - May change the availability status of resources.
            - Logs a warning if there were any complications.
        The events are actually handled in the engine via _make_decision
        """
        return []


class WaitingRequest(Event):

    def __init__(self, time: datetime):
        super().__init__(time)

    def handle(self, state: 'SimulationState') -> List['Event']:

        logger.sim(f'{state.current_time} | {self.__class__.__module__} | {self.__class__.__name__}')

        return [ActionEvent(self.time, None, "waiting")]


class InitialRouteEvent(Event):
    """
    Event triggered when a vehicle should start an initial route.
    This happens when a vehicle is idle and there are no orders to process.
    """
    def __init__(self, time: datetime, vehicle: Vehicle, dummy_order: Order):
        super().__init__(time)
        self.vehicle = vehicle
        self.dummy_order = dummy_order

    def handle(self, state: 'SimulationState') -> List['Event']:

        dummy_batch = Batch(arrival_time=self.time, due_time=self.dummy_order.due_time,
                            dummy_batch=self.dummy_order.dummy_order)
        dummy_batch.orders.append(self.dummy_order)

        logger.sim(f'{state.current_time} | {self.__class__.__module__} | {self.__class__.__name__}: '
            f'Initial Route created in dummy batch. Create BatchReady Event.')

        return [BatchReady(self.time, dummy_batch)]


class BatchAbort(Event):
    def __init__(self, time: datetime):
        super().__init__(time)

    def handle(self, state: 'SimulationState') -> List['Event']:

        batcher = state.batcher
        flushed_batched = batcher.make_decision(None, None, abort_event=True)
        if flushed_batched:
            flushed_batch = flushed_batched['batch']

            logger.sim(f'{state.current_time} | {self.__class__.__module__} | {self.__class__.__name__}: '
                        f'Batch aborted, Create BatchReady Event')

            return [BatchReady(self.time, flushed_batch)]
        else:
            # if no batch is present, no event is created
            return []


class OrderArrival(Event):
    """
    Represents the arrival of an order in the simulation.

    This event is triggered when a new order enters the system and needs to be processed.
    It generates an ActionEvent to handle the decision-making process for the order.

    Attributes:
        time (datetime): The time at which the order arrives.
        order (Order): The order that has arrived.
    """
    def __init__(self, time: datetime, order: Order):
        """
        Initialize an OrderArrival event.

        Args:
            time (datetime): The time at which the order arrives.
            order (Order): The order that has arrived.
        """
        super().__init__(time)
        self.order = order

    def handle(self, state: 'SimulationState') -> List[Event]:
        """
        Handle the arrival of an order.

        This method logs the arrival of the order and creates an ActionEvent
        to handle the decision-making process for batching.

        Args:
            state (SimulationState): The current state of the simulation.

        Returns:
            List[Event]: A list containing an ActionEvent for dispatch decision-making.
        """
        logger.sim(f'{state.current_time} | {self.__class__.__module__} | {self.__class__.__name__}: {self.order.id}')

        return [ActionEvent(self.time, self.order, "batching")]


class BatchReady(Event):
    def __init__(self, time: datetime, batch) -> None:
        super().__init__(time)
        self.batch = batch

    def handle(self, state: 'SimulationState') -> List[Event]:

        logger.sim(f'{state.current_time} | {self.__class__.__module__} | {self.__class__.__name__}: '
                    f'Batch {self.batch.id} with orders {[o.id for o in self.batch.orders]}')

        state.ready_batches.append(self.batch)
        return [ActionEvent(self.time, None, "dispatching")]


class PickStart(Event):
    def __init__(self, time: datetime, vehicle: Vehicle, batch):
        super().__init__(time)
        self.vehicle = vehicle
        self.batch = batch

    def handle(self, state: 'SimulationState') -> List[Event]:
        self.batch.pick_start_time = self.time
        self.batch.waiting_duration = self.time - state.last_pick_complete_time

        self.vehicle.transported_batch = self.batch

        logger.sim(f'{state.current_time} | {self.__class__.__module__} | {self.__class__.__name__}: '
                    f'Vehicle {self.vehicle.id} started batch {self.batch.id}')

        return [Travel(self.time, self.vehicle, self.batch)]


class Travel(Event):
    def __init__(self,time: datetime, vehicle: Vehicle, batch):
        super().__init__(time)
        self.vehicle = vehicle
        self.batch = batch

    def handle(self, state: 'SimulationState') -> List['Event']:

        logger.sim(f'{state.current_time} | {self.__class__.__module__} | {self.__class__.__name__}: '
                    f'Routing calculation for vehicle {self.vehicle.id} with batch {self.batch.id} started')

        return [ActionEvent(self.time, (self.batch, self.vehicle), "routing")]


class PickComplete(Event):
    def __init__(self, time: datetime, vehicle: Vehicle, batch):
        super().__init__(time)
        self.vehicle = vehicle
        self.batch = batch
        self.cancelled = False

    def handle(self, state: 'SimulationState') -> List[Event]:

        if self.cancelled:

            logger.sim(f'{state.current_time} | {self.__class__.__module__} | {self.__class__.__name__}: '
                        f'Event of batch {self.batch.id} cancelled, skip handling.')

            return []

        for order in self.batch.orders:
            order.fulfill(self.time)
            state.completed_orders.append(order)

        state.completed_batches.append(self.batch)
        self.batch.fulfill(self.time)
        self.batch.pick_complete_time = self.time
        self.batch.tour_duration = self.time - self.batch.pick_start_time

        state.add_batch_statistics(self.batch)
        state.add_order_statistics(self.vehicle.id, self.batch)

        logger.sim(f'{state.current_time} | {self.__class__.__module__} | {self.__class__.__name__}: '
                    f'Vehicle {self.vehicle.id} with batch {self.batch.id}')

        self.vehicle.available = True
        self.vehicle.transported_batch = None
        self.vehicle.location = state.warehouse.start_node

        state.last_pick_complete_time = self.time

        if state.waiting and not state.ready_batches:
            return [WaitingRequest(self.time)]

        else:
            # Standard: Fahrzeug wird wieder zum allgemeinen Dispatching freigegeben.
            return [ActionEvent(self.time, (None, self.vehicle), "dispatching")]