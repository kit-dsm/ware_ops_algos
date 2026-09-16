import copy
import heapq
import logging
from datetime import datetime, timedelta
from typing import List, Optional
import os
import pandas as pd
from Domain.entities import Vehicle, Order, Warehouse, Batch
from Domain.events import (Event, ActionEvent, PickStart, Travel, BatchAbort,
                           BatchReady, WaitingRequest, PickComplete, OrderArrival)

from Utils import utils
from Algorithms.batching import FIFOBatching, BatchingPolicy
from Algorithms.dispatching import GreedyDispatcher, DispatchingPolicy
from Algorithms.routing import Routing
from Algorithms.waiting import WaitingPolicy, WaitingOrderCountPolicy, StartImmediatelyPolicy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def reset_global_counters():
    """
    Reset global counters used in the simulation.

    This function resets the event counter in the Event class and the batch counter
    in the Batch class. It should be called before starting a new simulation to ensure
    that event and batch IDs start from 1.

    Returns:
        None
    """
    Event.event_counter = 0
    Batch.reset_counter()


class SimulationState:
    def __init__(self, warehouse: Warehouse, vehicles: List[Vehicle], batcher:BatchingPolicy,
                 dispatcher:DispatchingPolicy, router: Routing, waiting: Optional[WaitingPolicy] = None):
        self.warehouse = warehouse
        self.vehicles = vehicles
        self.dispatcher = dispatcher
        self.batcher = batcher
        self.current_time = datetime(2025, 1, 1, 0, 0)
        self.completed_orders = []
        self.ready_batches = []
        self.completed_batches = []
        self.order_statistics = [['batch_id', 'order_id', 'items', 'pick_nodes', 'vehicle_id', 'arrival_time',
                                  'fulfilled_at', 'due_time', 'completion_time', 'tardiness', 'dummy_order']]
        self.batch_statistics =[['batch_id', 'tour_length', 'waiting_duration', 'tour_duration', 'total_time',
                                 'nr_orderlines', 'nr_items', "time_per_node", 'dummy_batch']]
        self.router = router
        self.last_pick_complete_time = self.current_time
        self.waiting = waiting
        self.initial_route = None

    def add_order_statistics(self, vehicle_id: int, batch: Batch):
        for order in batch.orders:
            pick_nodes = []
            for item_id, amount in order.items.items():
                pick_node = self.warehouse.allocation.get(item_id)
                pick_nodes.append(pick_node)

            if order.fulfilled_at:
                completion_time = (order.fulfilled_at - order.arrival_time).total_seconds()
                tardiness = max(0, (order.fulfilled_at - order.due_time).total_seconds())
            else:
                logger.sim(f"No order.fulfilled_at {order.fulfilled_at}")
            self.order_statistics.append([
                batch.id,
                order.id,
                order.items,
                pick_nodes,
                vehicle_id,
                order.arrival_time.strftime("%Y-%m-%d %H:%M:%S"),
                order.fulfilled_at.strftime("%Y-%m-%d %H:%M:%S") if order.fulfilled_at else None,
                order.due_time.strftime("%Y-%m-%d %H:%M:%S"),
                completion_time if order.fulfilled_at else None,
                tardiness if order.fulfilled_at else None,
                order.dummy_order
            ])

    def add_batch_statistics(self, batch):
        waiting_sec = batch.waiting_duration.total_seconds() if batch.waiting_duration else 0
        tour_sec = batch.tour_duration.total_seconds() if batch.tour_duration else 0
        tour_length = batch.routing_result['tour_length'] if batch.routing_result['tour_length'] else 0

        total_items_count = 0
        total_items_sum = 0

        for order in batch.orders:
            if order.dummy_order: continue
            total_items_count += len(order.items)
            total_items_sum += len(order.items.values())

        if waiting_sec and tour_sec: total_time = waiting_sec + tour_sec

        elif tour_sec: total_time = tour_sec

        else: total_time = waiting_sec

        self.batch_statistics.append([
            batch.id,
            tour_length,
            waiting_sec,
            tour_sec,
            total_time,
            total_items_count,
            total_items_sum,
            total_time / total_items_count if total_items_count else None,
            batch.dummy_batch
        ])


class SimulationControl:
    def __init__(self):
        self.strategies = {}

    def register_strategy(self, decision_type: str, policy):
        self.strategies[decision_type] = policy


class SimulationEngine:
    """
    Manages the event queue and drives the simulation forward.

    The SimulationEngine is responsible for maintaining the heap of events,
    adding new events to the simulation, and processing events in chronological order.
    It works in conjunction with the SimulationState to update the state of the
    simulation as events are processed.

    Attributes:
        state (SimulationState): The current state of the simulation.
        events (List[Event]): A heap queue of events to be processed.
    """
    def __init__(self, state: SimulationState, control: SimulationControl, vis: bool = False):
        self.state = state
        self.control = control
        self.events = []
        self.log = []
        self.vis = vis


    def add_order(self, order: Order):
        """
        Add a new order to the simulation.

        This method creates an OrderArrival event for the given order and adds it
        to the event queue. The OrderArrival event is scheduled at the order's
        specified arrival time.

        Args:
            order (Order): The order to be added to the simulation.

        Side effects:
            - Creates a new OrderArrival event.
            - Adds the new event to the simulation's event queue.

        Example:
            engine = SimulationEngine(initial_state)
            new_order = Order(id=1, items=[(1, 2), (3, 1)], arrival_time=datetime.now(), due_time=datetime.now() + timedelta(hours=1))
            engine.add_order(new_order)
        """
        self.add_event(OrderArrival(order.arrival_time, order))


    def add_event(self, event: Event):
        """
        Adds a new event to the simulation's event queue.

        This method uses a heap queue (priority queue) to maintain the events
        in chronological order. The events are automatically sorted based on
        their occurrence time, ensuring that they will be processed in the
        correct temporal sequence during the simulation run.

        :param event: The Event object to be added to the simulation queue.
                      This should be an instance of a class derived from the
                      base Event class, containing at minimum a 'time' attribute.
        :type event: Event

        :return: None

        Example usage:
            engine = SimulationEngine(state)
            new_event = OrderArrival(datetime.now(), some_order)
            engine.add_event(new_event)
        """
        heapq.heappush(self.events, event)


    def run(self):
        """
        Executes the main simulation loop.

        This method runs the entire simulation by processing events in chronological order.
        It continues until all events in the queue have been handled. For each event:
        1. The event is removed from the queue.
        2. The simulation time is updated to the event's time.
        3. The event is handled, potentially generating new events.
        4. Any new events are added to the queue.

        After all events are processed, the method writes simulation statistics to a xlsx file.

        :return: None
        :rtype: NoneType

        Side effects:
        - Updates the simulation state (self.state)
        - Logs the start and completion of the simulation
        - Writes simulation statistics

        Example usage:
            engine = SimulationEngine(state)
            engine.run()
        """
        logger.sim(f"{self.state.current_time} | {self.__class__.__module__} | {self.__class__.__name__}: "
                    f"Starting simulation")

        if "waiting" in self.control.strategies and isinstance(self.control.strategies["waiting"], StartImmediatelyPolicy):
            self.add_event(WaitingRequest(self.state.current_time))

        while self.events:
            event = heapq.heappop(self.events)
            self.state.current_time = event.time
            self._update_vehicle_positions()
            if isinstance(event, ActionEvent):
                events_to_add = self._make_decision(event)
            else:
                events_to_add = event.handle(self.state)
            if events_to_add:
                for new_event in events_to_add:
                    self.add_event(new_event)

            if self.vis == True:
                if isinstance(event, (ActionEvent, PickComplete)):
                    self.log.append(copy.deepcopy({
                        "event": f"{str(event.id).zfill(5)} - {event.__class__.__name__}",
                        "time": self.state.current_time,
                        "pickers": self.state.vehicles
                    }))

        # Finaler Flush: Prüfen, ob noch ein unvollständiger Batch vorhanden ist
        batcher: FIFOBatching = self.control.strategies.get("batching")
        flushed_batch = batcher.flush_current_batch() if batcher else None
        if flushed_batch:

            logger.sim(f"{self.state.current_time} | {self.__class__.__module__} | {self.__class__.__name__}: "
                        f"No more orders to fulfill waiting criteria, flushing incomplete batch {flushed_batch.id}")

            flush_event = BatchReady(self.state.current_time, flushed_batch)
            for new_event in flush_event.handle(self.state):
                self.add_event(new_event)
            # Event-Queue erneut abarbeiten (falls noch Events generiert wurden)
            while self.events:
                event = heapq.heappop(self.events)
                self.state.current_time = event.time
                self._update_vehicle_positions()
                if isinstance(event, ActionEvent):
                    events_to_add = self._make_decision(event)
                else:
                    events_to_add = event.handle(self.state)
                for new_event in events_to_add:
                    self.add_event(new_event)

                if self.vis == True:
                    if isinstance(event, (ActionEvent, PickComplete)):
                        self.log.append(copy.deepcopy({
                            "event": f"{str(event.id).zfill(5)} - {event.__class__.__name__}",
                            "time": self.state.current_time,
                            "pickers": self.state.vehicles
                        }))

        logger.sim(f"{self.state.current_time} | {self.__class__.__module__} | {self.__class__.__name__}: "
                    f"Simulation complete")

        # Order-Daten speichern
        order_df = pd.DataFrame(self.state.order_statistics)
        order_df.columns = order_df.iloc[0]
        order_df = order_df[1:]

        # Batch-Daten speichern
        batch_df = pd.DataFrame(self.state.batch_statistics)
        batch_df.columns = batch_df.iloc[0]
        batch_df = batch_df[1:]

        return order_df, batch_df

    def _make_decision(self, event: ActionEvent):
        if event.decision_mode == "batching":
            return self.__create_event_on_batching_decision(event)

        elif event.decision_mode == "dispatching":
            return self.__create_event_on_dispatching_decision(event)

        elif event.decision_mode == "routing":
            return self.__create_event_on_routing_decision(event)

        elif event.decision_mode == "waiting":
            return self.__create_event_on_waiting_decision(event)

        return []

    def _update_vehicle_positions(self):
        for vehicle in self.state.vehicles:
            if vehicle.available == True:
                vehicle.location = self.state.warehouse.start_node

                logger.sim(f"{self.state.current_time} | {self.__class__.__module__} | {self.__class__.__name__} | "
                            f"Location Update: Vehicle {vehicle.id} in depot, position set to {vehicle.location}")


            elif not vehicle.available and vehicle.transported_batch:
                batch = vehicle.transported_batch
                if hasattr(batch, "routing_result") and hasattr(batch, "pick_start_time"):
                    routing_result = batch.routing_result
                    travel_tour = routing_result.get('tour')
                    time_to_pick = routing_result.get('time_to_pick', 1)
                    if travel_tour is not None:
                        if self.state.current_time == batch.pick_start_time:
                            current_pos = self.state.warehouse.start_node

                        else:
                            current_pos = utils.get_current_or_next_node(
                                tour=travel_tour,
                                orders=batch.orders,
                                dist_matrix=self.state.warehouse.dist_matrix,
                                time_to_pick=time_to_pick,
                                speed=vehicle.speed,
                                start_time=batch.pick_start_time,
                                current_time=self.state.current_time,
                                sku_storage_mapping=self.state.warehouse.allocation
                            )
                        vehicle.location = current_pos

                        logger.sim(
                            f"{self.state.current_time} | {self.__class__.__module__} | {self.__class__.__name__} | "
                            f"Location Update: Vehicle {vehicle.id} position set to {current_pos}")

                    else:
                        logger.sim(
                            f"{self.state.current_time} | {self.__class__.__module__} | {self.__class__.__name__} | "
                            f"Location Update: No tour found for vehicle {vehicle.id} in batch {batch.id}")

                else:
                    logger.sim(
                            f"{self.state.current_time} | {self.__class__.__module__} | {self.__class__.__name__} | "
                            f"Location Update: Missing routing_result or "
                            f"pick_start_time in batch {batch.id} for vehicle {vehicle.id}")

            else:
                logger.sim(f"{self.state.current_time} | {self.__class__.__module__} | {self.__class__.__name__} | "
                            f"Location Update: Conditions not met for vehicle {vehicle.id}")

    def __create_event_on_batching_decision(self, event: ActionEvent) -> List[Event]:

        batcher = self.control.strategies.get(event.decision_mode)

        batched = batcher.make_decision(event.entity, self.state)

        if batched:
            batch = batched['batch']
            if 'vehicle' in batched:
                vehicle = batched['vehicle']
                return [Travel(batch.pick_start_time, vehicle, batch)]
            return [BatchReady(event.time, batch)]
        else:
            return [WaitingRequest(event.time)]

    def __create_event_on_routing_decision(self, event: ActionEvent) -> List[Event]:
        batch, vehicle = event.entity
        routing_policy: Routing = self.control.strategies.get("routing")
        batched_list_df = utils.create_batched_list(batch, self.state.warehouse.allocation)
        picker_dict = [{"id": vehicle.id,
                       "speed": vehicle.speed,
                       }]

        router = routing_policy(network_graph = self.state.warehouse.graph,
                                batched_list = batched_list_df,
                                distance_matrix=self.state.warehouse.dist_matrix,
                                tour_matrix=self.state.warehouse.tour_matrix,
                                picker=picker_dict,
                                time_to_pick=1)

        # Verwende die Routingstrategie, die im SimulationState gespeichert ist.
        router.generate_routing()
        solution = router.complete_solutions[0]['results'][0]
        travel_time = solution['travel_time']
        travel_tour = solution['tour']
        travel_tour.append(self.state.warehouse.end_node)
        travel_tour_reverse = list(reversed(travel_tour))
        pick_nodes = batched_list_df["pick_node"].unique().tolist()

        batch.routing_result = {
            'tour': travel_tour_reverse,
            'pick_nodes': pick_nodes,
            'time_to_pick': router.time_to_pick,
            'tour_length': solution['distance']
        }

        logger.sim(f'{self.state.current_time} | {self.__class__.__module__} | {self.__class__.__name__} | '
                    f'Routing Decision: Route calculated with batch {batch.id} and '
                    f'vehicle {vehicle.id}. Calculated time: {travel_time} seconds.')

        pick_complete_time = event.time + timedelta(seconds=travel_time)

        if hasattr(batch, 'pick_complete_event') and batch.pick_complete_event is not None:
            batch.pick_complete_event.cancelled = True

        new_pick_complete_event = PickComplete(pick_complete_time, vehicle, batch)
        batch.pick_complete_event = new_pick_complete_event

        return [new_pick_complete_event]

    def __create_event_on_dispatching_decision(self, event: ActionEvent) -> List[Event]:
        events_to_add = []
        dispatcher: GreedyDispatcher = self.control.strategies.get("dispatching")
        # So lange Batches in der Warteschlange liegen und Fahrzeuge verfügbar sind, dispatchen wir
        while True:
            # Falls keine Batches warten, Ende
            if not self.state.ready_batches:
                break
            # GreedyDispatcher sucht das erste verfügbare Fahrzeug
            vehicle = dispatcher.make_decision(self.state.vehicles)
            if not vehicle:

                logger.sim(f'{event.time} | {self.__class__.__module__} | {self.__class__.__name__} | '
                            f'Dispatching Decision: No vehicle available.')

                # Kein Fahrzeug frei -> wir warten auf das nächste PickComplete
                break
            # Batch aus der Warteschlange holen

            logger.sim(f'{event.time} | {self.__class__.__module__} | {self.__class__.__name__} | '
                        f'Dispatching Decision: Vehicle {vehicle.id} dispatched.')

            batch = self.state.ready_batches.pop(0)
            vehicle.available = False
            events_to_add.append(PickStart(event.time, vehicle, batch))
        return events_to_add

    def __create_event_on_waiting_decision(self, event: ActionEvent) -> List[Event]:
        waiting_policy: WaitingPolicy = self.control.strategies.get("waiting")
        if waiting_policy:
            batcher = self.state.batcher
            if not batcher.current_batch:
                decision = waiting_policy.make_decision(self.state)

                if decision == True:
                    return [BatchAbort(event.time)]

                elif decision: return decision

                else: return []

            elif batcher.current_batch:
                batch_ready = waiting_policy.make_decision(self.state, batcher.current_batch)
                if batch_ready:

                    logger.sim(f'{self.state.current_time} | {self.__class__.__module__} | {self.__class__.__name__} | '
                                f'Waiting Decision: BatchAbort Event triggered.')

                    return [BatchAbort(event.time)]
                else:
                    return []
            else:

                logger.sim(f'{self.state.current_time} | {self.__class__.__module__} | {self.__class__.__name__} | '
                            f'Waiting Decision: No current batch available.')

                return []
        else:
            logger.sim(
                f'{self.state.current_time} | {self.__class__.__module__} | {self.__class__.__name__} | '
                f'Waiting Decision: No waiting strategy registered.')

            return []
