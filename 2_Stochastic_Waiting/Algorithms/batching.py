from Utils import logging_config
import logging
from typing import List, Optional
from abc import ABC, abstractmethod
from Domain.entities import Batch

logger = logging.getLogger(__name__)

class BatchingPolicy(ABC):
    def __init__(self, strategy_name: str, max_capacity: int):
        self.strategy_name = strategy_name
        self.max_capacity = max_capacity

    @abstractmethod
    def make_decision(self, order, state, abort_event: Optional[bool] = False):
        pass

class FIFOBatching(BatchingPolicy):
    def __init__(self, strategy_name: str = "FIFO", max_capacity: int = 4):
        super().__init__(strategy_name, max_capacity)
        self.current_batch = None


    def make_decision(self, order=None, state = None, abort_event: Optional[bool] = False):

        order_fits = None
        if state and state.waiting:
            for vehicle in state.vehicles:
                if not vehicle.available and vehicle.transported_batch:
                    vehicle.transported_batch.max_capacity = self.max_capacity
                    if (len(vehicle.transported_batch.orders) < vehicle.transported_batch.max_capacity or
                            (vehicle.transported_batch.dummy_batch == True and len(vehicle.transported_batch.orders) <=
                             vehicle.transported_batch.max_capacity)):

                        logger.sim(
                            f'{state.current_time} | {self.__class__.__module__} | '
                            f'{self.__class__.__bases__[0].__name__} | {self.__class__.__name__}: '
                            f'Vehicle {vehicle.id} has capacity left in current batch {vehicle.transported_batch.id}.')

                        batch = vehicle.transported_batch

                        if hasattr(batch, "routing_result"):
                            travel_tour = batch.routing_result['tour']

                            current_pos = vehicle.location
                            current_index = travel_tour.index(current_pos)
                            remaining_route = travel_tour[current_index:]

                            for pick_pos in order.items.keys():
                                article_loc = state.warehouse.allocation.get(pick_pos)
                                if article_loc not in remaining_route:
                                    order_fits = False

                                    logger.sim(
                                        f'{state.current_time} | {self.__class__.__module__} | '
                                        f'{self.__class__.__bases__[0].__name__} | {self.__class__.__name__}: '
                                        f'Article at position {article_loc} not on remaining route')
                                    break
                                else:
                                    order_fits = True

                                    logger.sim(
                                        f'{state.current_time} | {self.__class__.__module__} | '
                                        f'{self.__class__.__bases__[0].__name__} | {self.__class__.__name__}: '
                                        f'Article at position {article_loc} on remaining route')

                            if order_fits == True:
                                vehicle.transported_batch.orders.append(order)

                                logger.sim(
                                    f'{state.current_time} | {self.__class__.__module__} | '
                                    f'{self.__class__.__bases__[0].__name__} | {self.__class__.__name__}: '
                                    f'Current batch {batch.id} now contains {[o.id for o in batch.orders]}')

                                return {'batch': batch, 'vehicle': vehicle}

                        else:

                            logger.sim(
                                f'{state.current_time} | {self.__class__.__module__} | '
                                f'{self.__class__.__bases__[0].__name__} | {self.__class__.__name__}: '
                                f'No routing results found in {vehicle.transported_batch.id}. '
                                f'Check for remaining route failed.')

        if order_fits == False or not order_fits:
            if order:
                if self.current_batch is None:

                    logger.sim(
                        f'{state.current_time} | {self.__class__.__module__} | '
                        f'{self.__class__.__bases__[0].__name__} | {self.__class__.__name__}: '
                        f'Creating new batch for Order {order.id}')

                    self.current_batch = Batch(arrival_time=order.arrival_time,
                                               due_time=order.due_time,
                                               max_capacity=self.max_capacity)
                else:

                    logger.sim(
                        f'{state.current_time} | {self.__class__.__module__} | '
                        f'{self.__class__.__bases__[0].__name__} | {self.__class__.__name__}: '
                        f'Adding Order {order.id} to existing batch {self.current_batch.id}')

                self.current_batch.add_order(order)

                logger.sim(
                    f'{order.arrival_time} | {self.__class__.__module__} | '
                    f'{self.__class__.__bases__[0].__name__} | {self.__class__.__name__}: '
                    f'Current batch {self.current_batch.id} now contains {[o.id for o in self.current_batch.orders]}')

            if self.current_batch and (self.current_batch.is_full() or abort_event):
                completed_batch = self.current_batch
                self.current_batch = None

                if abort_event:
                    logger.sim(
                        f'{completed_batch.arrival_time} | {self.__class__.__module__} | '
                        f'{self.__class__.__bases__[0].__name__} | {self.__class__.__name__}: '
                        f'Batch {completed_batch.id} is aborted.')

                else:
                    logger.sim(
                        f'{completed_batch.arrival_time} | {self.__class__.__module__} | '
                        f'{self.__class__.__bases__[0].__name__} | {self.__class__.__name__}: '
                        f'Batch {completed_batch.id} is full.')

                return {'batch': completed_batch}
            return None
        else: return None

    def flush_current_batch(self):
        if self.current_batch and self.current_batch.orders:
            batch = self.current_batch
            self.current_batch = None
            return batch
        return None
