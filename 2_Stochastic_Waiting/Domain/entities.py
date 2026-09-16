from datetime import datetime, timedelta
from typing import Tuple, Optional
import networkx as nx
import pandas as pd

class Order:
    def __init__(self, id: int, items: dict, arrival_time: datetime, due_time: datetime, dummy_order: bool = False):
        self.id = id
        self.items = items  # List of (SKU, quantity) tuples
        self.arrival_time = arrival_time
        self.due_time = due_time
        self.fulfilled = False
        self.fulfilled_at = None
        self.dummy_order: bool = dummy_order

    def fulfill(self, time: datetime):
        self.fulfilled = True
        self.fulfilled_at = time


class Vehicle:
    def __init__(self, id: int, location: Tuple[int, int], speed: float = 1.0):
        self.id = id
        self.location = location
        self.speed = speed  # units per minute
        self.available = True
        self.transported_batch = None


class Warehouse:
    def __init__(self, allocation: dict, graph: nx.Graph, start_node: Tuple, end_node: Tuple, dist_matrix: pd.DataFrame,
                 tour_matrix: pd.DataFrame):
        self.allocation = allocation  # SKU to location mapping
        self.graph = graph
        self.start_node = start_node
        self.end_node = end_node
        self.dist_matrix = dist_matrix
        self.tour_matrix = tour_matrix

class Batch:
    batch_counter = 1

    def __init__(self, arrival_time: datetime, due_time: datetime, max_capacity: int = 4, dummy_batch: bool = False):
        self.id = Batch.batch_counter
        Batch.batch_counter += 1
        self.max_capacity = max_capacity
        self.orders = []
        self.arrival_time = arrival_time
        self.due_time = due_time
        self.fulfilled = False
        self.fulfilled_at = None
        self.pick_start_time: Optional[datetime] = None
        self.pick_complete_time: Optional[datetime] = None
        self.waiting_duration: Optional[timedelta] = None
        self.tour_duration: Optional[timedelta] = None
        self.tour_length: Optional[float] = None
        self.dummy_batch = dummy_batch

    def fulfill(self, time: datetime):
        self.fulfilled = True
        self.fulfilled_at = time

    def add_order(self, order) -> bool:
        if len(self.orders) < self.max_capacity:
            self.orders.append(order)
            return True
        return False

    def is_full(self) -> bool:
        return len(self.orders) >= self.max_capacity

    def __repr__(self):
        return f'Batch({self.orders})'

    @classmethod
    def reset_counter(cls):
        cls.batch_counter = 1