# time_window_henn.py

from typing import List
import time as _time

from algorithm_timeWindow import (
    TimeWindowAlgorithm,
    Warehouse,
    RoutingAlgorithm,
    Batch,
)


# Stubs for Order and BatchCollected, to illustrate structure.
# TODO: remove these stubs and import the actual implementations
class Order:
    """
    Placeholder representation of an Order.

    Expected attributes based on the Java code:
    - tiempo_llegada: arrival time (e.g., earliest start time or release time).
    - serviceTime: service time required to process the order.
    """
    def __init__(self, arrival_time: float, service_time: float) -> None:
        self.arrival_time = arrival_time
        self.service_time = service_time


class BatchCollected:
    """
    Placeholder representation of BatchCollected.

    In the Java code, it is constructed from `Batch` and `Warehouse`, and
    has a `routing(algoritmos)` method returning a double (routing time).
    #TODO: Adjust to actual implementation
    """

    def __init__(self, batch: Batch, warehouse: Warehouse) -> None:
        self.batch = batch
        self.warehouse = warehouse

    def routing(self, routing_algorithm: RoutingAlgorithm) -> float:
        """
        Compute the routing time for the batch using the given routing algorithm.

        In the real application, this should:
        - build a route for the batch within the warehouse,
        - return its travel time or distance as a floating-point number.

        Here, it is a stub.

        :param routing_algorithm: Routing algorithm used to evaluate routes.
        :return: Routing time/distance as a float.
        """
        # TODO: Replace this stub with the actual routing computation.
        return 0.0


# We also assume the Batch class provides method:
#   getOrderMaxserviceTime(wh, alg_routing) -> Order
# so we do not implement it here, only rely on its existence.


class TimeWindowHenn(TimeWindowAlgorithm):
    """
    Henn-based time window algorithm.

    Python equivalent of the Java class:
        timeWindow_henn extends algoritmos_timeWindow.

    Conceptual behavior
    -------------------
    The release decision depends on:
    - how many batches currently exist (len(batches)),
    - the order with the maximum service time in the single existing batch
      (when there is exactly one batch),
    - the routing time for the candidate batch (`batch_to_route`),
    - and a tuning parameter `alpha`.

    Cases:
    - If no batches exist (len(batches) == 0):
        -> return False (always wait).
    - If exactly one batch exists:
        -> Let b be the only batch.
        -> Let oi be the order in b with the maximum service time:
             oi = b.getOrderMaxserviceTime(warehouse, routing_algorithm)
        -> Compute routing time for the new batch:
             BCo = BatchCollected(batch_to_route, warehouse)
             t_routing = BCo.routing(routing_algorithm)
        -> Compute threshold:
             threshold = (1 + alpha) * oi.tiempo_llegada \
                         + alpha * oi.serviceTime \
                         - t_routing
        -> Let current_time be the current system time (ms since epoch).
        -> return current_time > threshold
    - If more than one batch exists:
        -> return True (no more waiting).

    Input:
    - List[Batch] batches: existing batches in the system.
    - Batch batch_to_route: the batch for which we want to decide release.

    Output:
    - bool: True if release allowed, False otherwise.
    """

    def __init__(
        self,
        warehouse: Warehouse,
        routing_algorithm: RoutingAlgorithm,
        alpha: float,
    ) -> None:
        """
        Initialize the Henn time window algorithm.

        :param warehouse: Warehouse context.
        :param routing_algorithm: Routing algorithm for evaluating routes.
        :param alpha: Tuning parameter controlling the weight of arrival and
                      service times in the threshold formula.
        """
        super().__init__(warehouse, routing_algorithm)

        # Current time (in ms) stored during each run; not strictly needed as a field,
        # but kept for structural similarity with the Java code.
        self.current_time_ms: float = 0.0

        # Henn parameter alpha
        self.alpha: float = alpha

    def run(self, batches: List[Batch], batch_to_route: Batch) -> bool:
        """
        Perform the Henn time window decision.

        This is the Python equivalent of the Java method:
            public boolean run(List<Batch> LB, Batch batchToRouting) throws Exception

        Waiting strategy:
        - If there are no existing batches:
            return False.
        - If there is exactly one existing batch:
            * Identify the order with the maximum service time.
            * Compute routing time for `batch_to_route`.
            * Compute the threshold based on arrival time, service time, alpha,
              and routing time.
            * Release if current_time_ms > threshold.
        - If there are more than one existing batches:
            return True.

        :param batches: List of existing batches in the system.
        :param batch_to_route: Batch that is being considered for release/routing.
        :return: True if release is allowed according to Henn's rule, False otherwise.
        :raises Exception: Included for compatibility with the Java signature.
                           This method does not explicitly raise an exception.
        """
        # Current time in milliseconds (similar to System.currentTimeMillis())
        self.current_time_ms = int(_time.time() * 1000)

        # Case 1: no batches -> always wait
        if not batches:
            return False

        # Case 2: exactly one batch -> apply Henn-based time formula
        if len(batches) == 1:
            # Single existing batch
            b = batches[0]

            # Get order with maximum service time from this batch
            # (Assuming this method exists on the Batch class, as in Java.)
            oi: Order = b.getOrderMaxserviceTime(self.warehouse, self.routing_algorithm)

            # Create BatchCollected wrapper for the batch_to_route and compute routing time
            b_collected = BatchCollected(batch_to_route, self.warehouse)
            t_routing: float = b_collected.routing(self.routing_algorithm)

            # Compute threshold according to Henn formula:
            #
            # threshold = ((1 + alpha) * oi.tiempo_llegada)
            #             + (alpha * oi.serviceTime)
            #             - t_routing
            threshold = ((1.0 + self.alpha) * oi.arrival_time) \
                        + (self.alpha * oi.service_time) \
                        - t_routing

            # Release allowed if current time is greater than threshold
            return self.current_time_ms > threshold

        # Case 3: two or more batches -> always release
        return True