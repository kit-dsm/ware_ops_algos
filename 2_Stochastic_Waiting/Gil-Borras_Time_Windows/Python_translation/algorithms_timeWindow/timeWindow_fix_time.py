# time_window_fix_time.py

from typing import List
import time  # for current time in seconds

from algorithm_timeWindow import (
    TimeWindowAlgorithm,
    Warehouse,
    RoutingAlgorithm,
    Batch,
)


# Stub for the Java 'helpers' class.
# In your real code, replace this with your actual helpers implementation.
class helpers:
    @staticmethod
    def calculate_objective_function_routing_distance(
        batch: Batch,
        warehouse: Warehouse,
        routing_algorithm: RoutingAlgorithm,
    ) -> int:
        """
        Stub for the Java method:
            helpers.calculateFuncionObjetivo_distancia_routing(batchToRouting, wh, alg_routing)

        In the real application, this should compute a time- or cost-based value
        (likely in milliseconds) derived from the routing distance/time needed
        to process 'batch' in 'warehouse' using 'routing_algorithm'.

        :param batch: Batch to evaluate.
        :param warehouse: Warehouse context.
        :param routing_algorithm: Routing algorithm.
        :return: Time/cost value in milliseconds.
        """
        # TODO: Replace with the real implementation.
        return 0


class TimeWindowFixTime(TimeWindowAlgorithm):
    """
    Time window algorithm with a fixed time-based release rule plus routing-dependent delay.

    Python equivalent of the Java class:
        timeWindow_fix_time extends algoritmos_timeWindow.

    Conceptual behavior
    -------------------
    The algorithm maintains an internal timestamp 'next_allowed_time' (in ms).
    On each call to 'run', it checks if the current time has passed this threshold:

    - If current_time_ms >= next_allowed_time:
        * The batch is allowed to be released (returns True).
        * 'next_allowed_time' is updated to:
              current_time_ms
            + n_minutes * 60,000 ms
            + routing_time_for(batch_to_route)

    - Otherwise:
        * The batch is not allowed yet (returns False).
        * 'next_allowed_time' remains unchanged.

    This results in a dynamic waiting strategy:
    - A fixed base waiting period (n_minutes),
    - plus an additional delay that depends on the routing distance/time of
      the last released batch.
    """

    def __init__(
        self,
        warehouse: Warehouse,
        routing_algorithm: RoutingAlgorithm,
        n_minutes: int,
    ) -> None:
        """
        Initialize the fixed-time time window algorithm.

        :param warehouse: Warehouse context.
        :param routing_algorithm: Routing algorithm used to evaluate route distances/times.
        :param n_minutes: Base waiting period in minutes.
        """
        super().__init__(warehouse, routing_algorithm)

        # Base waiting period in minutes
        self.n_minutes: int = n_minutes

        # Internal timestamp for the next allowed release (milliseconds since epoch).
        #
        # Java:
        #   this.time = System.currentTimeMillis() + (120000 * nMinutes);
        # where 120000 ms = 2 minutes.
        #
        # Python: time.time() returns seconds, so we convert to milliseconds.
        current_millis = int(time.time() * 1000)
        self.next_allowed_time: int = current_millis + (120000 * self.n_minutes)

    def run(self, batches: List[Batch], batch_to_route: Batch) -> bool:
        """
        Decide whether the given batch may be released at the current time.

        Python equivalent of the Java method:
            public boolean run(List<Batch> LB, Batch batchToRouting) throws Exception

        Waiting / release rule:
        - If current_time_ms >= self.next_allowed_time:
              self.next_allowed_time = current_time_ms
                                     + 60000 * n_minutes
                                     + routing_delay_ms(batch_to_route)
              return True
          else:
              return False

        :param batches: List of existing batches (not used by this algorithm,
                        but required by the interface).
        :param batch_to_route: Batch that is being considered for release/routing.
        :return: True if it is time to release the batch, False otherwise.
        :raises Exception: Included for interface compatibility with the Java
                           version. No explicit exception is raised here.
        """
        # Current time in ms, equivalent to Java's System.currentTimeMillis()
        current_millis = int(time.time() * 1000)

        # Check if the waiting period has elapsed
        if self.next_allowed_time <= current_millis:
            # Compute additional delay based on routing distance/time for the batch
            additional_delay_ms = helpers.calculate_objective_function_routing_distance(
                batch_to_route,
                self.warehouse,
                self.routing_algorithm,
            )

            # Update the internal timestamp for the next allowed release.
            #
            # Java:
            #   this.time = System.currentTimeMillis()
            #             + (60000 * nMinutes)
            #             + helpers.calculateFuncionObjetivo_distancia_routing(...)
            #
            # 60000 ms = 1 minute, so this is 'now + n_minutes minutes + routing_delay'.
            self.next_allowed_time = (
                current_millis
                + (60000 * self.n_minutes)
                + additional_delay_ms
            )

            # It is now allowed to release this batch.
            return True
        else:
            # Still within the waiting period; cannot release yet.
            return False