# time_window_fix_batches.py

from typing import List

from algorithm_timeWindow import (
    TimeWindowAlgorithm,
    Warehouse,
    RoutingAlgorithm,
    Batch,
)


class TimeWindowFixBatches(TimeWindowAlgorithm):
    """
    Time window algorithm with a fixed batch-count-based release rule.

    Python equivalent of the Java class:
        `timeWindow_fix_batches extends algoritmos_timeWindow`.

    Conceptual behavior
    -------------------
    This algorithm does not use clock time directly. Instead, it uses the
    *number of batches* in the system as a trigger:

    - It is configured with an integer 'n_batches'.
    - On each call to `run`, it looks at the size of the batch list 'batches'
      (Java: LB.size()).
    - It returns True if `len(batches) > n_batches`, and False otherwise.

    Interpretation:
    - The algorithm enforces a *wait* until more than 'n_batches' batches
      exist in the system.
    - Once the number of batches exceeds the threshold, the algorithm allows
      the given batch to proceed or be considered for routing.

    In short:
    - "Do not proceed while the system has too few batches; only proceed once
      there are more than 'n_batches' batches."
    """

    def __init__(
        self,
        warehouse: Warehouse,
        routing_algorithm: RoutingAlgorithm,
        n_batches: int,
    ) -> None:
        """
        Initialize the fixed-batches time window algorithm.

        :param warehouse: Warehouse context (layout, distances, etc.).
        :param routing_algorithm: Routing algorithm used for route computation.
        :param n_batches: Threshold for the number of batches. The algorithm
                          will only return True if the number of existing
                          batches is strictly greater than this value.
        """
        super().__init__(warehouse, routing_algorithm)

        # Threshold for the number of batches
        self.n_batches: int = n_batches

    def run(self, batches: List[Batch], batch_to_route: Batch) -> bool:
        """
        Evaluate whether the current batch may proceed based on batch count.

        Python equivalent of the Java method:
            public boolean run(List<Batch> LB, Batch batchToRouting) throws Exception

        Waiting / release rule:
        - If len(batches) > self.n_batches:
            return True  (release is allowed)
        - Else:
            return False (must wait)

        :param batches: List of existing batches in the system.
                        Only the length of this list is used.
        :param batch_to_route: Batch that is being considered for routing or
                               scheduling. Not used in this implementation,
                               but part of the common interface defined by the
                               base class.
        :return: True if len(batches) > n_batches, False otherwise.
        :raises Exception: Included for API compatibility with the Java version.
                           In this implementation, no exception is raised.
        """
        # Check if the number of existing batches is greater than the threshold.

        # Note: The original Java code uses "LB.size() > n_batches"
        if len(batches) >= self.n_batches:
            return True
        else:
            return False