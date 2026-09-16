from typing import List

from algorithm_timeWindow import (
    TimeWindowAlgorithm,
    Warehouse,
    RoutingAlgorithm,
    Batch,
)


class TimeWindowFixArrivePicker(TimeWindowAlgorithm):
    """
    Concrete time window algorithm with a trivial (always-true) time window rule.

    This is the Python equivalent of the Java class:
    `timeWindow_fix_arrivePicker extends algoritmos_timeWindow`.

    Conceptual behavior
    -------------------
    - The algorithm does not impose any time-based restriction.
    - Every call to `run` returns True, regardless of:
        * the current time,
        * the content of the batch list,
        * the batch to be routed.

    Interpretation:
    - It can be seen as a "no time window" policy: the picker is assumed to be
      always allowed to start (fixed arrival), or the arrival constraints are
      ignored.
    - Often used as:
        * a baseline strategy in comparisons,
        * a default/fallback when no specific time window algorithm is selected,
        * a simple test implementation to verify the framework.
    """

    def __init__(self, warehouse: Warehouse, routing_algorithm: RoutingAlgorithm) -> None:
        """
        Initialize the algorithm with the given warehouse and routing algorithm.

        :param warehouse: The warehouse context (layout, distances, etc.).
        :param routing_algorithm: The routing algorithm used for route computation.
        """
        # Delegate to the abstract base class constructor:
        # it will store `warehouse` and `routing_algorithm` as attributes.
        super().__init__(warehouse, routing_algorithm)

    def run(self, batches: List[Batch], batch_to_route: Batch) -> bool:
        """
        Execute the time window check for a given batch.

        This method is the Python counterpart of the Java method:

            public boolean run(List<Batch> LB, Batch batchToRouting) throws Exception

        Behavior:
        - Always returns True.
        - Does not inspect 'batches' or 'batch_to_route'.
        - Does not modify any state (no side effects).

        :param batches: List of existing batches in the system.
                        NOTE: This parameter is not used in this implementation,
                        but is part of the common interface defined by the
                        TimeWindowAlgorithm base class.
        :param batch_to_route: The batch to be routed/scheduled.
                               NOTE: Also not used in this implementation.
        :return: Always True, indicating that the batch is allowed to be routed
                 immediately (no time window restriction).
        :raises Exception: Included for API compatibility with the Java version.
                           In this implementation, no exception is thrown.
        """
        # No conditions, no state changes: always allow the batch.
        return True