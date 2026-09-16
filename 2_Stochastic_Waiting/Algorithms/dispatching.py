import logging
from typing import List, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class DispatchingPolicy(ABC):
    """
    Abstract base class for vehicle dispatching policies.

    This class defines the interface for dispatching policies used in the simulation.
    Subclasses should implement the dispatch method to define specific dispatching strategies.

    Attributes:
        strategy_name (str): A string identifier for the dispatching strategy.
    """

    def __init__(self, strategy_name: str):
        """
        Initialize the dispatching policy.

        Args:
            strategy_name (str): A string identifier for the dispatching strategy.
        """
        self.strategy_name = strategy_name

    @abstractmethod
    def make_decision(self, vehicles):
        """
        Abstract method to dispatch a vehicle.

        This method should be implemented by subclasses to define the specific
        dispatching logic for selecting an available vehicle.

        Args:
            vehicles (List[Vehicle]): A list of all vehicles in the simulation.

        Returns:
            Optional[Vehicle]: The selected vehicle, or None if no vehicle is available.

        Raises:
            NotImplementedError: If this method is not implemented by a subclass.
        """
        pass


class GreedyDispatcher(DispatchingPolicy):
    """
    A greedy dispatching policy that selects the first available vehicle.

    This policy implements a simple greedy strategy where it selects the first
    vehicle that is marked as available in the provided list of vehicles.

    Attributes:
        strategy_name (str): The name of the strategy, default is "greedy".
    """

    def __init__(self, strategy_name: str = "greedy"):
        """
        Initialize the GreedyDispatcher.

        Args:
            strategy_name (str, optional): The name of the strategy. Defaults to "greedy".
        """
        super().__init__(strategy_name)

    def make_decision(self, vehicles):
        """
        Dispatch a vehicle using the greedy strategy.

        This method selects the first available vehicle from the provided list.
        If a vehicle is selected, it is immediately marked as unavailable.

        Args:
            vehicles (List[Vehicle]): A list of all vehicles in the simulation.

        Returns:
            Optional[Vehicle]: The first available vehicle, or None if no vehicle is available.

        Example:
            dispatcher = GreedyDispatcher()
            selected_vehicle = dispatcher.dispatch(list_of_vehicles)
            if selected_vehicle:
                print(f"Vehicle {selected_vehicle.id} has been dispatched")
            else:
                print("No vehicle available")
        """
        available_vehicle = next((v for v in vehicles if v.available), None)
        if available_vehicle:
            return available_vehicle
        else:
            return None