from abc import ABC, abstractmethod
from typing import List, Optional
from ..entities.drone import Drone

class DroneRepository(ABC):
    """
    Abstract Interface for Drone data access.
    """
    
    @abstractmethod
    def get_drone(self, drone_id: str) -> Optional[Drone]:
        """Retrieve a drone by its ID."""
        pass

    @abstractmethod
    def save_drone(self, drone: Drone) -> None:
        """Save or update a drone's state."""
        pass

    @abstractmethod
    def get_all_drones(self) -> List[Drone]:
        """Retrieve all drones."""
        pass
