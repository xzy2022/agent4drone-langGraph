from abc import ABC, abstractmethod
from typing import List, Optional
from ..entities.position import Position
from ..entities.map import GridMap

class NavigationService(ABC):
    """
    Abstract Interface for Navigation capabilities.
    """

    @abstractmethod
    def compute_path(self, start: Position, end: Position, grid_map: GridMap) -> Optional[List[Position]]:
        """
        Compute a path from start to end using the provided map.
        Returns a list of Positions (waypoints) if a path is found, else None.
        """
        pass
