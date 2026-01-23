from typing import Optional
from ..entities.drone import Drone, DroneState
from ..entities.position import Position
from ..interfaces.drone_repository import DroneRepository

class DroneControlUseCase:
    """
    Application logic for controlling a drone.
    Orchestrates entity behavior and persistence.
    """
    
    def __init__(self, drone_repo: DroneRepository):
        self.drone_repo = drone_repo
        
    def takeoff(self, drone_id: str, altitude: float) -> bool:
        """Command a drone to take off to a specific altitude."""
        drone = self.drone_repo.get_drone(drone_id)
        if not drone:
            return False
            
        success = drone.takeoff(altitude)
        if success:
            self.drone_repo.save_drone(drone)
        return success
        
    def land(self, drone_id: str) -> bool:
        """Command a drone to land."""
        drone = self.drone_repo.get_drone(drone_id)
        if not drone:
            return False
            
        success = drone.land()
        if success:
            self.drone_repo.save_drone(drone)
        return success

    def move_line(self, drone_id: str, target_pos: Position) -> bool:
        """
        Move drone to a target position.
        Note: This is a discrete move/teleport for the high-level control.
        """
        drone = self.drone_repo.get_drone(drone_id)
        if not drone:
            return False
            
        success = drone.move(target_pos)
        if success:
            self.drone_repo.save_drone(drone)
        return success
        
    def hover(self, drone_id: str) -> bool:
        """Command drone to hover."""
        drone = self.drone_repo.get_drone(drone_id)
        if not drone:
            return False
        
        success = drone.hover()
        if success:
            self.drone_repo.save_drone(drone)
        return success
