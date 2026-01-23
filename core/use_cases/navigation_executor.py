from typing import List, Optional
from ..entities.task import Task, TaskStatus
from ..entities.drone import DroneState
from ..interfaces.drone_repository import DroneRepository
from ..entities.map import GridMap
from .drone_control import DroneControlUseCase
from .navigation import NavigationUseCase

class NavigationExecutor:
    """
    Coordinates navigation missions by decomposing them into atomic actions.
    """
    def __init__(self, 
                 drone_repo: DroneRepository,
                 drone_control: DroneControlUseCase,
                 navigation: NavigationUseCase,
                 grid_map: GridMap):
        self.drone_repo = drone_repo
        self.drone_control = drone_control
        self.navigation = navigation
        self.grid_map = grid_map
        # In a real app, we'd have a TaskRepository too.
        # For this scope, we'll assume tasks are passed in or we have a simple lookup if we added TaskRepository.
        # But the prompt didn't ask for TaskRepository interface, so I'll assume we pass Task object or manage it here.
        # I'll add a method `execute_task` that takes a Task object.

    def execute_task(self, drone_id: str, task: Task) -> bool:
        """
        Executes a task for a given drone.
        1. Plan path.
        2. Takeoff.
        3. Traverse path.
        4. Land (optional, depends on task type, assume yes for now or just reach target).
        """
        drone = self.drone_repo.get_drone(drone_id)
        if not drone:
            print(f"Drone {drone_id} not found.")
            return False

        if not task.position:
            print("Task has no target position.")
            return False

        # 1. Plan Path
        print(f"Planning path for Drone {drone_id} from {drone.position} to {task.position}...")
        path = self.navigation.compute_path(drone.position, task.position, self.grid_map)
        
        if not path:
            print("No path found.")
            task.status = TaskStatus.FAILED
            return False
            
        task.mark_in_progress(drone_id)
        
        # 2. Takeoff (if needed)
        # Assuming safe altitude is part of task or fixed. Let's say 10 meters.
        target_altitude = task.position.z if task.position.z > 0 else 10.0
        
        if drone.state == DroneState.LANDED:
            print(f"Taking off to {target_altitude}m...")
            if not self.drone_control.takeoff(drone_id, target_altitude):
                print("Takeoff failed.")
                task.status = TaskStatus.FAILED
                return False
                
        # 3. Traverse Path
        print(f"Traversing path with {len(path)} waypoints...")
        for waypoint in path:
            # We might want to update heading too
            # For now, just move
            if not self.drone_control.move_line(drone_id, waypoint):
                print(f"Failed to move to waypoint {waypoint}.")
                task.status = TaskStatus.FAILED
                return False
                
        # 4. Completion
        print("Target reached.")
        task.mark_completed()
        
        # Optional: Land?
        # self.drone_control.land(drone_id)
        
        return True
