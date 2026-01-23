from typing import List, Optional
from ..entities.task import Task, TaskStatus
from ..entities.drone import DroneState
from ..entities.position import Position
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

    def execute_task(self, drone_id: str, task: Task, default_altitude: float = 5.0) -> bool:
        """
        根据指定逻辑执行无人机任务。
        """
        drone = self.drone_repo.get_drone(drone_id)
        if not drone or not task.position:
            return False

        task.mark_in_progress(drone_id)
        target_z = task.position.z

        # --- 1. 判断是否为 LANDED，并执行 TAKEOFF ---
        if drone.state == DroneState.LANDED:
            # 如果 target.z > 0，则 takeoff 到 target.z，否则到默认高度
            takeoff_alt = target_z if target_z > 0 else default_altitude
            print(f"Drone {drone_id} taking off to {takeoff_alt}m...")
            if not self.drone_control.takeoff(drone_id, takeoff_alt):
                task.status = TaskStatus.FAILED
                return False

        # --- 2. 如果当前高度 < target.z，则调整高度至 target.z ---
        # 注意：这里是在原地进行垂直爬升，为后续水平导航提供安全高度
        if drone.position.z < target_z:
            print(f"Ascending: Adjusting altitude from {drone.position.z}m to {target_z}m...")
            climb_pos = Position(drone.position.x, drone.position.y, target_z)
            if not self.drone_control.move_line(drone_id, climb_pos):
                task.status = TaskStatus.FAILED
                return False

        # --- 3. 导航与路径追踪 ---
        # 在当前高度进行水平移动规划
        target_h = Position(task.position.x, task.position.y, drone.position.z)
        print(f"Planning path to ({target_h.x}, {target_h.y}) at altitude {target_h.z}m...")
        
        path = self.navigation.compute_path(drone.position, target_h, self.grid_map)
        if not path:
            print("No path found.")
            task.status = TaskStatus.FAILED
            return False

        print(f"Moving along path ({len(path)} waypoints)...")
        for waypoint in path:
            if not self.drone_control.move_line(drone_id, waypoint):
                task.status = TaskStatus.FAILED
                return False

        # --- 4. target.z > 0 且当前高度 > target.z，则调整高度至 target.z ---
        # 此时已到达目标点的上方，进行垂直下降
        if target_z > 0 and drone.position.z > target_z:
            print(f"Descending: Adjusting altitude from {drone.position.z}m to {target_z}m...")
            if not self.drone_control.move_line(drone_id, task.position):
                task.status = TaskStatus.FAILED
                return False

        # --- 5. 如果 target.z == 0，则执行 LAND ---
        if target_z == 0:
            print(f"Target Z is 0. Drone {drone_id} landing...")
            if not self.drone_control.land(drone_id):
                task.status = TaskStatus.FAILED
                return False

        print("Task execution completed successfully.")
        task.mark_completed()
        return True
