import sys
import os
from unittest.mock import MagicMock

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from core.entities.task import Task, TaskStatus
from core.entities.drone import Drone, DroneState
from core.entities.position import Position
from core.use_cases.navigation_executor import NavigationExecutor

def test_navigation_landing():
    print("Running test_navigation_landing...")
    
    # Mock dependencies
    drone_repo = MagicMock()
    drone_control = MagicMock()
    navigation = MagicMock()
    grid_map = MagicMock()
    
    executor = NavigationExecutor(drone_repo, drone_control, navigation, grid_map)
    
    drone_id = "test_drone"
    # 无人机初始状态为 HOVERING（悬停中），位置 (0, 0, 10)
    drone = Drone(id=drone_id, position=Position(0, 0, 10), state=DroneState.HOVERING)
    drone_repo.get_drone.return_value = drone
    
    # 任务目标位置为 (20, 20, 0) （Z = 0，意味着需要降落）
    target_pos = Position(20, 20, 0)
    task = Task(id="task_2", position=target_pos)
    
    # Mock horizontal path at altitude 10
    path = [Position(10, 10, 10), Position(20, 20, 10)]
    navigation.compute_path.return_value = path
    
    # Mock successful calls
    drone_control.move_line.return_value = True
    drone_control.land.return_value = True
    
    # Side effects to update drone position
    def move_line_side_effect(did, pos):
        drone.position = pos
        return True
    drone_control.move_line.side_effect = move_line_side_effect

    def land_side_effect(did):
        drone.position.z = 0
        drone.state = DroneState.LANDED
        return True
    drone_control.land.side_effect = land_side_effect

    # Execute
    result = executor.execute_task(drone_id, task)
    
    # 验证逻辑：
    assert result is True
    
    # 验证 不 应该调用 takeoff（因为已经在悬停）
    drone_control.takeoff.assert_not_called()
    
    # 验证先进行了水平移动
    navigation.compute_path.assert_called()
    assert drone_control.move_line.call_count == 2
    
    # 关键验证：在到达目标平面的上方后，必须调用了 drone_control.land(drone_id) 方法
    drone_control.land.assert_called_with(drone_id)
    
    # 验证任务最终状态为 COMPLETED
    assert task.status == TaskStatus.COMPLETED
    
    print("test_navigation_landing passed!")

if __name__ == "__main__":
    try:
        test_navigation_landing()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
