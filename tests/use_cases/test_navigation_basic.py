import sys
import os
from unittest.mock import MagicMock

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from core.entities.task import Task, TaskStatus
from core.entities.drone import Drone, DroneState
from core.entities.position import Position
from core.use_cases.navigation_executor import NavigationExecutor

def test_navigation_basic():
    print("Running test_navigation_basic...")
    
    # Mock dependencies
    drone_repo = MagicMock()
    drone_control = MagicMock()
    navigation = MagicMock()
    grid_map = MagicMock()
    
    executor = NavigationExecutor(drone_repo, drone_control, navigation, grid_map)
    
    drone_id = "test_drone"
    # 无人机初始状态为 LANDED，位置 (0, 0, 0)
    drone = Drone(id=drone_id, position=Position(0, 0, 0), state=DroneState.LANDED)
    drone_repo.get_drone.return_value = drone
    
    # 任务目标位置为 (10, 10, 5) （Z > 0）
    target_pos = Position(10, 10, 5)
    task = Task(id="task_1", position=target_pos)
    
    # navigation.compute_path 应该返回一个包含2个航点的列表
    path = [Position(5, 5, 5), Position(10, 10, 5)]
    navigation.compute_path.return_value = path
    
    # Mock successful calls
    drone_control.takeoff.return_value = True
    drone_control.move_line.return_value = True
    
    # Define side effect to update drone position and state when takeoff is called
    # To simulate the actual behavior so subsequent checks in executor work
    def takeoff_side_effect(did, alt):
        drone.position.z = alt
        drone.state = DroneState.IDLE # Or HOVERING/IDLE
        return True
    drone_control.takeoff.side_effect = takeoff_side_effect

    # Define side effect for move_line to update drone position
    def move_line_side_effect(did, pos):
        drone.position = pos
        return True
    drone_control.move_line.side_effect = move_line_side_effect

    # Execute
    result = executor.execute_task(drone_id, task)
    
    # 验证逻辑：
    # 验证 execute_task 返回 True
    assert result is True, "execute_task should return True"
    
    # 验证 drone_control.takeoff 被调用
    drone_control.takeoff.assert_called_with(drone_id, 5.0)
    
    # 验证 navigation.compute_path 被调用
    navigation.compute_path.assert_called()
    
    # 验证 drone_control.move_line 被调用了相应次数（针对路径点）
    # 在这个场景中，没有额外的爬升/下降，所以调用次数应该等于路径长度
    assert drone_control.move_line.call_count == 2, f"move_line should be called twice for waypoints, got {drone_control.move_line.call_count}"
    
    # 验证 task.status 最终变为 COMPLETED
    assert task.status == TaskStatus.COMPLETED, f"Task status should be COMPLETED, got {task.status}"
    
    print("test_navigation_basic passed!")

if __name__ == "__main__":
    try:
        test_navigation_basic()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
