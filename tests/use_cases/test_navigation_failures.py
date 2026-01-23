import sys
import os
from unittest.mock import MagicMock

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from core.entities.task import Task, TaskStatus
from core.entities.drone import Drone, DroneState
from core.entities.position import Position
from core.use_cases.navigation_executor import NavigationExecutor

def test_no_path():
    print("Running test_no_path...")
    drone_repo = MagicMock()
    drone_control = MagicMock()
    navigation = MagicMock()
    grid_map = MagicMock()
    executor = NavigationExecutor(drone_repo, drone_control, navigation, grid_map)
    
    drone_id = "test_drone"
    drone = Drone(id=drone_id, position=Position(0, 0, 5), state=DroneState.HOVERING)
    drone_repo.get_drone.return_value = drone
    
    task = Task(id="task_fail_path", position=Position(10, 10, 5))
    
    # 模拟 navigation.compute_path 返回 None 或空列表
    navigation.compute_path.return_value = None
    
    result = executor.execute_task(drone_id, task)
    
    # 验证 execute_task 返回 False
    assert result is False
    # 验证 task.status 被标记为 TaskStatus.FAILED
    assert task.status == TaskStatus.FAILED
    print("test_no_path passed!")

def test_takeoff_failure():
    print("Running test_takeoff_failure...")
    drone_repo = MagicMock()
    drone_control = MagicMock()
    navigation = MagicMock()
    grid_map = MagicMock()
    executor = NavigationExecutor(drone_repo, drone_control, navigation, grid_map)
    
    drone_id = "test_drone"
    drone = Drone(id=drone_id, position=Position(0, 0, 0), state=DroneState.LANDED)
    drone_repo.get_drone.return_value = drone
    
    task = Task(id="task_fail_takeoff", position=Position(10, 10, 5))
    
    # 模拟 drone_control.takeoff 返回 False
    drone_control.takeoff.return_value = False
    
    result = executor.execute_task(drone_id, task)
    
    assert result is False
    # 验证代码立即停止，不执行后续的 compute_path
    navigation.compute_path.assert_not_called()
    assert task.status == TaskStatus.FAILED
    print("test_takeoff_failure passed!")

def test_move_failure():
    print("Running test_move_failure...")
    drone_repo = MagicMock()
    drone_control = MagicMock()
    navigation = MagicMock()
    grid_map = MagicMock()
    executor = NavigationExecutor(drone_repo, drone_control, navigation, grid_map)
    
    drone_id = "test_drone"
    drone = Drone(id=drone_id, position=Position(0, 0, 5), state=DroneState.HOVERING)
    drone_repo.get_drone.return_value = drone
    
    task = Task(id="task_fail_move", position=Position(10, 10, 5))
    
    path = [Position(1, 1, 5), Position(2, 2, 5), Position(3, 3, 5)]
    navigation.compute_path.return_value = path
    
    # 模拟在处理第 2 个点时 move_line 返回 False
    def move_line_side_effect(did, pos):
        if pos.x == 2: # Second point
            return False
        return True
    drone_control.move_line.side_effect = move_line_side_effect
    
    result = executor.execute_task(drone_id, task)
    
    assert result is False
    # 验证任务最终状态为 FAILED
    assert task.status == TaskStatus.FAILED
    # 验证 move_line 被调用了 2 次 (第一次成功，第二次失败)
    assert drone_control.move_line.call_count == 2
    print("test_move_failure passed!")

if __name__ == "__main__":
    try:
        test_no_path()
        test_takeoff_failure()
        test_move_failure()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
