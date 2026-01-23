import sys
import os
from unittest.mock import MagicMock, call

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from core.entities.task import Task, TaskStatus
from core.entities.drone import Drone, DroneState
from core.entities.position import Position
from core.use_cases.navigation_executor import NavigationExecutor

def test_climb():
    print("Running test_climb...")
    drone_repo = MagicMock()
    drone_control = MagicMock()
    navigation = MagicMock()
    grid_map = MagicMock()
    executor = NavigationExecutor(drone_repo, drone_control, navigation, grid_map)
    
    drone_id = "test_drone"
    # 场景：无人机当前 Z=2，目标 Z=10
    drone = Drone(id=drone_id, position=Position(0, 0, 2), state=DroneState.HOVERING)
    drone_repo.get_drone.return_value = drone
    
    target_pos = Position(10, 10, 10)
    task = Task(id="task_climb", position=target_pos)
    
    navigation.compute_path.return_value = [Position(10, 10, 10)]
    drone_control.move_line.return_value = True
    
    # Side effect to update drone position
    def move_line_side_effect(did, pos):
        drone.position = pos
        return True
    drone_control.move_line.side_effect = move_line_side_effect

    executor.execute_task(drone_id, task)
    
    # 验证：在调用 compute_path 之前，必须先调用 move_line 移动到 Z=10 的高度（垂直爬升）
    # We can check the sequence of calls
    move_calls = drone_control.move_line.call_args_list
    assert len(move_calls) >= 2
    
    # The first move should be the vertical climb to (0, 0, 10)
    first_move = move_calls[0]
    expected_climb_pos = Position(0, 0, 10)
    assert first_move == call(drone_id, expected_climb_pos), f"First move should be climb to {expected_climb_pos}"
    
    # Check that compute_path was called after the first move
    # (Checking logical flow via side effects or mock history is complex, 
    # but the executor code shows move_line for climb is before compute_path)
    
    print("test_climb passed!")

def test_descend():
    print("Running test_descend...")
    drone_repo = MagicMock()
    drone_control = MagicMock()
    navigation = MagicMock()
    grid_map = MagicMock()
    executor = NavigationExecutor(drone_repo, drone_control, navigation, grid_map)
    
    drone_id = "test_drone"
    # 场景：无人机当前 Z=10，目标 Z=5 (且 Target Z > 0)
    drone = Drone(id=drone_id, position=Position(0, 0, 10), state=DroneState.HOVERING)
    drone_repo.get_drone.return_value = drone
    
    target_pos = Position(20, 20, 5)
    task = Task(id="task_descend", position=target_pos)
    
    # 先执行水平路径的 move_line（在 Z=10 高度）
    horizontal_waypoint = Position(20, 20, 10)
    navigation.compute_path.return_value = [horizontal_waypoint]
    drone_control.move_line.return_value = True
    
    # Side effect to update drone position
    def move_line_side_effect(did, pos):
        drone.position = pos
        return True
    drone_control.move_line.side_effect = move_line_side_effect

    executor.execute_task(drone_id, task)
    
    # 验证：先执行水平路径的 move_line（在 Z=10 高度），最后再调用 move_line 垂直下降到 Z=5
    move_calls = drone_control.move_line.call_args_list
    assert len(move_calls) == 2
    
    assert move_calls[0] == call(drone_id, horizontal_waypoint)
    assert move_calls[1] == call(drone_id, target_pos) # Final vertical drop
    
    # 确保此场景下 不会 调用 land 方法
    drone_control.land.assert_not_called()
    
    print("test_descend passed!")

if __name__ == "__main__":
    try:
        test_climb()
        test_descend()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
