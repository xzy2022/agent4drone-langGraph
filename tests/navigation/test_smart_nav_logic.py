import pytest
from src.navigation import GridMapManager
from src.uav_tools import SmartNavigateTool
from unittest.mock import MagicMock, patch

def test_grid_map_sparse_status():
    """测试栅格地图的稀疏状态表示"""
    gmm = GridMapManager(resolution=5.0, inflation=0)
    
    # 初始状态应该是未知 (-2.0)
    assert gmm.get_status(0, 0) == -2.0
    
    # 添加障碍物
    entities = {
        "obstacles": [{"id": "obs1", "type": "point", "position": {"x": 5, "y": 5}, "radius": 0, "height": 10.0}]
    }
    gmm.add_obstacles_from_entities(entities)
    assert gmm.get_status(1, 1) == 10.0
    
    # 标记已探索
    gmm.mark_explored_area(20.0, 20.0, 5.0)
    # (20,20) 对应于 (4,4) 栅格
    assert gmm.get_status(4, 4) == -1.0
    # (0,0) 仍然是未知
    assert gmm.get_status(0, 0) == -2.0

def test_grid_map_is_blocked_optimism():
    """测试 is_blocked 的乐观与悲观模式"""
    gmm = GridMapManager(resolution=5.0, inflation=0)
    
    # 未知栅格 (0,0)
    # 悲观：默认阻塞
    assert gmm.is_blocked(0, 0, 10.0, optimistic=False) == True
    # 乐观：不阻塞
    assert gmm.is_blocked(0, 0, 10.0, optimistic=True) == False
    
    # 已知空闲
    gmm.mark_explored_area(0, 0, 1)
    assert gmm.is_blocked(0, 0, 10.0, optimistic=False) == False
    
    # 障碍物
    entities = {"obstacles": [{"id": "obs1", "type": "point", "position": {"x": 10, "y": 10}, "radius": 0, "height": 20.0}]}
    gmm.add_obstacles_from_entities(entities)
    # 高度低于障碍物 -> 阻塞
    assert gmm.is_blocked(2, 2, 10.0, optimistic=True) == True
    # 高度高于障碍物 -> 不阻塞
    assert gmm.is_blocked(2, 2, 25.0, optimistic=True) == False

def test_optimistic_a_star_search():
    """测试 A* 搜索在乐观模式下可以穿越未知区域"""
    gmm = GridMapManager(resolution=5.0, inflation=0)
    
    # 起点 (0,0)，终点 (20,0)
    # 中间 (10,0) 是未知的
    start = {"x": 0.0, "y": 0.0, "z": 10.0}
    target = {"x": 20.0, "y": 0.0, "z": 10.0}
    
    # 乐观模式下应该能找到路径 (A* 默认使用 optimistic=True)
    path = gmm.a_star_search(start, target)
    assert path is not None
    assert len(path) > 0
    assert path[-1]["x"] == 20.0

def test_pessimistic_path_truncation():
    """测试悲观路径截断逻辑"""
    gmm = GridMapManager(resolution=5.0, inflation=0)
    tool = SmartNavigateTool(client=MagicMock())
    
    # 标记 (0,0) 和 (5,0) 为已探索
    gmm.mark_explored_area(0, 0, 6.0)
    
    # 路径：(0,0) -> (5,0) -> (10,0) -> (15,0) -> (20,0)
    # (10,0) 及以后是未知的
    full_path = [
        {"x": 0.0, "y": 0.0},
        {"x": 5.0, "y": 0.0},  # 已知
        {"x": 10.0, "y": 0.0}, # 未知
        {"x": 15.0, "y": 0.0},
        {"x": 20.0, "y": 0.0}
    ]
    
    truncated_path, is_truncated = tool._truncate_path_to_known(full_path, gmm)
    
    assert is_truncated == True
    # 在进入未知区域 (10,0) 前回退 1-2 个点
    # 索引 2 是 (10,0)，回退 2 个索引是 0
    # 所以应该是 full_path[:1], 包含起点
    assert len(truncated_path) < len(full_path)
    # 检查截断点是否在已知范围内
    last_p = truncated_path[-1]
    gx, gy = gmm._to_grid(last_p['x']), gmm._to_grid(last_p['y'])
    assert gmm.get_status(gx, gy) != -2.0 # 不是未知

def test_smart_navigate_loop_truncation():
    """测试 SmartNavigateTool 在执行循环中的截断与重规划流 (Mock 版本)"""
    client = MagicMock()
    tool = SmartNavigateTool(client=client)
    
    # 设置初始位置
    client.get_drone_status.return_value = {
        "position": {"x": 0.0, "y": 0.0, "z": 10.0},
        "perceived_radius": 5.0,
        "status": "flying"
    }
    client.get_nearby_entities.return_value = {"obstacles": []}
    client.move_to.return_value = {"status": "success"}
    client.change_altitude.return_value = {"status": "success"}
    
    # 执行导航到远处 (50, 0)
    # 第一次循环：A* 规划到 (50,0)，但路径会在 (5,0) 左右被截断 (因为感应半径是 5.0，标记之后 10.0 是未知的)
    # 预期：move_to 会被调用，且目标坐标应该在已探索范围内
    
    # 我们只运行一轮循环来检查逻辑
    # 注意：_execute 内部有 while 循环，我们需要让它停下来，可以通过 mock 返回值或抛出特定异常
    
    with patch.object(SmartNavigateTool, '_persistent_maps', {}):
        # 限制 loop_count 或使用 patch
        with patch('src.uav_tools.print') as mock_print:
            # 修改 move_to 让它在第二次调用时返回终止或抛出异常以跳出循环
            def side_effect_move(*args, **kwargs):
                return {"status": "success"} # 保持成功
            
            client.move_to.side_effect = side_effect_move
            
            # 手动执行一部分 logic
            result = tool._execute(drone_id="test", x=50.0, y=0.0, z=10.0)
            
            # 验证是否有截断日志
            truncated_logs = [call for call in mock_print.call_args_list if "截断" in str(call)]
            assert len(truncated_logs) > 0
