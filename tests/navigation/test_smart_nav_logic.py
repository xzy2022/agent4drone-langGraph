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

    # --- 1. 优化 Mock：模拟无人机移动，防止死循环 ---
    # 定义 side_effect，第一次调用返回起点，第二次调用返回移动后的点（模拟移动了）
    # 这样 _execute 中的 while 循环会检测到位置变化或在逻辑中完成判断
    client.get_drone_status.side_effect = [
        {"position": {"x": 0.0, "y": 0.0, "z": 10.0}, "perceived_radius": 5.0, "status": "flying"}, # 初始
        {"position": {"x": 5.0, "y": 0.0, "z": 10.0}, "perceived_radius": 5.0, "status": "flying"}, # 移动后
        {"position": {"x": 50.0, "y": 0.0, "z": 10.0}, "perceived_radius": 5.0, "status": "flying"}, # 到达目标(防止无限循环)
    ]
    
    client.get_nearby_entities.return_value = {"obstacles": []}
    client.move_to.return_value = {"status": "success"}
    client.change_altitude.return_value = {"status": "success"}

    # 这里的关键是把地图重置为空，确保它是全新的环境
    with patch.object(SmartNavigateTool, '_persistent_maps', {}):
        
        # --- 2. 修正 Patch：针对内置 print 函数 ---
        # 如果你的代码确实是用 print()，请使用 'builtins.print'
        # 如果你的代码是用 logger.info()，请不要用 patch，改用 caplog fixture
        with patch('builtins.print') as mock_print:
            
            # 执行导航
            # 为了防止 A* 搜索不到路径或逻辑卡死，我们可以限制 _execute 内部循环
            # 但这里我们主要依赖 side_effect 让循环自然结束
            try:
                tool._execute(drone_id="test", x=50.0, y=0.0, z=10.0)
            except StopIteration:
                # 捕获 side_effect 用尽的情况（如果逻辑跑太多次）
                pass

            # --- 3. 验证 ---
            # 打印所有调用的参数，方便调试（如果失败可以看到实际打印了什么）
            print("\nCaptured prints:", mock_print.call_args_list)

            # 检查是否有包含 "截断" 的打印
            # 遍历所有调用参数，检查第一个参数（args[0]）是否包含关键字
            truncated_logs = [
                call_args 
                for call_args in mock_print.call_args_list 
                if len(call_args[0]) > 0 and "截断" in str(call_args[0][0])
            ]
            
            # 断言
            assert len(truncated_logs) > 0, "未检测到路径截断的日志输出，请检查代码是否执行了截断逻辑或是否使用了 print"
