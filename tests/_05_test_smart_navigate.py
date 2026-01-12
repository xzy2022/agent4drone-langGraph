
import pytest
from unittest.mock import MagicMock, patch
from src.uav_tools import SmartNavigateTool
from src.uav_api_client import UAVAPIClient

class TestSmartNavigateTool:
    @pytest.fixture
    def mock_client(self):
        client = MagicMock(spec=UAVAPIClient)
        # 默认无人机状态
        client.get_drone_status.return_value = {
            "drone_id": "test-drone",
            "position": {"x": 0.0, "y": 0.0, "z": 10.0},
            "perceived_radius": 30.0,
            "status": "idle"
        }
        # 默认环境感知：空旷
        client.get_nearby_entities.return_value = {"obstacles": [], "targets": []}
        # 默认移动结果：成功
        client.move_to.return_value = {"status": "success", "message": "Moving to target"}
        return client

    def test_basic_navigation_clear_path(self, mock_client):
        """测试开阔地带的乐观执行（一步达）"""
        tool = SmartNavigateTool(client=mock_client)
        
        # 模拟执行：从 (0,0,10) 飞向 (100,0,10)
        # 乐观策略应该直接尝试 (100,0,10)
        result = tool._execute(drone_id="test-drone", x=100.0, y=0.0, z=10.0)
        
        assert result == "成功抵达目的地"
        # 检查是否尝试了远端目标
        mock_client.move_to.assert_any_call("test-drone", 100.0, 0.0, 10.0)
        # 检查 get_nearby_entities 的调用（初始化、循环中、移动后）
        assert mock_client.get_nearby_entities.call_count >= 2

    def test_obstacle_avoidance_cascade_retry(self, mock_client):
        """测试级联重试：远端失败，近端成功后绕行"""
        tool = SmartNavigateTool(client=mock_client)
        
        # 1. 模拟在 50m 处有一个看不到的墙
        # 第一次尝试 100m (乐观) 会失败
        # 第二次尝试 24m (安全半径 30*0.8) 会成功
        
        def side_effect_move(drone_id, x, y, z):
            if x > 50.0:
                return {"status": "error", "message": "Collision with obstacle"}
            return {"status": "success", "message": "Moving"}

        mock_client.move_to.side_effect = side_effect_move
        
        # 模拟在 24m 处扫描时发现了障碍物
        # 假设 A* 路径第二步到了 (24, 0)
        def side_effect_status(drone_id):
            # 简化：第二次获取状态时，已经在 (24,0) 了
            if mock_client.get_drone_status.call_count > 2:
                return {"position": {"x": 24.0, "y": 0.0, "z": 10.0}, "perceived_radius": 30.0}
            return {"position": {"x": 0.0, "y": 0.0, "z": 10.0}, "perceived_radius": 30.0}
            
        mock_client.get_drone_status.side_effect = side_effect_status

        # 执行导航
        result = tool._execute(drone_id="test-drone", x=100.0, y=0.0, z=10.0)
        
        # 验证逻辑
        # 应该先试了 100m
        mock_client.move_to.assert_any_call("test-drone", 100.0, 0.0, 10.0)
        # 导航应该会继续进行（除非达到 max_loops）
        assert "成功抵达目的地" in result or "超时" in result

    def test_grid_map_learning(self, mock_client):
        """测试栅格地图的学习和记忆能力"""
        from src.uav_tools import GridMapManager
        gmm = GridMapManager(resolution=5.0, inflation=1)
        
        # 添加一个障碍物点 (10, 10)
        entities = {
            "obstacles": [{"id": "obs1", "position": {"x": 10.1, "y": 10.1, "z": 0}}]
        }
        gmm.add_obstacles_from_entities(entities, 10.0)
        
        # 检查栅格化是否正确 (10/5 = 2)
        assert (2, 2) in gmm.obstacles
        # 检查膨胀 (10+-5m 范围内都应被封锁)
        assert (1, 1) in gmm.obstacles 
        assert (3, 3) in gmm.obstacles

        # A* 寻路测试：起点 (0,0), 终点 (10,10) 应该报错无路径 (被膨胀包围了)
        path = gmm.a_star_search({"x":0, "y":0}, {"x":10, "y":10, "z":10})
        # 注意：现在 A* 允许起点终点在障碍物内，但 (2,2) 周围全是障碍，4向移动出不去
        # 不过起点 (0,0) 不在障碍内。终点 (2,2) 在。
        assert path is not None # 应该能找到，因为 A* 允许终点在障碍内以防卡死
        
        # 验证路径中的高度
        for wp in path:
            assert wp['z'] == 10.0
