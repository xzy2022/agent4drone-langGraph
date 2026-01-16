
import pytest
from unittest.mock import MagicMock, patch
from src.uav_tools import SmartNavigateTool
from src.navigation import GridMapManager
from src.uav_api_client import UAVAPIClient

class TestSmartNavigateTool:
    @pytest.fixture(autouse=True)
    def clear_persistence(self):
        """每个测试前清空持久化地图缓存"""
        SmartNavigateTool._persistent_maps.clear()

    @pytest.fixture
    def mock_client(self):
        client = MagicMock(spec=UAVAPIClient)
        # 初始状态
        self.drone_pos = {"x": 0.0, "y": 0.0, "z": 10.0}
        
        def side_effect_status(drone_id):
            return {
                "drone_id": drone_id,
                "position": self.drone_pos.copy(),
                "perceived_radius": 30.0,
                "status": "flying"
            }
        
        def side_effect_move(drone_id, x, y, z):
            self.drone_pos = {"x": x, "y": y, "z": z}
            return {"status": "success", "message": "Moved"}

        def side_effect_alt(drone_id, altitude):
            self.drone_pos["z"] = altitude
            return {"status": "success", "message": "Alt changed"}

        client.get_drone_status.side_effect = side_effect_status
        client.move_to.side_effect = side_effect_move
        client.change_altitude.side_effect = side_effect_alt
        client.get_nearby_entities.return_value = {"obstacles": [], "targets": []}
        
        return client

    def test_basic_navigation_clear_path(self, mock_client):
        """测试开阔地带的巡航避障导航"""
        tool = SmartNavigateTool(client=mock_client)
        
        # 模拟执行：从 (0,0,10) 飞向 (100,0,10)
        result = tool._execute(drone_id="test-drone", x=100.0, y=0.0, z=10.0)
        
        assert "成功抵达目的地" in result
        assert self.drone_pos["x"] == 100.0
        assert "test-drone" in SmartNavigateTool._persistent_maps

    def test_three_phase_navigation_altitude_change(self, mock_client):
        """测试三阶段导航：先爬升，再巡航，后降落"""
        tool = SmartNavigateTool(client=mock_client)
        
        # 初始高度 10.0, 目标高度 5.0 -> cruise_z = 10.0 (取最大)
        # 初始高度 10.0, 目标高度 20.0 -> cruise_z = 20.0
        self.drone_pos = {"x": 0.0, "y": 0.0, "z": 10.0}
        
        result = tool._execute(drone_id="test-drone", x=100.0, y=0.0, z=5.0)
        
        assert "包含高度调整" in result
        assert self.drone_pos["z"] == 5.0
        assert self.drone_pos["x"] == 100.0

    def test_persistence_memory(self, mock_client):
        """测试地图持久化存储逻辑"""
        tool = SmartNavigateTool(client=mock_client)
        
        # 发现一个障碍物
        mock_client.get_nearby_entities.return_value = {
            "obstacles": [{"id": "wall", "type": "box", "position": {"x": 50, "y": 0, "z": 0}, "height": 20}]
        }
        tool._execute(drone_id="test-drone", x=10.0, y=0.0, z=10.0)
        
        grid_map = SmartNavigateTool._persistent_maps["test-drone"]
        gx, gy = grid_map._to_grid(50.0), grid_map._to_grid(0.0)
        assert grid_map.get_status(gx, gy) == 20.0
        
        # 修改感知，不再返回该障碍物，看之前的是否还在
        mock_client.get_nearby_entities.return_value = {"obstacles": []}
        tool._execute(drone_id="test-drone", x=20.0, y=0.0, z=10.0)
        
        assert grid_map.get_status(gx, gy) == 20.0

    def test_grid_map_new_features(self):
        """测试 GridMapManager 的新功能：多边形、椭圆、高度感知"""
        gmm = GridMapManager(resolution=5.0, inflation=0)
        
        entities = {
            "obstacles": [{
                "id": "poly1", 
                "type": "polygon", 
                "vertices": [{"x":0, "y":0}, {"x":10, "y":0}, {"x":10, "y":10}, {"x":0, "y":10}],
                "height": 15.0
            }]
        }
        gmm.add_obstacles_from_entities(entities)
        assert gmm.get_status(1, 1) == 15.0
        
        assert gmm.is_blocked(1, 1, 10.0, optimistic=True) == True
        assert gmm.is_blocked(1, 1, 20.0, optimistic=True) == False

        path = gmm.a_star_search({"x":0, "y":0, "z":20.0}, {"x":10, "y":10, "z":20.0})
        assert path is not None
        
        path_low = gmm.a_star_search({"x":0, "y":0, "z":5.0}, {"x":10, "y":10, "z":10.0})
        assert path_low is None

