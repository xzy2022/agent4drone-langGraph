import os
import sys
import unittest
from unittest.mock import MagicMock

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.uav_tools import SmartNavigateTool
from src.navigation.grid_map import GridMapManager

class TestMapCacheIntegration(unittest.TestCase):
    def setUp(self):
        self.client = MagicMock()
        self.tool = SmartNavigateTool(client=self.client)
        self.session_id = "test_session_123"
        self.drone_id = "drone_abc"
        self.cache_dir = ".map_cache"
        self.expected_path = os.path.join(self.cache_dir, f"map_{self.session_id}_{self.drone_id}.json")

    def tearDown(self):
        # Cleanup cache dir if it exists
        if os.path.exists(self.expected_path):
            os.remove(self.expected_path)
        if os.path.exists(self.cache_dir) and not os.listdir(self.cache_dir):
            os.rmdir(self.cache_dir)

    def test_cache_path_generation(self):
        """Test if the cache path is generated correctly"""
        path = self.tool._get_map_cache_path(self.session_id, self.drone_id)
        self.assertEqual(path, self.expected_path)

    def test_map_persistence_integration(self):
        """Test the full flow: Create -> Save -> Load"""
        # 1. Create a map
        gm = GridMapManager(resolution=5.0, inflation=1)
        gm.obstacles[(5, 5)] = 10.0
        gm.explored.add((1, 1))

        # 2. Save using the tool's logic path
        cache_path = self.tool._get_map_cache_path(self.session_id, self.drone_id)
        gm.save_to_disk(cache_path)

        # Verify file exists
        self.assertTrue(os.path.exists(cache_path))

        # 3. Load using the tool's logic path
        loaded_gm = GridMapManager.load_from_disk(cache_path)
        
        # Verify data
        self.assertIsNotNone(loaded_gm)
        self.assertEqual(loaded_gm.obstacles[(5, 5)], 10.0)
        self.assertIn((1, 1), loaded_gm.explored)
        self.assertEqual(len(loaded_gm.obstacles), 1)

    def test_persistent_maps_dict(self):
        """Test if the tool uses the (session_id, drone_id) key correctly"""
        # Clear the class-level dict for testing
        SmartNavigateTool._persistent_maps.clear()
        
        map_key = (self.session_id, self.drone_id)
        gm = GridMapManager()
        SmartNavigateTool._persistent_maps[map_key] = gm
        
        # Verify indexing
        self.assertIn(map_key, SmartNavigateTool._persistent_maps)
        self.assertEqual(SmartNavigateTool._persistent_maps[map_key], gm)

if __name__ == "__main__":
    unittest.main()
