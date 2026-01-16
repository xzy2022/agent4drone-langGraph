import os
import sys
import unittest
from unittest.mock import MagicMock
import shutil

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.navigation.target_manager import TargetManager
from src.uav_tools import get_target_manager, save_target_manager, GetKnownTargetsTool

class TestTargetTracking(unittest.TestCase):
    def setUp(self):
        self.test_dir = ".map_cache_test"
        # Mock the cache dir in uav_tools/target_manager implicitly by patching 'os.path.join' or just relying on cleanup
        # Since uav_tools hardcodes ".map_cache", we might want to temporarily rename it or just clean up carefuly.
        # Ideally we refactor uav_tools to accept cache dir, but for now let's just use the real one and clean up unique files.
        self.session_id = "test_target_session"
        self.drone_id = "drone_test_1"
        self.cache_path = os.path.join(".map_cache", f"targets_{self.session_id}_{self.drone_id}.json")
        
        # Clean start
        if os.path.exists(self.cache_path):
            os.remove(self.cache_path)

    def tearDown(self):
        if os.path.exists(self.cache_path):
            os.remove(self.cache_path)
            
        # Clear global state
        from src.uav_tools import _PERSISTENT_TARGETS
        if (self.session_id, self.drone_id) in _PERSISTENT_TARGETS:
            del _PERSISTENT_TARGETS[(self.session_id, self.drone_id)]

    def test_target_manager_logic(self):
        print("\nTesting TargetManager logic...")
        tm = TargetManager()
        
        # 1. Update with new target
        perception_data = {
            "targets": [
                {"id": "t1", "name": "Target 1", "position": {"x": 10, "y": 10}, "is_reached": False}
            ]
        }
        tm.update_from_perception(perception_data)
        self.assertEqual(len(tm.targets), 1)
        self.assertEqual(tm.get_target("t1")["is_reached"], False)
        
        # 2. Update existing target (status change)
        perception_data_update = {
            "targets": [
                {"id": "t1", "position": {"x": 10, "y": 10}, "is_reached": True},
                {"id": "t2", "name": "Target 2", "position": {"x": 20, "y": 20}}
            ]
        }
        tm.update_from_perception(perception_data_update)
        self.assertEqual(len(tm.targets), 2)
        self.assertEqual(tm.get_target("t1")["is_reached"], True) # Should update
        self.assertEqual(tm.get_target("t1")["name"], "Target 1") # Should preserve name if not present?
        # Note: in my implementation updated.update(t) blindly overwrites. 
        # If perception sends partial data, we might lose data.
        # But typically perception sends full entity objects.
        
    def test_start_to_end_integration(self):
        print("\nTesting integration with uav_tools functions...")
        
        # 1. Get manager
        tm = get_target_manager(self.session_id, self.drone_id)
        
        # 2. Update it
        tm.update_from_perception({
            "targets": [{"id": "t99", "name": "Target 99", "is_reached": True}]
        })
        
        # 3. Save
        save_target_manager(self.session_id, self.drone_id)
        self.assertTrue(os.path.exists(self.cache_path))
        
        # 4. Clear memory to force reload
        from src.uav_tools import _PERSISTENT_TARGETS
        if (self.session_id, self.drone_id) in _PERSISTENT_TARGETS:
            del _PERSISTENT_TARGETS[(self.session_id, self.drone_id)]
            
        # 5. Reload
        tm2 = get_target_manager(self.session_id, self.drone_id)
        target = tm2.get_target("t99")
        self.assertIsNotNone(target)
        self.assertEqual(target["name"], "Target 99")
        self.assertTrue(target["is_reached"])
        
    def test_tool_execution(self):
        print("\nTesting GetKnownTargetsTool...")
        client_mock = MagicMock()
        client_mock.get_current_session.return_value = {"id": self.session_id}
        
        tool = GetKnownTargetsTool(client=client_mock)
        
        # Pre-populate
        tm = get_target_manager(self.session_id, self.drone_id)
        tm.update_from_perception({"targets": [{"id": "t_tool", "val": 123}]})
        
        result = tool._execute(self.drone_id)
        # Result should be a list of dicts
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], "t_tool")

if __name__ == "__main__":
    unittest.main()
