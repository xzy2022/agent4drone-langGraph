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
        self.session_id = "test_shared_session"
        self.drone_A = "drone_A"
        self.drone_B = "drone_B"
        
        # Shared cache path
        self.cache_path = os.path.join(".map_cache", f"targets_{self.session_id}.json")
        
        # Clean start
        if os.path.exists(self.cache_path):
            os.remove(self.cache_path)

    def tearDown(self):
        if os.path.exists(self.cache_path):
            os.remove(self.cache_path)
            
        # Clear global state
        from src.uav_tools import _PERSISTENT_TARGETS
        if self.session_id in _PERSISTENT_TARGETS:
            del _PERSISTENT_TARGETS[self.session_id]

    # ... test_target_manager_logic omitted as it tests the class in isolation ...

    def test_start_to_end_integration_shared(self):
        print("\nTesting shared integration with uav_tools functions...")
        
        # 1. Drone A updates targets
        # Note: get_target_manager now only takes session_id
        tm_a = get_target_manager(self.session_id) 
        tm_a.update_from_perception({
            "targets": [{"id": "t_shared", "name": "Shared Target", "is_reached": True}]
        })
        save_target_manager(self.session_id)
        
        # 2. Drone B accesses targets (should get same manager instance in memory)
        tm_b = get_target_manager(self.session_id)
        target = tm_b.get_target("t_shared")
        
        self.assertIsNotNone(target)
        self.assertEqual(target["name"], "Shared Target")
        self.assertTrue(target["is_reached"])
        
        # 3. Verify disk persistence is shared
        # Force reload by clearing memory
        from src.uav_tools import _PERSISTENT_TARGETS
        del _PERSISTENT_TARGETS[self.session_id]
        
        tm_c = get_target_manager(self.session_id) # New instance from disk
        target_reloaded = tm_c.get_target("t_shared")
        self.assertIsNotNone(target_reloaded)
        
    def test_tool_execution(self):
        print("\nTesting GetKnownTargetsTool...")
        client_mock = MagicMock()
        client_mock.get_current_session.return_value = {"id": self.session_id}
        
        tool = GetKnownTargetsTool(client=client_mock)
        
        # Pre-populate using Drone A's view
        tm = get_target_manager(self.session_id)
        tm.update_from_perception({"targets": [{"id": "t_tool", "val": 123}]})
        
        # Query using Drone B's ID (should access same session data)
        result = tool._execute(self.drone_B)
        
        # Result should show the target
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], "t_tool")

if __name__ == "__main__":
    unittest.main()
