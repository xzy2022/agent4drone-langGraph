
import unittest
import sys
import os
import math

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from core.entities.map import GridMap
from core.entities.drone import Drone
from core.entities.position import Position
from core.use_cases.navigation.adaptive_navigation import AdaptiveNavigation
from tests.navigation.viz_utils import visualize_navigation_result

class TestAspect2Truncation(unittest.TestCase):
    def setUp(self):
        self.grid_map = GridMap(resolution=1.0)
        self.drone = Drone("d1", Position(0,0,0))
        self.nav = AdaptiveNavigation()

    def test_truncation_straight(self):
        """
        Scenario: Start (0,0) -> (10,0).
        Max Dist: 5.5
        Expect: Path truncated exactly at (5.5, 0).
        """
        end_pos = Position(10, 0, 0)
        max_dist = 5.5
        
        path = self.nav.navigate(self.drone, end_pos, self.grid_map, max_dist)
        
        print("\n--- Aspect 2: Truncation ---")
        if path:
            print(f"Path: {path}")

        visualize_navigation_result(self.grid_map, self.drone.position, end_pos, path, max_dist, "Aspect 2: Truncation")
        
        self.assertIsNotNone(path)
        self.assertGreater(len(path), 0)
        
        final_pt = path[-1]
        
        # 1. Check Total Length
        total_len = 0.0
        for i in range(1, len(path)):
            p1, p2 = path[i-1], path[i]
            total_len += math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
            
        print(f"Total Path Length: {total_len}")
        
        # We expect it to be very close to 5.5.
        # Due to float precision, it might be 5.499999 or 5.5000001
        self.assertLessEqual(total_len, 5.5 + 1e-5, "Path length exceeded max distance.")
        self.assertAlmostEqual(total_len, 5.5, places=2, msg="Path length should be practically equal to max_dist")
        
        # 2. Check Final Point Coordinates (Interpolation Check)
        self.assertAlmostEqual(final_pt.x, 5.5, places=2)
        self.assertAlmostEqual(final_pt.y, 0.0, places=2)
        
        # 3. Check it does not contain original target
        dist_to_orig = math.sqrt((final_pt.x - end_pos.x)**2 + (final_pt.y - end_pos.y)**2)
        self.assertGreater(dist_to_orig, 1.0, "Path should not reach original target.")

if __name__ == '__main__':
    unittest.main()
