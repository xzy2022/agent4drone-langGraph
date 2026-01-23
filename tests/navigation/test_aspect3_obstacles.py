
import unittest
import sys
import os
import math

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from core.entities.map import GridMap
from core.entities.drone import Drone
from core.entities.position import Position
from core.entities.obstacle import PolygonObstacle
from core.use_cases.navigation.adaptive_navigation import AdaptiveNavigation
from tests.navigation.viz_utils import visualize_navigation_result

"""
inflation膨胀后的障碍物直接被绘制了出来，因此视觉效果是贴着障碍物飞的，但是实际上有安全距离。
"""

class TestAspect3Obstacles(unittest.TestCase):
    def setUp(self):
        self.grid_map = GridMap(resolution=0.5, inflation=2)
        self.drone = Drone("d1", Position(0,0,0))
        self.nav = AdaptiveNavigation()

    def test_truncation_around_obstacle(self):
        """
        Scenario: Start (0,0) -> (0,4).
        Wall blocking direct path at y=2.
        Path must go around.
        Direct distance: 4.
        Path distance: > 6.
        Max Move Dist: 5.0 (Sufficient for straight, insufficient for detour).
        Values:
        Wall at x=[-2, 2], y=[1.5, 2.5] (approx grid y=2)
        """
        # Create obstacle
        # A wide wall centered at (0, 2)
        # Vertices: (-3, 1.5), (3, 1.5), (3, 2.5), (-3, 2.5)
        # This blocks x=-3 to 3.
        # Drone must go to x=4 or x=-4 to pass? 
        # Grid resolution is 1.0.
        # Wall at grid x=-3..3, y=2.
        # Path will likely find x=4, y=2 as passage.
        pass_x = 4.0 
        
        obs = PolygonObstacle(vertices=[
            Position(-3.5, 1.5, 0),
            Position(3.5, 1.5, 0),
            Position(3.5, 2.5, 0),
            Position(-3.5, 2.5, 0)
        ], height=10.0)
        self.grid_map.add_obstacle(obs)
        
        end_pos = Position(0, 5, 0)
        # Direct dist is 4.
        # Detour approx (0,0)->(4,2)->(0,4) = sqrt(16+4)*2 = 4.47*2 = 8.9.
        # Let's set max_dist = 6.0.
        # It should pass valid straight range (4.0) but fail path range (8.9).
        max_dist = 10.0
        
        path = self.nav.navigate(self.drone, end_pos, self.grid_map, max_dist)
        
        print("\n--- Aspect 3: Obstacle Truncation ---")
        if path:
            print(f"Path: {path}")

        visualize_navigation_result(self.grid_map, self.drone.position, end_pos, path, max_dist, "Aspect 3: Obstacle Truncation")
        
        self.assertIsNotNone(path)
        self.assertGreater(len(path), 0)
        
        final_pt = path[-1]
        
        # 1. Verify Truncation Length
        total_len = 0.0
        for i in range(1, len(path)):
            p1, p2 = path[i-1], path[i]
            total_len += math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
            
        print(f"Total Path Length: {total_len}")
        self.assertAlmostEqual(total_len, max_dist, delta=0.2) # Allow slight float error
        
        # 2. Verify it did NOT reach target
        dist_to_goal = math.sqrt((final_pt.x - end_pos.x)**2 + (final_pt.y - end_pos.y)**2)
        print(f"Distance to Goal: {dist_to_goal}")
        self.assertGreater(dist_to_goal, 0.5)

if __name__ == '__main__':
    unittest.main()
