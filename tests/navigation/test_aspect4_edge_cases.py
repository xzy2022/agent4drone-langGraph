
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

class TestAspect4EdgeCases(unittest.TestCase):
    def setUp(self):
        self.grid_map = GridMap(resolution=1.0)
        self.drone = Drone("d1", Position(0,0,0))
        self.nav = AdaptiveNavigation()

    def test_case1_unreachable(self):
        """
        Case 1: Target surrounded by obstacles.
        Expect: None.
        """
        # Create a box around (5,5)
        # Vertices covering x=4..6, y=4..6
        obs1 = PolygonObstacle(vertices=[
            Position(3.5, 3.5, 0), Position(6.5, 3.5, 0),
            Position(6.5, 4.5, 0), Position(3.5, 4.5, 0)
        ], height=10) # Bottom wall
        obs2 = PolygonObstacle(vertices=[
            Position(3.5, 5.5, 0), Position(6.5, 5.5, 0),
            Position(6.5, 6.5, 0), Position(3.5, 6.5, 0)
        ], height=10) # Top wall
        obs3 = PolygonObstacle(vertices=[
            Position(3.5, 4.5, 0), Position(4.5, 4.5, 0),
            Position(4.5, 5.5, 0), Position(3.5, 5.5, 0)
        ], height=10) # Left wall
        obs4 = PolygonObstacle(vertices=[
            Position(5.5, 4.5, 0), Position(6.5, 4.5, 0),
            Position(6.5, 5.5, 0), Position(5.5, 5.5, 0)
        ], height=10) # Right wall
        
        self.grid_map.add_obstacle(obs1)
        self.grid_map.add_obstacle(obs2)
        self.grid_map.add_obstacle(obs3)
        self.grid_map.add_obstacle(obs4)
        
        end_pos = Position(5, 5, 0)
        max_dist = 1000.0
        
        # A* might return None, or empty list
        path = self.nav.navigate(self.drone, end_pos, self.grid_map, max_dist)
        
        print("\n--- Edge Case 1: Unreachable ---")
        if path is None:
            print("Path is None (Expected)")
        else:
            print(f"Path: {path}")

        # Visualization might fail if path is None, so careful
        try:
             visualize_navigation_result(self.grid_map, self.drone.position, end_pos, path, max_dist, "Edge Case 1: Unreachable")
        except Exception as e:
            print(f"Viz skipped or failed: {e}")

        self.assertIsNone(path)

    def test_case2_zero_distance(self):
        """
        Case 2: Start == End.
        Expect: List with length 0, 1 or 2 (start/end same).
        """
        end_pos = Position(0, 0, 0) # Same as start
        max_dist = 100.0
        
        path = self.nav.navigate(self.drone, end_pos, self.grid_map, max_dist)
        
        print("\n--- Edge Case 2: Zero Distance ---")
        print(f"Path: {path}")
        
        if path:
             visualize_navigation_result(self.grid_map, self.drone.position, end_pos, path, max_dist, "Edge Case 2: Zero Distance")
        
        self.assertIsNotNone(path)
        # Should be essentially empty path logic or just start point
        # A* for same start/end should return cost 0, path [start]
        # Our truncate logic handles path len 1? 
        # AdaptiveNav._truncate_path checks: if not path: return []
        # if len is 1, loop range(1, 1) is empty. Returns [path[0]].
        # Simplify path if len <=2 returns path.
        # So expects [Start].
        self.assertTrue(len(path) <= 2)
        if len(path) > 0:
            self.assertEqual(path[0], self.drone.position)

    def test_case3_exact_distance(self):
        """
        Case 3: Path length == max_dist.
        Expect: Full path reached.
        """
        end_pos = Position(0, 5, 0)
        max_dist = 5.0
        
        path = self.nav.navigate(self.drone, end_pos, self.grid_map, max_dist)
        
        print("\n--- Edge Case 3: Exact Distance ---")
        print(f"Path: {path}")
        
        visualize_navigation_result(self.grid_map, self.drone.position, end_pos, path, max_dist, "Edge Case 3: Exact Distance")
        
        self.assertIsNotNone(path)
        self.assertEqual(len(path), 2)
        self.assertEqual(path[-1], end_pos)

if __name__ == '__main__':
    unittest.main()
