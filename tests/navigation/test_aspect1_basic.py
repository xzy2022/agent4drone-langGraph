
import unittest
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from core.entities.map import GridMap
from core.entities.drone import Drone
from core.entities.position import Position
from core.entities.obstacle import PolygonObstacle
from core.use_cases.navigation.adaptive_navigation import AdaptiveNavigation
from tests.navigation.viz_utils import visualize_navigation_result

class TestAspect1Basic(unittest.TestCase):
    def setUp(self):
        self.grid_map = GridMap(resolution=1.0, inflation=0) # Resolution 1m for easier reasoning
        self.drone = Drone("d1", Position(0,0,0))
        self.nav = AdaptiveNavigation()

    def test_scenario_a_straight_line(self):
        """
        Scenario A: Start (0,0) -> (0,10), no obstacles.
        Expect: [Start, End] (2 points).
        """
        end_pos = Position(0, 10, 0)
        max_dist = 1000.0
        
        path = self.nav.navigate(self.drone, end_pos, self.grid_map, max_dist)
        
        print("\n--- Scenario A: Straight Line ---")
        if path:
            print(f"Path: {path}")
            
        visualize_navigation_result(self.grid_map, self.drone.position, end_pos, path, max_dist, "Aspect 1 - Scenario A: Straight Line")
        
        self.assertIsNotNone(path)
        self.assertEqual(len(path), 2, f"Path should have 2 points (Start, End), got {len(path)}: {path}")
        self.assertEqual(path[0], self.drone.position)
        self.assertEqual(path[-1], end_pos)

    def test_scenario_b_simple_corner(self):
        """
        Scenario B: Start (0,0) -> (5,5).
        Obstacle: Wall at x=2, y=[0, 1, 2, 3].
        This forces the drone to go up to y>3 to cross x=2.
        """
        # Create a wall x from 1.5 to 2.5, y from -1 to 3.5
        # Grid coords: x=2, y=0,1,2,3
        # Start (0,0) is x=0, y=0.
        # Desitnation (5,5) is x=5, y=5.
        
        # Add obstacle
        obs = PolygonObstacle(vertices=[
            Position(1.5, -1.0, 0),
            Position(2.5, -1.0, 0),
            Position(2.5, 3.5, 0),
            Position(1.5, 3.5, 0)
        ], height=10.0)
        self.grid_map.add_obstacle(obs)
        
        end_pos = Position(5, 5, 0)
        max_dist = 1000.0
        
        path = self.nav.navigate(self.drone, end_pos, self.grid_map, max_dist)
        
        print("\n--- Scenario B: Simple Corner ---")
        if path:
            print(f"Path: {path}")

        visualize_navigation_result(self.grid_map, self.drone.position, end_pos, path, max_dist, "Aspect 1 - Scenario B: Simple Corner")
        
        self.assertIsNotNone(path)
        # Should have Start, maybe 1 or 2 corners, End.
        # Definitely > 2 points if strictly straight line logic used, but here we expect simplified corners.
        # The path should not contain collinear points.
        self.assertGreater(len(path), 2, "Path should have at least one corner.")
        
        # Verify simplification: No 3 consecutive collinear points?
        # Our simplification code compares directions.
        # Just checking lengths is basic sanity check.
        self.assertEqual(path[0], self.drone.position)
        self.assertEqual(path[-1], end_pos)

if __name__ == '__main__':
    unittest.main()
