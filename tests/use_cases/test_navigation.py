import unittest
from core.entities.map import GridMap
from core.entities.position import Position
from core.entities.obstacle import PolygonObstacle
from core.use_cases.navigation import NavigationUseCase

class TestNavigationUseCase(unittest.TestCase):
    def setUp(self):
        self.nav = NavigationUseCase()
        self.map = GridMap(resolution=1.0)
        # Empty map by default

    def test_simple_path(self):
        start = Position(0, 0, 10)
        end = Position(5, 5, 10)
        
        path = self.nav.compute_path(start, end, self.map)
        self.assertIsNotNone(path)
        self.assertTrue(len(path) > 0)
        
        # Check start and end proximity (due to grid discretization, might not be exact float match)
        # But my reconstruct_path returns center of grid.
        # Start(0,0) -> Grid(0,0) -> Real(0.5, 0.5)
        # So first point should be close to 0.5, 0.5
        first = path[0]
        last = path[-1]
        
        self.assertAlmostEqual(first.x, 0.5)
        self.assertAlmostEqual(first.y, 0.5)
        
        # End(5,5) -> Grid(5,5) -> Real(5.5, 5.5)
        self.assertAlmostEqual(last.x, 5.5)
        self.assertAlmostEqual(last.y, 5.5)

    def test_obstacle_avoidance(self):
        # Create a wall at x=2
        # Use coordinates that safely enclose the grid points (2,0) to (2,5)
        # Grid 2 -> 2.0. Polygon from 1.5 to 2.5 covers it.
        obs = PolygonObstacle(vertices=[
            Position(1.5, -0.5, 0), Position(2.5, -0.5, 0), 
            Position(2.5, 5.5, 0), Position(1.5, 5.5, 0)
        ], height=20.0)
        self.map.add_obstacle(obs)
        
        start = Position(0, 2, 10)
        end = Position(4, 2, 10)
        
        path = self.nav.compute_path(start, end, self.map)
        self.assertIsNotNone(path)
        
        # Verify no point in path is in the obstacle (x approx 2)
        for p in path:
            gx = self.map.to_grid(p.x)
            gy = self.map.to_grid(p.y)
            # Check if this specific cell is blocked
            status = self.map.get_status(gx, gy)
            # Blocked if status > 0 and height >= drone.z
            is_blocked = (status > 0 and status >= p.z)
            self.assertFalse(is_blocked, f"Path point {p} traverses blocked grid ({gx}, {gy})")

    def test_no_path(self):
        # Enclose start completely
        # Block all 8 neighbors of (0,0)
        neighbors = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]
        for gx, gy in neighbors:
            self.map._update_grid_cell(gx, gy, 20.0)
            
        # Debug: print obstacles
        # print("Obstacles:", self.map.obstacles)

        start = Position(0, 0, 10)
        end = Position(5, 5, 10)
        path = self.nav.compute_path(start, end, self.map)
        
        if path is not None:
            print(f"DEBUG NO_PATH FAILED: Path found: {path}")
            # Check why path is valid
            for p in path:
                gx, gy = self.map.to_grid(p.x), self.map.to_grid(p.y)
                obs_status = self.map.get_status(gx, gy)
                is_blocked = self.map.is_blocked(gx, gy, 10.0, optimistic=True)
                print(f"Point {p} -> Grid({gx}, {gy}), Status: {obs_status}, Blocked: {is_blocked}")

        self.assertIsNone(path, f"Path found {path} but should be blocked")

if __name__ == '__main__':
    unittest.main()
