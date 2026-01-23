import pytest
from core.use_cases.navigation.adaptive_navigation import AdaptiveNavigation
from core.entities.map import GridMap
from core.entities.drone import Drone, DroneState
from core.entities.position import Position

class TestAdaptiveNavigation:
    
    @pytest.fixture
    def grid_map(self):
        # 100x100m map, resolution 1.0
        gm = GridMap(resolution=1.0, inflation=0)
        return gm

    @pytest.fixture
    def drone(self):
        d = Drone(id="d1", position=Position(0,0,10))
        d.state = DroneState.HOVERING
        return d
        
    @pytest.fixture
    def nav_algo(self):
        return AdaptiveNavigation()

    def test_navigate_short_distance(self, nav_algo, drone, grid_map):
        """
        Target is 10m away, max_move is 20m.
        Should return full path to target.
        Path should be simplified (start and end only for straight line).
        """
        target = Position(10, 0, 10)
        max_dist = 20.0
        
        path = nav_algo.navigate(drone, target, grid_map, max_dist)
        
        assert path is not None
        # Due to grid centering, path might have small zig-zags at start/end
        # e.g. (0,0) -> (0.5, 0.5) -> ... -> (10,0)
        assert len(path) >= 2 
        assert len(path) <= 4 # Should be relatively straight
        assert path[0].x == 0 and path[0].y == 0
        assert path[-1].x == 10 and path[-1].y == 0
        
    def test_navigate_long_distance_truncation(self, nav_algo, drone, grid_map):
        """
        Target is 20m away, max_move is 10m.
        Should return truncated path stopping at 10m.
        """
        target = Position(20, 0, 10)
        max_dist = 10.0
        
        path = nav_algo.navigate(drone, target, grid_map, max_dist)
        
        assert path is not None
        # Check endpoint
        end_pt = path[-1]
        # Due to grid path (zigzag), limited distance might land us slightly off ideal (10,0)
        # x might be slightly less than 10, y might be off by up to 0.5 (half grid)
        assert abs(end_pt.x - 10.0) < 1.0 
        assert abs(end_pt.y - 0.0) < 1.0
        
        # Should be simplified to start and truncated end (straight line)
        assert len(path) >= 2 

    def test_path_simplify_check_inflection(self, nav_algo, drone, grid_map):
        """
        Force a path that requires turns.
        Map:
        Obstacle at (5, 0)
        Start (0,0), Target (10,0)
        
        Should go around obstacle.
        Path: (0,0) -> ... -> (10,0)
        Expected inflection points only.
        """
        # Add obstacle blocking direct path
        # Wall at x=5, y=-5 to 5
        # It's a grid map.
        # Add infinite obstacle
        for y in range(-5, 6):
            grid_map._update_grid_cell(5, y, 0.0) # 0.0 is infinite height
            
        target = Position(10, 0, 10)
        max_dist = 50.0 # Plenty
        
        path = nav_algo.navigate(drone, target, grid_map, max_dist)
        
        assert path is not None
        assert len(path) > 2 # Should have intermediate points to go around
        assert path[0] == drone.position
        assert path[-1] == target
        
        # Verify all points are inflection or start/end.
        # It's hard to assert strict "inflection" without looking at internal logic,
        # but we can verify it's NOT just the dense A* grid list.
        # A* would return tons of points. Simplification should reduce it drastically.
        # For a simple detour, it might be ~3-5 points depending on resolution.
        # If it was raw A*, it would be ~14 points or more.
        assert len(path) < 15 

    def test_blocked_path(self, nav_algo, drone, grid_map):
        # Fully encircle target
        for x in range(8, 12):
            for y in range(-2, 3):
                if x==8 or x==11 or y==-2 or y==2:
                     grid_map._update_grid_cell(x, y, 0.0)
        
        target = Position(10, 0, 10) # Inside box
        max_dist = 50.0
        
        # Ensure target is actually blocked from (0,0)
        # (0,0) is outside.
        
        path = nav_algo.navigate(drone, target, grid_map, max_dist)
        
        assert path is None
