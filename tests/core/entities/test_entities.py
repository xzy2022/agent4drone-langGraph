import math
from core.entities.position import Position
from core.entities.obstacle import PolygonObstacle, CircleObstacle, EllipseObstacle, create_obstacle
from core.entities.map import GridMap
from core.entities.drone import Drone, DroneState
from core.entities.task import Task, TaskStatus

# Helper to print test results
def run_test(name, func):
    print(f"Running {name}...")
    try:
        func()
        print(f"✅ {name} Passed")
    except AssertionError as e:
        print(f"❌ {name} Failed: {e}")
    except Exception as e:
        print(f"❌ {name} Error: {e}")

# --- Position Tests ---
def test_position_distance():
    p1 = Position(0, 0, 0)
    p2 = Position(3, 4, 0)
    assert p1.distance_to(p2) == 5.0
    assert p1.distance_2d(p2) == 5.0

# --- Obstacle Tests ---
def test_circle_obstacle():
    obs = CircleObstacle(height=10.0, center=Position(10, 10, 0), radius=5.0)
    assert obs.contains(10, 10)
    assert obs.contains(13, 14) 
    assert not obs.contains(20, 20)
    
def test_ellipse_obstacle():
    # Ellipse elongated along X axis
    # Major (X) = 4 (semi=2), Minor (Y) = 2 (semi=1)
    obs = EllipseObstacle(
        height=5.0, 
        center=Position(0,0,0), 
        major_axis=2.0, # semi-major
        minor_axis=1.0, # semi-minor
        orientation=0.0
    )
    assert obs.contains(1.5, 0)   # Within long axis
    assert obs.contains(0, 0.8)   # Within short axis
    assert not obs.contains(0, 1.2) # Outside short axis
    
    # Rotated 90 degrees
    obs_rot = EllipseObstacle(
        height=5.0,
        center=Position(0,0,0),
        major_axis=2.0,
        minor_axis=1.0, 
        orientation=math.pi/2
    )
    assert obs_rot.contains(0, 1.5) # Now valid along Y
    assert not obs_rot.contains(1.5, 0) # Now invalid along X

def test_create_obstacle_factory():
    data_ellipse = {
        'type': 'ellipse',
        'height': 5.0,
        'position': {'x': 0, 'y': 0},
        'major_axis': 2.0,
        'minor_axis': 1.0,
        'orientation': 0.0
    }
    obs = create_obstacle(data_ellipse)
    assert isinstance(obs, EllipseObstacle)
    assert obs.major_axis == 2.0

# --- Map Tests ---
def test_grid_map_obstacles():
    grid = GridMap(resolution=1.0, inflation=0) # Use finer resolution for better point checking
    
    # Add Square at (0,0) to (1,1) - matching the user's visualization concern
    # (0,0) (1,0) (1,1) (0,1)
    sq_verts = [
        Position(0, 0, 0),
        Position(1, 0, 0),
        Position(1, 1, 0),
        Position(0, 1, 0)
    ]
    grid.add_obstacle(PolygonObstacle(height=5.0, vertices=sq_verts))

    # Add Circle
    obs_c = CircleObstacle(height=10.0, center=Position(5, 5, 0), radius=1.0)
    grid.add_obstacle(obs_c)
    
    # Check specific points
    points_to_check = [
        (0.5, 0.5), # Center of square -> Should be obstacle
        (0.4, 0.6), # Inside square -> Should be obstacle
        (0.0, 0.0), # Corner of square -> Should be obstacle
        (1.0, 1.0), # Corner of square -> Should be obstacle
        (1.1, 1.1), # Just outside square -> Should be free/unknown (-2.0 if not explored)
        (5.0, 5.0), # Center of circle -> Obstacle
        (5.0, 5.9), # Inside circle -> Obstacle
        (5.0, 6.1), # Outside circle -> Unknown
    ]
    
    print("\n[Map Point Checks]")
    for x, y in points_to_check:
        gx, gy = grid._to_grid(x), grid._to_grid(y)
        # print(f"Point ({x}, {y}) -> Grid({gx}, {gy})")  
        status = grid.get_status(gx, gy)
        is_blocked = grid.is_blocked(gx, gy, z=1.0)
        
        status_str = "Obstacle" if status >= 0 else ("Free" if status == -1 else "Unknown")
        print(f"Point ({x}, {y}) -> Grid({gx}, {gy}): Status={status} ({status_str}), Blocked={is_blocked}")

    # Assertions
    # Note: (0,0) might depend on rounding and rasterization logic
    gx0, gy0 = grid._to_grid(0), grid._to_grid(0)
    assert grid.get_status(gx0, gy0) == 5.0, f"Expected (0,0) to be obstacle, got {grid.get_status(gx0, gy0)}"
    
    gx_mid, gy_mid = grid._to_grid(0.5), grid._to_grid(0.5)
    assert grid.get_status(gx_mid, gy_mid) == 5.0

# --- Drone Tests (Updated Logic) ---
def test_drone_state_transitions():
    d = Drone(id="d1")
    assert d.state == DroneState.IDLE
    
    d.state = DroneState.LANDED
    d.position = Position(0,0,0)
    
    assert d.takeoff(altitude=10.0)
    assert d.state == DroneState.IDLE
    assert d.position.z == 10.0
    
    assert d.move(Position(10, 10, 10))
    assert d.position == Position(10, 10, 10)
    
    assert d.hover()
    assert d.state == DroneState.HOVERING
    
    d.state = DroneState.IDLE 
    assert d.land()
    assert d.state == DroneState.LANDED
    assert d.position.z == 0.0

def test_drone_battery():
    d = Drone(id="d1")
    assert d.update_battery(80.0)
    assert d.battery == 80.0
    assert not d.update_battery(101.0) # Invalid

if __name__ == "__main__":
    print("=== Running Core Entity Tests ===")
    run_test("Position Distance", test_position_distance)
    run_test("Circle Obstacle", test_circle_obstacle)
    run_test("Ellipse Obstacle", test_ellipse_obstacle)
    run_test("Obstacle Factory", test_create_obstacle_factory)
    run_test("Grid Map Obstacles & Points", test_grid_map_obstacles)
    run_test("Drone State Transitions", test_drone_state_transitions)
    run_test("Drone Battery", test_drone_battery)
    print("=== All Tests Finished ===")
