import pytest
import math
from core.entities.position import Position
from core.entities.obstacle import PolygonObstacle, CircleObstacle, create_obstacle
from core.entities.map import GridMap
from core.entities.drone import Drone, DroneState
from core.entities.task import Task, TaskStatus

# --- Position Tests ---
def test_position_distance():
    p1 = Position(0, 0, 0)
    p2 = Position(3, 4, 0)
    assert p1.distance_to(p2) == 5.0
    assert p1.distance_2d(p2) == 5.0

    p3 = Position(0, 0, 10)
    assert p1.distance_to(p3) == 10.0
    assert p1.distance_2d(p3) == 0.0

def test_position_dict_conversion():
    p1 = Position(1.1, 2.2, 3.3)
    data = p1.to_dict()
    assert data == {"x": 1.1, "y": 2.2, "z": 3.3}
    
    p2 = Position.from_dict(data)
    assert p1 == p2

# --- Obstacle Tests ---
def test_circle_obstacle():
    obs = CircleObstacle(height=10.0, center=Position(10, 10, 0), radius=5.0)
    
    # Inside
    assert obs.contains(10, 10)
    assert obs.contains(13, 14) # 3^2 + 4^2 = 25 <= 25
    
    # Outside
    assert not obs.contains(20, 20)
    
    # Infinite height check
    assert not obs.is_infinite_height  # 10.0 > 0
    obs_inf = CircleObstacle(height=0.0, radius=1.0)
    assert obs_inf.is_infinite_height

def test_polygon_obstacle():
    # Square 10x10 at (0,0) to (10,10)
    vertices = [
        Position(0, 0, 0),
        Position(10, 0, 0),
        Position(10, 10, 0),
        Position(0, 10, 0)
    ]
    obs = PolygonObstacle(height=5.0, vertices=vertices)
    
    assert obs.contains(5, 5)
    assert obs.contains(1, 1)
    assert not obs.contains(-1, 5)
    assert not obs.contains(11, 5)

def test_create_obstacle_factory():
    data_poly = {
        'type': 'polygon',
        'height': 20.0,
        'vertices': [{'x': 0, 'y': 0}, {'x': 10, 'y': 0}, {'x': 0, 'y': 10}]
    }
    obs_poly = create_obstacle(data_poly)
    assert isinstance(obs_poly, PolygonObstacle)
    assert len(obs_poly.vertices) == 3
    
    data_circle = {
        'type': 'circle',
        'height': 0.0,
        'position': {'x': 5, 'y': 5, 'z': 0},
        'radius': 3.0
    }
    obs_circle = create_obstacle(data_circle)
    assert isinstance(obs_circle, CircleObstacle)
    assert obs_circle.radius == 3.0
    assert obs_circle.center == Position(5, 5, 0)

# --- Map Tests ---
def test_grid_map_basic():
    grid = GridMap(resolution=1.0, inflation=0)
    
    # Add an obstacle at (5,5) with radius 1
    # Grid cells around (5,5) should be blocked
    obs = CircleObstacle(height=10.0, center=Position(5.0, 5.0, 0), radius=0.9) # radius slightly less than 1
    # 0.9 radius at res 1.0 -> grid radius 0.
    # only (5,5) blocked.
    
    grid.add_obstacle(obs)
    
    # Check status
    gx, gy = grid._to_grid(5.0), grid._to_grid(5.0)
    assert grid.get_status(gx, gy) == 10.0
    assert grid.is_blocked(gx, gy, z=5.0)  # 5 <= 10 blocked
    assert not grid.is_blocked(gx, gy, z=15.0) # 15 > 10 safe
    
    # Unknown area (0,0)
    assert grid.get_status(0, 0) == -2.0
    assert grid.is_blocked(0, 0, z=5.0, optimistic=False) # Pessimistic: blocked
    assert not grid.is_blocked(0, 0, z=5.0, optimistic=True) # Optimistic: free

def test_grid_map_exploration():
    grid = GridMap(resolution=1.0)
    grid.mark_explored(0, 0, radius=2.0)
    
    # (0,0) and neighbors should be -1.0
    gx, gy = grid._to_grid(0), grid._to_grid(0)
    assert grid.get_status(gx, gy) == -1.0
    assert not grid.is_blocked(gx, gy, z=0)

# --- Drone Tests ---
def test_drone_update():
    d = Drone(id="d1")
    assert d.state == DroneState.IDLE
    
    d.update_status(Position(10,10,10), 90.0, 180.0, "FLYING")
    assert d.position == Position(10,10,10)
    assert d.state == DroneState.FLYING
    assert d.battery == 90.0

# --- Task Tests ---
def test_task_flow():
    t = Task(id="t1", position=Position(100, 100, 0))
    assert t.status == TaskStatus.PENDING
    
    t.mark_in_progress("d1")
    assert t.status == TaskStatus.IN_PROGRESS
    assert t.assigned_drone_id == "d1"
    
    t.mark_completed()
    assert t.status == TaskStatus.COMPLETED
