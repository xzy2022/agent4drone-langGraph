import math
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from .position import Position

@dataclass
class Obstacle(ABC):
    """
    Abstract base class for obstacles.
    """
    height: float = 0.0 # 0.0 usually means infinite height in this domain, or we can use None. 
                        # Based on existing logic: 0.0 is infinite.
    
    @abstractmethod
    def contains(self, x: float, y: float) -> bool:
        """Check if a 2D point (x, y) is inside the obstacle."""
        pass

    @property
    def is_infinite_height(self) -> bool:
        return self.height <= 0.0

@dataclass
class PolygonObstacle(Obstacle):
    vertices: List[Position] = field(default_factory=list)

    def contains(self, x: float, y: float) -> bool:
        """Ray casting algorithm for point in polygon."""
        n = len(self.vertices)
        if n < 3: return False
        inside = False
        p1x, p1y = self.vertices[0].x, self.vertices[0].y
        for i in range(n + 1):
            p2x, p2y = self.vertices[i % n].x, self.vertices[i % n].y
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                            if p1x == p2x or x <= xinters:
                                inside = not inside
            p1x, p1y = p2x, p2y
        return inside

@dataclass
class CircleObstacle(Obstacle):
    center: Position = field(default_factory=lambda: Position(0,0,0))
    radius: float = 0.0

    def contains(self, x: float, y: float) -> bool:
        """Check if point is within radius of center."""
        dist_sq = (x - self.center.x)**2 + (y - self.center.y)**2
        return dist_sq <= self.radius**2

@dataclass
class CylinderObstacle(CircleObstacle):
    """
    Logically same as CircleObstacle for 2D contains check, 
    but explicit type name might be useful.
    """
    pass

@dataclass
class EllipseObstacle(Obstacle):
    center: Position = field(default_factory=lambda: Position(0,0,0))
    major_axis: float = 0.0
    minor_axis: float = 0.0
    orientation: float = 0.0  # radians

    def contains(self, x: float, y: float) -> bool:
        """Check if point is within the rotated ellipse."""
        dx = x - self.center.x
        dy = y - self.center.y
        
        cos_theta = math.cos(self.orientation)
        sin_theta = math.sin(self.orientation)
        
        # Transform point to ellipse local coordinates
        x_local = dx * cos_theta + dy * sin_theta
        y_local = -dx * sin_theta + dy * cos_theta
        
        if self.major_axis == 0 or self.minor_axis == 0:
            return False
            
        return (x_local**2 / self.major_axis**2) + (y_local**2 / self.minor_axis**2) <= 1.0

def create_obstacle(data: Dict[str, Any]) -> Optional[Obstacle]:
    """Factory function to create obstacle from dict data."""
    obs_type = data.get('type', 'unknown')
    height = float(data.get('height', 0.0))
    
    if obs_type == 'polygon':
        verts_data = data.get('vertices', [])
        vertices = [Position(v['x'], v['y'], v.get('z', 0)) for v in verts_data]
        return PolygonObstacle(height=height, vertices=vertices)
    
    elif obs_type in ('circle', 'cylinder', 'point'):
        # 'point' is treated as a small circle/cylinder in legacy logic
        pos_data = data.get('position', {})
        center = Position(pos_data.get('x', 0), pos_data.get('y', 0), pos_data.get('z', 0))
        radius = float(data.get('radius', 0.0))
        return CylinderObstacle(height=height, center=center, radius=radius)
    
    elif obs_type == 'ellipse':
        pos_data = data.get('position', {})
        center = Position(pos_data.get('x', 0), pos_data.get('y', 0), pos_data.get('z', 0))
        major = float(data.get('major_axis', 0.0))
        minor = float(data.get('minor_axis', 0.0))
        orientation = float(data.get('orientation', 0.0))
        return EllipseObstacle(height=height, center=center, major_axis=major, minor_axis=minor, orientation=orientation)
    
    return None
