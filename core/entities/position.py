from dataclasses import dataclass
from typing import Dict, Any
import math

@dataclass
class Position:
    """
    Represents a 3D position in the world.
    Value Object.
    """
    x: float
    y: float
    z: float

    def distance_to(self, other: 'Position') -> float:
        """Calculate Euclidean distance to another position."""
        return math.sqrt(
            (self.x - other.x)**2 + 
            (self.y - other.y)**2 + 
            (self.z - other.z)**2
        )
    
    def distance_2d(self, other: 'Position') -> float:
        """Calculate 2D Euclidean distance (ignoring z)."""
        return math.sqrt(
            (self.x - other.x)**2 + 
            (self.y - other.y)**2
        )

    def to_dict(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Position':
        return cls(
            x=float(data.get("x", 0.0)),
            y=float(data.get("y", 0.0)),
            z=float(data.get("z", 0.0))
        )
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Position):
            return NotImplemented
        # Use a small epsilon for float comparison if needed, 
        # but pure equality is often better for Value Objects unless fuzzy match is explicitly needed.
        # Here we stick to exact equality for simplicity, or use math.isclose if strictness causes issues.
        return (math.isclose(self.x, other.x, abs_tol=1e-9) and 
                math.isclose(self.y, other.y, abs_tol=1e-9) and 
                math.isclose(self.z, other.z, abs_tol=1e-9))
